# MultiDeviceAI

A browser POC that physically splits a **Qwen3 Q8_0** model between two WebGPU
devices and chats through the combined model over WebRTC.

The host owns the embedding table, lower transformer layers, final norm, and LM
head. The worker owns the upper layers. For every token, a hidden state travels
host → worker → host over a reliable ordered WebRTC data channel. Weights and KV
caches stay on the device that owns their layers.

## Models

The host picks a model before assigning layers. Each choice is range-probed for
its real shape, then measured against both devices' memory budgets:

| Model | Download | Layers | Hidden | Verdict on a typical laptop + phone |
| --- | --- | --- | --- | --- |
| Qwen3 0.6B Q8_0 | 639 MB | 28 | 1024 | fits on one device |
| Qwen3 1.7B Q8_0 | 1.83 GB | 28 | 2048 | fits on one device |
| Qwen3 4B Q8_0 | 4.28 GB | 36 | 2560 | **needs both devices** |

4B is the point of the project: at roughly 4.05 GB of weights, KV caches, and
working buffers it does not fit in either browser alone, so the split stops
being a demonstration and becomes the only way to run it.

The catalogue is Q8_0-only. The loader repacks 2-D Q8_0 tensors and dequantizes
everything else to f32; there is no K-quant path, so a Q4_K entry would fail at
load.

> This is an experimental correctness-first POC. It supports exactly one host
> and one worker, greedy decoding, a 256-token context, and a 64-token answer.
> It has not yet been validated against golden logits on every GPU/browser, and
> weights are **not cached** — every session re-downloads its shard.

## Try it on a laptop first

```bash
npm install
npm run dev
```

Open the printed localhost URL in a WebGPU-capable browser. The application is
the room/chat interface; there is no longer a vector-add demo.

## Deploy it

To use the app across two physical devices you must serve it over **HTTPS**.
WebGPU is only available in a secure context: `localhost` counts as secure on
the laptop, but a phone opening `http://<laptop-ip>:5173` does **not**, so plain
LAN dev will not give the phone WebGPU.

The app is a fully static Vite bundle. Model weights are range-fetched directly
from Hugging Face by each browser, so the host only serves ~170 KB of static
assets — any static HTTPS host works (Vercel, Netlify, Cloudflare Pages, GitHub
Pages, etc.). Cross-origin isolation (COOP/COEP) is **not** required by the app,
so no special response headers are needed.

Build the production bundle:

```bash
npm run build   # outputs dist/
```

### Deploy to Vercel (used for the reference deployment)

```bash
npm i -g vercel        # if the CLI is not installed
vercel login           # once
vercel deploy ./dist --prod --yes
```

The CLI prints a production URL such as
`https://<project>-<hash>-<team>.vercel.app`.

> **Important — turn off Deployment Protection.** New Vercel projects often
> enable "Vercel Authentication", which puts an SSO login wall in front of the
> deployment (requests return `302 → vercel.com/sso-api`). Your phone cannot get
> past it. Disable it under
> **Project → Settings → Deployment Protection → Vercel Authentication → Disabled**,
> then redeploy or simply reload. Verify the URL is public with:
>
> ```bash
> curl -sS -o /dev/null -w "%{http_code}\n" https://<your-deployment-url>
> # expect: 200 (not 302)
> ```

To ship code changes later, rebuild and redeploy:

```bash
npm run build && vercel deploy ./dist --prod --yes
```

### Deploy to any other static host

Upload the contents of `dist/` to the host of your choice. No server-side
runtime, environment variables, or custom headers are required. Just make sure
the deployment is served over HTTPS.

## Use it once deployed

Open the **same HTTPS URL** on both devices, in a WebGPU-capable browser
(Chrome/Edge on desktop and Android; Safari on iOS 26+ or with the WebGPU flag
enabled). Then:

1. **Laptop:** open the URL, click **Create room**, then **Copy invite**.
2. **Phone:** open the invite link (send it to yourself via chat/notes, or type
   the room code shown on the laptop).
3. **Phone:** tap **Join room**.
4. **Laptop:** pick a model. The card reports whether it fits on one device or
   needs both, and each device shows an editable **memory budget** — the browser
   exposes no real VRAM figure, so correct it if a load fails.
5. **Laptop:** choose a split (`14` means laptop layers 0–13 and phone layers
   14–27) and click **Assign & load**. **By device power** picks the split with
   the most headroom on the tighter device; **Balance** evens out the two
   downloads. A split that does not fit is refused, and the button says which
   device is short.
6. Each peer range-downloads only its assigned Qwen tensors. For 0.6B at a 14/14
   split the host downloads roughly 381 MB and the worker roughly 223 MB (the
   host is larger because it also owns embedding/head weights); for 4B at the
   recommended split it is roughly 2.2 GB and 1.8 GB. Use Wi-Fi.
7. Wait until both model cards say **ready**, then chat from either device. Use
   **Stop** to interrupt generation and **Reset** to clear the conversation.

Both devices must run the **same build**: the room protocol is versioned, and a
peer on a cached older bundle is rejected with a message telling you to reload,
rather than silently loading a different model.

PeerJS provides signaling; model activations travel directly over WebRTC. The
default setup has public STUN but no TURN service, so strict NAT/firewall pairs
(e.g. phone on cellular, laptop behind a restrictive router) may fail to
connect. Putting **both devices on the same Wi-Fi** is the most reliable path.

## Token path

```text
host
  tokenize → embedding → layers 0..split-1
       │
       └── hidden state over WebRTC ──► worker layers split..L-1
                                             │
host ◄──────── returned hidden state ─────────┘
  final norm → LM head → greedy sample → next token → repeat
```

During prompt prefill, both peers build the KV caches for only their own layers.
The host runs the LM head only after the final prompt token and each generated
token.

## Architecture

```text
src/engine/
  gguf.ts             GGUF metadata parsing and concurrent range tensor loading
  quant.ts            Q8_0 repacking/dequantization
  headChunks.ts       row-chunking so a big LM head fits WebGPU binding limits
  tokenizer.ts        byte-level BPE and Qwen chat template
  wgsl/               dense Qwen WebGPU kernels
  denseEngine.ts      layer-range execution and local KV caches
  model.ts            model catalogue, GGUF profiling, role-aware shard loader
  placement.ts        memory budgets, feasibility verdicts, split selection
  runtimeAdapter.ts   chat/runtime integration and generation loop

src/runtime/
  capabilities.ts     shared WebGPU capability and memory-budget probing
  peer.ts             PeerJS signaling and WebRTC transport
  protocol.ts         versioned controls and binary f32 activation frames
  room.ts             two-peer assignment, readiness, and request correlation

src/ui/
  app.ts              room controls and shared chat rendering
  styles.css          responsive laptop/phone interface
```

More design details are in `system_architecture.md`.

## Verification

```bash
npm run verify
```

This runs strict TypeScript checking, CPU tests for GGUF/Q8/tokenization, the
shard loader, head chunking, the model catalogue, capacity placement and the
wire protocol, then a production build. Browser GPU execution must be tested on
real hardware.

There is also a two-browser transport smoke test, which needs a running
preview server:

```bash
npm run preview &
npm run test:e2e
```

## Current limitations

- Exactly two peers; no reconnect recovery or automatic multi-device planner.
- Qwen3 0.6B Q8_0 only.
- Tensor loading is range-based but does not yet persist weights in Cache API.
- Decode performs CPU readback each token and has no batched prefill or
  speculative decoding.
- No TURN credentials are bundled.
- Activations are sent as exact f32 for correctness, not compressed f16.

## License

Apache-2.0 for MultiDeviceAI code.
