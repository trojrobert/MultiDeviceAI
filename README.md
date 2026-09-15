# LocalClusterAI

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
| Qwen3 4B Instruct 2507 Q8_0 | 4.28 GB | 36 | 2560 | **needs both devices** |
| Qwen3 4B Thinking 2507 Q8_0 | 4.28 GB | 36 | 2560 | **needs both devices** |

4B is the point of the project: at roughly 4.05 GB of weights, KV caches, and
working buffers it does not fit in either browser alone, so the split stops
being a demonstration and becomes the only way to run it.

The two 2507 fine-tunes are pulled from unsloth, which publishes them as
single-file Q8_0 GGUFs. They share the 4B geometry exactly, so they split the
same way. Their headers advertise a 262144 context, but the engine clamps to
`min(512, contextLength)` like every other entry.

The catalogue is Q8_0-only. The loader repacks 2-D Q8_0 tensors and dequantizes
everything else to f32; there is no K-quant path, so a Q4_K entry would fail at
load. Entries must also be Qwen3 proper: the layer loader requires the
`attn_q_norm`/`attn_k_norm` tensors, so a Llama- or Gemma-architecture GGUF
would fail even at Q8_0.

> This is an experimental correctness-first POC. It supports exactly one host
> and one worker, greedy decoding, a 256-token context, and a 64-token answer.
> It has not yet been validated against golden logits on every GPU/browser.

## Try it on a laptop first

```bash
npm install
npm run dev
```

Open the printed localhost URL in a WebGPU-capable browser. The application is
the cluster/chat interface; there is no longer a vector-add demo.

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

1. **Laptop:** open the URL, click **Create cluster**, then **Copy invite**.
2. **Phone:** open the invite link (send it to yourself via chat/notes, or type
   the cluster code shown on the laptop).
3. **Phone:** tap **Join cluster**.
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

Weights are cached, so step 6 is a one-time cost per device per model. The
**Weights** card shows how much of this device's shard is stored; a later
session restores it instead of downloading, and the status line says
"restored from cache" rather than "downloading". The **Performance** card
appears with the first token.

Both devices must run the **same build**: the cluster protocol is versioned, and a
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

## Weight caching

Every tensor is stored in Cache Storage under its `(model URL, tensor name,
byte range)` key, so a second session restores its shard instead of
re-downloading it — at 4B that is roughly 2.2 GB on the host and 1.8 GB on the
worker saved per session. Parsed GGUF headers are cached the same way, which
also makes the model picker's probe instant after the first look.

Keying on the byte range means a re-published GGUF that moves a tensor misses
rather than returning the wrong bytes, and the loader re-checks every entry's
length before use, so a truncated or partially evicted entry is re-fetched
rather than handed to the engine. The app asks for persistent storage so a
shard is not evicted because a background tab needed room.

Caching degrades quietly rather than failing: an insecure origin, private
browsing, or a full quota all just mean tensors are fetched as before. The
**Weights** card says which of those you are in, and can clear the store.

## Performance panel

Each token is attributed to four stages, so the cost of splitting is visible
rather than assumed:

| Stage | What it covers |
| --- | --- |
| Host layers | Embedding and the host's own layer range |
| Network | The round trip minus the peer's reported compute |
| Peer layers | The worker's layer range, timed on the worker itself |
| Head + sample | Final norm, LM head, and sampling |

The worker times its own layers and reports that figure in the activation
response, which is what protocol v3 widened the frame header for. Without it
the host can only see a round trip and cannot tell a slow peer from a slow
link — the one distinction that decides whether a given split is worth making.

The panel also reports tokens per second, time to first token, per-token
latency over a recent window, and activation bytes per token in each direction.
The worker sees its own panel, fed by the frames it answered.

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
  weightCache.ts      persistent per-tensor store, keyed by model and byte range
  runtimeAdapter.ts   chat/runtime integration and instrumented generation loop

src/runtime/
  capabilities.ts     shared WebGPU capability and memory-budget probing
  metrics.ts          per-token stage accounting and run summaries
  peer.ts             PeerJS signaling and WebRTC transport
  protocol.ts         versioned controls and binary f32 activation frames
  cluster.ts             two-peer assignment, readiness, and request correlation

src/ui/
  app.ts              cluster controls and shared chat rendering
  styles.css          responsive laptop/phone interface
```

More design details are in `system_architecture.md`.

## Verification

```bash
npm run verify
```

This runs strict TypeScript checking, CPU tests for GGUF/Q8/tokenization, the
shard loader, head chunking, the model catalogue, capacity placement, the
weight cache, per-token metrics and the wire protocol, then a production build.
Browser GPU execution must be tested on real hardware.

There are also two-browser smoke tests, which need a running preview server:

```bash
npm run preview &
npm run test:e2e           # cluster creation, pairing, capability exchange
npm run test:e2e:metrics   # v3 activation frames and the performance panel
```

The metrics test stubs the engine at the cluster's boundary, so it exercises real
WebRTC frames, the worker's compute reporting, and the panel without needing a
GPU or a multi-gigabyte download.

## Current limitations

- Exactly two peers; no reconnect recovery or automatic multi-device planner.
- Decode performs CPU readback each token and has no batched prefill or
  speculative decoding, so prefill costs one round trip per prompt token.
- No TURN credentials are bundled.
- Activations are sent as exact f32 for correctness, not compressed f16.

## License

Apache-2.0 for LocalClusterAI code.
