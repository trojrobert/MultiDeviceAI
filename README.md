# MultiDeviceAI

A browser POC that physically splits **Qwen3 0.6B Q8_0** between two WebGPU
devices and chats through the combined model over WebRTC.

The host owns the embedding table, lower transformer layers, final norm, and LM
head. The worker owns the upper layers. For every token, a 1,024-float hidden
state travels host → worker → host over a reliable ordered WebRTC data channel.
Weights and KV caches stay on the device that owns their layers.

> This is an experimental correctness-first POC. It supports exactly one host
> and one worker, greedy decoding, a 256-token context, and a 64-token answer.
> It has not yet been validated against golden logits on every GPU/browser.

## Try it on a laptop first

```bash
npm install
npm run dev
```

Open the printed localhost URL in a WebGPU-capable browser. The application is
the room/chat interface; there is no longer a vector-add demo.

## Try it with a phone and laptop

Both devices must open the **same HTTPS deployment**. WebGPU works on
`localhost`, but a phone opening `http://<laptop-ip>:5173` is not a secure
context and normally cannot use WebGPU.

One simple path is to deploy the static Vite application to any HTTPS host:

```bash
npm run build
# deploy the generated dist/ directory to Vercel, Netlify, Cloudflare Pages, etc.
```

Then:

1. Open the HTTPS URL on the laptop and choose **Create room**.
2. Copy the invite link and open it on the phone.
3. On the phone, choose **Join room**.
4. On the laptop, choose a split (14 means laptop layers 0–13 and phone layers
   14–27) and click **Assign & load**.
5. Each peer range-downloads only its assigned Qwen tensors. With a 14/14 split,
   the host downloads roughly 381 MB and the worker roughly 223 MB; the host is
   larger because it also owns embedding/head weights.
6. Wait until both model cards say ready, then chat from either device.

PeerJS provides signaling; model activations travel directly over WebRTC. The
default setup has public STUN but no TURN service, so strict NAT/firewall pairs
may fail to connect.

## Token path

```text
host
  tokenize → embedding → layers 0..split-1
       │
       └── hidden state over WebRTC ──► worker layers split..27
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
  gguf.ts             GGUF metadata parsing and HTTP range tensor loading
  quant.ts            Q8_0 repacking/dequantization
  tokenizer.ts        byte-level BPE and Qwen chat template
  wgsl/               dense Qwen WebGPU kernels
  denseEngine.ts      layer-range execution and local KV caches
  model.ts            role-aware Qwen shard loader
  runtimeAdapter.ts   chat/runtime integration and generation loop

src/runtime/
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

This runs strict TypeScript checking, CPU tests for GGUF/Q8/tokenization and the
wire protocol, then a production build. Browser GPU execution must be tested on
real hardware.

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
