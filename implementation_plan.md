# End-to-End Split-Model Chat POC

## Goal

Replace the WebGPU vector-add demo with a working browser chat application where
two devices join the same room, each loads a different contiguous range of a
real model's transformer layers, and token generation passes hidden states
between them over WebRTC.

The first supported model will be **Qwen3 0.6B Q8_0**. It is small enough to
debug and download repeatedly, but uses the same dense transformer architecture,
GGUF quantization, layer sharding, and activation transport needed by larger
Qwen3 models. The POC proves physical model splitting; it is not intended to
prove that 0.6B needs pooled memory.

## POC boundaries

- Exactly two compute peers: one host and one worker.
- Host owns embedding, the first half of transformer layers, final norm, LM
  head, tokenizer, sampler, and chat UI.
- Worker owns the second half of transformer layers.
- Each browser range-fetches and uploads only the tensors it owns.
- Hidden states cross a reliable ordered WebRTC data channel as binary data.
- Both devices display room state and the shared chat transcript.
- Generation is greedy initially for deterministic debugging.
- Context and output lengths are deliberately small for the first POC.
- No speculative decoding, multi-token batching, automatic N-device placement,
  TURN deployment, or reconnect recovery in this pass.

## Runtime flow

1. Host creates a room and shows a shareable URL/code.
2. Phone opens the HTTPS URL and joins as worker.
3. Peers exchange WebGPU limits and memory estimates.
4. Host assigns `[0, split)` to itself and `[split, layerCount)` to worker.
5. Both range-fetch GGUF metadata and only their owned tensors.
6. Host tokenizes the chat prompt and performs embedding + host layers.
7. Host sends `{requestId, position, hiddenState}` to worker.
8. Worker runs its layers and sends the resulting hidden state back.
9. Host runs final norm + LM head, greedily samples, decodes, and repeats.
10. Token and status events update both chat interfaces.

## Files

### Project shell and UI

- `[MODIFY] package.json`
  - Add PeerJS client dependency.
  - Add scripts for LAN hosting and verification.
- `[MODIFY] vite.config.ts`
  - Keep isolation headers.
  - Support an HTTPS/LAN development path; document that phone WebGPU requires
    a secure origin.
- `[REPLACE] index.html`
  - Chat application shell: room controls, peer/device status, model loading
    progress, transcript, prompt composer, stop/reset controls.
- `[REPLACE] src/main.ts`
  - Bootstrap the app controller instead of running vector addition.
- `[NEW] src/ui/app.ts`
  - UI rendering and event wiring.
- `[NEW] src/ui/styles.css`
  - Responsive phone/laptop chat layout.

### Model engine

- `[MODIFY] src/engine/device.ts`
  - Request large-buffer/storage limits from the adapter, report capabilities,
    and expose structured device-loss/validation errors.
- `[NEW] src/engine/types.ts`
  - Model config, layer range, weight-entry, progress, and engine interfaces.
- `[NEW] src/engine/wgsl/base.ts`
  - Dense Qwen kernels: f32/Q8 matvec, RMSNorm, Q/K head norm, RoPE,
    attention scores/softmax/output, SiLU gate, and residual add.
- `[NEW] src/engine/wgsl/cooperative.ts`
  - Cooperative Q8 GEMV used by decode.
- `[NEW] src/engine/buffers.ts`
  - Mapped initialization, staging readback, alignment, and safe destruction.
- `[NEW] src/engine/quant.ts`
  - Q8_0 repacking/dequantization helpers.
- `[NEW] src/engine/gguf.ts`
  - GGUF header parsing, tensor-name mapping, HTTP range loading, Q8_0
    repacking, and tokenizer/config extraction.
- `[NEW] src/engine/tokenizer.ts`
  - Qwen tokenizer encoding, chat-template application, and incremental decode.
- `[NEW] src/engine/sampling.ts`
  - Deterministic greedy sampling and stop-token handling.
- `[NEW] src/engine/denseEngine.ts`
  - Layer-range-aware Qwen3 execution with local KV caches.
  - `prefillToken`, `runHidden`, and `headFromHidden` split-mode APIs.
- `[NEW] src/engine/model.ts`
  - Qwen3 0.6B model URL, memory estimate, layer split, and loader orchestration.
- `[DELETE] src/engine/ops/vectorAdd.ts`
- `[DELETE] src/engine/shaders/vectorAdd.ts`

### Peer-to-peer room

- `[NEW] src/runtime/protocol.ts`
  - Versioned JSON control messages and binary hidden-state frame encoding.
- `[NEW] src/runtime/peer.ts`
  - PeerJS signaling and reliable ordered WebRTC data-channel lifecycle.
- `[NEW] src/runtime/room.ts`
  - Host/join roles, capability exchange, model assignment, readiness, transcript
    broadcast, and request/response correlation.
- `[NEW] src/runtime/generation.ts`
  - Distributed prompt prefill and autoregressive generation loop.

### Tests and documentation

- `[DELETE] tests/vectorAdd.test.ts`
- `[NEW] tests/protocol.test.ts`
  - Binary hidden-state and control-message round trips.
- `[NEW] tests/tokenizer.test.ts`
  - Chat-template and known-token checks.
- `[NEW] tests/quant.test.ts`
  - Q8_0 repacking/dequantization against CPU fixtures.
- `[NEW] tests/split.test.ts`
  - Synthetic small transformer comparison: whole-engine output versus two
    contiguous engine slices.
- `[NEW] system_architecture.md`
  - Durable description of engine/room boundaries and generation protocol.
- `[MODIFY] README.md`
  - POC setup, HTTPS requirement, two-device testing instructions, architecture,
    model download size, and limitations.

## Correctness gates

1. Every WGSL module compiles with zero errors on Apple/Metal.
2. Individual kernels match CPU references within declared numeric tolerance.
3. A synthetic full model and the same model split into two layer ranges produce
   equivalent logits.
4. Protocol tests preserve every f32 hidden-state value and request identifier.
5. Real Qwen3 0.6B local single-device logits match a known golden fixture.
6. Two browser contexts complete model load and produce a deterministic response.
7. Typecheck, unit tests, production build, and browser end-to-end smoke test pass.

## Testing on phone and laptop

WebGPU requires a secure browser context. `localhost` works only on the laptop;
opening a plain `http://<laptop-ip>` URL on the phone is not sufficient. The
POC must therefore be tested from an HTTPS deployment (or a locally trusted
HTTPS certificate). Both devices open the same deployed application; PeerJS is
used only for signaling, while hidden states travel directly over WebRTC.

