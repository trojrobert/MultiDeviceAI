# Split-model chat POC execution

- [x] Adapt the Qwen3 dense WebGPU engine with explicit MIT attribution.
- [x] Adapt GGUF range loading, Q8_0 repacking, tokenizer, and sampling.
- [x] Add model-role loading for host and worker layer ranges.
- [x] Add the versioned WebRTC room protocol and PeerJS transport.
- [x] Add distributed prefill/decode orchestration.
- [x] Replace the vector demo with the responsive room/chat interface.
- [x] Add deterministic protocol and engine correctness tests.
- [x] Document secure-origin deployment and two-device testing.
- [ ] Run a real two-device browser smoke test (requires user WebGPU devices).
- [x] Run typecheck, tests, production build, and real GGUF parser validation.

