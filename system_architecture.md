# System architecture

## Product boundary

MultiDeviceAI is a browser application that runs one model across cooperating
WebGPU devices. It does not route independent whole-model requests. Each peer
owns a contiguous transformer-layer range and keeps those weights and KV caches
locally; intermediate hidden states move between peers over WebRTC.

## Roles in the first POC

The POC supports exactly two compute peers:

- **Host:** owns the tokenizer, embedding table, lower transformer layers,
  final normalization, language-model head, sampler, generation loop, model
  selection, and room authority.
- **Worker:** owns the upper transformer layers and transforms hidden states
  received from the host.

For a model with `L` layers and split point `S`:

```text
host:   embedding + layers [0, S) + final norm/head
worker: layers [S, L)
```

The host deliberately keeps both model edges. This makes token lookup and
sampling local while only a hidden vector crosses the network in each
direction.

## Engine boundary

The engine has no room or UI knowledge. A layer-range engine exposes operations
equivalent to:

- embed a token and run the local lower-layer range;
- run a received hidden state through the owned range;
- run final normalization and the LM head on a returned hidden state;
- reset local sequence/KV-cache position.

Weights are loaded according to role. A worker must never fetch embedding,
final-normalization, or LM-head tensors. The host must never fetch worker-owned
layer tensors.

## Generation protocol

For each prompt or generated token at position `p`:

1. Host embeds the token and runs layers `[0, S)`.
2. Host sends the resulting hidden state and `p` to the worker.
3. Worker runs layers `[S, L)` using its local KV caches.
4. Worker returns the resulting hidden state and `p`.
5. During prefill, the host advances without running the LM head.
6. For the final prompt token and every decode token, the host runs the head,
   samples the next token, and broadcasts transcript state.

Requests carry IDs and positions so stale or mismatched responses fail loudly.
The first POC uses ordered, reliable WebRTC delivery and permits one in-flight
hidden-state request.

## Model and format

Models are dense Qwen3 Q8_0 GGUFs, selected from a catalogue at run time. The
format is fixed: the loader repacks 2-D Q8_0 tensors and dequantizes everything
else to f32, so there is no K-quant path.

Nothing may assume a model's shape. Layer count, hidden size, and per-tensor
byte sizes are read from the chosen GGUF's header before layers are assigned,
because they differ per model (28 layers and hidden 1024 at 0.6B; 36 and 2560 at
4B). The assignment carries the model URL, hidden size, and a fingerprint so the
two peers cannot end up on different models.

Qwen3 ties the language-model head to the embedding table, which makes it the
only tensor large enough to exceed `maxStorageBufferBindingSize` — 148 MiB of
quants at 0.6B but 371 MiB at 4B, against a 256 MiB default. The head is
therefore uploaded as row chunks, each with its own weight buffers and its own
slice of the logits buffer. No per-layer tensor comes close to the limit.

## Capacity

Supporting a model larger than one device is the capacity milestone, and Qwen3
4B Q8_0 meets it: roughly 4.05 GB of weights, KV caches, and working buffers,
which fits in neither browser alone.

Placement is computed, not guessed. Each device reports a memory budget, and for
every legal split the host estimates both roles' footprints and reports one of
three verdicts: the model fits on one device, it needs both, or it fits neither.
The recommended split maximises the tighter device's headroom rather than
filling one device to its ceiling, since these budgets are estimates.

The browser exposes no real VRAM figure. `maxBufferSize` is a per-allocation
limit, not a capacity, and `navigator.deviceMemory` is coarse and
Chromium-only. The budget is therefore presented as an editable estimate with
its source labelled, rather than as a measurement.

## Security and deployment

WebGPU is available only in secure browser contexts. `localhost` is treated as
secure on the laptop, but a phone opening `http://<laptop-ip>` is not. Real
two-device tests therefore require HTTPS with a trusted certificate or a hosted
HTTPS deployment.

PeerJS performs signaling only. Inference payloads travel over the negotiated
WebRTC data connection. Participants can observe shared prompts, responses, and
their local activations; the POC does not claim protection from a malicious
room peer.

## Correctness policy

- GPU kernels are tested against CPU references.
- Whole-model and split-model paths must produce equivalent logits within a
  declared tolerance.
- WGSL compilation messages and WebGPU validation errors are surfaced.
- Generation starts with greedy sampling for deterministic comparison.
- Performance work begins only after real-model golden correctness.

