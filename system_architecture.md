# System architecture

## Product boundary

LocalClusterAI is a browser application that runs one model across cooperating
WebGPU devices. It does not route independent whole-model requests. Each peer
owns a contiguous transformer-layer range and keeps those weights and KV caches
locally; intermediate hidden states move between peers over WebRTC.

## Roles in the first POC

The POC supports exactly two compute peers:

- **Host:** owns the tokenizer, embedding table, lower transformer layers,
  final normalization, language-model head, sampler, generation loop, model
  selection, and cluster authority.
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

The engine has no cluster or UI knowledge. A layer-range engine exposes operations
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

A response also carries the time the responding peer spent inside its own
layers. This is the only way the host can separate a slow peer from a slow
link: on its own, a round trip conflates the two, and that distinction is
exactly what decides whether a given split is worth making. The frame header
carries it rather than a control message so the figure cannot drift away from
the activation it describes.

## Weight persistence

Tensors are stored under a `(model URL, tensor name, byte range)` key, holding
raw GGUF bytes rather than repacked ones, so the store is independent of how a
tensor is later dequantized. Keying on the range means a re-published file that
moves a tensor misses instead of returning the wrong bytes.

The loader re-checks every cached entry's length before use. A cache is an
optimization and may be implemented by anything, so trusting one to return a
complete body would let a truncated entry reach the engine as silent garbage;
a miss only costs a re-fetch.

Caching is best-effort throughout. An insecure origin, private browsing, or an
exhausted quota each degrade to the uncached path rather than failing a load,
and a write failure disables further writes rather than costing a rejected
promise per tensor across a several-hundred-tensor shard.

## Measurement

Every token is attributed to four stages — host layers, network, peer layers,
head and sampling — where the network's share is the round trip's remainder
after the peer's reported compute. It is a residual rather than a direct
measurement, and is documented as such, because it is the number that answers
what the split costs.

The recorder is pure and its clock injectable, so the accounting is
unit-testable without a browser. Samples are coalesced before reaching the UI:
a 4B prefill is hundreds of samples arriving back to back, and re-rendering per
sample would cost more than the inference being measured.

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
The recommended split maximises the tighter device's headXCLUSTERPLACEHOLDERX rather than
filling one device to its ceiling, since these budgets are estimates.

The browser exposes no real VRAM figure. `maxBufferSize` is a per-allocation
limit, not a capacity, and `navigator.deviceMemory` is coarse and
Chromium-only. The budget is therefore presented as an editable estimate with
its source labelled, rather than as a measurement.

## Interface

The interface has one surface: the cluster drawn as a graph. Two device orbs sit
at fixed points with a fan of filaments between them, one per transformer
layer, belonging to whichever device owns it. An arc over the top carries
activations out and an arc under the bottom carries hidden state back. A
divider between the two fans is the split control, so the thing being
configured and the thing being looked at are the same object.

The fan is the loading progress as well as the control: a filament is dim while
its tensors are pending and lit once they have arrived.

The visual system is flat, neutral, and high-contrast. The ground is a single
near-black — no gradient, no ambient motion, no glass — because live text and a
live graph both sit on top of it and a tinted wash costs both of them
legibility. Surfaces are matte panels one step lighter, separated by hairlines
rather than glow. Type is a single grotesque throughout, including figures,
which use tabular numerals rather than a second typeface. Buttons are matte
pills, and exactly one filled accent button is on screen at a time.

Colour is reserved for meaning, and one palette is reused wherever the same
fact appears. An acid accent carries brand and primary action and is never used
for data. Ownership keeps two hues: host-owned layers and outbound activations
are amber, worker-owned layers and returning states are blue. The network is
coral because it is the cost the split adds, and the head is violet because it
belongs to neither layer range. The same four values drive the canvas, the
stage bar, its legend, and the device orbs.

Everything else floats on the canvas. A dock at the bottom carries four live
figures — throughput, first token, token count, and bytes per token — joined to
the prompt box, because those four are what decide whether a split was worth
making and the reader is already looking there. A conversation appears over the
graph on a scrim rather than inside a panel, and the graph fades well back so
no stroke crosses a line of text.

Settings live in a right-hand inspector holding the detail of whatever is
selected, in four groups: Devices, Model & split, Performance, Session. Exactly
one group is visible at a time, and a panel within it shows only when its own
data exists. Clicking a node on the canvas opens the group that describes it.
On a narrow screen the inspector becomes a sheet so the graph keeps the screen.

Pairing has no panel of its own: the second node does not exist yet, so the QR
card takes the canvas and the worker orb is drawn as a ghost until it joins.

`src/ui/canvas.ts` builds its SVG skeleton once per layer count and thereafter
repaints by attribute. Rebuilding per snapshot would restart the arc animations
several times a second during generation.

Every element id is part of the contract with `tests/e2e/*.mjs`, which drive the
app through the DOM. Presentation may be rearranged freely; ids may not be
renamed without updating those drivers. Because the inspector shows one group at
a time, a driver has to open a group before asserting that a panel inside it is
visible.

## Security and deployment

WebGPU is available only in secure browser contexts. `localhost` is treated as
secure on the laptop, but a phone opening `http://<laptop-ip>` is not. Real
two-device tests therefore require HTTPS with a trusted certificate or a hosted
HTTPS deployment.

PeerJS performs signaling only. Inference payloads travel over the negotiated
WebRTC data connection. Participants can observe shared prompts, responses, and
their local activations; the POC does not claim protection from a malicious
cluster peer.

## Correctness policy

- GPU kernels are tested against CPU references.
- Whole-model and split-model paths must produce equivalent logits within a
  declared tolerance.
- WGSL compilation messages and WebGPU validation errors are surfaced.
- Generation starts with greedy sampling for deterministic comparison.
- Performance work begins only after real-model golden correctness.

