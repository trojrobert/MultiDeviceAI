# MultiDeviceLLM

> **Every device brings a slice. Together they run the whole model.**

Run **one** large language model across **all** your devices — phone, laptop,
tablet, spare mini-PC — by pooling their memory into a single virtual machine.
A model that's too big for any one device can run when the devices cooperate.

MultiDeviceLLM is a friendly control layer on top of
[**exo**](https://github.com/exo-explore/exo), an open-source distributed
inference engine. exo does the hard part (peer-to-peer discovery, topology-aware
sharding, MLX-distributed execution); MultiDeviceLLM gives you:

- **`mdllm doctor`** — one command to verify prerequisites.
- **`mdllm bootstrap`** — clone + build the engine automatically.
- **`mdllm up`** — join this device to the cluster.
- **`mdllm status`** — see every connected device and total pooled memory.
- **`mdllm fit MODEL`** — *"can my cluster run this model, and how should it split?"*
- **`mdllm run MODEL`** — load a model across the cluster and chat, in one step.

> **What this buys you:** *capacity*, not raw speed. Splitting a model across a
> phone and a laptop lets you run a model that wouldn't fit on either one alone.
> Network latency between devices means it won't be faster than a single device
> that already fits the model — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Requirements

- **Best experience:** Apple Silicon Macs (exo uses Apple's MLX backend / Metal GPU).
- **Also works:** Linux nodes (currently CPU-only in exo), and iPhone/iPad via the
  exo app. See [docs/DEVICES.md](docs/DEVICES.md) for the exact, per-device path
  and current caveats.
- All devices must be on the **same local network**.
- Enough **combined** memory across devices to hold the whole model.

## Install the control CLI

```bash
cd MultiDeviceLLM
uv sync                 # or: pip install -e .
uv run mdllm --help     # or just: mdllm --help
```

## Quick start (2 Macs, or Mac + supported phone)

```bash
# 1. On your main laptop: check tools, then build the engine
mdllm doctor
mdllm bootstrap          # clones + builds exo into ./vendor/exo

# 2. Create a private "room" and start the cluster on the laptop
mdllm room new           # prints a shareable code, e.g. swift-otter-4821
mdllm up                 # dashboard + API at http://localhost:52415

# 3. On your other device (see docs/DEVICES.md), join the same room:
mdllm up --room swift-otter-4821    # auto-discovers the laptop, no other config

# 4. Back on the laptop, confirm both devices joined and see pooled memory
mdllm status

# 5. Check a model fits, then run + chat with it across the cluster
mdllm fit llama-3.2-1b
mdllm run llama-3.2-1b
```

### Rooms — share a code, form a swarm

Inspired by [SwarmLLM](https://github.com/Nehanth/swarmllm)'s room model. A
"room" is a private cluster: only devices using the same code join together, so
you can share a network with others without accidentally merging swarms.

```bash
mdllm room new                  # create + activate a room (auto-generated code)
mdllm room new my-swarm         # or pick your own code
mdllm room show                 # show the active room
mdllm room join swift-otter-42  # save a code someone shared with you
mdllm up                        # start; uses the saved room automatically
mdllm room clear                # go back to open LAN clustering
```

Under the hood a room maps to exo's `EXO_LIBP2P_NAMESPACE`.

## Commands

| Command | What it does |
|---|---|
| `mdllm doctor` | Check prerequisites (git, uv, node, rust, Xcode, macmon). |
| `mdllm bootstrap [--dry-run]` | Clone + build the exo engine into `./vendor/exo`. |
| `mdllm room new/show/join/clear` | Manage shareable swarm room codes. |
| `mdllm up [--room CODE] [--no-worker] [--dry-run]` | Start this device as a cluster node. |
| `mdllm status` | Show connected devices + pooled memory + loaded models. |
| `mdllm models [--search Q] [--downloaded]` | List / search models. |
| `mdllm fit MODEL` | Show whether/how the cluster can run a model. |
| `mdllm load MODEL [-c N]` | Load a model across the cluster. |
| `mdllm chat MODEL` | Chat with an already-loaded model. |
| `mdllm run MODEL [-c N]` | Load + chat in one step. |

## How it compares

Several projects split or pool a model across devices. They mainly differ in
**what you install** and **where the model runs**.

| | Approach | Devices | Install | Network |
|---|---|---|---|---|
| **MultiDeviceLLM** | control layer + rooms over **exo** | Apple Silicon Macs (best), iPhone/iPad, Linux/Pi | one-command bootstrap (Python/MLX) | same LAN (rooms isolate swarms) |
| [exo](https://github.com/exo-explore/exo) | slice of layers, MLX-distributed | machines running Python + MLX/tinygrad | Python package per node | one network |
| [SwarmLLM](https://github.com/Nehanth/swarmllm) | slice of layers, from-scratch WebGPU + WebRTC | any device with a WebGPU browser | **none — open a URL** | LAN *or* across the internet |
| llama.cpp `rpc-server` | slice of layers | machines running the binary | binary + open port per node | LAN in practice |
| [Petals](https://github.com/bigscience-workshop/petals) | slice of layers | server GPUs in a public swarm | Python client/server | public internet swarm |
| Ollama / LM Studio | the whole model | one machine per request | native app | no splitting |

**Why we chose exo:** best-in-class on Apple Silicon (MLX + Metal GPU), true
peer-to-peer with no master node, and an OpenAI-compatible API. If zero-install
and browser/phone reach matter more to you than raw Mac performance, SwarmLLM's
WebGPU approach is well worth a look — see the note at the bottom.

## Configuration

Everything is overridable via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MDLLM_EXO_URL` | `http://localhost:52415` | exo API/dashboard address. |
| `MDLLM_EXO_DIR` | `./vendor/exo` | Where the engine is cloned/built. |
| `MDLLM_EXO_REPO` | `https://github.com/exo-explore/exo` | Engine source repo. |
| `MDLLM_ROOM` | *(unset)* | Room code to use (overrides the saved room). |
| `MDLLM_HTTP_TIMEOUT` | `30` | Control-plane request timeout (seconds). |

## How it fits together

```
   your phone ┐
   your laptop ┼──  exo (P2P discovery + sharding + MLX-distributed)  ──►  OpenAI-compatible API :52415
   spare mini ┘                          ▲
                                          │
                              MultiDeviceLLM  (mdllm CLI: bootstrap, status, fit, run/chat)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and
[docs/DEVICES.md](docs/DEVICES.md) for per-device onboarding.

## An alternative worth knowing: the browser path

[SwarmLLM](https://github.com/Nehanth/swarmllm) runs models across devices
*entirely in browser tabs* using a from-scratch WebGPU engine and WebRTC — no
install, works on any OS with a WebGPU browser (including iPhone Safari), and
rooms can even span the public internet. It has demoed a 27B model across a
MacBook + iPhone. The trade-off is a younger engine vs. exo's mature MLX
backend. If "open a URL on every device" is your priority over peak Mac
throughput, that's the project to study. The **room** UX here is borrowed from it.

## Credits & license

- Distributed inference engine: [exo](https://github.com/exo-explore/exo) (Apache-2.0), by exo labs.
- Room UX inspiration: [SwarmLLM](https://github.com/Nehanth/swarmllm) (MIT), by Nehanth Narendrula.
- MultiDeviceLLM control layer: Apache-2.0.
