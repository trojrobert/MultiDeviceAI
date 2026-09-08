# MultiDeviceLLM

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

# 2. Start the cluster on the laptop
mdllm up                 # dashboard + API at http://localhost:52415

# 3. On your other device, install exo and start it too (see docs/DEVICES.md).
#    It auto-discovers the laptop — no config needed.

# 4. Back on the laptop, confirm both devices joined and see pooled memory
mdllm status

# 5. Check a model fits, then run + chat with it across the cluster
mdllm fit llama-3.2-1b
mdllm run llama-3.2-1b
```

## Commands

| Command | What it does |
|---|---|
| `mdllm doctor` | Check prerequisites (git, uv, node, rust, Xcode, macmon). |
| `mdllm bootstrap [--dry-run]` | Clone + build the exo engine into `./vendor/exo`. |
| `mdllm up [--no-worker] [--dry-run]` | Start this device as a cluster node. |
| `mdllm status` | Show connected devices + pooled memory + loaded models. |
| `mdllm models [--search Q] [--downloaded]` | List / search models. |
| `mdllm fit MODEL` | Show whether/how the cluster can run a model. |
| `mdllm load MODEL [-c N]` | Load a model across the cluster. |
| `mdllm chat MODEL` | Chat with an already-loaded model. |
| `mdllm run MODEL [-c N]` | Load + chat in one step. |

## Configuration

Everything is overridable via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MDLLM_EXO_URL` | `http://localhost:52415` | exo API/dashboard address. |
| `MDLLM_EXO_DIR` | `./vendor/exo` | Where the engine is cloned/built. |
| `MDLLM_EXO_REPO` | `https://github.com/exo-explore/exo` | Engine source repo. |
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

## Credits & license

- Distributed inference engine: [exo](https://github.com/exo-explore/exo) (Apache-2.0), by exo labs.
- MultiDeviceLLM control layer: Apache-2.0.
