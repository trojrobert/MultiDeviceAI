# Architecture

## The core idea

A transformer model is a stack of layers. You don't need every layer in one
device's memory at the same time — you only need each layer *somewhere*, and you
need to pass the intermediate activations from one layer to the next.

**Pipeline sharding** exploits this: device A holds layers 0–15, device B holds
layers 16–31. A token's activations flow A → B → (sample) → repeat. Each device
only needs enough memory for *its* layers. Pool two 8 GB devices and you can hold
a ~16 GB model that fits on neither alone.

That is what MultiDeviceLLM gives you via exo: **capacity through pooling.**

## Why it's about capacity, not speed

Between layers, activations must cross the network. On a home Wi-Fi/LAN, that hop
costs milliseconds; on Apple's Thunderbolt-5 RDMA it can be microseconds. Either
way, a model that already fits on a single device will run *faster* on that
single device, because you avoid the hop entirely.

So the rule of thumb:

- **Model fits on one device already?** Run it there. Don't shard.
- **Model too big for any single device?** Pool devices — a slow run beats no run.

`exo` also supports **tensor parallelism** (splitting each layer across devices so
they compute simultaneously) for up to ~1.8× on 2 devices / ~3.2× on 4 — but only
over a fast link. On a phone-over-Wi-Fi link, pipeline sharding for capacity is
the realistic win. `mdllm fit` ranks placements with this bias (fewest hops).

## The stack

```
┌─────────────────────────────────────────────────────────────┐
│  MultiDeviceLLM  (this repo)                                  │
│  - mdllm CLI: doctor / bootstrap / up / status / fit / run   │
│  - exo API client + cluster fit planner                      │
└───────────────▲──────────────────────────────────────────────┘
                │ HTTP (localhost:52415)
┌───────────────┴──────────────────────────────────────────────┐
│  exo engine  (github.com/exo-explore/exo, Apache-2.0)         │
│  - libp2p peer-to-peer device discovery (no master node)     │
│  - topology-aware auto-partitioning (memory-weighted ring)   │
│  - MLX + MLX-distributed execution backend                   │
│  - OpenAI / Claude / Ollama-compatible API + web dashboard   │
└───────────────▲──────────────────────────────────────────────┘
                │ TCP / RDMA-over-Thunderbolt
        ┌───────┴───────┬───────────────┐
     laptop           phone          mini-PC        (auto-discovered peers)
```

## What MultiDeviceLLM adds on top of exo

exo's raw model-loading flow is powerful but multi-step:

1. `GET /instance/previews?model_id=…` — enumerate valid ways to shard.
2. `POST /instance` — create the instance for a chosen placement.
3. `GET /instance/await?model_id=…` — SSE stream until every shard is loaded.
4. `POST /v1/chat/completions` — finally chat.

`mdllm run MODEL` collapses all four into one command, choosing the best
placement automatically (see `planner.py`). `mdllm fit` exposes step 1 as a
plain-English "will it fit?" answer, and `mdllm status` reads `/state` to show
the live topology and pooled memory.

## Request lifecycle (`mdllm run llama-3.2-1b`)

```
mdllm ──previews──► exo ──► [placement A: laptop only]
                              [placement B: laptop+phone]   ◄─ picked if A doesn't fit
mdllm ──create instance──► exo ──► shards assigned to devices
mdllm ──await (SSE)──────► exo ──► each device loads its layers ──► "ready"
mdllm ──chat (stream)────► exo ──► tokens flow across the ring ──► deltas back to your terminal
```

## Design choices

- **No re-implementing exo's dashboard.** exo already ships a good web dashboard
  at `:52415`. MultiDeviceLLM focuses on CLI ergonomics, setup automation, and
  planning — the gaps around the engine, not the engine itself.
- **Liberal state parsing.** exo's `/state` schema evolves; `status` accepts
  several shapes (`nodes`/`topology`/`devices`) so it keeps working across
  versions.
- **Everything env-overridable.** The same CLI drives the hosting laptop and any
  worker node by changing `MDLLM_EXO_URL` / `MDLLM_EXO_DIR`.
