# Adding devices to your cluster

All devices must be on the **same local network**. exo discovers peers
automatically over libp2p — there is **no master node and no manual config**.
As long as each device is running exo, it joins the pool.

> **Reality check (2026):** exo's best-supported target today is **Apple Silicon
> Macs** (MLX + Metal GPU). Other device types work with caveats noted below.
> When in doubt, start with two Macs, confirm the flow, then add other devices.

---

## macOS (Apple Silicon) — recommended

**Option A — from source (what `mdllm bootstrap` does):**

```bash
mdllm doctor        # verify git, uv, node, rust, Xcode, macmon
mdllm bootstrap     # clone + build exo into ./vendor/exo
mdllm up            # join the cluster
```

**Option B — the official macOS app (background menu-bar app):**

```bash
brew install --cask exo
```

Requires macOS Tahoe 26.2+. The app runs exo in the background and asks to
install a network profile. To keep this Mac isolated to *your* cluster, set a
custom namespace in the app's Advanced settings (or `EXO_LIBP2P_NAMESPACE`).

**Faster links (optional):** Macs with Thunderbolt 5 can enable RDMA for ~99%
lower inter-device latency. Boot into Recovery, open Terminal, run `rdma_ctl
enable`, reboot. All devices in an RDMA cluster must be cabled to each other with
TB5 cables and run the *exact* same macOS version. See exo's README for caveats.

---

## iPhone / iPad

exo has historically supported iOS/iPadOS as cluster members. Support and the
distribution channel change frequently, so:

1. Check the [exo repo](https://github.com/exo-explore/exo) and
   [exo labs](https://x.com/exolabs) for the current iOS app / TestFlight link.
2. Install it on the phone/tablet, join the **same Wi-Fi** as your laptop.
3. Open the app so exo is running; it auto-discovers the laptop.
4. On the laptop: `mdllm status` — the phone should appear with its memory.

Caveats: iOS gives apps a limited memory budget, so a phone contributes a
smaller shard than its total RAM. It's great as an *extra* pool member, less so
as the main workhorse. If your phone doesn't yet have a working exo build, run
the cluster across your Mac(s) and treat the phone as a future add-on.

---

## Linux (laptop / mini-PC / server)

Works as a cluster member, but exo currently runs **CPU-only** on Linux (GPU
support in progress). Great as a memory donor / coordinator.

```bash
# prerequisites: uv, node (>=18), rust nightly
git clone https://github.com/exo-explore/exo
cd exo/dashboard && npm install && npm run build && cd ..
uv sync --extra mlx-cpu
uv run exo                 # or `uv run exo --no-worker` for coordinator-only
```

Set `MDLLM_EXO_DIR` to this checkout and you can still drive it with `mdllm up`.

---

## Raspberry Pi / SBCs

Low-memory ARM boards can join as small pool members (CPU inference). Follow the
Linux instructions above (`--extra mlx-cpu`). Expect them to hold a small number
of layers; useful for squeezing a model over the line, not for speed.

---

## Verifying the cluster

On the hosting laptop:

```bash
mdllm status
```

You should see each device with its contributed memory and the **total pooled
memory**. Then:

```bash
mdllm fit <model>     # does it fit? which split is best?
mdllm run <model>     # load across the cluster + chat
```

## Isolating your cluster

Sharing a network with other exo users? Give every one of *your* devices the
same namespace so they only cluster with each other:

```bash
EXO_LIBP2P_NAMESPACE=roberts-cluster mdllm up
```

## Troubleshooting

- **Devices don't see each other:** confirm same subnet/Wi-Fi, no client
  isolation on the router, and the same `EXO_LIBP2P_NAMESPACE` (or none) on all.
- **Version handshake errors:** pin all devices to the same exo commit
  (`cd vendor/exo && git pull` on each, then rebuild).
- **Model won't fit:** `mdllm fit MODEL` will say so — add a device or pick a
  smaller / more-quantized model (e.g. a `-4bit` variant).
