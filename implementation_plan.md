# Implementation plan — canvas-first interface

A full redesign, not a restyle. The previous interface was a chat column with a
permanent rail of six setting cards beside it; the cluster itself was never drawn.
The new one makes the cluster the surface and pushes every setting into a
contextual panel.

## What changed structurally

| Before | After |
| --- | --- |
| Left rail, always open, six stacked cards | Right inspector showing one group at a time |
| Four-step stepper card | The canvas itself shows the phase; status sits in the topbar |
| Layer ribbon inside a rail card | A fan of layer filaments on the canvas, dragged in place |
| Chat as a bordered column | Conversation floats over the canvas on a scrim |
| Invite panel replacing the chat area | Pairing card takes the canvas; the worker node is a ghost |
| Metrics card in the rail | Four live figures docked to the composer, detail in the inspector |
| No picture of the cluster | The cluster is the picture |

## The canvas

`src/ui/canvas.ts` owns one SVG in a fixed 1000×560 viewBox:

- Two device orbs at fixed points, each with a name, its layer count, and its
  download size.
- One filament per transformer layer, fanning out of the orb that owns it and
  converging on a divider between the two fans. Dim while the layer's tensors
  are pending, lit once they have arrived, so the fan is the split control and
  the loading progress at once.
- A gold arc over the top carrying activations out and a cyan arc under the
  bottom carrying hidden state back, their dashes travelling at the pace of the
  run.
- A draggable divider. Moving it moves layers between the devices.

The skeleton is built once and repainted by attribute. Rebuilding it per
snapshot would restart the arc animations several times a second during
generation.

## The inspector

Four groups — Devices, Model & split, Performance, Session — reached from the
dock chips or by clicking a node on the canvas. Exactly one group is visible;
a panel shows when its group is selected *and* its own data exists. Pressing
the chip that is already showing closes the panel and returns the width to the
canvas. On a narrow screen the inspector is a bottom sheet over the graph.

## Files

- `[NEW] src/ui/canvas.ts` — the graph: geometry, skeleton, repaint, and the
  pointer-to-split conversion through the SVG's own matrix.
- `[MODIFY] src/ui/app.ts` — new shell and render pass; inspector groups; the
  dock; pairing on the canvas; the floating reading layer.
- `[MODIFY] src/ui/styles.css` — new layout; canvas part styling; dock;
  inspector; panels.
- `[MODIFY] tests/e2e/cluster.mjs`, `tests/e2e/metrics.mjs` — open the inspector
  group before asserting a panel inside it is visible.

## Constraints honoured

Every element id the end-to-end drivers use is unchanged: `#create-cluster`,
`#join-cluster`, `#cluster-view`, `#cluster-code-display`, `#role-display`,
`#cluster-status`, `#join-status`, `#error-banner`, `#remote-name`,
`#remote-phase`, `#model-card`, `#model-select`, `#cache-card`, `#cache-state`,
`#assign-layers`, `#prompt-input`, `#composer-hint`, `#telemetry`,
`#metric-live`, `#metric-tokens`, `#metric-bytes`, `#stage-bar`,
`#stage-worker-share`, `#spark-line`. The only test change is that a panel now
has to be asked for before it is visible, which is what a user does too.
