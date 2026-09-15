/**
 * The cluster drawn as a graph, and the primary surface of the application.
 *
 * Two device nodes sit at fixed points with a fan of filaments between them —
 * one filament per transformer layer, belonging to whichever device owns it.
 * A divider between the fans is the split control: dragging it moves layers
 * across, so the thing being configured and the thing being looked at are the
 * same object.
 *
 * The skeleton is built once per layer count and then repainted by attribute,
 * never rebuilt. Rebuilding would restart the CSS animations on the flow arcs
 * every time a snapshot arrives, which during generation is several times a
 * second.
 */

/** Everything the drawing needs, derived by the caller from a cluster snapshot. */
export interface CanvasModel {
  layerCount: number;
  /** Layers owned by the host — proposed while unassigned, actual after. */
  hostLayers: number;
  /** How many of each side's layers have finished loading. */
  hostLoaded: number;
  workerLoaded: number;
  hostName: string;
  workerName: string;
  hostDetail: string;
  workerDetail: string;
  hostIsLocal: boolean;
  hostPhase: string;
  workerPhase: string;
  connected: boolean;
  generating: boolean;
  /** Whether the divider may be dragged. */
  interactive: boolean;
  selected?: NodeTarget;
}

export type NodeTarget = "host" | "worker" | "split";

/* Geometry. A fixed viewBox scaled to fit, so every figure below is in one
   coordinate system regardless of the window. */
const W = 1000;
const H = 560;
const CY = 260;
const ORB_R = 52;
const HOST_CX = 170;
const WORKER_CX = 830;
/** How far the divider may travel. Kept clear of both orbs' halos. */
const DIV_MIN = 300;
const DIV_MAX = 700;
/**
 * Vertical room a fan may spread into, and the widest gap between filaments.
 * The span is bounded well inside the flow arcs so a 36-layer fan never
 * reaches the arcs or their labels.
 */
const FAN_SPAN = 260;
const MAX_SPACING = 13;
/** Half-height of the split marker, and where its label sits above it. */
const DIV_REACH = 120;

export class ClusterCanvas {
  private layerCount = -1;
  private filaments: SVGPathElement[] = [];

  constructor(private readonly svg: SVGSVGElement) {
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    this.svg.innerHTML = skeleton();
  }

  /** Rebuilds the filament layer only, and only when the model changes shape. */
  private ensureFilaments(layerCount: number): void {
    if (this.layerCount === layerCount) return;
    this.layerCount = layerCount;
    const group = this.pick("filaments");
    group.replaceChildren();
    this.filaments = [];
    for (let i = 0; i < layerCount; i++) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "filament");
      group.append(path);
      this.filaments.push(path);
    }
  }

  paint(model: CanvasModel): void {
    this.ensureFilaments(model.layerCount);

    const { layerCount, hostLayers } = model;
    const workerLayers = Math.max(0, layerCount - hostLayers);
    const dividerX =
      layerCount > 0
        ? DIV_MIN + (hostLayers / layerCount) * (DIV_MAX - DIV_MIN)
        : (DIV_MIN + DIV_MAX) / 2;
    const spacing = Math.min(MAX_SPACING, FAN_SPAN / Math.max(layerCount, 1));

    for (let i = 0; i < this.filaments.length; i++) {
      const path = this.filaments[i]!;
      const onHost = i < hostLayers;
      const indexInSide = onHost ? i : i - hostLayers;
      const sideCount = onHost ? hostLayers : workerLayers;
      const loaded = onHost
        ? indexInSide < model.hostLoaded
        : indexInSide < model.workerLoaded;
      path.setAttribute("d", filamentPath(indexInSide, sideCount, spacing, dividerX, onHost));
      path.setAttribute(
        "class",
        `filament ${onHost ? "host" : "worker"}${loaded ? " loaded" : ""}`,
      );
    }

    // Divider and its hit band.
    this.pick("divider").setAttribute("transform", `translate(${dividerX} 0)`);
    this.pick("split-hit").setAttribute("x", String(dividerX - 34));
    this.svg.dataset.interactive = String(model.interactive);
    // Nothing to divide until a model has been probed, so the marker and its
    // hit band stay out of the drawing rather than parking at the left edge.
    this.svg.dataset.hasLayers = String(layerCount > 0);
    this.svg.dataset.generating = String(model.generating);
    this.svg.dataset.connected = String(model.connected);
    this.svg.dataset.selected = model.selected ?? "";

    // Nodes.
    this.node("host", model.hostName, model.hostDetail, model.hostPhase, model.hostIsLocal);
    this.node("worker", model.workerName, model.workerDetail, model.workerPhase, !model.hostIsLocal);
    this.pick("worker-node").setAttribute(
      "class",
      `node worker${model.connected ? "" : " ghost"}`,
    );

    this.svg.setAttribute(
      "aria-label",
      layerCount === 0
        ? "Cluster topology — no model selected yet"
        : `${model.hostName} owns ${hostLayers} of ${layerCount} layers, ` +
          `${model.workerName} owns ${workerLayers}`,
    );
    this.pick("split-hit").setAttribute("aria-valuenow", String(hostLayers));
    this.pick("split-hit").setAttribute("aria-valuemax", String(Math.max(1, layerCount - 1)));
  }

  private node(
    side: "host" | "worker",
    name: string,
    detail: string,
    phase: string,
    mine: boolean,
  ): void {
    this.pick(`${side}-name`).textContent = mine ? `${name} · you` : name;
    this.pick(`${side}-detail`).textContent = detail;
    this.pick(`${side}-orb`).setAttribute("class", `node-orb ${side} ${tone(phase)}`);
  }

  /**
   * Split implied by a pointer position. Converted through the SVG's own
   * matrix rather than its bounding box, because the drawing is letterboxed
   * inside the element whenever the stage is not the viewBox's aspect ratio.
   */
  splitFromPointer(clientX: number, clientY: number, fallback: number): number {
    const matrix = this.svg.getScreenCTM();
    if (!matrix || this.layerCount <= 1) return fallback;
    const point = this.svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const local = point.matrixTransform(matrix.inverse());
    const fraction = (local.x - DIV_MIN) / (DIV_MAX - DIV_MIN);
    const split = Math.round(fraction * this.layerCount);
    return Math.min(this.layerCount - 1, Math.max(1, split));
  }

  /** Which part of the graph a pointer event landed on, if any. */
  targetAt(event: Event): NodeTarget | undefined {
    const hit = (event.target as Element | null)?.closest("[data-hit]");
    const value = hit?.getAttribute("data-hit");
    return value === "host" || value === "worker" || value === "split"
      ? value
      : undefined;
  }

  private pick(name: string): SVGElement {
    const element = this.svg.querySelector<SVGElement>(`[data-part="${name}"]`);
    if (!element) throw new Error(`Missing canvas part ${name}`);
    return element;
  }
}

function filamentPath(
  index: number,
  count: number,
  spacing: number,
  dividerX: number,
  onHost: boolean,
): string {
  if (count === 0) return "";
  const y = CY + (index - (count - 1) / 2) * spacing;
  const orbX = onHost ? HOST_CX + ORB_R - 6 : WORKER_CX - ORB_R + 6;
  const endX = onHost ? dividerX - 16 : dividerX + 16;
  const reach = endX - orbX;
  return (
    `M${orbX.toFixed(1)},${CY}` +
    `C${(orbX + reach * 0.42).toFixed(1)},${CY} ` +
    `${(endX - reach * 0.38).toFixed(1)},${y.toFixed(1)} ` +
    `${endX.toFixed(1)},${y.toFixed(1)}`
  );
}

function tone(phase: string): string {
  if (phase === "ready") return "ok";
  if (phase === "generating") return "live";
  if (phase === "error") return "bad";
  if (phase === "loading") return "busy";
  return "idle";
}

/**
 * The parts that never change shape: gradients, the two flow arcs, the orbs,
 * their labels, the divider, and the transparent hit targets.
 */
function skeleton(): string {
  // Both arcs are drawn twice — once faint as the path itself, once as the
  // travelling dashes — so the geometry is named rather than repeated.
  const ARC_OUT =
    `M${HOST_CX + 30},${CY - 52} C380,46 620,46 ${WORKER_CX - 30},${CY - 52}`;
  const ARC_BACK =
    `M${WORKER_CX - 30},${CY + 52} C620,${H - 46} 380,${H - 46} ${HOST_CX + 30},${CY + 52}`;

  const orb = (side: "host" | "worker", cx: number): string => `
    <g data-part="${side}-node" class="node ${side}">
      <circle class="node-glow" cx="${cx}" cy="${CY}" r="${ORB_R * 2.1}" fill="url(#grad-${side})"/>
      <circle class="node-ring dotted" cx="${cx}" cy="${CY}" r="${ORB_R + 22}"/>
      <circle class="node-ring" cx="${cx}" cy="${CY}" r="${ORB_R + 10}"/>
      <circle data-part="${side}-orb" class="node-orb ${side} idle"
              cx="${cx}" cy="${CY}" r="${ORB_R}" fill="url(#grad-${side})"/>
      <text data-part="${side}-name" class="node-name" x="${cx}" y="${CY - ORB_R - 40}"
            text-anchor="middle">—</text>
      <text data-part="${side}-detail" class="node-detail" x="${cx}" y="${CY + ORB_R + 46}"
            text-anchor="middle">—</text>
      <circle data-hit="${side}" class="node-hit" cx="${cx}" cy="${CY}" r="${ORB_R + 26}"/>
    </g>`;

  return `
  <defs>
    <radialGradient id="grad-host" cx="34%" cy="28%" r="70%">
      <stop offset="0%" stop-color="#fff" stop-opacity="0.9"/>
      <stop offset="40%" stop-color="var(--host)"/>
      <stop offset="100%" stop-color="var(--host)" stop-opacity="0.12"/>
    </radialGradient>
    <radialGradient id="grad-worker" cx="34%" cy="28%" r="70%">
      <stop offset="0%" stop-color="#fff" stop-opacity="0.9"/>
      <stop offset="40%" stop-color="var(--worker)"/>
      <stop offset="100%" stop-color="var(--worker)" stop-opacity="0.12"/>
    </radialGradient>
  </defs>

  <!-- Every token's path: gold out with the activation, cyan back with the
       transformed hidden state. -->
  <g class="arcs">
    <path class="arc-base" d="${ARC_OUT}"/>
    <path class="arc-base" d="${ARC_BACK}"/>
    <path class="arc out" d="${ARC_OUT}"/>
    <path class="arc back" d="${ARC_BACK}"/>
    <text class="arc-label" x="500" y="52" text-anchor="middle">activation out</text>
    <text class="arc-label" x="500" y="${H - 44}" text-anchor="middle">hidden state back</text>
  </g>

  <g data-part="filaments" class="filaments"></g>

  <g data-part="divider" class="divider">
    <line class="divider-line" x1="0" y1="${CY - DIV_REACH}" x2="0" y2="${CY + DIV_REACH}"/>
    <circle class="divider-grip" cx="0" cy="${CY}" r="9"/>
    <text class="divider-label" x="0" y="${CY - DIV_REACH - 14}" text-anchor="middle">split</text>
  </g>

  ${orb("host", HOST_CX)}
  ${orb("worker", WORKER_CX)}

  <rect data-hit="split" data-part="split-hit" class="split-hit"
        x="466" y="0" width="68" height="${H}" role="slider" tabindex="0"
        aria-label="Transformer layer split point" aria-valuemin="1"/>`;
}
