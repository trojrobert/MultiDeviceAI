import { formatCount, formatMicros, STAGE_NAMES } from "../runtime/metrics.ts";
import { createClusterCode, normalizeClusterCode } from "../runtime/peer.ts";
import { ClusterController, type ClusterSnapshot } from "../runtime/cluster.ts";
import type { DeviceCapabilities } from "../runtime/protocol.ts";
import type { PlacementResult, SplitCandidate } from "../engine/placement.ts";
import { ClusterCanvas, type NodeTarget } from "./canvas.ts";
import { encodeQR, qrToSvg } from "./qr.ts";

const THEME_KEY = "lcai-theme";
const NAME_KEY = "lcai-device-name";
const BUDGET_KEY = "lcai-memory-budget";

type Theme = "dark" | "light";

/**
 * Which set of controls the inspector is showing. The canvas is the primary
 * surface and the inspector holds the detail of whatever is selected on it, so
 * exactly one group is visible at a time rather than a permanent stack of
 * every setting the application has.
 */
type Group = "devices" | "model" | "perf" | "session";

const GROUP_TITLES: Record<Group, string> = {
  devices: "Devices",
  model: "Model & split",
  perf: "Performance",
  session: "Session",
};

export function mountApp(root: HTMLElement): () => void {
  root.innerHTML = shell();

  const cluster = new ClusterController({ onChange: render });
  let snapshot = cluster.snapshot;
  let budgetRestored = false;

  // ── Element handles ─────────────────────────────────────────────────────
  const joinView = get("join-view");
  const clusterView = get("cluster-view");
  const nameInput = input("device-name");
  const codeInput = input("cluster-code");
  const joinStatus = get("join-status");
  const createButton = button("create-cluster");
  const joinButton = button("join-cluster");
  const themeToggle = button("theme-toggle");
  const copyButton = button("copy-link");
  const shareButton = button("share-link");
  const inviteLink = get("invite-link");
  const qrHolder = get("qr-holder");
  const invitePanel = get("invite-panel");
  const stageCaption = get("stage-caption");
  const chatLayer = get("chat-layer");
  const assignButton = button("assign-layers");
  const balanceButton = button("balance-split");
  const modelSelect = get("model-select") as HTMLSelectElement;
  const loadSlider = input("load-slider");
  const loadEvenButton = button("load-even");
  const loadPowerButton = button("load-power");
  const promptForm = get("prompt-form") as HTMLFormElement;
  const promptInput = get("prompt-input") as HTMLTextAreaElement;
  const stopButton = button("stop-generation");
  const resetButton = button("reset-chat");
  const leaveButton = button("leave-cluster");
  const transcriptEl = get("transcript");
  const inspector = get("inspector");
  const inspectorToggle = button("inspector-toggle");
  const inspectorClose = button("inspector-close");
  const scrim = get("scrim");
  const clearCacheButton = button("clear-cache");
  const toasts = get("toasts");
  const canvasEl = get("canvas");
  const canvas = new ClusterCanvas(canvasEl as unknown as SVGSVGElement);

  // ── Local UI state ──────────────────────────────────────────────────────
  let proposedSplit = 0;
  let layerCount = 0;
  let dragging = false;
  let lastInviteCode = "";
  let group: Group = "devices";
  let modelGroupOffered = false;
  let selected: NodeTarget | undefined;
  // The inspector is docked beside the canvas on a wide screen and a sheet on
  // a narrow one, so it starts open only where it costs no canvas.
  let inspectorOpen = !window.matchMedia?.("(max-width: 900px)").matches;

  // ── Theme ───────────────────────────────────────────────────────────────
  const storedTheme = localStorage.getItem(THEME_KEY);
  let theme: Theme =
    storedTheme === "light" || storedTheme === "dark"
      ? storedTheme
      : window.matchMedia?.("(prefers-color-scheme: light)").matches
        ? "light"
        : "dark";
  applyTheme();

  themeToggle.addEventListener("click", () => {
    theme = theme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, theme);
    applyTheme();
  });

  function applyTheme(): void {
    document.documentElement.dataset.theme = theme;
    themeToggle.textContent = theme === "dark" ? "☾" : "☀";
    themeToggle.setAttribute(
      "aria-label",
      theme === "dark" ? "Switch to light appearance" : "Switch to dark appearance",
    );
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#0A0A0B" : "#F6F6F7");
  }

  // ── Join view ───────────────────────────────────────────────────────────
  const urlCode = normalizeClusterCode(
    new URLSearchParams(location.search).get("cluster") ?? "",
  );
  if (urlCode) codeInput.value = urlCode;
  nameInput.value =
    localStorage.getItem(NAME_KEY) ??
    (/Mobi|Android/i.test(navigator.userAgent) ? "Phone" : "Laptop");

  if (urlCode) {
    joinStatus.textContent = `Invite for cluster ${urlCode} detected. Join to become the worker.`;
    joinButton.classList.add("primary");
    createButton.classList.remove("primary");
  }

  createButton.addEventListener("click", () => {
    const code = createClusterCode();
    codeInput.value = code;
    void start("host", code);
  });
  joinButton.addEventListener("click", () => {
    const code = normalizeClusterCode(codeInput.value);
    if (!code) {
      joinStatus.textContent = "Enter the cluster code shown on the host.";
      codeInput.focus();
      return;
    }
    void start("worker", code);
  });
  codeInput.addEventListener("input", () => {
    codeInput.value = normalizeClusterCode(codeInput.value);
  });
  codeInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") joinButton.click();
  });

  async function start(role: "host" | "worker", code: string): Promise<void> {
    const name = nameInput.value.trim() || (role === "host" ? "Host" : "Worker");
    localStorage.setItem(NAME_KEY, name);
    setJoinBusy(true);
    try {
      await cluster.start(role, code, name);
      const url = new URL(location.href);
      url.searchParams.set("cluster", code);
      history.replaceState({}, "", url);
    } catch (error) {
      joinStatus.textContent = message(error);
      setJoinBusy(false);
    }
  }

  function setJoinBusy(busy: boolean): void {
    createButton.disabled = busy;
    joinButton.disabled = busy;
    createButton.classList.toggle("busy", busy);
    joinButton.classList.toggle("busy", busy);
  }

  // ── Invite sharing ──────────────────────────────────────────────────────
  function inviteUrl(): string {
    const url = new URL(location.href);
    url.searchParams.set("cluster", snapshot.clusterCode ?? "");
    url.hash = "";
    return url.toString();
  }

  copyButton.addEventListener("click", async () => {
    const url = inviteUrl();
    try {
      await navigator.clipboard.writeText(url);
      toast("Invite link copied");
    } catch {
      window.prompt("Copy this invite link", url);
    }
  });

  shareButton.addEventListener("click", async () => {
    const url = inviteUrl();
    try {
      await navigator.share({
        title: "LocalCluster AI",
        text: "Join my split-model cluster",
        url,
      });
    } catch {
      /* share cancelled or unsupported */
    }
  });

  leaveButton.addEventListener("click", () => {
    void cluster.leave();
    const url = new URL(location.href);
    url.searchParams.delete("cluster");
    history.replaceState({}, "", url);
    setJoinBusy(false);
  });

  // ── Inspector ───────────────────────────────────────────────────────────
  function setGroup(next: Group, open = true): void {
    group = next;
    inspectorOpen = open;
    applyInspector();
  }

  function applyInspector(): void {
    inspector.hidden = !inspectorOpen;
    clusterView.classList.toggle("inspector-open", inspectorOpen);
    inspectorToggle.setAttribute("aria-expanded", String(inspectorOpen));
    text("inspector-title", GROUP_TITLES[group]);
    for (const chip of root.querySelectorAll<HTMLElement>("[data-inspect]")) {
      const on = inspectorOpen && chip.dataset.inspect === group;
      chip.classList.toggle("on", on);
      chip.setAttribute("aria-pressed", String(on));
    }
    paintPanels();
  }

  /** A panel shows when its group is selected and its own data exists. */
  function paintPanels(): void {
    const next = snapshot;
    const tokens = next.perf.promptTokens + next.perf.decodeTokens;
    const available: Record<string, boolean> = {
      "model-card": next.role === "host" && next.catalogue.length > 0,
      "cache-card": next.engineAvailable && Boolean(next.cache),
      telemetry: tokens > 0,
    };
    for (const panel of root.querySelectorAll<HTMLElement>(".panel[data-group]")) {
      const owner = panel.dataset.group as Group;
      panel.hidden = !inspectorOpen || owner !== group || available[panel.id] === false;
    }
  }

  inspectorToggle.addEventListener("click", () => {
    inspectorOpen = !inspectorOpen;
    applyInspector();
  });
  inspectorClose.addEventListener("click", () => {
    inspectorOpen = false;
    applyInspector();
  });
  scrim.addEventListener("click", () => {
    inspectorOpen = false;
    applyInspector();
  });
  for (const chip of root.querySelectorAll<HTMLElement>("[data-inspect]")) {
    chip.addEventListener("click", () => {
      const target = chip.dataset.inspect as Group;
      // Pressing the group already showing closes the panel, so the canvas can
      // be cleared without hunting for a close button.
      setGroup(target, !(inspectorOpen && group === target));
    });
  }

  // ── Canvas: selection and the split drag ────────────────────────────────
  function splitInteractive(next: Readonly<ClusterSnapshot>): boolean {
    return (
      next.role === "host" &&
      next.connected &&
      next.engineAvailable &&
      next.localPhase !== "loading" &&
      !next.generating
    );
  }

  function setSplit(value: number): void {
    const next = clamp(value, 1, Math.max(1, layerCount - 1));
    if (next === proposedSplit) return;
    proposedSplit = next;
    paintCanvas(snapshot);
    paintSplit(snapshot);
  }

  canvasEl.addEventListener("pointerdown", (event) => {
    const target = canvas.targetAt(event);
    // Empty canvas clears the selection, so the highlight always names what
    // the inspector is currently showing.
    if (!target) {
      if (selected === undefined) return;
      selected = undefined;
      paintCanvas(snapshot);
      return;
    }
    selected = target;
    if (target === "split") {
      setGroup("model");
      if (!splitInteractive(snapshot)) {
        paintCanvas(snapshot);
        return;
      }
      dragging = true;
      canvasEl.setPointerCapture(event.pointerId);
      setSplit(canvas.splitFromPointer(event.clientX, event.clientY, proposedSplit));
      event.preventDefault();
    } else {
      setGroup("devices");
    }
    paintCanvas(snapshot);
  });
  canvasEl.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setSplit(canvas.splitFromPointer(event.clientX, event.clientY, proposedSplit));
  });
  const endDrag = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    if (canvasEl.hasPointerCapture(event.pointerId)) {
      canvasEl.releasePointerCapture(event.pointerId);
    }
  };
  canvasEl.addEventListener("pointerup", endDrag);
  canvasEl.addEventListener("pointercancel", endDrag);

  canvasEl.addEventListener("keydown", (event) => {
    if (!splitInteractive(snapshot)) return;
    const step = event.shiftKey ? 4 : 1;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      setSplit(proposedSplit - step);
    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      setSplit(proposedSplit + step);
    } else if (event.key === "Home") {
      setSplit(1);
    } else if (event.key === "End") {
      setSplit(layerCount - 1);
    } else {
      return;
    }
    event.preventDefault();
  });

  // ── Split controls ──────────────────────────────────────────────────────
  balanceButton.addEventListener("click", () => {
    const balanced = snapshot.placement?.balanced;
    if (!balanced) return;
    setSplit(balanced.split);
    toast("Split balanced by download size");
  });

  loadSlider.addEventListener("input", () => {
    if (!splitInteractive(snapshot)) return;
    setSplit(Math.round((Number(loadSlider.value) / 100) * layerCount));
  });

  loadEvenButton.addEventListener("click", () => {
    setSplit(Math.round(layerCount / 2));
    toast("Workload split evenly");
  });

  loadPowerButton.addEventListener("click", () => {
    const recommended = snapshot.placement?.recommended;
    if (!recommended) return;
    setSplit(recommended.split);
    toast("Workload matched to device memory");
  });

  modelSelect.addEventListener("change", () => {
    // A different model means a different layer count, so the proposed split
    // has to be re-derived from the new placement rather than carried over.
    proposedSplit = 0;
    void cluster.selectModel(modelSelect.value);
  });

  (get("local-budget") as HTMLInputElement).addEventListener("change", (event) => {
    const gb = Number((event.target as HTMLInputElement).value);
    if (!Number.isFinite(gb) || gb <= 0) return;
    localStorage.setItem(BUDGET_KEY, String(gb));
    cluster.setBudgetBytes(gb * 1024 ** 3);
  });

  assignButton.addEventListener("click", () => {
    void cluster.assignSplit(proposedSplit);
  });

  // ── Composer ────────────────────────────────────────────────────────────
  function resizeComposer(): void {
    promptInput.style.height = "auto";
    promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
  }
  promptInput.addEventListener("input", resizeComposer);
  promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      promptForm.requestSubmit();
    }
  });
  promptForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = promptInput.value;
    if (!value.trim() || promptInput.disabled) return;
    promptInput.value = "";
    resizeComposer();
    void cluster.submitPrompt(value);
  });
  stopButton.addEventListener("click", () => cluster.stop());
  resetButton.addEventListener("click", () => void cluster.reset());

  clearCacheButton.addEventListener("click", async () => {
    clearCacheButton.disabled = true;
    await cluster.clearWeightCache();
    toast("Cached weights cleared");
  });

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && inspectorOpen) {
      inspectorOpen = false;
      applyInspector();
    }
  };
  window.addEventListener("keydown", onKeyDown);

  // ── Toasts ──────────────────────────────────────────────────────────────
  function toast(text: string): void {
    const node = document.createElement("div");
    node.className = "toast";
    node.textContent = text;
    toasts.append(node);
    setTimeout(() => {
      node.classList.add("out");
      setTimeout(() => node.remove(), 240);
    }, 2000);
  }

  // ── Render ──────────────────────────────────────────────────────────────
  function render(next: Readonly<ClusterSnapshot>): void {
    snapshot = next;
    const active = Boolean(next.role);
    joinView.hidden = active;
    clusterView.hidden = !active;
    if (!active) return;

    // Layer count is per-model, so the graph carries no filaments until the
    // host has probed the GGUF header or the worker has been told its range.
    layerCount =
      next.modelProfile?.layerCount ??
      next.localAssignment?.layerCount ??
      next.remoteAssignment?.layerCount ??
      0;
    if (proposedSplit === 0 && layerCount > 0) {
      proposedSplit = next.placement?.recommended?.split ?? Math.round(layerCount / 2);
    }

    // Choosing the split is the one thing that must happen right after
    // pairing, so the inspector offers it once instead of waiting to be found.
    if (!modelGroupOffered && next.role === "host" && next.catalogue.length > 0) {
      modelGroupOffered = true;
      group = "model";
    }

    text("cluster-code-display", next.clusterCode ?? "—");
    text("session-code", next.clusterCode ?? "—");
    text("role-display", next.role === "host" ? "Host" : "Worker");
    text("cluster-status", next.status);
    text("engine-mode", next.engineAvailable ? "engine live" : "transport only");
    get("engine-mode").className = `chip ${next.engineAvailable ? "ok" : "warn"}`;
    get("status-dot").className = `dot ${phaseTone(next.localPhase)}`;
    const banner = get("error-banner");
    banner.hidden = !next.error;
    text("error-banner", next.error ?? "");

    renderPairing(next);
    renderModel(next);
    // The user's own budget outlives the session; apply it as soon as the first
    // capability probe gives us something to override.
    if (!budgetRestored && next.localCapabilities) {
      budgetRestored = true;
      const stored = Number(localStorage.getItem(BUDGET_KEY));
      if (Number.isFinite(stored) && stored > 0) cluster.setBudgetBytes(stored * 1024 ** 3);
    }

    // Which half of the model a device owns decides its colour everywhere.
    const localSide = next.role === "host" ? "host" : "worker";
    const remoteSide = next.role === "host" ? "worker" : "host";
    renderPeer("local", next.localCapabilities, next.localPhase, next.localProgress, true, localSide);
    renderPeer("remote", next.remoteCapabilities, next.remotePhase, next.remoteProgress, false, remoteSide);

    paintCanvas(next);
    paintSplit(next);
    renderCache(next);
    renderTelemetry(next);
    renderTranscript(next);
    renderCaption(next);
    applyInspector();

    promptInput.disabled = !next.ready || next.generating;
    promptInput.placeholder = next.ready
      ? next.role === "host"
        ? "Message the split model…"
        : "Send a prompt through the host…"
      : next.engineAvailable
        ? "Waiting for both shards to finish loading…"
        : "Engine unavailable — transport preview only";
    stopButton.hidden = !next.generating;
    get("send-button").hidden = next.generating;
    resetButton.disabled = next.transcript.length === 0 || next.generating;
    text(
      "composer-hint",
      next.generating
        ? "Generating — activations in flight"
        : next.ready
          ? "Enter to send · Shift+Enter for a new line"
          : "Chat unlocks when both shards report ready",
    );
  }

  /** Host-owned layer count for the current snapshot, assigned or proposed. */
  function hostLayersOf(next: Readonly<ClusterSnapshot>): number {
    const assigned = next.localAssignment;
    if (!assigned) return Math.min(proposedSplit, Math.max(layerCount, 1));
    return next.role === "host" ? assigned.end : assigned.start;
  }

  /** Download size of each side at the current split, assigned or estimated. */
  function sideBytes(
    next: Readonly<ClusterSnapshot>,
    hostLayers: number,
  ): { host?: number; worker?: number } {
    const hostIsLocal = next.role === "host";
    const candidate = candidateAt(next.placement, hostLayers);
    if (!next.localAssignment) {
      return { host: candidate?.hostDownloadBytes, worker: candidate?.workerDownloadBytes };
    }
    const local = next.localAssignment.bytes;
    const remote = next.remoteAssignment?.bytes;
    return hostIsLocal ? { host: local, worker: remote } : { host: remote, worker: local };
  }

  function paintCanvas(next: Readonly<ClusterSnapshot>): void {
    const hostLayers = hostLayersOf(next);
    const workerLayers = Math.max(0, layerCount - hostLayers);
    const hostIsLocal = next.role === "host";
    const assigned = Boolean(next.localAssignment);
    const hostProgress = hostIsLocal ? next.localProgress : next.remoteProgress;
    const workerProgress = hostIsLocal ? next.remoteProgress : next.localProgress;
    const hostCaps = hostIsLocal ? next.localCapabilities : next.remoteCapabilities;
    const workerCaps = hostIsLocal ? next.remoteCapabilities : next.localCapabilities;
    const bytes = sideBytes(next, hostLayers);

    canvas.paint({
      layerCount,
      hostLayers,
      hostLoaded: assigned ? Math.round(hostProgress * hostLayers) : 0,
      workerLoaded: assigned ? Math.round(workerProgress * workerLayers) : 0,
      hostName: hostCaps?.label ?? "Host",
      workerName: next.connected ? (workerCaps?.label ?? "Worker") : "No second device",
      hostDetail: nodeDetail(hostLayers, layerCount, bytes.host),
      workerDetail: next.connected
        ? nodeDetail(workerLayers, layerCount, bytes.worker)
        : "waiting to pair",
      hostIsLocal,
      hostPhase: hostIsLocal ? next.localPhase : next.remotePhase,
      workerPhase: hostIsLocal ? next.remotePhase : next.localPhase,
      connected: next.connected,
      generating: next.generating,
      interactive: splitInteractive(next),
      selected,
    });
  }

  /** The split panel's readouts, which mirror the canvas rather than lead it. */
  function paintSplit(next: Readonly<ClusterSnapshot>): void {
    const hostLayers = hostLayersOf(next);
    const workerLayers = Math.max(0, layerCount - hostLayers);
    const hostIsLocal = next.role === "host";
    const assigned = Boolean(next.localAssignment);
    const interactive = splitInteractive(next);
    const bytes = sideBytes(next, hostLayers);

    text("host-range", hostLayers > 0 ? `0–${hostLayers - 1}` : "—");
    text("worker-range", workerLayers > 0 ? `${hostLayers}–${layerCount - 1}` : "—");
    text("host-size", bytes.host === undefined ? "—" : formatBytes(bytes.host));
    text("worker-size", bytes.worker === undefined ? "—" : formatBytes(bytes.worker));
    text("host-side-role", hostIsLocal ? "Host · you" : "Host · peer");
    text("worker-side-role", hostIsLocal ? "Worker · peer" : "Worker · you");

    const candidate = candidateAt(next.placement, hostLayers);
    assignButton.textContent = assigned ? "Reassign & reload" : "Assign & load";
    const fits = !candidate || (candidate.hostFits && candidate.workerFits);
    assignButton.disabled = !interactive || !fits;
    balanceButton.disabled = !interactive || !next.placement?.balanced;

    text(
      "split-hint",
      assigned || !candidate || fits
        ? ""
        : !candidate.hostFits && !candidate.workerFits
          ? "Both devices are over budget at this split."
          : candidate.hostFits
            ? `Worker needs ${formatBytes(candidate.workerBytes)} here — over its budget.`
            : `Host needs ${formatBytes(candidate.hostBytes)} here — over its budget.`,
    );

    const hostPercent = layerCount > 0 ? Math.round((hostLayers / layerCount) * 100) : 50;
    loadSlider.style.setProperty("--load", String(hostPercent / 100));
    // Don't yank the thumb while the user is dragging the slider itself; the
    // canvas divider and the presets still push their value in here.
    if (document.activeElement !== loadSlider) {
      loadSlider.value = String(hostPercent);
    }
    loadSlider.disabled = !interactive;
    loadSlider.setAttribute(
      "aria-valuetext",
      `Host handles ${hostPercent}%, worker handles ${100 - hostPercent}%`,
    );
    loadEvenButton.disabled = !interactive;
    loadPowerButton.disabled = !interactive || !next.placement?.recommended;
    text("host-load-pct", `${hostPercent}%`);
    text("worker-load-pct", `${100 - hostPercent}%`);
  }

  /**
   * Pairing takes over the canvas rather than hiding in a side panel: the
   * second node does not exist yet, and scanning the code is the only thing
   * left to do.
   */
  function renderPairing(next: Readonly<ClusterSnapshot>): void {
    const show = next.role === "host" && !next.connected;
    invitePanel.hidden = !show;
    clusterView.classList.toggle("pairing", show);
    if (!show) return;
    const url = inviteUrl();
    inviteLink.textContent = url.replace(/^https?:\/\//, "");
    shareButton.hidden = typeof navigator.share !== "function";
    text("invite-code", next.clusterCode ?? "—");
    if (lastInviteCode === url) return;
    lastInviteCode = url;
    try {
      qrHolder.innerHTML = qrToSvg(encodeQR(url), 2);
    } catch {
      qrHolder.replaceChildren();
    }
  }

  function renderModel(next: Readonly<ClusterSnapshot>): void {
    // Only the host chooses; the worker is told what to load.
    if (next.role !== "host" || next.catalogue.length === 0) return;

    if (modelSelect.options.length !== next.catalogue.length) {
      modelSelect.replaceChildren(
        ...next.catalogue.map((entry) => {
          const option = document.createElement("option");
          option.value = entry.id;
          option.textContent = `${entry.label} · ${formatBytes(entry.approxBytes)}`;
          return option;
        }),
      );
    }
    if (modelSelect.value !== next.modelId) modelSelect.value = next.modelId;
    // Switching models mid-load would strand a half-loaded shard on the peer.
    modelSelect.disabled =
      next.modelPhase === "probing" || next.localPhase === "loading" || next.generating;

    const verdict =
      next.modelPhase === "probing"
        ? { text: "reading…", tone: "busy" }
        : next.modelPhase === "error"
          ? { text: "unavailable", tone: "bad" }
          : verdictLabel(next.placement);
    text("model-verdict", verdict.text);
    get("model-verdict").className = `chip ${verdict.tone}`;

    text(
      "model-summary",
      next.modelPhase === "probing"
        ? "Reading the model header…"
        : (next.placement?.summary ??
          "Waiting for the peer to report how much memory it can spend."),
    );
  }

  function renderPeer(
    prefix: "local" | "remote",
    capabilities: DeviceCapabilities | undefined,
    phase: string,
    progress: number,
    isLocal: boolean,
    side: "host" | "worker",
  ): void {
    // The tone class only drives the halo's motion; the side owns the colour,
    // so an idle device is still visibly the host or the worker.
    const tone = phaseTone(phase);
    get(`${prefix}-orb`).className = capabilities
      ? `orb ${side}${tone === "idle" ? "" : ` ${tone}`}`
      : "orb idle";
    text(
      `${prefix}-name`,
      capabilities?.label ?? (isLocal ? "This device" : "Waiting for peer…"),
    );
    text(`${prefix}-phase`, phase);
    get(`${prefix}-phase`).className = `phase ${tone}`;
    text(
      `${prefix}-gpu`,
      capabilities
        ? capabilities.webgpu
          ? (capabilities.gpu ?? "WebGPU ready")
          : "WebGPU unavailable"
        : "Probing capabilities…",
    );
    text(
      `${prefix}-memory`,
      capabilities?.maxBufferSize
        ? `${formatBytes(capabilities.maxBufferSize)} max buffer`
        : "—",
    );

    const budgetInput = get(`${prefix}-budget`) as HTMLInputElement;
    // The peer's budget is theirs to set; this device only displays it.
    budgetInput.disabled = !isLocal || !capabilities;
    if (capabilities && document.activeElement !== budgetInput) {
      budgetInput.value = (capabilities.budgetBytes / 1024 ** 3).toFixed(2);
    }
    // Short enough to sit on the budget's own line; the full wording is the
    // tooltip, so the panel does not grow a second row per device.
    const budgetSource = capabilities?.budgetSource;
    const budgetNote =
      budgetSource === "user"
        ? { short: "manual", long: "Set by you." }
        : budgetSource === "device-memory"
          ? { short: "auto", long: "Estimated from the memory this device reports." }
          : { short: "estimate", long: "Rough estimate — adjust it if loading fails." };
    text(`${prefix}-budget-src`, capabilities ? budgetNote.short : "");
    get(`${prefix}-budget-src`).title = capabilities ? budgetNote.long : "";
    const percent = Math.round(clamp(progress, 0, 1) * 100);
    const fill = get(`${prefix}-progress`);
    fill.style.width = `${percent}%`;
    fill.parentElement?.setAttribute("aria-valuenow", String(percent));
    text(`${prefix}-percent`, phase === "loading" ? `${percent}%` : "");
  }

  function renderTelemetry(next: Readonly<ClusterSnapshot>): void {
    const perf = next.perf;
    const tokens = perf.promptTokens + perf.decodeTokens;
    // The strip appears the moment the first activation moves and stays up
    // afterwards: the numbers from a finished run are the point.
    get("hud").hidden = tokens === 0 || !invitePanel.hidden;
    if (tokens === 0) return;

    const live = get("metric-live");
    live.textContent = perf.running ? "live" : "last run";
    live.className = `chip ${perf.running ? "live" : "ok"}`;

    text("metric-rate", perf.tokensPerSecond > 0 ? perf.tokensPerSecond.toFixed(1) : "—");
    text("metric-ttft", perf.ttftMicros ? formatMicros(perf.ttftMicros) : "—");
    text("metric-tokens", String(tokens));
    text("metric-per-token", formatMicros(Math.round(perf.totalMicros / tokens)));

    // Stage bar and legend share one pass so a stage's width and its printed
    // share can never disagree.
    const bar = get("stage-bar");
    const parts: string[] = [];
    for (const stage of STAGE_NAMES) {
      const stat = perf.stages[stage];
      const segment = bar.querySelector<HTMLElement>(`i[data-stage="${stage}"]`);
      if (segment) segment.style.flexGrow = String(Math.max(stat.share, 0));
      text(`stage-${stage}-time`, formatMicros(stat.meanMicros));
      text(`stage-${stage}-share`, `${Math.round(stat.share * 100)}%`);
      parts.push(`${stage} ${Math.round(stat.share * 100)}%`);
    }
    bar.setAttribute("aria-label", `Stage breakdown: ${parts.join(", ")}`);
    // A run whose time is mostly network is the interesting failure mode, so
    // name it rather than leaving the reader to compare four percentages.
    get("stage-legend").className = `stage-legend dominant-${perf.dominantStage ?? "none"}`;

    renderSpark(perf.recentMicros);

    text(
      "metric-bytes",
      perf.bytesPerToken > 0 ? `${formatCount(perf.bytesPerToken)}/token` : "—",
    );
    text(
      "metric-bytes-total",
      `${formatCount(perf.bytesOut)} · ${formatCount(perf.bytesIn)}`,
    );

    text("hud-rate", perf.tokensPerSecond > 0 ? perf.tokensPerSecond.toFixed(1) : "—");
    text("hud-ttft", perf.ttftMicros ? formatMicros(perf.ttftMicros) : "—");
    text("hud-tokens", String(tokens));
    text(
      "hud-wire",
      perf.bytesPerToken > 0 ? `${formatCount(perf.bytesPerToken)}/tok` : "—",
    );
  }

  /**
   * Latency per token, oldest left. Scaled to the window's own peak: the shape
   * of the variation matters more here than an absolute axis, and the peak is
   * printed beside it.
   */
  function renderSpark(values: readonly number[]): void {
    const line = get("spark-line");
    const area = get("spark-area");
    if (values.length < 2) {
      line.setAttribute("d", "");
      area.setAttribute("d", "");
      text("metric-spark-peak", values.length === 1 ? formatMicros(values[0]!) : "—");
      return;
    }

    const peak = Math.max(...values);
    const scale = peak > 0 ? peak : 1;
    const stepX = 100 / (values.length - 1);
    const points = values.map((value, i) => {
      const x = i * stepX;
      // 1px of padding top and bottom so the peak is not clipped by the stroke.
      const y = 27 - (value / scale) * 26;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    line.setAttribute("d", `M${points.join("L")}`);
    area.setAttribute("d", `M0,28L${points.join("L")}L100,28Z`);
    text("metric-spark-peak", `peak ${formatMicros(peak)}`);
  }

  function renderCache(next: Readonly<ClusterSnapshot>): void {
    const report = next.cache;
    if (!report) return;

    const state = get("cache-state");
    const assigned = next.localAssignment?.bytes ?? 0;
    const held = report.shard.bytes;
    // Against the assigned shard when there is one, so the bar answers "will
    // this session re-download?" rather than "how full is the disk?".
    const fraction = assigned > 0 ? clamp(held / assigned, 0, 1) : 0;

    if (!report.available) {
      state.textContent = "unavailable";
      state.className = "chip warn";
      text(
        "cache-summary",
        "This browser will not store weights — every session re-downloads its shard. " +
          "Private browsing and insecure origins both block the cache.",
      );
    } else if (held === 0) {
      state.textContent = "empty";
      state.className = "chip idle";
      text(
        "cache-summary",
        "Nothing stored yet. The first load fills the cache, and later sessions " +
          "restore from it instead of downloading again.",
      );
    } else {
      const complete = assigned > 0 && held >= assigned;
      state.textContent = complete ? "complete" : "partial";
      state.className = `chip ${complete ? "ok" : "live"}`;
      text(
        "cache-summary",
        `${formatBytes(held)} stored for this model` +
          (assigned > 0 ? ` of the ${formatBytes(assigned)} this device owns` : "") +
          `. ${formatBytes(report.total.bytes)} across all models` +
          (report.persisted ? ", kept through storage pressure." : "."),
      );
    }

    get("cache-fill").style.width = `${Math.round(fraction * 100)}%`;
    clearCacheButton.disabled = !report.available || report.total.bytes === 0;
  }

  function renderTranscript(next: Readonly<ClusterSnapshot>): void {
    // With no conversation the graph is the content; the reading layer only
    // exists once there is something to read.
    const empty = next.transcript.length === 0;
    chatLayer.hidden = empty || !invitePanel.hidden;
    clusterView.classList.toggle("reading", !chatLayer.hidden);
    if (empty) {
      transcriptEl.replaceChildren();
      return;
    }

    const nearBottom =
      transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight <
      120;

    transcriptEl.replaceChildren(
      ...next.transcript.map((entry) => {
        const article = document.createElement("article");
        article.className = `msg ${entry.role}`;

        const meta = document.createElement("div");
        meta.className = "msg-meta";
        const who = document.createElement("span");
        who.textContent =
          entry.role === "assistant"
            ? "Split model"
            : entry.role === "user"
              ? "You"
              : "System";
        meta.append(who);

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        if (entry.pending && !entry.text) {
          const dots = document.createElement("span");
          dots.className = "dots";
          dots.setAttribute("aria-label", "Generating");
          dots.append(
            document.createElement("i"),
            document.createElement("i"),
            document.createElement("i"),
          );
          bubble.append(dots);
        } else {
          bubble.textContent = entry.text;
          if (entry.pending) {
            const caret = document.createElement("span");
            caret.className = "caret";
            bubble.append(caret);
          }
        }

        if (!entry.pending && entry.text) {
          const copy = document.createElement("button");
          copy.type = "button";
          copy.className = "msg-copy";
          copy.textContent = "Copy";
          copy.addEventListener("click", async () => {
            try {
              await navigator.clipboard.writeText(entry.text);
              toast("Message copied");
            } catch {
              /* clipboard unavailable */
            }
          });
          meta.append(copy);
        }

        article.append(meta, bubble);
        return article;
      }),
    );

    if (nearBottom) {
      requestAnimationFrame(() => {
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      });
    }
  }

  /**
   * One caption under the graph, naming the next action. It replaces the old
   * empty-state card and disappears as soon as there is a conversation.
   */
  function renderCaption(next: Readonly<ClusterSnapshot>): void {
    stageCaption.hidden = next.transcript.length > 0 || !invitePanel.hidden;
    if (stageCaption.hidden) return;

    let title: string;
    let body: string;
    if (!next.connected) {
      title = "Waiting for the second device";
      body = "Connecting to the host cluster…";
    } else if (!next.localAssignment) {
      title = "Divide the model";
      body =
        next.role === "host"
          ? "Drag the marker between the devices to move layers, then assign and load."
          : "The host is choosing which layers this device will own.";
    } else if (!next.ready) {
      title = "Loading shards";
      body = "Each device downloads only the tensors for the layers it owns.";
    } else {
      title = "Both devices ready";
      body = "Every token crosses the network. Send a message to watch it happen.";
    }
    text("caption-title", title);
    text("caption-body", body);
  }

  // ── Helpers ─────────────────────────────────────────────────────────────
  function get(id: string): HTMLElement {
    const element = root.querySelector<HTMLElement>(`#${id}`);
    if (!element) throw new Error(`Missing UI element #${id}`);
    return element;
  }
  function input(id: string): HTMLInputElement {
    return get(id) as HTMLInputElement;
  }
  function button(id: string): HTMLButtonElement {
    return get(id) as HTMLButtonElement;
  }
  function text(id: string, value: string): void {
    get(id).textContent = value;
  }

  applyInspector();
  render(snapshot);
  resizeComposer();

  return () => {
    window.removeEventListener("keydown", onKeyDown);
    void cluster.leave();
  };
}

/** The line under a canvas node: how much of the model it carries. */
function nodeDetail(layers: number, layerCount: number, bytes: number | undefined): string {
  if (layerCount === 0) return "no model selected";
  return `${layers} layers${bytes === undefined ? "" : ` · ${formatBytes(bytes)}`}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** The candidate describing a given split, or undefined before the model is probed. */
function candidateAt(
  placement: PlacementResult | undefined,
  split: number,
): SplitCandidate | undefined {
  return placement?.candidates[split - 1];
}

/**
 * Short enough to sit beside the panel title at any width. The full reasoning
 * is the summary sentence directly below it, so the chip only has to carry the
 * verdict itself.
 */
function verdictLabel(placement: PlacementResult | undefined): { text: string; tone: string } {
  switch (placement?.verdict) {
    case "single-device":
      return { text: "fits one device", tone: "ok" };
    case "needs-both":
      return { text: "needs both", tone: "live" };
    case "infeasible":
      return { text: "won't fit", tone: "bad" };
    default:
      return { text: "sizing…", tone: "idle" };
  }
}

function phaseTone(phase: string): string {
  if (phase === "ready") return "ok";
  if (phase === "generating") return "live";
  if (phase === "error") return "bad";
  if (phase === "loading") return "busy";
  return "idle";
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Outline glyphs for the stat tiles. Single-path where possible so they stay
 * legible at 15px, and stroked in `currentColor` so the tile owns the colour.
 */
const ICONS = {
  clock: `<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>`,
  stack: `<path d="M12 4l8 4-8 4-8-4 8-4Z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>`,
  gauge: `<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>`,
  swap: `<path d="M4 9h13l-3.5-3.5M20 15H7l3.5 3.5"/>`,
} as const;

function icon(path: string, size = 15): string {
  return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none"
    stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">${path}</svg>`;
}

function metricTile(
  id: string,
  label: string,
  glyph: string,
  initial: string,
  tight = false,
): string {
  return `
    <div class="tile${tight ? " tight" : ""}">
      ${icon(glyph)}
      <b id="${id}" class="mono">${initial}</b>
      <span>${label}</span>
    </div>`;
}

function hudStat(id: string, label: string, unit: string): string {
  return `
    <div class="hud-stat">
      <span class="hud-label">${label}</span>
      <span class="hud-value">
        <b id="${id}" class="mono">—</b>${unit ? `<i>${unit}</i>` : ""}
      </span>
    </div>`;
}

function stageLegendRow(stage: string, label: string): string {
  return `
    <li data-stage="${stage}">
      <i class="swatch"></i>
      <span class="stage-name">${label}</span>
      <b id="stage-${stage}-time" class="mono">—</b>
      <span id="stage-${stage}-share" class="stage-share mono">—</span>
    </li>`;
}

function peerRow(prefix: string, fallback: string): string {
  return `
    <article class="peer">
      <div class="peer-top">
        <span id="${prefix}-orb" class="orb idle" aria-hidden="true"></span>
        <h4 id="${prefix}-name">${fallback}</h4>
        <span id="${prefix}-phase" class="phase">connecting</span>
      </div>
      <p id="${prefix}-gpu" class="peer-gpu">Probing capabilities…</p>
      <div class="peer-foot">
        <span id="${prefix}-memory" class="mono">—</span>
        <span id="${prefix}-percent" class="mono pct"></span>
      </div>
      <div class="peer-budget">
        <label for="${prefix}-budget">Budget</label>
        <input id="${prefix}-budget" class="budget-input mono" type="number"
               min="0.25" step="0.25" inputmode="decimal" aria-describedby="${prefix}-budget-src">
        <span class="budget-unit">GB</span>
        <span id="${prefix}-budget-src" class="budget-src"></span>
      </div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100">
        <i id="${prefix}-progress"></i>
      </div>
    </article>`;
}

function panelHead(title: string, extra = ""): string {
  return `<div class="panel-head"><h3>${title}</h3>${extra}</div>`;
}

function shell(): string {
  return `
  <div class="backdrop" aria-hidden="true"><i></i><i></i><i></i></div>
  <div id="toasts" class="toasts" role="status" aria-live="polite"></div>

  <section id="join-view" class="join">
    <div class="join-inner">
      <div class="join-brand">
        <div class="logo" aria-hidden="true"><i></i><i></i><i></i></div>
        <p class="join-appname">Local<b>Cluster AI</b></p>
      </div>
      <h1>One Model<br><em>Run AI locally on multiple devices</em></h1>
      <p class="lede">
        Split a language model across your laptop, phone, or tablet.
        Each device processes its own layers with no cloud and no subscriptions.
      </p>

      <div class="card join-card">
        <div class="fields">
          <label class="field" for="device-name">
            <span>Device name</span>
            <input id="device-name" maxlength="32" autocomplete="nickname" placeholder="Laptop">
          </label>
          <label class="field" for="cluster-code">
            <span>Cluster code</span>
            <input id="cluster-code" class="mono code" maxlength="8" placeholder="ABC123" autocomplete="off" spellcheck="false">
          </label>
        </div>
        <div class="join-actions">
          <button id="create-cluster" class="btn primary">Create cluster</button>
          <button id="join-cluster" class="btn">Join cluster</button>
        </div>
        <p id="join-status" class="hint" aria-live="polite">
          Start a cluster on any device, then share the invite link with a second device.
        </p>
      </div>

      <ul class="facts">
        <li><b>100%</b><span>peer-to-peer</span></li>
        <li><b>$0</b><span>no subscription</span></li>
        <li><b>Local</b><span>no cloud</span></li>
      </ul>
      <p class="secure">WebGPU on a second device requires an HTTPS origin</p>
    </div>
  </section>

  <section id="cluster-view" class="cluster" hidden>
    <header class="topbar">
      <a class="wordmark" href="./" aria-label="LocalCluster AI home">
        <span class="logo sm" aria-hidden="true"><i></i><i></i><i></i></span>
        local<b>cluster</b>
      </a>
      <div class="topbar-status">
        <span id="status-dot" class="dot"></span>
        <p id="cluster-status" class="topbar-line" aria-live="polite"></p>
      </div>
      <div class="cluster-pill"><span>Cluster</span><strong id="cluster-code-display" class="mono">—</strong></div>
      <b id="role-display" class="role">—</b>
      <span id="engine-mode" class="chip"></span>
      <button id="theme-toggle" class="btn icon" aria-label="Toggle appearance">☾</button>
      <button id="inspector-toggle" class="btn icon" aria-expanded="true" aria-label="Toggle the detail panel">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
             stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
          <rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M14.5 4.5v15"/>
        </svg>
      </button>
    </header>

    <div id="error-banner" class="error" role="alert" hidden></div>

    <div class="workspace">
      <!-- The graph is the application surface; everything else floats on it. -->
      <div class="stage">
        <svg id="canvas" class="canvas" role="img" tabindex="-1"></svg>

        <div id="stage-caption" class="stage-caption" aria-live="polite">
          <h2 id="caption-title"></h2>
          <p id="caption-body"></p>
        </div>

        <section id="invite-panel" class="pairing-card card" hidden>
          <p class="eyebrow">Pair a second device</p>
          <h2 class="pairing-title">Scan to join as the worker</h2>
          <div id="qr-holder" class="qr"></div>
          <p class="pairing-code mono" id="invite-code">—</p>
          <p class="pairing-url mono" id="invite-link"></p>
          <div class="pairing-actions">
            <button id="copy-link" class="btn sm">Copy invite link</button>
            <button id="share-link" class="btn sm" hidden>Share</button>
          </div>
        </section>

        <div id="chat-layer" class="chat-layer" hidden>
          <div id="transcript" class="transcript" aria-live="polite"></div>
        </div>

        <!-- One floating cluster: live figures, the prompt, and the four
             groups of detail the inspector can show. -->
        <div class="dock">
          <div id="hud" class="hud" role="group" aria-label="Live generation metrics" hidden>
            ${hudStat("hud-rate", "Throughput", "tok/s")}
            ${hudStat("hud-ttft", "First token", "")}
            ${hudStat("hud-tokens", "Tokens", "")}
            ${hudStat("hud-wire", "On the wire", "")}
          </div>

          <form id="prompt-form" class="composer">
            <div class="composer-box">
              <textarea id="prompt-input" rows="1" placeholder="Waiting for both shards…"
                        disabled spellcheck="true"></textarea>
              <button id="send-button" type="submit" class="round send" aria-label="Send prompt">
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path d="M12 19V5M5 12l7-7 7 7" fill="none" stroke="currentColor"
                        stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
              <button id="stop-generation" type="button" class="round stop" hidden aria-label="Stop generation">
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/>
                </svg>
              </button>
            </div>
          </form>

          <div class="dock-rail">
            <button class="dock-chip" type="button" data-inspect="devices" aria-pressed="false">Devices</button>
            <button class="dock-chip" type="button" data-inspect="model" aria-pressed="false">Model &amp; split</button>
            <button class="dock-chip" type="button" data-inspect="perf" aria-pressed="false">Performance</button>
            <button class="dock-chip" type="button" data-inspect="session" aria-pressed="false">Session</button>
          </div>
          <small id="composer-hint" class="hint"></small>
        </div>
      </div>

      <div id="scrim" class="scrim"></div>

      <aside id="inspector" class="inspector" aria-label="Detail panel">
        <header class="inspector-head">
          <h2 id="inspector-title">Devices</h2>
          <button id="inspector-close" class="btn icon" aria-label="Close the detail panel">✕</button>
        </header>

        <div class="inspector-body">
          <section id="devices-panel" class="panel" data-group="devices">
            ${panelHead("Both devices")}
            ${peerRow("local", "This device")}
            <div class="wire" aria-hidden="true"><span></span><b>reliable · ordered</b><span></span></div>
            ${peerRow("remote", "Waiting for peer…")}
          </section>

          <section id="model-card" class="panel" data-group="model" hidden>
            ${panelHead("Model", `<span id="model-verdict" class="chip idle">sizing…</span>`)}
            <label class="sr-only" for="model-select">Model</label>
            <select id="model-select" class="model-select"></select>
            <p id="model-summary" class="status-text" aria-live="polite">Reading model header…</p>
          </section>

          <section id="split-panel" class="panel" data-group="model">
            ${panelHead(
              "Split",
              `<div class="preset-group" role="group" aria-label="Layer split presets">
                 <button id="load-even" class="preset" type="button"
                         title="Give each device the same number of layers">Even</button>
                 <button id="balance-split" class="preset" type="button"
                         title="Give each device a similar download size">Balanced</button>
                 <button id="load-power" class="preset" type="button"
                         title="Leave the tighter device the most memory headroom">Best fit</button>
               </div>`,
            )}
            <div class="workload">
              <div class="workload-head">
                <span class="workload-label">Workload</span>
                <span class="workload-readout mono">
                  <b id="host-load-pct" class="host-load">50%</b>
                  <span class="workload-sep">·</span>
                  <b id="worker-load-pct" class="worker-load">50%</b>
                </span>
              </div>
              <input id="load-slider" class="load-slider" type="range" min="1" max="99"
                     value="50" aria-label="Share of the model assigned to the host device">
            </div>

            <div class="split-legend">
              <div class="side host">
                <span class="side-role" id="host-side-role">Host</span>
                <div class="side-line">
                  <b class="mono" id="host-range">—</b>
                  <span class="side-size mono" id="host-size">—</span>
                </div>
              </div>
              <div class="side worker">
                <span class="side-role" id="worker-side-role">Worker</span>
                <div class="side-line">
                  <b class="mono" id="worker-range">—</b>
                  <span class="side-size mono" id="worker-size">—</span>
                </div>
              </div>
            </div>

            <button id="assign-layers" class="btn primary wide">Assign &amp; load</button>
            <p id="split-hint" class="status-text" aria-live="polite"></p>
          </section>

          <section id="cache-card" class="panel" data-group="model" hidden>
            ${panelHead("Weights", `<span id="cache-state" class="chip idle">checking…</span>`)}
            <p id="cache-summary" class="status-text" aria-live="polite"></p>
            <div class="cache-bar" role="img" aria-label="Share of this shard held in the cache">
              <i id="cache-fill"></i>
            </div>
            <button id="clear-cache" class="btn ghost sm wide">Clear cached weights</button>
          </section>

          <section id="telemetry" class="panel" data-group="perf" hidden>
            ${panelHead("This run", `<span id="metric-live" class="chip idle">idle</span>`)}
            <div class="metric-hero">
              <div class="stage-head">
                <span class="stage-title">Throughput</span>
                <b id="metric-spark-peak" class="mono">—</b>
              </div>
              <div class="hero-row">
                <b id="metric-rate" class="hero-value mono">—</b>
                <span class="hero-unit">tok/s</span>
                <svg id="latency-spark" class="spark" viewBox="0 0 100 28"
                     preserveAspectRatio="none" role="img"
                     aria-label="Latency of each recent token">
                  <path id="spark-area" class="spark-area" d=""></path>
                  <path id="spark-line" class="spark-line" d=""></path>
                </svg>
              </div>
            </div>

            <div class="metric-tiles">
              ${metricTile("metric-ttft", "First token", ICONS.clock, "—")}
              ${metricTile("metric-tokens", "Tokens", ICONS.stack, "0")}
              ${metricTile("metric-per-token", "Per token", ICONS.gauge, "—")}
              ${metricTile("metric-bytes", "On the wire", ICONS.swap, "—", true)}
            </div>

            <div class="stage-block">
              <div class="stage-head">
                <span class="stage-title">Where a token's time goes</span>
              </div>
              <div id="stage-bar" class="stage-bar" role="img" aria-label="Stage breakdown">
                ${["host", "wire", "worker", "head"]
                  .map((stage) => `<i data-stage="${stage}"></i>`)
                  .join("")}
              </div>
              <ul id="stage-legend" class="stage-legend">
                ${stageLegendRow("host", "Host layers")}
                ${stageLegendRow("wire", "Network")}
                ${stageLegendRow("worker", "Peer layers")}
                ${stageLegendRow("head", "Head + sample")}
              </ul>
            </div>

            <dl class="wire-stats">
              <div><dt>Sent · received</dt><dd id="metric-bytes-total" class="mono">—</dd></div>
            </dl>
          </section>

          <section id="session-panel" class="panel" data-group="session">
            ${panelHead("Cluster")}
            <dl class="wire-stats">
              <div><dt>Code</dt><dd id="session-code" class="mono">—</dd></div>
            </dl>
            <p class="status-text">
              Signaling runs through PeerJS; every activation then travels directly
              between the two browsers over WebRTC.
            </p>
            <button id="reset-chat" class="btn ghost sm wide" disabled>Reset conversation</button>
            <button id="leave-cluster" class="btn ghost sm wide">Leave cluster</button>
          </section>
        </div>
      </aside>
    </div>
  </section>`;
}
