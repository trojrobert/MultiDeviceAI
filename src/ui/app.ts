import { createRoomCode, normalizeRoomCode } from "../runtime/peer.ts";
import { RoomController, type RoomSnapshot } from "../runtime/room.ts";
import type { DeviceCapabilities } from "../runtime/protocol.ts";
import type { PlacementResult, SplitCandidate } from "../engine/placement.ts";
import { encodeQR, qrToSvg } from "./qr.ts";

const THEME_KEY = "mdai-theme";
const NAME_KEY = "mdllm-device-name";
const BUDGET_KEY = "mdai-memory-budget";
const COLLAPSED_KEY = "mdai-collapsed-cards";

type Theme = "dark" | "light";

export function mountApp(root: HTMLElement): () => void {
  root.innerHTML = shell();

  const room = new RoomController({ onChange: render });
  let snapshot = room.snapshot;
  let budgetRestored = false;

  // ── Element handles ─────────────────────────────────────────────────────
  const joinView = get("join-view");
  const roomView = get("room-view");
  const nameInput = input("device-name");
  const codeInput = input("room-code");
  const joinStatus = get("join-status");
  const createButton = button("create-room");
  const joinButton = button("join-room");
  const themeToggle = button("theme-toggle");
  const copyButton = button("copy-link");
  const shareButton = button("share-link");
  const inviteLink = get("invite-link");
  const qrHolder = get("qr-holder");
  const invitePanel = get("invite-panel");
  const ribbon = get("layer-ribbon");
  const ribbonCells = get("ribbon-cells");
  const assignButton = button("assign-layers");
  const balanceButton = button("balance-split");
  const modelSelect = get("model-select") as HTMLSelectElement;
  const localBudgetInput = get("local-budget") as HTMLInputElement;
  const loadSlider = input("load-slider");
  const loadEvenButton = button("load-even");
  const loadPowerButton = button("load-power");
  const promptForm = get("prompt-form") as HTMLFormElement;
  const promptInput = get("prompt-input") as HTMLTextAreaElement;
  const stopButton = button("stop-generation");
  const resetButton = button("reset-chat");
  const transcriptEl = get("transcript");
  const clusterToggle = button("cluster-toggle");
  const sheetBackdrop = get("sheet-backdrop");
  const toasts = get("toasts");

  // ── Local UI state ──────────────────────────────────────────────────────
  let proposedSplit = 0;
  let ribbonLayerCount = 0;
  let dragging = false;
  let lastInviteCode = "";
  let generationStart = 0;
  let tokenCount = 0;
  let lastPendingText = "";
  let wasGenerating = false;
  let tokenTimes: number[] = [];

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
      ?.setAttribute("content", theme === "dark" ? "#07090F" : "#EEF1F7");
  }

  // ── Join view ───────────────────────────────────────────────────────────
  const urlCode = normalizeRoomCode(
    new URLSearchParams(location.search).get("room") ?? "",
  );
  if (urlCode) codeInput.value = urlCode;
  nameInput.value =
    localStorage.getItem(NAME_KEY) ??
    (/Mobi|Android/i.test(navigator.userAgent) ? "Phone" : "Laptop");

  if (urlCode) {
    joinStatus.textContent = `Invite for room ${urlCode} detected. Join to become the worker.`;
    joinButton.classList.add("primary");
    createButton.classList.remove("primary");
  }

  createButton.addEventListener("click", () => {
    const code = createRoomCode();
    codeInput.value = code;
    void start("host", code);
  });
  joinButton.addEventListener("click", () => {
    const code = normalizeRoomCode(codeInput.value);
    if (!code) {
      joinStatus.textContent = "Enter the room code shown on the host.";
      codeInput.focus();
      return;
    }
    void start("worker", code);
  });
  codeInput.addEventListener("input", () => {
    codeInput.value = normalizeRoomCode(codeInput.value);
  });
  codeInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") joinButton.click();
  });

  async function start(role: "host" | "worker", code: string): Promise<void> {
    const name = nameInput.value.trim() || (role === "host" ? "Host" : "Worker");
    localStorage.setItem(NAME_KEY, name);
    setJoinBusy(true);
    try {
      await room.start(role, code, name);
      const url = new URL(location.href);
      url.searchParams.set("room", code);
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
    url.searchParams.set("room", snapshot.roomCode ?? "");
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
        title: "MultiDevice AI",
        text: "Join my split-model room",
        url,
      });
    } catch {
      /* share cancelled or unsupported */
    }
  });

  // ── Layer ribbon ────────────────────────────────────────────────────────
  function ensureRibbon(layerCount: number): void {
    if (ribbonLayerCount === layerCount) return;
    ribbonLayerCount = layerCount;
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < layerCount; i++) {
      const cell = document.createElement("span");
      cell.className = "layer";
      cell.title = `Layer ${i}`;
      fragment.append(cell);
    }
    ribbonCells.replaceChildren(fragment);
  }

  function ribbonInteractive(next: Readonly<RoomSnapshot>): boolean {
    return (
      next.role === "host" &&
      next.connected &&
      next.engineAvailable &&
      next.localPhase !== "loading" &&
      !next.generating
    );
  }

  function splitFromPointer(clientX: number): number {
    const rect = ribbonCells.getBoundingClientRect();
    if (rect.width === 0) return proposedSplit;
    const ratio = (clientX - rect.left) / rect.width;
    return clamp(Math.round(ratio * ribbonLayerCount), 1, ribbonLayerCount - 1);
  }

  function setSplit(value: number): void {
    const next = clamp(value, 1, Math.max(1, ribbonLayerCount - 1));
    if (next === proposedSplit) return;
    proposedSplit = next;
    paintRibbon(snapshot);
  }

  ribbon.addEventListener("pointerdown", (event) => {
    if (!ribbonInteractive(snapshot)) return;
    dragging = true;
    ribbon.setPointerCapture(event.pointerId);
    setSplit(splitFromPointer(event.clientX));
    event.preventDefault();
  });
  ribbon.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setSplit(splitFromPointer(event.clientX));
  });
  const endDrag = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    if (ribbon.hasPointerCapture(event.pointerId)) {
      ribbon.releasePointerCapture(event.pointerId);
    }
  };
  ribbon.addEventListener("pointerup", endDrag);
  ribbon.addEventListener("pointercancel", endDrag);

  ribbon.addEventListener("keydown", (event) => {
    if (!ribbonInteractive(snapshot)) return;
    const step = event.shiftKey ? 4 : 1;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      setSplit(proposedSplit - step);
    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      setSplit(proposedSplit + step);
    } else if (event.key === "Home") {
      setSplit(1);
    } else if (event.key === "End") {
      setSplit(ribbonLayerCount - 1);
    } else {
      return;
    }
    event.preventDefault();
  });

  balanceButton.addEventListener("click", () => {
    const balanced = snapshot.placement?.balanced;
    if (!balanced) return;
    setSplit(balanced.split);
    toast("Split balanced by download size");
  });

  loadSlider.addEventListener("input", () => {
    if (!ribbonInteractive(snapshot)) return;
    setSplit(Math.round((Number(loadSlider.value) / 100) * ribbonLayerCount));
  });

  loadEvenButton.addEventListener("click", () => {
    setSplit(Math.round(ribbonLayerCount / 2));
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
    void room.selectModel(modelSelect.value);
  });

  localBudgetInput.addEventListener("change", () => {
    const gb = Number(localBudgetInput.value);
    if (!Number.isFinite(gb) || gb <= 0) return;
    localStorage.setItem(BUDGET_KEY, String(gb));
    room.setBudgetBytes(gb * 1024 ** 3);
  });

  assignButton.addEventListener("click", () => {
    void room.assignSplit(proposedSplit);
  });

  function paintRibbon(next: Readonly<RoomSnapshot>): void {
    const layerCount = ribbonLayerCount;
    if (layerCount === 0) return;

    const assigned = next.localAssignment;
    const hostLayers = assigned
      ? next.role === "host"
        ? assigned.end
        : assigned.start
      : proposedSplit;
    const workerLayers = layerCount - hostLayers;

    const hostIsLocal = next.role === "host";
    const hostProgress = hostIsLocal ? next.localProgress : next.remoteProgress;
    const workerProgress = hostIsLocal ? next.remoteProgress : next.localProgress;
    const hostLoaded = assigned ? Math.round(hostProgress * hostLayers) : 0;
    const workerLoaded = assigned ? Math.round(workerProgress * workerLayers) : 0;

    const cells = ribbonCells.children;
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i] as HTMLElement;
      const onHost = i < hostLayers;
      const indexInSide = onHost ? i : i - hostLayers;
      const loaded = onHost ? indexInSide < hostLoaded : indexInSide < workerLoaded;
      cell.className = `layer ${onHost ? "host" : "worker"}${loaded ? " loaded" : ""}${
        onHost && i === hostLayers - 1 ? " edge" : ""
      }`;
    }

    const interactive = ribbonInteractive(next);
    ribbon.classList.toggle("interactive", interactive);
    ribbon.classList.toggle("locked", !interactive);
    ribbon.classList.toggle("streaming", next.generating);
    ribbon.tabIndex = interactive ? 0 : -1;
    ribbon.setAttribute("aria-valuenow", String(hostLayers));
    ribbon.setAttribute("aria-valuemin", "1");
    ribbon.setAttribute("aria-valuemax", String(Math.max(1, layerCount - 1)));
    ribbon.setAttribute(
      "aria-valuetext",
      `Host owns ${hostLayers} layers, worker owns ${workerLayers}`,
    );
    ribbon.style.setProperty("--split", String(hostLayers / layerCount));

    text("host-range", hostLayers > 0 ? `0–${hostLayers - 1}` : "—");
    text("worker-range", workerLayers > 0 ? `${hostLayers}–${layerCount - 1}` : "—");
    // Exact numbers once assigned, otherwise the probed model's own byte layout.
    const candidate = candidateAt(next.placement, hostLayers);
    const hostBytes = assigned
      ? next.role === "host"
        ? next.localAssignment?.bytes
        : next.remoteAssignment?.bytes
      : candidate?.hostDownloadBytes;
    const workerBytes = assigned
      ? next.role === "host"
        ? next.remoteAssignment?.bytes
        : next.localAssignment?.bytes
      : candidate?.workerDownloadBytes;
    text("host-size", hostBytes === undefined ? "—" : formatBytes(hostBytes));
    text("worker-size", workerBytes === undefined ? "—" : formatBytes(workerBytes));
    text("host-side-role", hostIsLocal ? "Host · you" : "Host · peer");
    text("worker-side-role", hostIsLocal ? "Worker · peer" : "Worker · you");

    const assignLabel = assigned ? "Reassign & reload" : "Assign & load";
    assignButton.textContent = assignLabel;
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

    syncLoadControl(hostLayers, layerCount, interactive);
  }

  function syncLoadControl(
    hostLayers: number,
    layerCount: number,
    interactive: boolean,
  ): void {
    const hostPercent = Math.round((hostLayers / layerCount) * 100);
    loadSlider.style.setProperty("--load", String(hostLayers / layerCount));
    // Don't yank the thumb while the user is dragging the slider itself; other
    // controls (ribbon, presets, Balance) still push their value in here.
    if (document.activeElement !== loadSlider) {
      loadSlider.value = String(hostPercent);
    }
    loadSlider.disabled = !interactive;
    loadSlider.setAttribute(
      "aria-valuetext",
      `Host handles ${hostPercent}%, worker handles ${100 - hostPercent}%`,
    );
    loadEvenButton.disabled = !interactive;
    loadPowerButton.disabled = !interactive || !snapshot.placement?.recommended;
    text("host-load-pct", `${hostPercent}%`);
    text("worker-load-pct", `${100 - hostPercent}%`);
  }

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
    void room.submitPrompt(value);
  });
  stopButton.addEventListener("click", () => room.stop());
  resetButton.addEventListener("click", () => void room.reset());

  // ── Collapsible settings cards ──────────────────────────────────────────
  setupCollapsibleCards(root);

  // ── Mobile cluster sheet ────────────────────────────────────────────────
  const setSheet = (open: boolean) => {
    roomView.classList.toggle("sheet-open", open);
    clusterToggle.setAttribute("aria-expanded", String(open));
  };
  clusterToggle.addEventListener("click", () =>
    setSheet(!roomView.classList.contains("sheet-open")),
  );
  sheetBackdrop.addEventListener("click", () => setSheet(false));
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") setSheet(false);
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
  function render(next: Readonly<RoomSnapshot>): void {
    snapshot = next;
    const active = Boolean(next.role);
    joinView.hidden = active;
    roomView.hidden = !active;
    if (!active) return;

    // Layer count is per-model, so the ribbon stays empty until the host has
    // probed the GGUF header or the worker has been told its assignment.
    const layerCount =
      next.modelProfile?.layerCount ??
      next.localAssignment?.layerCount ??
      next.remoteAssignment?.layerCount ??
      0;
    ensureRibbon(layerCount);
    if (proposedSplit === 0 && layerCount > 0) {
      proposedSplit = next.placement?.recommended?.split ?? Math.round(layerCount / 2);
    }

    text("room-code-display", next.roomCode ?? "—");
    text("role-display", next.role === "host" ? "Host" : "Worker");
    text("room-status", next.status);
    text("engine-mode", next.engineAvailable ? "engine live" : "transport only");
    get("engine-mode").className = `chip ${next.engineAvailable ? "ok" : "warn"}`;
    get("status-dot").className = `dot ${phaseTone(next.localPhase)}`;
    const banner = get("error-banner");
    banner.hidden = !next.error;
    text("error-banner", next.error ?? "");

    renderStepper(next);
    renderInvite(next);
    renderModel(next);
    // The user's own budget outlives the session; apply it as soon as the first
    // capability probe gives us something to override.
    if (!budgetRestored && next.localCapabilities) {
      budgetRestored = true;
      const stored = Number(localStorage.getItem(BUDGET_KEY));
      if (Number.isFinite(stored) && stored > 0) room.setBudgetBytes(stored * 1024 ** 3);
    }

    renderPeer("local", next.localCapabilities, next.localPhase, next.localProgress, true);
    renderPeer("remote", next.remoteCapabilities, next.remotePhase, next.remoteProgress, false);
    paintRibbon(next);
    renderTelemetry(next);
    renderTranscript(next);

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

  function renderStepper(next: Readonly<RoomSnapshot>): void {
    const done = [
      true,
      next.connected,
      Boolean(next.localAssignment),
      next.ready,
    ];
    const current = done.findIndex((value) => !value);
    const steps = get("stepper").children;
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i] as HTMLElement;
      step.className = `step ${
        done[i] ? "done" : i === current ? "active" : "todo"
      }`;
    }
  }

  function renderInvite(next: Readonly<RoomSnapshot>): void {
    const show = next.role === "host" && !next.connected;
    invitePanel.hidden = !show;
    transcriptEl.hidden = show;
    promptForm.hidden = show;
    if (!show) return;
    const url = inviteUrl();
    inviteLink.textContent = url.replace(/^https?:\/\//, "");
    shareButton.hidden = typeof navigator.share !== "function";
    if (lastInviteCode === url) return;
    lastInviteCode = url;
    try {
      qrHolder.innerHTML = qrToSvg(encodeQR(url), 2);
    } catch {
      qrHolder.replaceChildren();
    }
  }

  function renderModel(next: Readonly<RoomSnapshot>): void {
    const card = get("model-card");
    // Only the host chooses; the worker is told what to load.
    card.hidden = next.role !== "host" || next.catalogue.length === 0;
    if (card.hidden) return;

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
    modelSelect.disabled = next.modelPhase === "probing" || next.localPhase === "loading" || next.generating;

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
  ): void {
    text(
      `${prefix}-name`,
      capabilities?.label ?? (isLocal ? "This device" : "Waiting for peer…"),
    );
    text(`${prefix}-phase`, phase);
    get(`${prefix}-phase`).className = `phase ${phaseTone(phase)}`;
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
    // tooltip, so the rail does not grow a second row per device.
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

  function renderTelemetry(next: Readonly<RoomSnapshot>): void {
    const pending = [...next.transcript].reverse().find((entry) => entry.pending);

    if (next.generating && !wasGenerating) {
      generationStart = performance.now();
      tokenCount = 0;
      tokenTimes = [];
      lastPendingText = "";
    }
    if (next.generating && pending && pending.text !== lastPendingText) {
      lastPendingText = pending.text;
      tokenCount++;
      tokenTimes.push(performance.now());
      if (tokenTimes.length > 24) tokenTimes.shift();
    }
    wasGenerating = next.generating;

    let rate = 0;
    if (tokenTimes.length >= 2) {
      const span = tokenTimes[tokenTimes.length - 1] - tokenTimes[0];
      if (span > 0) rate = ((tokenTimes.length - 1) / span) * 1000;
    }

    const hasRun = tokenCount > 0;
    get("telemetry").hidden = !hasRun;
    text("metric-tokens", String(tokenCount));
    text("metric-rate", rate > 0 ? rate.toFixed(1) : "—");
    text(
      "metric-elapsed",
      hasRun ? `${((performance.now() - generationStart) / 1000).toFixed(1)}s` : "—",
    );
  }

  function renderTranscript(next: Readonly<RoomSnapshot>): void {
    const nearBottom =
      transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight <
      120;

    if (next.transcript.length === 0) {
      transcriptEl.replaceChildren(emptyState(next));
    } else {
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
    }

    if (nearBottom) {
      requestAnimationFrame(() => {
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      });
    }
  }

  function emptyState(next: Readonly<RoomSnapshot>): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "empty";
    const visual = document.createElement("div");
    visual.className = "empty-visual";
    visual.innerHTML = `<span></span><span></span><span></span>`;
    const title = document.createElement("h3");
    const body = document.createElement("p");

    if (!next.connected) {
      title.textContent = "Waiting for the second device";
      body.textContent =
        next.role === "host"
          ? "Scan the invite code with your phone to pair it as the worker."
          : "Connecting to the host room…";
    } else if (!next.localAssignment) {
      title.textContent = "Choose how to divide the model";
      body.textContent =
        next.role === "host"
          ? "Drag the ribbon to set the split point, then assign and load."
          : "The host is choosing which layers this device will own.";
    } else if (!next.ready) {
      title.textContent = "Loading model shards";
      body.textContent =
        "Each device downloads only the tensors for the layers it owns.";
    } else {
      title.textContent = "All devices ready";
      body.textContent =
        "Every token crosses the network. Send a message to watch it happen.";
    }

    wrap.append(visual, title, body);
    return wrap;
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

  render(snapshot);
  resizeComposer();

  return () => {
    window.removeEventListener("keydown", onKeyDown);
    void room.leave();
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Every card the rail folds into its single scrollable settings panel. */
function setupCollapsibleCards(root: HTMLElement): void {
  let collapsed: Set<string>;
  try {
    collapsed = new Set(JSON.parse(localStorage.getItem(COLLAPSED_KEY) ?? "[]"));
  } catch {
    collapsed = new Set();
  }

  root.querySelectorAll<HTMLElement>(".card[id] > .card-head > .card-toggle").forEach(
    (toggle) => {
      const card = toggle.closest<HTMLElement>(".card[id]");
      if (!card) return;
      const id = card.id;

      const apply = (isCollapsed: boolean) => {
        card.classList.toggle("collapsed", isCollapsed);
        toggle.setAttribute("aria-expanded", String(!isCollapsed));
        toggle.setAttribute("aria-label", isCollapsed ? "Expand section" : "Collapse section");
      };
      apply(collapsed.has(id));

      toggle.addEventListener("click", () => {
        const next = !card.classList.contains("collapsed");
        apply(next);
        if (next) collapsed.add(id);
        else collapsed.delete(id);
        localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...collapsed]));
      });
    },
  );
}

function cardToggle(): string {
  return `<button class="card-toggle" type="button" aria-expanded="true" aria-label="Collapse section">
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </button>`;
}

/** The candidate describing a given split, or undefined before the model is probed. */
function candidateAt(placement: PlacementResult | undefined, split: number): SplitCandidate | undefined {
  return placement?.candidates[split - 1];
}

/**
 * Short enough to sit beside the card title at any rail width. The full
 * reasoning is the summary sentence directly below it, so the chip only has to
 * carry the verdict itself.
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

function shell(): string {
  return `
  <div class="backdrop" aria-hidden="true"><i></i><i></i><i></i></div>
  <div id="toasts" class="toasts" role="status" aria-live="polite"></div>

  <section id="join-view" class="join">
    <div class="join-inner">
      <div class="join-brand">
        <div class="logo" aria-hidden="true"><i></i><i></i><i></i></div>
        <p class="join-appname">Multi<b>Device AI</b></p>
      </div>
      <h1>One Model<br><em>Run locally on multiple devices</em></h1>
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
          <label class="field" for="room-code">
            <span>Room code</span>
            <input id="room-code" class="mono code" maxlength="8" placeholder="ABC123" autocomplete="off" spellcheck="false">
          </label>
        </div>
        <div class="join-actions">
          <button id="create-room" class="btn primary">Create room</button>
          <button id="join-room" class="btn">Join room</button>
        </div>
        <p id="join-status" class="hint" aria-live="polite">
          Start a room on any device, then share the invite link with a second device.
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

  <section id="room-view" class="room" hidden>
    <header class="topbar">
      <a class="wordmark" href="./" aria-label="MultiDevice AI home">
        <span class="logo sm" aria-hidden="true"><i></i><i></i><i></i></span>
        multi<b>device</b>
      </a>
      <div class="topbar-spacer"></div>
      <div class="room-pill"><span>Room</span><strong id="room-code-display" class="mono">—</strong></div>
      <span id="engine-mode" class="chip"></span>
      <button id="copy-link" class="btn ghost sm">Copy invite</button>
      <button id="theme-toggle" class="btn icon" aria-label="Toggle appearance">☾</button>
      <button id="cluster-toggle" class="btn icon cluster" aria-expanded="false" aria-label="Show cluster details">
        <span id="status-dot" class="dot"></span>
      </button>
    </header>

    <div id="error-banner" class="error" role="alert" hidden></div>

    <div class="workspace">
      <div id="sheet-backdrop" class="sheet-backdrop"></div>

      <aside class="rail">
        <ol id="stepper" class="stepper">
          <li class="step"><i></i><span>Room</span></li>
          <li class="step"><i></i><span>Pair</span></li>
          <li class="step"><i></i><span>Split</span></li>
          <li class="step"><i></i><span>Ready</span></li>
        </ol>

        <section id="status-card" class="card status-card">
          <div class="card-head">
            <h2>Session</h2>
            <b id="role-display" class="role">—</b>
            ${cardToggle()}
          </div>
          <div class="card-body">
            <p id="room-status" class="status-text" aria-live="polite"></p>
          </div>
        </section>

        <section id="peers-card" class="card peers">
          <div class="card-head">
            <h2>Devices</h2>
            ${cardToggle()}
          </div>
          <div class="card-body">
            ${peerRow("local", "This device")}
            <div class="wire" aria-hidden="true"><span></span><b>reliable · ordered</b><span></span></div>
            ${peerRow("remote", "Waiting for peer…")}
          </div>
        </section>

        <section id="model-card" class="card model-card" hidden>
          <div class="card-head">
            <h2>Model</h2>
            <span id="model-verdict" class="chip idle">sizing…</span>
            ${cardToggle()}
          </div>
          <div class="card-body">
            <label class="sr-only" for="model-select">Model</label>
            <select id="model-select" class="model-select"></select>
            <p id="model-summary" class="status-text" aria-live="polite">Reading model header…</p>
          </div>
        </section>

        <section id="split-card" class="card split-card">
          <div class="card-head">
            <h2>Split</h2>
            <div class="preset-group" role="group" aria-label="Layer split presets">
              <button id="load-even" class="preset" type="button"
                      title="Give each device the same number of layers">Even</button>
              <button id="balance-split" class="preset" type="button"
                      title="Give each device a similar download size">Balanced</button>
              <button id="load-power" class="preset" type="button"
                      title="Leave the tighter device the most memory headroom">Best fit</button>
            </div>
            ${cardToggle()}
          </div>
          <div class="card-body">
            <div class="ribbon-wrap">
              <div id="layer-ribbon" class="ribbon" role="slider" tabindex="0"
                   aria-label="Transformer layer split point">
                <div id="ribbon-cells" class="ribbon-cells"></div>
                <div class="ribbon-handle" aria-hidden="true"></div>
                <div class="ribbon-pulse" aria-hidden="true"></div>
              </div>
            </div>

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
          </div>
        </section>

        <section id="telemetry" class="card metrics" hidden>
          <div class="metric-grid">
            <div><b id="metric-tokens" class="mono">0</b><span>tokens</span></div>
            <div><b id="metric-rate" class="mono">—</b><span>tok/s</span></div>
            <div><b id="metric-elapsed" class="mono">—</b><span>elapsed</span></div>
          </div>
        </section>
      </aside>

      <section class="chat">
        <div class="chat-head">
          <div>
            <p class="eyebrow">Distributed session</p>
            <h2>Split model <span class="tag mono">P2P</span></h2>
          </div>
          <button id="reset-chat" class="btn ghost sm" disabled>Reset</button>
        </div>

        <section id="invite-panel" class="invite" hidden>
          <h2 class="invite-title">Pair your phone</h2>
          <p class="invite-sub">Scan this code with your phone's camera to join as the second device.</p>
          <div id="qr-holder" class="qr"></div>
          <p class="invite-url mono" id="invite-link"></p>
          <div class="invite-actions">
            <button id="share-link" class="btn sm" hidden>Share</button>
          </div>
        </section>

        <div id="transcript" class="transcript" aria-live="polite"></div>

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
          <small id="composer-hint" class="hint"></small>
        </form>
      </section>
    </div>
  </section>`;
}

function peerRow(prefix: string, fallback: string): string {
  return `
    <article class="peer">
      <div class="peer-top">
        <h3 id="${prefix}-name">${fallback}</h3>
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
