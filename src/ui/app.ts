import {
  createRoomCode,
  normalizeRoomCode,
} from "../runtime/peer.ts";
import {
  RoomController,
  type RoomSnapshot,
} from "../runtime/room.ts";
import type { DeviceCapabilities } from "../runtime/protocol.ts";

export function mountApp(root: HTMLElement): () => void {
  root.innerHTML = shell();
  const room = new RoomController({ onChange: render });
  let snapshot = room.snapshot;

  const joinView = get("join-view");
  const roomView = get("room-view");
  const nameInput = input("device-name");
  const codeInput = input("room-code");
  const joinStatus = get("join-status");
  const createButton = button("create-room");
  const joinButton = button("join-room");
  const copyButton = button("copy-link");
  const splitInput = input("split-input");
  const assignButton = button("assign-layers");
  const promptForm = get("prompt-form") as HTMLFormElement;
  const promptInput = input("prompt-input");
  const stopButton = button("stop-generation");
  const resetButton = button("reset-chat");
  const emptyChat = get("empty-chat");

  const urlCode = normalizeRoomCode(
    new URLSearchParams(location.search).get("room") ?? "",
  );
  if (urlCode) codeInput.value = urlCode;
  nameInput.value =
    localStorage.getItem("mdllm-device-name") ??
    (/Mobi|Android/i.test(navigator.userAgent) ? "phone" : "laptop");

  createButton.addEventListener("click", () => {
    const code = createRoomCode();
    codeInput.value = code;
    void start("host", code);
  });
  joinButton.addEventListener("click", () => {
    const code = normalizeRoomCode(codeInput.value);
    if (!code) {
      joinStatus.textContent = "Enter the room code from the host.";
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
  copyButton.addEventListener("click", async () => {
    const url = new URL(location.href);
    url.searchParams.set("room", snapshot.roomCode ?? "");
    try {
      await navigator.clipboard.writeText(url.toString());
      copyButton.textContent = "Copied";
      setTimeout(() => (copyButton.textContent = "Copy invite"), 1400);
    } catch {
      window.prompt("Copy this invite link", url.toString());
    }
  });
  assignButton.addEventListener("click", () => {
    void room.assignSplit(Number(splitInput.value));
  });
  promptForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = promptInput.value;
    if (!text.trim()) return;
    promptInput.value = "";
    void room.submitPrompt(text);
  });
  stopButton.addEventListener("click", () => room.stop());
  resetButton.addEventListener("click", () => void room.reset());

  async function start(role: "host" | "worker", code: string): Promise<void> {
    const name = nameInput.value.trim() || role;
    localStorage.setItem("mdllm-device-name", name);
    setJoinBusy(true);
    try {
      await room.start(role, code, name);
      const url = new URL(location.href);
      url.searchParams.set("room", code);
      history.replaceState({}, "", url);
    } catch (error) {
      joinStatus.textContent =
        error instanceof Error ? error.message : String(error);
      setJoinBusy(false);
    }
  }

  function setJoinBusy(busy: boolean): void {
    createButton.disabled = busy;
    joinButton.disabled = busy;
  }

  function render(next: Readonly<RoomSnapshot>): void {
    snapshot = next;
    const active = Boolean(next.role);
    joinView.hidden = active;
    roomView.hidden = !active;
    if (!active) return;

    text("room-code-display", next.roomCode ?? "—");
    text("role-display", next.role ?? "—");
    text("room-status", next.status);
    text("engine-mode", next.engineAvailable ? "engine connected" : "transport preview");
    get("status-dot").className = `status-dot ${phaseTone(next.localPhase)}`;
    get("error-banner").hidden = !next.error;
    text("error-banner", next.error ?? "");

    renderPeer("local", next.localCapabilities, next.localPhase, next.localProgress);
    renderPeer("remote", next.remoteCapabilities, next.remotePhase, next.remoteProgress);

    const hostControls = get("host-controls");
    hostControls.hidden = next.role !== "host";
    const layerCount =
      next.localAssignment?.layerCount ??
      next.remoteAssignment?.layerCount ??
      28;
    splitInput.max = String(layerCount - 1);
    if (!next.localAssignment) splitInput.value = String(Math.ceil(layerCount / 2));
    assignButton.disabled =
      !next.connected || next.localPhase === "loading" || !next.engineAvailable;
    text(
      "assignment-summary",
      assignmentText(next),
    );

    renderTranscript(next);
    promptInput.disabled = !next.ready || next.generating;
    promptInput.placeholder = next.ready
      ? next.role === "host"
        ? "Message the distributed model…"
        : "Send a prompt to the host…"
      : next.engineAvailable
        ? "Waiting for both model shards…"
        : "Connect the engine implementation to enable chat";
    stopButton.disabled = !next.generating;
    resetButton.disabled = next.transcript.length === 0;
    text("composer-hint", next.ready ? "Enter to send" : "Chat unlocks when both shards are ready");
  }

  function renderPeer(
    prefix: "local" | "remote",
    capabilities: DeviceCapabilities | undefined,
    phase: string,
    progress: number,
  ): void {
    text(`${prefix}-name`, capabilities?.label ?? (prefix === "local" ? "This device" : "Waiting for peer"));
    text(`${prefix}-phase`, phase);
    text(
      `${prefix}-gpu`,
      capabilities
        ? capabilities.webgpu
          ? capabilities.gpu ?? "WebGPU available"
          : "WebGPU unavailable"
        : "Capabilities pending",
    );
    text(
      `${prefix}-memory`,
      capabilities?.maxBufferSize
        ? `${formatBytes(capabilities.maxBufferSize)} max buffer`
        : "memory unknown",
    );
    const fill = get(`${prefix}-progress`);
    fill.style.width = `${Math.round(progress * 100)}%`;
    fill.parentElement?.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
  }

  function renderTranscript(next: Readonly<RoomSnapshot>): void {
    const transcript = get("transcript");
    if (next.transcript.length === 0) {
      emptyChat.hidden = false;
      transcript.replaceChildren(emptyChat);
    } else {
      emptyChat.hidden = true;
      transcript.replaceChildren(
        ...next.transcript.map((entry) => {
        const article = document.createElement("article");
        article.className = `message ${entry.role}`;
        const label = document.createElement("span");
        label.className = "message-label";
        label.textContent = entry.role === "assistant" ? "Distributed model" : entry.role;
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        if (entry.pending && !entry.text) {
          const typing = document.createElement("span");
          typing.className = "typing";
          typing.setAttribute("aria-label", "Generating response");
          typing.append(
            document.createElement("i"),
            document.createElement("i"),
            document.createElement("i"),
          );
          bubble.append(typing);
        } else {
          bubble.textContent = entry.text;
          if (entry.pending) {
            const cursor = document.createElement("span");
            cursor.className = "cursor";
            bubble.append(cursor);
          }
        }
        article.append(label, bubble);
        return article;
        }),
      );
    }
    requestAnimationFrame(() => {
      transcript.scrollTop = transcript.scrollHeight;
    });
  }

  function assignmentText(next: Readonly<RoomSnapshot>): string {
    if (!next.localAssignment) {
      return next.role === "host"
        ? "Choose where the contiguous layer ranges split."
        : "The host will assign this device a contiguous layer range.";
    }
    const local = next.localAssignment;
    const remote = next.remoteAssignment;
    const own = `You: layers ${local.start}–${local.end - 1}`;
    return remote
      ? `${own} · Peer: layers ${remote.start}–${remote.end - 1}`
      : own;
  }

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
  return () => void room.leave();
}

function phaseTone(phase: string): string {
  if (phase === "ready" || phase === "generating") return "good";
  if (phase === "error") return "bad";
  return "warm";
}

function formatBytes(bytes: number): string {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(1)} GB`;
  return `${Math.round(bytes / 2 ** 20)} MB`;
}

function shell(): string {
  return `
    <main class="app-shell">
      <div class="aurora" aria-hidden="true"></div>
      <section id="join-view" class="join-view">
        <div class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></div>
        <p class="eyebrow">MultiDevice AI</p>
        <h1>One model.<br><em>Two devices.</em></h1>
        <p class="lede">Split Qwen across a laptop and phone, then pass activations directly over WebRTC.</p>
        <div class="join-card">
          <div class="input-group">
            <label class="input-cell" for="device-name">
              <span class="field-label">Device name</span>
              <input id="device-name" maxlength="32" autocomplete="nickname" placeholder="My laptop">
            </label>
            <div class="input-divider"></div>
            <label class="input-cell" for="room-code">
              <span class="field-label">Room code</span>
              <input id="room-code" class="room-code-input" maxlength="8" placeholder="ABC123" autocomplete="off">
            </label>
          </div>
          <div class="join-actions">
            <button id="create-room" class="primary">Create room</button>
            <button id="join-room">Join room</button>
          </div>
          <p id="join-status" class="form-status" aria-live="polite">Create on the host, then open its invite on the worker.</p>
        </div>
        <p class="secure-note">WebGPU on a second device requires an HTTPS origin.</p>
      </section>

      <section id="room-view" class="room-view" hidden>
        <header class="topbar">
          <a class="wordmark" href="./">multi<span>device</span></a>
          <div class="room-pill"><span>Room</span><strong id="room-code-display">—</strong></div>
          <button id="copy-link" class="quiet">Copy invite</button>
          <span id="engine-mode" class="mode-chip"></span>
        </header>
        <div id="error-banner" class="error-banner" role="alert" hidden></div>
        <div class="workspace">
          <aside class="sidebar">
            <section class="status-panel">
              <div class="section-heading"><span class="status-dot" id="status-dot"></span><h2>Room status</h2><b id="role-display"></b></div>
              <p id="room-status" aria-live="polite"></p>
            </section>
            <div class="peer-list">
              ${peerCard("local", "THIS DEVICE")}
              <div class="link-line"><span></span><b>reliable · ordered · p2p</b><span></span></div>
              ${peerCard("remote", "PEER DEVICE")}
            </div>
            <section id="host-controls" class="split-panel" hidden>
              <div class="section-heading"><h2>Layer assignment</h2></div>
              <label for="split-input"><span>First worker layer</span><input id="split-input" type="number" min="1" value="14"></label>
              <button id="assign-layers" class="primary">Assign &amp; load</button>
            </section>
            <p id="assignment-summary" class="assignment-summary"></p>
          </aside>

          <section class="chat-panel">
            <div class="chat-heading">
              <div><p class="eyebrow">Distributed session</p><h2>Qwen3 0.6B <span>Q8_0</span></h2></div>
              <button id="reset-chat" class="quiet" disabled>Reset</button>
            </div>
            <div id="transcript" class="transcript" aria-live="polite">
              <div id="empty-chat" class="empty-chat">
                <div class="orb"><span></span><span></span><span></span></div>
                <h3>Ready to become one machine.</h3>
                <p>Connect both devices, assign the layer split, and each peer will load only its model shard.</p>
              </div>
            </div>
            <form id="prompt-form" class="composer">
              <input id="prompt-input" placeholder="Waiting for both model shards…" disabled>
              <button id="stop-generation" type="button" class="stop" disabled aria-label="Stop generation">■</button>
              <button type="submit" class="send" aria-label="Send prompt">↑</button>
              <small id="composer-hint">Chat unlocks when both shards are ready</small>
            </form>
          </section>
        </div>
      </section>
    </main>`;
}

function peerCard(prefix: string, eyebrow: string): string {
  return `
    <article class="peer-card">
      <div class="peer-top"><p class="eyebrow">${eyebrow}</p><span id="${prefix}-phase" class="phase">connecting</span></div>
      <h3 id="${prefix}-name">${prefix === "local" ? "This device" : "Waiting for peer"}</h3>
      <p id="${prefix}-gpu" class="gpu">Capabilities pending</p>
      <p id="${prefix}-memory" class="memory">memory unknown</p>
      <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100"><i id="${prefix}-progress"></i></div>
    </article>`;
}
