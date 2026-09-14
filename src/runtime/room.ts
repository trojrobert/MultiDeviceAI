import { DEFAULT_MODEL_ID, roleDownloadBytes, type ModelEntry, type ModelProfile } from "../engine/model.ts";
import { planPlacement, type PlacementResult } from "../engine/placement.ts";
import { unavailableCapabilities, withUserBudget } from "./capabilities.ts";
import {
  probeCapabilities,
  resolveEngineFactory,
  type DistributedEngine,
  type EngineFactory,
} from "./engine.ts";
import { PeerTransport } from "./peer.ts";
import {
  control,
  modelFingerprint,
  PROTOCOL_VERSION,
  type ControlMessage,
  type DeviceCapabilities,
  type HiddenStateFrame,
  type LayerAssignment,
  type PeerPhase,
  type RoomRole,
  type TranscriptEntry,
} from "./protocol.ts";

/**
 * A worker pass on a phone can be slow on the first token while shaders warm
 * up, and a 36-layer shard is slower still, so this is deliberately generous.
 */
const HIDDEN_REQUEST_TIMEOUT_MS = 60_000;

/** Must match `MAX_SEQUENCE_LENGTH` in the engine adapter; drives KV-cache estimates. */
const MAX_SEQUENCE_LENGTH = 256;

export type ModelPhase = "idle" | "probing" | "ready" | "error";

export interface RoomSnapshot {
  role?: RoomRole;
  roomCode?: string;
  /** Models this build offers; empty until an engine is available. */
  catalogue: readonly ModelEntry[];
  modelId: string;
  modelPhase: ModelPhase;
  /** Real shape and byte layout of the selected model, once probed. */
  modelProfile?: ModelProfile;
  placement?: PlacementResult;
  localCapabilities?: DeviceCapabilities;
  remoteCapabilities?: DeviceCapabilities;
  localAssignment?: LayerAssignment;
  remoteAssignment?: LayerAssignment;
  localProgress: number;
  remoteProgress: number;
  localPhase: PeerPhase;
  remotePhase: PeerPhase;
  status: string;
  transcript: TranscriptEntry[];
  connected: boolean;
  ready: boolean;
  generating: boolean;
  engineAvailable: boolean;
  error?: string;
}

export interface RoomControllerOptions {
  onChange(snapshot: Readonly<RoomSnapshot>): void;
  engineFactory?: EngineFactory;
}

export class RoomController {
  private transport?: PeerTransport;
  private engine?: DistributedEngine;
  private loadAbort?: AbortController;
  private generationAbort?: AbortController;
  private nextRequestId = 1;
  private pendingHidden = new Map<
    number,
    {
      resolve: (values: Float32Array) => void;
      reject: (error: Error) => void;
      timeout: ReturnType<typeof setTimeout>;
    }
  >();

  private state: RoomSnapshot;
  private readonly options: RoomControllerOptions;

  constructor(options: RoomControllerOptions) {
    this.options = options;
    this.state = {
      localProgress: 0,
      remoteProgress: 0,
      localPhase: "connecting",
      remotePhase: "connecting",
      status: "Create a room or join one from a shared link.",
      catalogue: [],
      modelId: DEFAULT_MODEL_ID,
      modelPhase: "idle",
      transcript: [],
      connected: false,
      ready: false,
      generating: false,
      engineAvailable: Boolean(
        this.options.engineFactory ?? resolveEngineFactory(),
      ),
    };
  }

  get snapshot(): Readonly<RoomSnapshot> {
    return this.state;
  }

  async start(role: RoomRole, roomCode: string, label: string): Promise<void> {
    await this.leave();
    this.patch({
      role,
      roomCode,
      status:
        role === "host"
          ? "Opening room…"
          : `Looking for room ${roomCode}…`,
      localPhase: "connecting",
      remotePhase: "connecting",
      localProgress: 0,
      remoteProgress: 0,
      transcript: [],
      connected: false,
      ready: false,
      error: undefined,
    });
    const factory = this.options.engineFactory ?? resolveEngineFactory();
    if (factory) {
      try {
        this.engine = await factory();
      } catch (error) {
        this.fail(error);
      }
    }
    this.patch({
      localCapabilities: unavailableCapabilities(label),
      catalogue: this.engine?.catalogue ?? [],
      engineAvailable: Boolean(this.engine),
    });
    const signal = new URLSearchParams(location.search).get("signal") ?? undefined;
    this.transport = new PeerTransport({
      role,
      roomCode,
      signal,
      events: {
        onOpen: () => this.onPeerOpen(),
        onControl: (message) => void this.onControl(message),
        onHidden: (frame) => void this.onHidden(frame),
        onStatus: (status) => this.patch({ status }),
        onError: (error) => this.fail(error),
        onClose: () => this.onClose(),
      },
    });
    await this.transport.connect();
    // Layer count and hidden size are per-model, so nothing downstream — the
    // split ribbon included — can render until the header has been read.
    if (role === "host") void this.selectModel(this.state.modelId);
    const capabilityPromise = this.engine
      ? this.engine.getCapabilities(label)
      : probeCapabilities(label);
    void capabilityPromise.then((capabilities) => {
      if (this.state.roomCode !== roomCode || this.state.role !== role) return;
      this.patch({ localCapabilities: capabilities });
      if (this.state.connected) {
        this.send(control({ type: "hello", role, capabilities }));
      }
    }).catch((error) => {
      if (this.state.roomCode !== roomCode || this.state.role !== role) return;
      this.patch({
        status: `Room connected; GPU probe failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      });
    });
  }

  /** Host-only: probe a model's GGUF header so its real shape is known. */
  async selectModel(modelId: string): Promise<void> {
    if (this.state.role !== "host" || !this.engine) return;
    this.patch({ modelId, modelPhase: "probing", modelProfile: undefined, placement: undefined });
    try {
      const profile = await this.engine.probeModel(modelId);
      if (this.state.modelId !== modelId) return;
      this.patch({ modelProfile: profile, modelPhase: "ready" });
      this.recomputePlacement();
    } catch (error) {
      this.patch({ modelPhase: "error" });
      this.fail(error);
    }
  }

  /** Override this device's memory estimate and re-plan against it. */
  setBudgetBytes(bytes: number): void {
    const capabilities = this.state.localCapabilities;
    if (!capabilities) return;
    this.patch({ localCapabilities: withUserBudget(capabilities, bytes) });
    if (this.state.connected) {
      this.send(control({ type: "hello", role: this.state.role!, capabilities: this.state.localCapabilities! }));
    }
    this.recomputePlacement();
  }

  private recomputePlacement(): void {
    const profile = this.state.modelProfile;
    if (!profile || this.state.role !== "host") return;
    this.patch({
      placement: planPlacement({
        profile,
        hostBudgetBytes: this.state.localCapabilities?.budgetBytes ?? 0,
        workerBudgetBytes: this.state.remoteCapabilities?.budgetBytes ?? 0,
        maxSequenceLength: MAX_SEQUENCE_LENGTH,
      }),
    });
  }

  async assignSplit(split: number): Promise<void> {
    if (this.state.role !== "host") return;
    const profile = this.state.modelProfile;
    if (!profile) {
      this.fail(new Error("Choose a model before assigning layers"));
      return;
    }
    const { layerCount } = profile;
    const bounded = Math.max(1, Math.min(layerCount - 1, Math.round(split)));
    const shared = {
      layerCount,
      modelId: profile.modelId,
      modelUrl: profile.url,
      hiddenSize: profile.hiddenSize,
      fingerprint: modelFingerprint(profile),
    };
    const hostAssignment: LayerAssignment = {
      ...shared,
      start: 0,
      end: bounded,
      ownsEmbedding: true,
      ownsHead: true,
      bytes: roleDownloadBytes(profile, { layerRange: [0, bounded], hasEmbedding: true, hasHead: true }),
    };
    const workerAssignment: LayerAssignment = {
      ...shared,
      start: bounded,
      end: layerCount,
      ownsEmbedding: false,
      ownsHead: false,
      bytes: roleDownloadBytes(profile, {
        layerRange: [bounded, layerCount],
        hasEmbedding: false,
        hasHead: false,
      }),
    };
    this.patch({
      localAssignment: hostAssignment,
      remoteAssignment: workerAssignment,
      ready: false,
    });
    this.send(control({ type: "assignment", assignment: workerAssignment }));
    await this.loadAssignment(hostAssignment);
  }

  async submitPrompt(text: string): Promise<void> {
    const prompt = text.trim();
    if (!prompt || this.state.generating) return;
    if (this.state.role === "worker") {
      this.send(control({ type: "prompt", id: randomId(), text: prompt }));
      return;
    }
    await this.generate(prompt);
  }

  stop(): void {
    this.generationAbort?.abort();
    this.send(control({ type: "stop" }));
    this.patch({ generating: false, status: "Generation stopped" });
  }

  async reset(): Promise<void> {
    this.generationAbort?.abort();
    await this.engine?.reset();
    this.patch({
      transcript: [],
      generating: false,
      status: "Conversation reset",
    });
    this.send(control({ type: "reset" }));
    this.broadcastTranscript();
  }

  async leave(): Promise<void> {
    this.loadAbort?.abort();
    this.generationAbort?.abort();
    this.transport?.close();
    this.transport = undefined;
    for (const pending of this.pendingHidden.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error("Room closed"));
    }
    this.pendingHidden.clear();
    await this.engine?.dispose?.();
    this.engine = undefined;
  }

  requestWorkerHidden(
    values: Float32Array,
    position: number,
  ): Promise<Float32Array> {
    const requestId = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingHidden.delete(requestId);
        reject(new Error(`Hidden-state request ${requestId} timed out`));
      }, HIDDEN_REQUEST_TIMEOUT_MS);
      this.pendingHidden.set(requestId, { resolve, reject, timeout });
      try {
        this.transport?.sendHidden({
          kind: "request",
          requestId,
          position,
          fingerprint: this.state.localAssignment?.fingerprint ?? 0,
          values,
        });
      } catch (error) {
        clearTimeout(timeout);
        this.pendingHidden.delete(requestId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private onPeerOpen(): void {
    this.patch({
      connected: true,
      localPhase: "connected",
      remotePhase: "connected",
      status: "Peer connected; exchanging capabilities…",
    });
    this.send(
      control({
        type: "hello",
        role: this.state.role!,
        capabilities:
          this.state.localCapabilities ?? unavailableCapabilities(this.state.role ?? "device"),
      }),
    );
  }

  private async onControl(message: ControlMessage): Promise<void> {
    switch (message.type) {
      case "hello":
        this.patch({
          remoteCapabilities: message.capabilities,
          remotePhase: "connected",
          status:
            this.state.role === "host"
              ? "Worker connected. Confirm the layer split to load."
              : "Connected to host. Waiting for layer assignment…",
        });
        // The peer's budget is half of the placement input, so the plan is only
        // meaningful once its hello has landed.
        this.recomputePlacement();
        break;
      case "assignment":
        this.patch({ localAssignment: message.assignment, ready: false });
        await this.loadAssignment(message.assignment);
        break;
      case "load-progress":
        this.patch({
          remoteProgress: message.progress,
          remotePhase: "loading",
          status: message.detail ?? "Peer is loading its shard…",
        });
        break;
      case "ready":
        this.patch({
          remoteAssignment: message.assignment,
          remoteProgress: 1,
          remotePhase: "ready",
        });
        this.updateReady();
        break;
      case "prompt":
        if (this.state.role === "host") await this.generate(message.text);
        break;
      case "transcript":
        this.patch({ transcript: message.entries });
        break;
      case "token":
        this.applyToken(message);
        break;
      case "status":
        this.patch({ remotePhase: message.phase, status: message.detail });
        break;
      case "stop":
        this.generationAbort?.abort();
        this.patch({ generating: false, remotePhase: "ready" });
        break;
      case "reset":
        await this.engine?.reset();
        this.patch({ transcript: [], generating: false });
        break;
      case "error":
        this.patch({
          remotePhase: "error",
          error: message.message,
          status: `Peer error: ${message.message}`,
        });
        break;
    }
  }

  private async loadAssignment(assignment: LayerAssignment): Promise<void> {
    this.loadAbort?.abort();
    this.loadAbort = new AbortController();
    this.patch({
      localPhase: "loading",
      localProgress: 0,
      status: this.engine
        ? `Loading layers ${assignment.start}–${assignment.end - 1}…`
        : "Engine integration is not available; transport is in UI-only mode.",
    });
    if (!this.engine) return;
    try {
      await this.engine.load({
        assignment,
        signal: this.loadAbort.signal,
        onProgress: (progress, detail) => {
          const normalized = Math.max(0, Math.min(1, progress));
          this.patch({
            localProgress: normalized,
            status: detail ?? "Loading model shard…",
          });
          this.send(
            control({
              type: "load-progress",
              progress: normalized,
              detail,
            }),
          );
        },
      });
      this.patch({ localProgress: 1, localPhase: "ready" });
      this.send(control({ type: "ready", assignment }));
      this.updateReady();
    } catch (error) {
      if (!this.loadAbort.signal.aborted) this.fail(error);
    }
  }

  private async onHidden(frame: HiddenStateFrame): Promise<void> {
    if (frame.kind === "response") {
      const pending = this.pendingHidden.get(frame.requestId);
      if (!pending) return;
      clearTimeout(pending.timeout);
      this.pendingHidden.delete(frame.requestId);
      const expected = this.state.localAssignment;
      if (expected && frame.fingerprint !== expected.fingerprint) {
        pending.reject(
          new Error(
            `Peer returned a hidden state from a different model ` +
              `(${frame.fingerprint} vs ${expected.fingerprint})`,
          ),
        );
        return;
      }
      pending.resolve(frame.values);
      return;
    }
    if (!this.engine || this.state.role !== "worker") {
      this.send(
        control({
          type: "error",
          code: "engine-unavailable",
          message: "Worker engine is not loaded",
        }),
      );
      return;
    }

    // Fail loudly on a model mismatch rather than transforming a hidden state
    // through the wrong layers. The width check alone is not enough — two
    // models can share a hidden size — so the fingerprint is authoritative.
    const assignment = this.state.localAssignment;
    if (assignment && frame.fingerprint !== assignment.fingerprint) {
      const detail = `peer sent model ${frame.fingerprint}, this device loaded ${assignment.fingerprint}`;
      this.send(
        control({ type: "error", code: "model-mismatch", message: `Devices are running different models (${detail})` }),
      );
      this.fail(new Error(`Model mismatch between devices: ${detail}`));
      return;
    }
    if (assignment && frame.values.length !== assignment.hiddenSize) {
      const detail = `got ${frame.values.length} floats, expected ${assignment.hiddenSize}`;
      this.send(
        control({ type: "error", code: "width-mismatch", message: `Hidden state has the wrong width (${detail})` }),
      );
      this.fail(new Error(`Hidden-state width mismatch: ${detail}`));
      return;
    }

    try {
      const values = await this.engine.runHidden(frame.values, frame.position);
      this.transport?.sendHidden({ ...frame, kind: "response", values });
    } catch (error) {
      this.fail(error);
    }
  }

  private async generate(prompt: string): Promise<void> {
    const user: TranscriptEntry = {
      id: randomId(),
      role: "user",
      text: prompt,
      createdAt: Date.now(),
    };
    const assistant: TranscriptEntry = {
      id: randomId(),
      role: "assistant",
      text: "",
      createdAt: Date.now() + 1,
      pending: true,
    };
    this.patch({
      transcript: [...this.state.transcript, user, assistant],
      generating: true,
      localPhase: "generating",
      status: "Generating across both devices…",
    });
    this.broadcastTranscript();
    if (!this.engine?.generate) {
      this.patch({
        generating: false,
        localPhase: this.state.ready ? "ready" : "connected",
        status: "Engine generation hook is not connected yet.",
      });
      this.send(
        control({
          type: "status",
          phase: this.state.localPhase,
          detail: "Engine generation hook is not connected yet.",
        }),
      );
      return;
    }
    this.generationAbort = new AbortController();
    try {
      await this.engine.generate(this.state.transcript.slice(0, -1), {
        signal: this.generationAbort.signal,
        runRemoteHidden: (hidden, position) =>
          this.requestWorkerHidden(hidden, position),
        onToken: (tokenId, text) => {
          this.applyToken({
            v: PROTOCOL_VERSION,
            type: "token",
            entryId: assistant.id,
            tokenId,
            text,
            done: false,
          });
          this.send(
            control({
              type: "token",
              entryId: assistant.id,
              tokenId,
              text,
              done: false,
            }),
          );
        },
      });
      this.finishAssistant(assistant.id);
      this.send(
        control({
          type: "token",
          entryId: assistant.id,
          tokenId: -1,
          text: "",
          done: true,
        }),
      );
    } catch (error) {
      if (!this.generationAbort.signal.aborted) this.fail(error);
    }
  }

  private applyToken(
    message: Extract<ControlMessage, { type: "token" }>,
  ): void {
    const transcript = this.state.transcript.map((entry) =>
      entry.id === message.entryId
        ? {
            ...entry,
            text: entry.text + message.text,
            pending: !message.done,
          }
        : entry,
    );
    this.patch({
      transcript,
      generating: !message.done,
      remotePhase: message.done ? "ready" : "generating",
    });
  }

  private finishAssistant(entryId: string): void {
    this.patch({
      transcript: this.state.transcript.map((entry) =>
        entry.id === entryId ? { ...entry, pending: false } : entry,
      ),
      generating: false,
      localPhase: "ready",
      status: "Ready",
    });
    this.broadcastTranscript();
  }

  private broadcastTranscript(): void {
    this.send(
      control({ type: "transcript", entries: this.state.transcript }),
    );
  }

  private updateReady(): void {
    const ready =
      this.state.localPhase === "ready" &&
      this.state.remotePhase === "ready";
    this.patch({
      ready,
      status: ready ? "Both model shards are ready." : this.state.status,
    });
  }

  private onClose(): void {
    if (!this.state.connected) return;
    this.patch({
      connected: false,
      ready: false,
      remotePhase: "error",
      status: "Peer disconnected. Reconnect recovery is outside this POC.",
    });
  }

  private send(message: ControlMessage): void {
    try {
      this.transport?.sendControl(message);
    } catch (error) {
      this.fail(error);
    }
  }

  private fail(value: unknown): void {
    const error = value instanceof Error ? value : new Error(String(value));
    this.patch({
      error: error.message,
      status: error.message,
      localPhase: "error",
      generating: false,
    });
  }

  private patch(patch: Partial<RoomSnapshot>): void {
    this.state = { ...this.state, ...patch };
    this.emit();
  }

  private emit(): void {
    this.options.onChange(this.state);
  }
}

function randomId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}
