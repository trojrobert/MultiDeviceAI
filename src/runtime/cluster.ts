import { DEFAULT_MODEL_ID, roleDownloadBytes, type ModelEntry, type ModelProfile } from "../engine/model.ts";
import { planPlacement, type PlacementResult } from "../engine/placement.ts";
import { unavailableCapabilities, withUserBudget } from "./capabilities.ts";
import {
  probeCapabilities,
  resolveEngineFactory,
  type DistributedEngine,
  type EngineCacheReport,
  type EngineFactory,
  type RemoteHiddenResult,
} from "./engine.ts";
import { emptySummary, PerfRecorder, type PerfSummary } from "./metrics.ts";
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
  type ClusterRole,
  type TranscriptEntry,
} from "./protocol.ts";

/**
 * A worker pass on a phone can be slow on the first token while shaders warm
 * up, and a 36-layer shard is slower still, so this is deliberately generous.
 */
const HIDDEN_REQUEST_TIMEOUT_MS = 60_000;

/** Must match `MAX_SEQUENCE_LENGTH` in the engine adapter; drives KV-cache estimates. */
const MAX_SEQUENCE_LENGTH = 256;

/** How often stage timings reach the UI while tokens are in flight. */
const PERF_PUBLISH_INTERVAL_MS = 250;

export type ModelPhase = "idle" | "probing" | "ready" | "error";

export interface ClusterSnapshot {
  role?: ClusterRole;
  clusterCode?: string;
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
  /** Stage timings for the most recent run; zeroed until a token has moved. */
  perf: PerfSummary;
  /** Persistent weight-cache occupancy; undefined until the engine reports it. */
  cache?: EngineCacheReport;
  error?: string;
}

export interface ClusterControllerOptions {
  onChange(snapshot: Readonly<ClusterSnapshot>): void;
  engineFactory?: EngineFactory;
}

export class ClusterController {
  private transport?: PeerTransport;
  private engine?: DistributedEngine;
  private loadAbort?: AbortController;
  private generationAbort?: AbortController;
  private nextRequestId = 1;
  private pendingHidden = new Map<
    number,
    {
      resolve: (result: RemoteHiddenResult) => void;
      reject: (error: Error) => void;
      timeout: ReturnType<typeof setTimeout>;
      /** `performance.now()` at send, and the framed size that went out. */
      sentAt: number;
      bytesOut: number;
    }
  >();

  private state: ClusterSnapshot;
  private readonly options: ClusterControllerOptions;
  private readonly perf = new PerfRecorder();
  /** Throttles snapshot emission while tokens are flowing; see `publishPerf`. */
  private perfDirty = false;
  private perfTimer?: ReturnType<typeof setTimeout>;

  constructor(options: ClusterControllerOptions) {
    this.options = options;
    this.state = {
      localProgress: 0,
      remoteProgress: 0,
      localPhase: "connecting",
      remotePhase: "connecting",
      status: "Create a cluster or join one from a shared link.",
      catalogue: [],
      modelId: DEFAULT_MODEL_ID,
      modelPhase: "idle",
      transcript: [],
      connected: false,
      ready: false,
      generating: false,
      perf: emptySummary(),
      engineAvailable: Boolean(
        this.options.engineFactory ?? resolveEngineFactory(),
      ),
    };
  }

  get snapshot(): Readonly<ClusterSnapshot> {
    return this.state;
  }

  async start(role: ClusterRole, clusterCode: string, label: string): Promise<void> {
    await this.leave();
    this.patch({
      role,
      clusterCode,
      status:
        role === "host"
          ? "Opening cluster…"
          : `Looking for cluster ${clusterCode}…`,
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
    // Tells the user up front whether this session will re-download its shard.
    void this.refreshCacheReport();
    const signal = new URLSearchParams(location.search).get("signal") ?? undefined;
    this.transport = new PeerTransport({
      role,
      clusterCode,
      signal,
      events: {
        onOpen: () => this.onPeerOpen(),
        onControl: (message) => void this.onControl(message),
        onHidden: (frame, byteLength) => void this.onHidden(frame, byteLength),
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
      if (this.state.clusterCode !== clusterCode || this.state.role !== role) return;
      this.patch({ localCapabilities: capabilities });
      if (this.state.connected) {
        this.send(control({ type: "hello", role, capabilities }));
      }
    }).catch((error) => {
      if (this.state.clusterCode !== clusterCode || this.state.role !== role) return;
      this.patch({
        status: `Cluster connected; GPU probe failed: ${
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
    this.perf.reset();
    this.flushPerf();
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
    if (this.perfTimer) clearTimeout(this.perfTimer);
    this.perfTimer = undefined;
    this.perf.reset();
    this.transport?.close();
    this.transport = undefined;
    for (const pending of this.pendingHidden.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error("Cluster closed"));
    }
    this.pendingHidden.clear();
    await this.engine?.dispose?.();
    this.engine = undefined;
  }

  requestWorkerHidden(
    values: Float32Array,
    position: number,
    prefill = false,
  ): Promise<RemoteHiddenResult> {
    const requestId = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingHidden.delete(requestId);
        reject(new Error(`Hidden-state request ${requestId} timed out`));
      }, HIDDEN_REQUEST_TIMEOUT_MS);
      // Registered before the send so a synchronously delivered response — a
      // loopback transport in a test — still finds its pending entry.
      const pending = {
        resolve,
        reject,
        timeout,
        sentAt: performance.now(),
        bytesOut: 0,
      };
      this.pendingHidden.set(requestId, pending);
      try {
        pending.bytesOut =
          this.transport?.sendHidden({
            kind: "request",
            requestId,
            position,
            prefill,
            fingerprint: this.state.localAssignment?.fingerprint ?? 0,
            values,
          }) ?? 0;
      } catch (error) {
        clearTimeout(timeout);
        this.pendingHidden.delete(requestId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  /** Read the persistent weight cache's occupancy into the snapshot. */
  async refreshCacheReport(): Promise<void> {
    if (!this.engine?.cacheReport) return;
    try {
      this.patch({ cache: await this.engine.cacheReport() });
    } catch {
      /* the readout is informational; a failure must not break the cluster */
    }
  }

  /** Drop every cached tensor, then re-read the occupancy. */
  async clearWeightCache(): Promise<void> {
    if (!this.engine?.clearCache) return;
    await this.engine.clearCache();
    await this.refreshCacheReport();
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
      // The shard just filled the cache, so the readout is stale.
      void this.refreshCacheReport();
    } catch (error) {
      if (!this.loadAbort.signal.aborted) this.fail(error);
    }
  }

  private async onHidden(frame: HiddenStateFrame, byteLength: number): Promise<void> {
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
      pending.resolve({
        values: frame.values,
        roundTripMicros: Math.max(0, Math.round((performance.now() - pending.sentAt) * 1000)),
        workerMicros: frame.computeMicros ?? 0,
        bytesOut: pending.bytesOut,
        bytesIn: byteLength,
      });
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
      const started = performance.now();
      const values = await this.engine.runHidden(frame.values, frame.position);
      const computeMicros = Math.max(0, Math.round((performance.now() - started) * 1000));
      const bytesOut =
        this.transport?.sendHidden({
          ...frame,
          kind: "response",
          values,
          computeMicros,
        }) ?? 0;
      // The worker never runs `generate`, so this is the only place its own
      // panel learns anything. The phase flag rides in on the request.
      this.perf.record({
        phase: frame.prefill ? "prefill" : "decode",
        hostMicros: 0,
        wireMicros: 0,
        workerMicros: computeMicros,
        headMicros: 0,
        bytesOut,
        bytesIn: byteLength,
      });
      this.publishPerf();
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * Coalesce metric updates. A 4B prefill is hundreds of samples arriving back
   * to back, and re-rendering the whole cluster for each one would cost more than
   * the inference it is measuring.
   */
  private publishPerf(): void {
    this.perfDirty = true;
    if (this.perfTimer) return;
    this.perfTimer = setTimeout(() => {
      this.perfTimer = undefined;
      if (!this.perfDirty) return;
      this.perfDirty = false;
      this.patch({ perf: this.perf.summary() });
    }, PERF_PUBLISH_INTERVAL_MS);
  }

  /** Emit the final numbers immediately, rather than up to one interval late. */
  private flushPerf(): void {
    if (this.perfTimer) {
      clearTimeout(this.perfTimer);
      this.perfTimer = undefined;
    }
    this.perfDirty = false;
    this.patch({ perf: this.perf.summary() });
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
    this.perf.begin();
    this.flushPerf();
    try {
      await this.engine.generate(this.state.transcript.slice(0, -1), {
        signal: this.generationAbort.signal,
        runRemoteHidden: (hidden, position, prefill) =>
          this.requestWorkerHidden(hidden, position, prefill),
        onStage: (sample) => {
          this.perf.record(sample);
          this.publishPerf();
        },
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
    } finally {
      // The run's numbers stay on screen after it ends; only the live flag drops.
      this.perf.end();
      this.flushPerf();
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

  private patch(patch: Partial<ClusterSnapshot>): void {
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
