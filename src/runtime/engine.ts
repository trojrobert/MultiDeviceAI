import type { ModelEntry, ModelProfile } from "../engine/model.ts";
import type { StorageEstimate, WeightCacheStats } from "../engine/weightCache.ts";
import { describeAdapter } from "./capabilities.ts";
import type { TokenSample } from "./metrics.ts";
import type {
  DeviceCapabilities,
  LayerAssignment,
  TranscriptEntry,
} from "./protocol.ts";

export interface EngineLoadOptions {
  assignment: LayerAssignment;
  signal: AbortSignal;
  onProgress(progress: number, detail?: string): void;
}

/** What the host learns from one activation round trip. */
export interface RemoteHiddenResult {
  values: Float32Array;
  /** Host-observed request-to-response time, in microseconds. */
  roundTripMicros: number;
  /** Time the peer reported spending in its own layers, in microseconds. */
  workerMicros: number;
  bytesOut: number;
  bytesIn: number;
}

export interface GenerationOptions {
  signal: AbortSignal;
  maxNewTokens?: number;
  onToken(tokenId: number, text: string): void;
  /** One call per token, with that token's four-stage breakdown. */
  onStage?(sample: TokenSample): void;
  runRemoteHidden(
    hidden: Float32Array,
    position: number,
    prefill: boolean,
  ): Promise<RemoteHiddenResult>;
}

/** What the persistent tensor cache holds for the model this device loaded. */
export interface EngineCacheReport {
  available: boolean;
  /** True when the browser agreed not to evict this origin under pressure. */
  persisted: boolean;
  shard: WeightCacheStats;
  total: WeightCacheStats;
  /** Whole-origin usage and quota, when the browser reports them. */
  storage?: StorageEstimate;
}

/**
 * Narrow integration boundary implemented by the concurrent model-engine work.
 * Host-only methods may throw when called on a worker assignment.
 */
export interface DistributedEngine {
  /** Models this build can load. */
  readonly catalogue: readonly ModelEntry[];
  /** Profile of the model currently selected, once `probeModel` has run. */
  readonly activeProfile?: ModelProfile;
  /**
   * Range-fetch a model's GGUF header and derive its real shape and byte
   * layout. Layer count and hidden size vary per model, so nothing may assume
   * them before this resolves.
   */
  probeModel(modelId: string, signal?: AbortSignal): Promise<ModelProfile>;
  getCapabilities(label: string): Promise<DeviceCapabilities>;
  load(options: EngineLoadOptions): Promise<void>;
  tokenize?(transcript: readonly TranscriptEntry[]): Promise<Int32Array>;
  embedToken?(tokenId: number, position: number): Promise<Float32Array>;
  runHidden(hidden: Float32Array, position: number): Promise<Float32Array>;
  sampleFromHidden?(
    hidden: Float32Array,
    position: number,
  ): Promise<{ tokenId: number; text: string; stop: boolean }>;
  generate?(
    transcript: readonly TranscriptEntry[],
    options: GenerationOptions,
  ): Promise<void>;
  /** Persistent weight-cache occupancy, for the UI's cache readout. */
  cacheReport?(): Promise<EngineCacheReport>;
  /** Drop every cached tensor for every model. */
  clearCache?(): Promise<void>;
  reset(): Promise<void> | void;
  dispose?(): Promise<void> | void;
}

export type EngineFactory = () => Promise<DistributedEngine>;

export const ENGINE_FACTORY_GLOBAL = "__LOCAL_CLUSTER_AI_ENGINE_FACTORY__";

declare global {
  interface Window {
    __LOCAL_CLUSTER_AI_ENGINE_FACTORY__?: EngineFactory;
  }
}

export function resolveEngineFactory(): EngineFactory | undefined {
  return typeof window === "undefined"
    ? undefined
    : window.__LOCAL_CLUSTER_AI_ENGINE_FACTORY__;
}

export async function probeCapabilities(
  label: string,
): Promise<DeviceCapabilities> {
  return describeAdapter(label);
}
