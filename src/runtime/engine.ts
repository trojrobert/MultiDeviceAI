import type { ModelEntry, ModelProfile } from "../engine/model.ts";
import { describeAdapter } from "./capabilities.ts";
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

export interface GenerationOptions {
  signal: AbortSignal;
  maxNewTokens?: number;
  onToken(tokenId: number, text: string): void;
  runRemoteHidden(
    hidden: Float32Array,
    position: number,
  ): Promise<Float32Array>;
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
  reset(): Promise<void> | void;
  dispose?(): Promise<void> | void;
}

export type EngineFactory = () => Promise<DistributedEngine>;

export const ENGINE_FACTORY_GLOBAL = "__MULTIDEVICE_AI_ENGINE_FACTORY__";

declare global {
  interface Window {
    __MULTIDEVICE_AI_ENGINE_FACTORY__?: EngineFactory;
  }
}

export function resolveEngineFactory(): EngineFactory | undefined {
  return typeof window === "undefined"
    ? undefined
    : window.__MULTIDEVICE_AI_ENGINE_FACTORY__;
}

export async function probeCapabilities(
  label: string,
): Promise<DeviceCapabilities> {
  return describeAdapter(label);
}
