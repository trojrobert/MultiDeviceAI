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
  readonly layerCount: number;
  readonly modelId: string;
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
  const capabilities: DeviceCapabilities = {
    label,
    userAgent: navigator.userAgent,
    webgpu: "gpu" in navigator,
    maxBufferSize: 0,
  };
  if (!navigator.gpu) return capabilities;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return capabilities;
    const info = adapter.info;
    capabilities.gpu =
      [info.vendor, info.architecture, info.device]
        .filter(Boolean)
        .join(" · ") || "WebGPU adapter";
    capabilities.maxBufferSize = adapter.limits.maxBufferSize;
    capabilities.estimatedMemoryBytes = adapter.limits.maxBufferSize;
  } catch {
    capabilities.webgpu = false;
  }
  return capabilities;
}
