// Engine contracts for the split Qwen3 POC.

import type { WeightCache } from "./weightCache.ts";

export type LayerRange = readonly [start: number, end: number];

export interface Qwen3Config {
  hiddenSize: number;
  intermediateSize: number;
  layerCount: number;
  attentionHeads: number;
  keyValueHeads: number;
  headDim: number;
  vocabSize: number;
  rmsNormEpsilon: number;
  ropeTheta: number;
  contextLength: number;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  ggmlType: number;
  nElements: number;
  byteOffset: number;
  byteLength: number;
}

export interface GGUFIndex {
  url?: string;
  metadata: Record<string, GGUFValue>;
  tensors: Record<string, TensorInfo>;
  dataStart: number;
  headerBytes: number;
}

export type GGUFScalar = string | number | bigint | boolean;
export type GGUFValue = GGUFScalar | GGUFScalar[];

export interface F32Weight {
  kind: "f32";
  shape: number[];
  data: Float32Array;
}

export interface Q8Weight {
  kind: "q8";
  shape: number[];
  quants: Uint8Array;
  scales: Uint32Array;
}

export type Weight = F32Weight | Q8Weight;

export interface LayerWeights {
  inputNorm: F32Weight;
  query: Weight;
  key: Weight;
  value: Weight;
  output: Weight;
  queryNorm: F32Weight;
  keyNorm: F32Weight;
  postAttentionNorm: F32Weight;
  gate: Weight;
  up: Weight;
  down: Weight;
}

export interface ModelWeights {
  layers: LayerWeights[];
  embedding?: Weight;
  finalNorm?: F32Weight;
  head?: Weight;
}

export interface ModelRole {
  layerRange: LayerRange;
  hasEmbedding: boolean;
  hasHead: boolean;
}

export interface LoadProgress {
  phase: "header" | "weights";
  loadedBytes: number;
  totalBytes: number;
  /** Of `loadedBytes`, how many came from the persistent tensor cache. */
  cachedBytes?: number;
  /** Of `loadedBytes`, how many crossed the network this session. */
  downloadedBytes?: number;
  tensor?: string;
  /** Whether this particular tensor was a cache hit. */
  fromCache?: boolean;
}

export type ProgressCallback = (progress: LoadProgress) => void;

export interface EngineLoadOptions {
  device: GPUDevice;
  modelUrl?: string;
  index?: GGUFIndex;
  role: ModelRole;
  maxSequenceLength?: number;
  onProgress?: ProgressCallback;
  fetchFn?: typeof fetch;
  /** Persistent tensor store, so a second session does not re-download a shard. */
  cache?: WeightCache;
}

export interface SplitEngine {
  readonly config: Qwen3Config;
  readonly role: ModelRole;
  readonly position: number;
  reset(): void;
  prefillToken(tokenId: number, position?: number): Promise<Float32Array>;
  runHidden(hidden: Float32Array, position: number): Promise<Float32Array>;
  headFromHidden(hidden: Float32Array): Promise<Float32Array>;
  destroy(): void;
}
