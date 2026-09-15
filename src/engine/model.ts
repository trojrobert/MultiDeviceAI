// Qwen3 Q8_0 model catalogue, GGUF profiling, and layer-role loader.

import { DenseQwen3Engine } from "./denseEngine.ts";
import {
  fetchGGUFIndex,
  loadModelWeights,
  qwen3ConfigFromGGUF,
  shardDownloadBytes,
} from "./gguf.ts";
import { createQwenTokenizer, tokenizerDefinitionFromGGUF, type QwenTokenizer } from "./tokenizer.ts";
import type {
  EngineLoadOptions,
  GGUFIndex,
  ModelRole,
  Qwen3Config,
} from "./types.ts";

export interface ModelEntry {
  id: string;
  label: string;
  /** Parameter count, for display only. */
  params: string;
  url: string;
  quantization: "Q8_0";
  /** Size of the whole GGUF file, used before the header has been probed. */
  approxBytes: number;
}

/**
 * Catalogue of supported models.
 *
 * Q8_0 only, and deliberately so: `loadWeight` in gguf.ts repacks 2-D Q8_0
 * tensors and falls through to `tensorToF32` for everything else, so there is
 * no K-quant dequantization path. Adding a Q4_K entry here would fail at load.
 *
 * Qwen3 8B Q8_0 exists (8.71 GB) but is omitted: without persistent weight
 * caching the download is impractical in a browser session.
 *
 * The 2507 pair comes from unsloth rather than Qwen because that is where the
 * single-file Q8_0 build lives. Both were probed with `fetchGGUFIndex`: they
 * report `general.architecture = qwen3` and carry the `attn_q_norm`/
 * `attn_k_norm` tensors `qwenLayerTensorNames` requires, with the same 253
 * Q8_0 / 145 F32 tensor split as the Qwen-published 4B. They advertise a
 * 262144 context, but `DenseQwen3Engine` clamps to `min(512, contextLength)`,
 * so the KV cache costs the same as every other entry here.
 *
 * unsloth's plain Qwen3-0.6B/1.7B/4B GGUFs are deliberately absent: they are
 * re-uploads of the weights Qwen already publishes above (byte sizes match to
 * within ~1 KB of header metadata), so listing them would only duplicate the
 * picker.
 */
export const MODEL_CATALOGUE: readonly ModelEntry[] = [
  {
    id: "qwen3-0.6b-q8_0",
    label: "Qwen3 0.6B",
    params: "0.6B",
    url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    quantization: "Q8_0",
    approxBytes: 639_446_688,
  },
  {
    id: "qwen3-1.7b-q8_0",
    label: "Qwen3 1.7B",
    params: "1.7B",
    url: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    quantization: "Q8_0",
    approxBytes: 1_834_426_016,
  },
  {
    id: "qwen3-4b-q8_0",
    label: "Qwen3 4B",
    params: "4B",
    url: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    quantization: "Q8_0",
    approxBytes: 4_280_404_704,
  },
  {
    id: "qwen3-4b-instruct-2507-q8_0",
    label: "Qwen3 4B Instruct 2507",
    params: "4B",
    url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf",
    quantization: "Q8_0",
    approxBytes: 4_280_405_600,
  },
  {
    id: "qwen3-4b-thinking-2507-q8_0",
    label: "Qwen3 4B Thinking 2507",
    params: "4B",
    url: "https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF/resolve/main/Qwen3-4B-Thinking-2507-Q8_0.gguf",
    quantization: "Q8_0",
    approxBytes: 4_280_405_632,
  },
];

export const DEFAULT_MODEL_ID = "qwen3-0.6b-q8_0";

export function findModel(id: string): ModelEntry | undefined {
  return MODEL_CATALOGUE.find((entry) => entry.id === id);
}

export function requireModel(id: string): ModelEntry {
  const entry = findModel(id);
  if (!entry) throw new Error(`unknown model ${id}`);
  return entry;
}

/** Measured shape and byte layout of a model, derived from its GGUF header. */
export interface ModelProfile {
  modelId: string;
  label: string;
  url: string;
  layerCount: number;
  hiddenSize: number;
  vocabSize: number;
  /** Download size of every transformer layer, indexed by layer. */
  layerBytes: number[];
  /** Embedding, final norm, and language-model head together. */
  edgeBytes: number;
  /** Whole model on a single device: every layer plus the edges. */
  totalBytes: number;
  config: Qwen3Config;
}

/** Only the model edges: an empty layer range still selects embedding/norm/head. */
const EDGE_ROLE: ModelRole = { layerRange: [0, 0], hasEmbedding: true, hasHead: true };

export function profileModel(index: GGUFIndex, entry: ModelEntry): ModelProfile {
  const config = qwen3ConfigFromGGUF(index);
  const layerBytes: number[] = [];
  for (let layer = 0; layer < config.layerCount; layer++) {
    layerBytes.push(
      shardDownloadBytes(index, { layerRange: [layer, layer + 1], hasEmbedding: false, hasHead: false }),
    );
  }
  const edgeBytes = shardDownloadBytes(index, EDGE_ROLE);
  return {
    modelId: entry.id,
    label: entry.label,
    url: entry.url,
    layerCount: config.layerCount,
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    layerBytes,
    edgeBytes,
    totalBytes: layerBytes.reduce((total, bytes) => total + bytes, 0) + edgeBytes,
    config,
  };
}

/** @deprecated Use `MODEL_CATALOGUE`; kept so existing call sites keep working. */
export const QWEN3_06B_Q8_URL = MODEL_CATALOGUE[0]!.url;

export const QWEN3_06B_LAYER_COUNT = 28;

export interface LoadedQwen3Engine {
  engine: DenseQwen3Engine;
  index: GGUFIndex;
  config: Qwen3Config;
  tokenizer?: QwenTokenizer;
  downloadBytes: number;
}

export function validateRole(role: ModelRole, layerCount: number): void {
  const [start, end] = role.layerRange;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > layerCount) {
    throw new Error(`invalid layer range [${start}, ${end}) for ${layerCount} layers`);
  }
  if (role.hasEmbedding && start !== 0) throw new Error("only a range starting at layer 0 may own embeddings");
}

export function splitQwen3Layers(layerCount: number, hostLayerCount = Math.ceil(layerCount / 2)): {
  host: ModelRole;
  worker: ModelRole;
} {
  if (!Number.isInteger(hostLayerCount) || hostLayerCount <= 0 || hostLayerCount >= layerCount) {
    throw new Error(`split must be between layers 1 and ${layerCount - 1}`);
  }
  return {
    host: { layerRange: [0, hostLayerCount], hasEmbedding: true, hasHead: true },
    worker: { layerRange: [hostLayerCount, layerCount], hasEmbedding: false, hasHead: false },
  };
}

/**
 * Bytes a role downloads: its own layers, plus the model edges when it owns
 * them. Mirrors `shardTensorNames` in gguf.ts, and `tests/placement.test.ts`
 * pins the two together so they cannot drift.
 */
export function roleDownloadBytes(profile: ModelProfile, role: ModelRole): number {
  let bytes = 0;
  for (let layer = role.layerRange[0]; layer < role.layerRange[1]; layer++) {
    const layerBytes = profile.layerBytes[layer];
    if (layerBytes === undefined) throw new RangeError(`layer ${layer} is outside the model`);
    bytes += layerBytes;
  }
  // Qwen3 ties the head to the embedding table, so a head role adds no separate
  // output tensor; `edgeBytes` already covers embedding, final norm, and head.
  if (role.hasEmbedding || role.hasHead) bytes += profile.edgeBytes;
  return bytes;
}

/**
 * Device memory a role needs: weights, its own KV caches, the working buffers,
 * and the logits buffer when it owns the head.
 */
export function estimateRoleMemoryBytes(
  profile: ModelProfile,
  role: ModelRole,
  maxSequenceLength = 512,
): number {
  const config = profile.config;
  const layers = role.layerRange[1] - role.layerRange[0];
  const kvBytes = layers * maxSequenceLength * config.keyValueHeads * config.headDim * 2 * 4;
  const logitsBytes = role.hasHead ? config.vocabSize * 4 : 0;
  const workingBytes =
    (config.hiddenSize * 4 + config.intermediateSize * 2 + config.attentionHeads * maxSequenceLength) * 4;
  return roleDownloadBytes(profile, role) + kvBytes + workingBytes + logitsBytes;
}

export async function loadQwen3Engine(options: EngineLoadOptions): Promise<LoadedQwen3Engine> {
  const modelUrl = options.modelUrl ?? QWEN3_06B_Q8_URL;
  const fetchFn = options.fetchFn ?? fetch;
  const index = options.index ?? await fetchGGUFIndex(modelUrl, {
    fetchFn,
    skipTokenizer: !options.role.hasEmbedding,
    cache: options.cache,
  });
  const config = qwen3ConfigFromGGUF(index);
  validateRole(options.role, config.layerCount);
  const downloadBytes = shardDownloadBytes(index, options.role);
  const weights = await loadModelWeights(index, options.role, {
    fetchFn,
    onProgress: options.onProgress,
    cache: options.cache,
  });
  const engine = await DenseQwen3Engine.create({
    device: options.device,
    config,
    weights,
    role: options.role,
    maxSequenceLength: options.maxSequenceLength,
  });
  const tokenizer = options.role.hasEmbedding
    ? createQwenTokenizer(tokenizerDefinitionFromGGUF(index.metadata))
    : undefined;
  return { engine, index, config, tokenizer, downloadBytes };
}
