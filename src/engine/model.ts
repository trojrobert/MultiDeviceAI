// Qwen3 0.6B Q8_0 catalogue and layer-role loader.

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

export const QWEN3_06B_Q8_URL =
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";

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

export function estimateRoleMemoryBytes(index: GGUFIndex, role: ModelRole, maxSequenceLength = 512): number {
  const config = qwen3ConfigFromGGUF(index);
  const weightBytes = shardDownloadBytes(index, role);
  const layers = role.layerRange[1] - role.layerRange[0];
  const kvBytes = layers * maxSequenceLength * config.keyValueHeads * config.headDim * 2 * 4;
  const workingBytes =
    (config.hiddenSize * 4 + config.intermediateSize * 2 + config.attentionHeads * maxSequenceLength) * 4;
  return weightBytes + kvBytes + workingBytes;
}

export async function loadQwen3Engine(options: EngineLoadOptions): Promise<LoadedQwen3Engine> {
  const modelUrl = options.modelUrl ?? QWEN3_06B_Q8_URL;
  const fetchFn = options.fetchFn ?? fetch;
  const index = options.index ?? await fetchGGUFIndex(modelUrl, {
    fetchFn,
    skipTokenizer: !options.role.hasEmbedding,
  });
  const config = qwen3ConfigFromGGUF(index);
  validateRole(options.role, config.layerCount);
  const downloadBytes = shardDownloadBytes(index, options.role);
  const weights = await loadModelWeights(index, options.role, fetchFn, options.onProgress);
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
