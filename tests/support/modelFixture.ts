/*
 * Synthetic Qwen3 GGUF indexes for CPU tests.
 *
 * Shaped exactly like the real thing — same tensor names, same Q8_0 byte
 * arithmetic, same tied-embedding layout (no `output.weight`) — so tests can
 * exercise profiling and placement without touching the network.
 */

import { qwenLayerTensorNames } from "../../src/engine/gguf.ts";
import type { GGUFIndex, TensorInfo } from "../../src/engine/types.ts";

const GGML_F32 = 0;
const GGML_Q8_0 = 8;
const Q8_BLOCK_SIZE = 32;
const Q8_BLOCK_BYTES = 34;

export interface FixtureShape {
  layerCount: number;
  hiddenSize: number;
  intermediateSize: number;
  attentionHeads: number;
  keyValueHeads: number;
  headDim: number;
  vocabSize: number;
}

/** Real Qwen3 Q8_0 geometry, read from the published GGUF headers. */
export const QWEN3_SHAPES: Record<"0.6B" | "1.7B" | "4B", FixtureShape> = {
  "0.6B": { layerCount: 28, hiddenSize: 1024, intermediateSize: 3072, attentionHeads: 16, keyValueHeads: 8, headDim: 128, vocabSize: 151936 },
  "1.7B": { layerCount: 28, hiddenSize: 2048, intermediateSize: 6144, attentionHeads: 16, keyValueHeads: 8, headDim: 128, vocabSize: 151936 },
  "4B": { layerCount: 36, hiddenSize: 2560, intermediateSize: 9728, attentionHeads: 32, keyValueHeads: 8, headDim: 128, vocabSize: 151936 },
};

function byteLength(ggmlType: number, nElements: number): number {
  if (ggmlType === GGML_F32) return nElements * 4;
  return (nElements / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES;
}

export function buildGGUFIndex(shape: FixtureShape, url = "https://example.invalid/model.gguf"): GGUFIndex {
  const tensors: Record<string, TensorInfo> = {};
  let byteOffset = 0;

  const add = (name: string, dims: number[], ggmlType: number): void => {
    const nElements = dims.reduce((product, value) => product * value, 1);
    const length = byteLength(ggmlType, nElements);
    tensors[name] = { name, shape: dims, ggmlType, nElements, byteOffset, byteLength: length };
    byteOffset += length;
  };

  const queryDim = shape.attentionHeads * shape.headDim;
  const kvDim = shape.keyValueHeads * shape.headDim;

  for (let layer = 0; layer < shape.layerCount; layer++) {
    const names = qwenLayerTensorNames(layer);
    add(names.inputNorm, [shape.hiddenSize], GGML_F32);
    add(names.query, [queryDim, shape.hiddenSize], GGML_Q8_0);
    add(names.key, [kvDim, shape.hiddenSize], GGML_Q8_0);
    add(names.value, [kvDim, shape.hiddenSize], GGML_Q8_0);
    add(names.output, [shape.hiddenSize, queryDim], GGML_Q8_0);
    add(names.queryNorm, [shape.headDim], GGML_F32);
    add(names.keyNorm, [shape.headDim], GGML_F32);
    add(names.postAttentionNorm, [shape.hiddenSize], GGML_F32);
    add(names.gate, [shape.intermediateSize, shape.hiddenSize], GGML_Q8_0);
    add(names.up, [shape.intermediateSize, shape.hiddenSize], GGML_Q8_0);
    add(names.down, [shape.hiddenSize, shape.intermediateSize], GGML_Q8_0);
  }

  // Tied embeddings: every catalogue model omits `output.weight`.
  add("token_embd.weight", [shape.vocabSize, shape.hiddenSize], GGML_Q8_0);
  add("output_norm.weight", [shape.hiddenSize], GGML_F32);

  return {
    url,
    dataStart: 0,
    headerBytes: 0,
    tensors,
    metadata: {
      "general.architecture": "qwen3",
      "qwen3.block_count": shape.layerCount,
      "qwen3.embedding_length": shape.hiddenSize,
      "qwen3.feed_forward_length": shape.intermediateSize,
      "qwen3.attention.head_count": shape.attentionHeads,
      "qwen3.attention.head_count_kv": shape.keyValueHeads,
      "qwen3.attention.key_length": shape.headDim,
      "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
      "qwen3.rope.freq_base": 1_000_000,
      "qwen3.context_length": 40960,
    },
  };
}
