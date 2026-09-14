/*
 * GGUF model loading and tensor indexing.
 */

import { float16ToFloat32, repackQ8, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE } from "./quant.ts";
import type {
  F32Weight,
  GGUFIndex,
  GGUFScalar,
  GGUFValue,
  ModelRole,
  ModelWeights,
  ProgressCallback,
  Qwen3Config,
  TensorInfo,
  Weight,
} from "./types.ts";

const GGUF_MAGIC = 0x46554747;
const TYPE_UINT8 = 0;
const TYPE_INT8 = 1;
const TYPE_UINT16 = 2;
const TYPE_INT16 = 3;
const TYPE_UINT32 = 4;
const TYPE_INT32 = 5;
const TYPE_FLOAT32 = 6;
const TYPE_BOOL = 7;
const TYPE_STRING = 8;
const TYPE_ARRAY = 9;
const TYPE_UINT64 = 10;
const TYPE_INT64 = 11;
const TYPE_FLOAT64 = 12;

export const GGML_F32 = 0;
export const GGML_F16 = 1;
export const GGML_Q8_0 = 8;

const textDecoder = new TextDecoder();

function checkedNumber(value: bigint, label: string): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new Error(`${label} exceeds JavaScript's safe integer range`);
  return number;
}

class GGUFReader {
  private offset = 0;
  private readonly view: DataView;

  constructor(view: DataView) {
    this.view = view;
  }

  get position(): number {
    return this.offset;
  }

  private require(bytes: number): void {
    if (this.offset + bytes > this.view.byteLength) throw new RangeError("incomplete GGUF header");
  }

  uint8(): number { this.require(1); return this.view.getUint8(this.offset++); }
  int8(): number { this.require(1); return this.view.getInt8(this.offset++); }
  uint16(): number { this.require(2); const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
  int16(): number { this.require(2); const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
  uint32(): number { this.require(4); const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  int32(): number { this.require(4); const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
  float32(): number { this.require(4); const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
  float64(): number { this.require(8); const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }
  uint64(): bigint { this.require(8); const v = this.view.getBigUint64(this.offset, true); this.offset += 8; return v; }
  int64(): bigint { this.require(8); const v = this.view.getBigInt64(this.offset, true); this.offset += 8; return v; }

  string(skip = false): string {
    const length = checkedNumber(this.uint64(), "GGUF string length");
    this.require(length);
    const value = skip
      ? ""
      : textDecoder.decode(new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, length));
    this.offset += length;
    return value;
  }

  value(type: number, skip = false): GGUFValue {
    switch (type) {
      case TYPE_UINT8: return this.uint8();
      case TYPE_INT8: return this.int8();
      case TYPE_UINT16: return this.uint16();
      case TYPE_INT16: return this.int16();
      case TYPE_UINT32: return this.uint32();
      case TYPE_INT32: return this.int32();
      case TYPE_FLOAT32: return this.float32();
      case TYPE_BOOL: return this.uint8() !== 0;
      case TYPE_STRING: return this.string(skip);
      case TYPE_UINT64: return this.uint64();
      case TYPE_INT64: return this.int64();
      case TYPE_FLOAT64: return this.float64();
      case TYPE_ARRAY: {
        const elementType = this.uint32();
        const length = checkedNumber(this.uint64(), "GGUF array length");
        if (skip) {
          for (let i = 0; i < length; i++) this.value(elementType, true);
          return [];
        }
        const values: GGUFScalar[] = [];
        for (let i = 0; i < length; i++) {
          const value = this.value(elementType);
          if (Array.isArray(value)) throw new Error("nested GGUF arrays are unsupported");
          values.push(value);
        }
        return values;
      }
      default: throw new Error(`unsupported GGUF metadata type ${type}`);
    }
  }
}

export interface ParseGGUFOptions {
  skipTokenizer?: boolean;
}

export function ggmlByteLength(type: number, elements: number): number {
  switch (type) {
    case GGML_F32: return elements * 4;
    case GGML_F16: return elements * 2;
    case GGML_Q8_0:
      if (elements % Q8_BLOCK_SIZE !== 0) throw new Error("Q8_0 tensor is not block aligned");
      return (elements / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES;
    default: throw new Error(`unsupported GGML tensor type ${type}; this POC requires F32, F16, or Q8_0`);
  }
}

export function parseGGUFHeader(buffer: ArrayBuffer, options: ParseGGUFOptions = {}): GGUFIndex {
  const reader = new GGUFReader(new DataView(buffer));
  if (reader.uint32() !== GGUF_MAGIC) throw new Error("not a GGUF file");
  const version = reader.uint32();
  if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version}`);
  const tensorCount = checkedNumber(reader.uint64(), "tensor count");
  const metadataCount = checkedNumber(reader.uint64(), "metadata count");
  const metadata: Record<string, GGUFValue> = {};
  for (let i = 0; i < metadataCount; i++) {
    const key = reader.string();
    const skip = Boolean(options.skipTokenizer && key.startsWith("tokenizer."));
    const value = reader.value(reader.uint32(), skip);
    if (!skip) metadata[key] = value;
  }
  const raw: Array<Omit<TensorInfo, "nElements" | "byteOffset" | "byteLength"> & { relativeOffset: number }> = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.string();
    const dimensions = reader.uint32();
    const shape: number[] = [];
    for (let d = 0; d < dimensions; d++) shape.push(checkedNumber(reader.uint64(), `${name} dimension`));
    raw.push({
      name,
      shape: shape.reverse(),
      ggmlType: reader.uint32(),
      relativeOffset: checkedNumber(reader.uint64(), `${name} offset`),
    });
  }
  const alignment = Number(metadata["general.alignment"] ?? 32);
  if (!Number.isSafeInteger(alignment) || alignment <= 0) throw new Error(`invalid GGUF alignment ${alignment}`);
  const dataStart = Math.ceil(reader.position / alignment) * alignment;
  const tensors: Record<string, TensorInfo> = {};
  for (const tensor of raw) {
    const nElements = tensor.shape.reduce((product, value) => product * value, 1);
    tensors[tensor.name] = {
      name: tensor.name,
      shape: tensor.shape,
      ggmlType: tensor.ggmlType,
      nElements,
      byteOffset: dataStart + tensor.relativeOffset,
      byteLength: ggmlByteLength(tensor.ggmlType, nElements),
    };
  }
  return { metadata, tensors, dataStart, headerBytes: reader.position };
}

function contentRangeTotal(response: Response): number | undefined {
  const match = response.headers.get("content-range")?.match(/\/(\d+)$/);
  return match ? Number(match[1]) : undefined;
}

export async function fetchGGUFIndex(
  url: string,
  options: ParseGGUFOptions & { fetchFn?: typeof fetch; initialBytes?: number } = {},
): Promise<GGUFIndex> {
  const fetchFn = options.fetchFn ?? fetch;
  let size = options.initialBytes ?? 2 * 1024 * 1024;
  for (let attempt = 0; attempt < 8; attempt++) {
    const response = await fetchFn(url, { headers: { Range: `bytes=0-${size - 1}` } });
    if (!response.ok) throw new Error(`GGUF header request failed: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (response.status === 200 && bytes.byteLength > size) {
      const index = parseGGUFHeader(bytes, options);
      return { ...index, url };
    }
    try {
      const index = parseGGUFHeader(bytes, options);
      return { ...index, url };
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      const total = contentRangeTotal(response);
      if (total !== undefined && bytes.byteLength >= total) throw new Error("GGUF header is truncated");
      size *= 2;
    }
  }
  throw new Error("GGUF header exceeds the 256 MiB safety limit");
}

export async function fetchTensorRange(
  index: GGUFIndex,
  tensor: TensorInfo,
  fetchFn: typeof fetch = fetch,
): Promise<Uint8Array> {
  if (!index.url) throw new Error("GGUF index has no source URL");
  const end = tensor.byteOffset + tensor.byteLength - 1;
  const response = await fetchFn(index.url, {
    headers: { Range: `bytes=${tensor.byteOffset}-${end}` },
  });
  if (!response.ok) throw new Error(`tensor ${tensor.name} request failed: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (response.status === 200) {
    if (bytes.byteLength < end + 1) throw new Error(`server ignored range and returned a short file for ${tensor.name}`);
    return bytes.slice(tensor.byteOffset, end + 1);
  }
  if (bytes.byteLength !== tensor.byteLength) {
    throw new Error(`short range for ${tensor.name}: ${bytes.byteLength}/${tensor.byteLength}`);
  }
  return bytes;
}

export function tensorToF32(info: TensorInfo, bytes: Uint8Array): Float32Array {
  if (bytes.byteLength !== info.byteLength) throw new Error(`short tensor ${info.name}`);
  if (info.ggmlType === GGML_F32) {
    const copy = bytes.slice();
    return new Float32Array(copy.buffer, copy.byteOffset, info.nElements);
  }
  if (info.ggmlType === GGML_F16) {
    const source = new Uint16Array(bytes.slice().buffer);
    const output = new Float32Array(info.nElements);
    for (let i = 0; i < output.length; i++) output[i] = float16ToFloat32(source[i]!);
    return output;
  }
  if (info.ggmlType === GGML_Q8_0) {
    const packed = repackQ8(info, bytes);
    const output = new Float32Array(info.nElements);
    const scaleBits = new Uint16Array(packed.scales.buffer);
    for (let block = 0; block < scaleBits.length; block++) {
      const scale = float16ToFloat32(scaleBits[block]!);
      for (let i = 0; i < Q8_BLOCK_SIZE; i++) {
        const byte = packed.quants[block * Q8_BLOCK_SIZE + i]!;
        output[block * Q8_BLOCK_SIZE + i] = scale * (byte > 127 ? byte - 256 : byte);
      }
    }
    return output;
  }
  throw new Error(`unsupported tensor type ${info.ggmlType}`);
}

export function qwenLayerTensorNames(layer: number) {
  const prefix = `blk.${layer}.`;
  return {
    inputNorm: `${prefix}attn_norm.weight`,
    query: `${prefix}attn_q.weight`,
    key: `${prefix}attn_k.weight`,
    value: `${prefix}attn_v.weight`,
    output: `${prefix}attn_output.weight`,
    queryNorm: `${prefix}attn_q_norm.weight`,
    keyNorm: `${prefix}attn_k_norm.weight`,
    postAttentionNorm: `${prefix}ffn_norm.weight`,
    gate: `${prefix}ffn_gate.weight`,
    up: `${prefix}ffn_up.weight`,
    down: `${prefix}ffn_down.weight`,
  } as const;
}

export const EMBEDDING_TENSOR = "token_embd.weight";
export const FINAL_NORM_TENSOR = "output_norm.weight";
export const OUTPUT_TENSOR = "output.weight";

function metadataNumber(metadata: Record<string, GGUFValue>, keys: string[], fallback?: number): number {
  for (const key of keys) {
    const value = metadata[key];
    if (typeof value === "number" || typeof value === "bigint") return Number(value);
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`GGUF metadata is missing ${keys.join(" or ")}`);
}

export function qwen3ConfigFromGGUF(index: GGUFIndex): Qwen3Config {
  const metadata = index.metadata;
  const hiddenSize = metadataNumber(metadata, ["qwen3.embedding_length", "llama.embedding_length"]);
  const attentionHeads = metadataNumber(metadata, ["qwen3.attention.head_count", "llama.attention.head_count"]);
  return {
    hiddenSize,
    intermediateSize: metadataNumber(metadata, ["qwen3.feed_forward_length", "llama.feed_forward_length"]),
    layerCount: metadataNumber(metadata, ["qwen3.block_count", "llama.block_count"]),
    attentionHeads,
    keyValueHeads: metadataNumber(metadata, ["qwen3.attention.head_count_kv", "llama.attention.head_count_kv"]),
    headDim: metadataNumber(
      metadata,
      ["qwen3.attention.key_length", "llama.attention.key_length"],
      hiddenSize / attentionHeads,
    ),
    vocabSize: metadataNumber(metadata, ["qwen3.vocab_size", "llama.vocab_size"], index.tensors[EMBEDDING_TENSOR]?.shape[0]),
    rmsNormEpsilon: metadataNumber(metadata, ["qwen3.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"]),
    ropeTheta: metadataNumber(metadata, ["qwen3.rope.freq_base", "llama.rope.freq_base"], 1_000_000),
    contextLength: metadataNumber(metadata, ["qwen3.context_length", "llama.context_length"], 40960),
  };
}

export function shardTensorNames(index: GGUFIndex, role: ModelRole): string[] {
  const names: string[] = [];
  for (let layer = role.layerRange[0]; layer < role.layerRange[1]; layer++) {
    names.push(...Object.values(qwenLayerTensorNames(layer)));
  }
  if (role.hasEmbedding || role.hasHead) names.push(EMBEDDING_TENSOR);
  if (role.hasHead) {
    names.push(FINAL_NORM_TENSOR);
    if (index.tensors[OUTPUT_TENSOR]) names.push(OUTPUT_TENSOR);
  }
  return [...new Set(names)];
}

export function shardDownloadBytes(index: GGUFIndex, role: ModelRole): number {
  return shardTensorNames(index, role).reduce((total, name) => {
    const tensor = index.tensors[name];
    if (!tensor) throw new Error(`missing tensor ${name}`);
    return total + tensor.byteLength;
  }, 0);
}

async function loadWeight(index: GGUFIndex, name: string, fetchFn: typeof fetch): Promise<Weight> {
  const info = index.tensors[name];
  if (!info) throw new Error(`missing tensor ${name}`);
  const bytes = await fetchTensorRange(index, info, fetchFn);
  if (info.shape.length === 2 && info.ggmlType === GGML_Q8_0) return repackQ8(info, bytes);
  return { kind: "f32", shape: info.shape, data: tensorToF32(info, bytes) };
}

function requireF32(weight: Weight, name: string): F32Weight {
  if (weight.kind !== "f32") throw new Error(`${name} must be an F32/F16 vector`);
  return weight;
}

/**
 * Range requests are independent, so the shard downloads far faster with a few
 * in flight at once. Kept modest: a 4B host shard is ~380 tensors, and browsers
 * cap concurrent connections per origin anyway.
 */
export const DEFAULT_FETCH_CONCURRENCY = 6;

export async function loadModelWeights(
  index: GGUFIndex,
  role: ModelRole,
  fetchFn: typeof fetch = fetch,
  onProgress?: ProgressCallback,
  concurrency: number = DEFAULT_FETCH_CONCURRENCY,
): Promise<ModelWeights> {
  const names = shardTensorNames(index, role);
  const totalBytes = shardDownloadBytes(index, role);
  let loadedBytes = 0;
  const loaded = new Map<string, Weight>();

  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (cursor < names.length) {
      const name = names[cursor++]!;
      const weight = await loadWeight(index, name, fetchFn);
      loaded.set(name, weight);
      loadedBytes += index.tensors[name]!.byteLength;
      // Progress is by bytes completed, so out-of-order arrival is fine; the
      // callback may throw to abort the load.
      onProgress?.({ phase: "weights", loadedBytes, totalBytes, tensor: name });
    }
  };
  const lanes = Math.max(1, Math.min(Math.floor(concurrency) || 1, names.length));
  await Promise.all(Array.from({ length: lanes }, worker));

  const take = (name: string): Weight => {
    const weight = loaded.get(name);
    if (!weight) throw new Error(`missing tensor ${name}`);
    return weight;
  };

  const layers = [];
  for (let layer = role.layerRange[0]; layer < role.layerRange[1]; layer++) {
    const names = qwenLayerTensorNames(layer);
    layers.push({
      inputNorm: requireF32(take(names.inputNorm), names.inputNorm),
      query: take(names.query),
      key: take(names.key),
      value: take(names.value),
      output: take(names.output),
      queryNorm: requireF32(take(names.queryNorm), names.queryNorm),
      keyNorm: requireF32(take(names.keyNorm), names.keyNorm),
      postAttentionNorm: requireF32(take(names.postAttentionNorm), names.postAttentionNorm),
      gate: take(names.gate),
      up: take(names.up),
      down: take(names.down),
    });
  }
  const weights: ModelWeights = { layers };
  if (role.hasEmbedding || role.hasHead) weights.embedding = take(EMBEDDING_TENSOR);
  if (role.hasHead) {
    weights.finalNorm = requireF32(take(FINAL_NORM_TENSOR), FINAL_NORM_TENSOR);
    if (index.tensors[OUTPUT_TENSOR]) weights.head = take(OUTPUT_TENSOR);
  }
  return weights;
}
