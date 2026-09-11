/*
 * Adapted from SwarmLLM engine/gguf.js and engine/quant.js.
 * Copyright (c) 2026 Nehanth Narendrula. MIT License.
 */

import type { Q8Weight, TensorInfo } from "./types.ts";

export const Q8_BLOCK_SIZE = 32;
export const Q8_BLOCK_BYTES = 34;

const floatScratch = new Float32Array(1);
const uintScratch = new Uint32Array(floatScratch.buffer);

export function float32ToFloat16(value: number): number {
  floatScratch[0] = value;
  const bits = uintScratch[0]!;
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  exponent = exponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | ((mantissa + 0x1000) >> 13);
  }
  return sign | ((exponent << 10) + ((mantissa + 0x1000) >> 13));
}

export function float16ToFloat32(value: number): number {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const mantissa = value & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function assertQ8Elements(n: number): void {
  if (!Number.isSafeInteger(n) || n < 0 || n % Q8_BLOCK_SIZE !== 0) {
    throw new Error(`Q8_0 element count must be a multiple of ${Q8_BLOCK_SIZE}, got ${n}`);
  }
}

export function repackQ8(info: TensorInfo, bytes: Uint8Array): Q8Weight {
  assertQ8Elements(info.nElements);
  if (bytes.byteLength !== info.byteLength) {
    throw new Error(`Q8_0 tensor ${info.name} expected ${info.byteLength} bytes, got ${bytes.byteLength}`);
  }
  const blockCount = info.nElements / Q8_BLOCK_SIZE;
  const quants = new Uint8Array(info.nElements);
  const scales = new Uint32Array(Math.ceil(blockCount / 2));
  const scales16 = new Uint16Array(scales.buffer);
  for (let block = 0; block < blockCount; block++) {
    const offset = block * Q8_BLOCK_BYTES;
    scales16[block] = bytes[offset]! | (bytes[offset + 1]! << 8);
    quants.set(bytes.subarray(offset + 2, offset + Q8_BLOCK_BYTES), block * Q8_BLOCK_SIZE);
  }
  return { kind: "q8", shape: info.shape, quants, scales };
}

export function dequantizeQ8(weight: Q8Weight): Float32Array {
  assertQ8Elements(weight.quants.length);
  const output = new Float32Array(weight.quants.length);
  const scales16 = new Uint16Array(
    weight.scales.buffer,
    weight.scales.byteOffset,
    Math.ceil(weight.quants.length / Q8_BLOCK_SIZE),
  );
  for (let block = 0; block < scales16.length; block++) {
    const scale = float16ToFloat32(scales16[block]!);
    const base = block * Q8_BLOCK_SIZE;
    for (let i = 0; i < Q8_BLOCK_SIZE; i++) {
      const byte = weight.quants[base + i]!;
      output[base + i] = scale * (byte > 127 ? byte - 256 : byte);
    }
  }
  return output;
}

export function quantizeQ8(data: Float32Array, shape: number[] = [data.length]): Q8Weight {
  if (data.length % Q8_BLOCK_SIZE !== 0) {
    throw new Error(`Q8_0 input length must be a multiple of ${Q8_BLOCK_SIZE}`);
  }
  const blockCount = data.length / Q8_BLOCK_SIZE;
  const quants = new Uint8Array(data.length);
  const scales = new Uint32Array(Math.ceil(blockCount / 2));
  const scales16 = new Uint16Array(scales.buffer);
  for (let block = 0; block < blockCount; block++) {
    const base = block * Q8_BLOCK_SIZE;
    let maximum = 0;
    for (let i = 0; i < Q8_BLOCK_SIZE; i++) maximum = Math.max(maximum, Math.abs(data[base + i]!));
    const scaleBits = float32ToFloat16(maximum / 127 || 1);
    scales16[block] = scaleBits;
    const scale = float16ToFloat32(scaleBits);
    for (let i = 0; i < Q8_BLOCK_SIZE; i++) {
      const quant = Math.max(-127, Math.min(127, Math.round(data[base + i]! / scale)));
      quants[base + i] = quant & 0xff;
    }
  }
  return { kind: "q8", shape, quants, scales };
}

export function dequantizeQ8Row(weight: Q8Weight, row: number): Float32Array {
  const [rows, columns] = weight.shape;
  if (weight.shape.length !== 2 || rows === undefined || columns === undefined) {
    throw new Error("Q8 row lookup requires a 2D weight");
  }
  if (row < 0 || row >= rows || !Number.isInteger(row)) throw new RangeError(`row ${row} is out of range`);
  if (columns % Q8_BLOCK_SIZE !== 0) throw new Error("Q8 row width must be block aligned");
  const output = new Float32Array(columns);
  const blocksPerRow = columns / Q8_BLOCK_SIZE;
  const scales16 = new Uint16Array(weight.scales.buffer, weight.scales.byteOffset);
  for (let block = 0; block < blocksPerRow; block++) {
    const globalBlock = row * blocksPerRow + block;
    const scale = float16ToFloat32(scales16[globalBlock]!);
    const quantBase = globalBlock * Q8_BLOCK_SIZE;
    for (let i = 0; i < Q8_BLOCK_SIZE; i++) {
      const byte = weight.quants[quantBase + i]!;
      output[block * Q8_BLOCK_SIZE + i] = scale * (byte > 127 ? byte - 256 : byte);
    }
  }
  return output;
}
