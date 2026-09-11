import assert from "node:assert/strict";
import { test } from "node:test";
import {
  dequantizeQ8,
  dequantizeQ8Row,
  float16ToFloat32,
  float32ToFloat16,
  quantizeQ8,
  repackQ8,
} from "../src/engine/quant.ts";
import type { TensorInfo } from "../src/engine/types.ts";

test("float16 conversion preserves representative values", () => {
  for (const value of [0, -0, 1, -2, 0.125, 65504, Number.POSITIVE_INFINITY]) {
    const roundTrip = float16ToFloat32(float32ToFloat16(value));
    assert.equal(roundTrip, value);
  }
});

test("repackQ8 extracts GGUF scales and signed quant bytes", () => {
  const bytes = new Uint8Array(68);
  new DataView(bytes.buffer).setUint16(0, float32ToFloat16(0.5), true);
  new DataView(bytes.buffer).setUint16(34, float32ToFloat16(0.25), true);
  for (let i = 0; i < 32; i++) {
    bytes[2 + i] = (i - 16) & 0xff;
    bytes[36 + i] = (16 - i) & 0xff;
  }
  const info: TensorInfo = {
    name: "fixture",
    shape: [2, 32],
    ggmlType: 8,
    nElements: 64,
    byteOffset: 0,
    byteLength: 68,
  };
  const packed = repackQ8(info, bytes);
  const values = dequantizeQ8(packed);
  assert.equal(values[0], -8);
  assert.equal(values[31], 7.5);
  assert.equal(values[32], 4);
  assert.equal(values[63], -3.75);
  assert.deepEqual([...dequantizeQ8Row(packed, 1)], [...values.slice(32)]);
});

test("quantizeQ8 round trips within one block scale", () => {
  const input = Float32Array.from({ length: 64 }, (_, i) => Math.sin(i / 5) * 3);
  const quantized = quantizeQ8(input, [2, 32]);
  const output = dequantizeQ8(quantized);
  const scales16 = new Uint16Array(quantized.scales.buffer);
  for (let i = 0; i < input.length; i++) {
    const tolerance = float16ToFloat32(scales16[Math.floor(i / 32)]!) / 2 + 1e-6;
    assert.ok(Math.abs(input[i]! - output[i]!) <= tolerance, `index ${i}`);
  }
});
