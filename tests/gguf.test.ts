import assert from "node:assert/strict";
import { test } from "node:test";
import { parseGGUFHeader } from "../src/engine/gguf.ts";

function buildMinimalGGUF(tensorType = 8): ArrayBuffer {
  const bytes = new Uint8Array(256);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  const u32 = (value: number) => { view.setUint32(offset, value, true); offset += 4; };
  const u64 = (value: number) => { view.setBigUint64(offset, BigInt(value), true); offset += 8; };
  const string = (value: string) => {
    const encoded = new TextEncoder().encode(value);
    u64(encoded.length);
    bytes.set(encoded, offset);
    offset += encoded.length;
  };
  u32(0x46554747);
  u32(3);
  u64(1);
  u64(1);
  string("general.alignment");
  u32(4);
  u32(32);
  string("token_embd.weight");
  u32(2);
  u64(32);
  u64(2);
  u32(tensorType);
  u64(0);
  return bytes.buffer;
}

test("parseGGUFHeader indexes Q8 tensor byte ranges", () => {
  const index = parseGGUFHeader(buildMinimalGGUF());
  const tensor = index.tensors["token_embd.weight"]!;
  assert.deepEqual(tensor.shape, [2, 32]);
  assert.equal(tensor.nElements, 64);
  assert.equal(tensor.byteLength, 68);
  assert.equal(tensor.byteOffset % 32, 0);
});

test("parseGGUFHeader rejects unsupported tensor quantization", () => {
  assert.throws(() => parseGGUFHeader(buildMinimalGGUF(2)), /unsupported GGML tensor type/);
});
