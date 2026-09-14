import assert from "node:assert/strict";
import { test } from "node:test";
import {
  planHeadChunks,
  sliceQ8Rows,
  STORAGE_OFFSET_ALIGNMENT,
} from "../src/engine/headChunks.ts";
import { dequantizeQ8, float32ToFloat16 } from "../src/engine/quant.ts";
import type { Q8Weight } from "../src/engine/types.ts";

const VOCAB = 151936;
const MIB = 1024 * 1024;

/** The three catalogue models, at the WebGPU default 256 MiB binding cap. */
const MODELS: ReadonlyArray<{ label: string; hidden: number; chunked: boolean }> = [
  { label: "0.6B", hidden: 1024, chunked: false },
  { label: "1.7B", hidden: 2048, chunked: true },
  { label: "4B", hidden: 2560, chunked: true },
];

function assertTiles(chunks: ReturnType<typeof planHeadChunks>, rows: number, columns: number, limit: number): void {
  assert.ok(chunks.length > 0, "expected at least one chunk");
  let expectedOffset = 0;
  for (const chunk of chunks) {
    assert.equal(chunk.rowOffset, expectedOffset, "chunks must tile without gaps or overlap");
    assert.ok(chunk.rowCount > 0, "every chunk must cover at least one row");
    assert.ok(chunk.rowCount * columns <= limit, "every chunk must fit the binding limit");
    assert.equal(
      (chunk.rowOffset * Float32Array.BYTES_PER_ELEMENT) % STORAGE_OFFSET_ALIGNMENT,
      0,
      "every logits slice must start on an aligned offset",
    );
    expectedOffset += chunk.rowCount;
  }
  assert.equal(expectedOffset, rows, "chunks must cover every row exactly once");
}

test("chunks tile the vocabulary exactly and respect the binding limit", () => {
  for (const { label, hidden } of MODELS) {
    const limit = 256 * MIB;
    const chunks = planHeadChunks(VOCAB, hidden, limit);
    assertTiles(chunks, VOCAB, hidden, limit);
    assert.ok(chunks.length >= 1, `${label} produced no chunks`);
  }
});

test("only models above 0.6B need more than one chunk at the 256 MiB default", () => {
  for (const { label, hidden, chunked } of MODELS) {
    const chunks = planHeadChunks(VOCAB, hidden, 256 * MIB);
    assert.equal(chunks.length > 1, chunked, `${label} chunk count did not match expectation`);
  }
});

test("a limit larger than the whole matrix yields a single chunk", () => {
  const chunks = planHeadChunks(VOCAB, 2560, 2 * 1024 * MIB);
  assert.deepEqual(chunks, [{ rowOffset: 0, rowCount: VOCAB }]);
});

test("chunking holds across a sweep of limits and hidden sizes", () => {
  for (const hidden of [1024, 2048, 2560, 4096]) {
    for (const limit of [64 * MIB, 128 * MIB, 256 * MIB, 512 * MIB, 1024 * MIB]) {
      assertTiles(planHeadChunks(VOCAB, hidden, limit), VOCAB, hidden, limit);
    }
  }
});

test("a vocabulary smaller than one aligned chunk stays whole", () => {
  assert.deepEqual(planHeadChunks(7, 32, 256 * MIB), [{ rowOffset: 0, rowCount: 7 }]);
});

test("planHeadChunks rejects unusable inputs", () => {
  assert.throws(() => planHeadChunks(0, 1024, MIB), RangeError);
  assert.throws(() => planHeadChunks(VOCAB, 0, MIB), RangeError);
  assert.throws(() => planHeadChunks(VOCAB, 1000, MIB), RangeError, "columns must be a multiple of 32");
  assert.throws(() => planHeadChunks(VOCAB, 1024, 0), RangeError);
  // 64 aligned rows of 1024 columns need 64 KiB; a 1 KiB cap cannot hold them.
  assert.throws(() => planHeadChunks(VOCAB, 1024, 1024), RangeError);
});

/** Build a Q8 weight whose quant byte and scale both encode their own index. */
function syntheticQ8(rows: number, columns: number): Q8Weight {
  const blocksPerRow = columns / 32;
  const quants = new Uint8Array(rows * columns);
  for (let i = 0; i < quants.length; i++) quants[i] = i % 251;
  const scales = new Uint32Array(Math.ceil((rows * blocksPerRow) / 2));
  const scales16 = new Uint16Array(scales.buffer);
  for (let block = 0; block < rows * blocksPerRow; block++) scales16[block] = float32ToFloat16(block % 1024);
  return { kind: "q8", shape: [rows, columns], quants, scales };
}

test("sliceQ8Rows reproduces each chunk's quants and scales", () => {
  const rows = 640;
  const columns = 64;
  const weight = syntheticQ8(rows, columns);
  const blocksPerRow = columns / 32;
  const chunks = planHeadChunks(rows, columns, 128 * columns);
  assert.ok(chunks.length > 1, "expected this fixture to chunk");

  const rebuilt = new Uint8Array(rows * columns);
  for (const chunk of chunks) {
    const slice = sliceQ8Rows(weight, columns, chunk);
    assert.deepEqual(slice.shape, [chunk.rowCount, columns]);
    assert.equal(slice.quants.length, chunk.rowCount * columns);
    rebuilt.set(slice.quants, chunk.rowOffset * columns);

    const sliceScales = new Uint16Array(slice.scales.buffer);
    const sourceScales = new Uint16Array(weight.scales.buffer);
    for (let block = 0; block < chunk.rowCount * blocksPerRow; block++) {
      assert.equal(sliceScales[block], sourceScales[chunk.rowOffset * blocksPerRow + block]);
    }
  }
  assert.deepEqual(rebuilt, weight.quants, "reassembled chunks must equal the original matrix");
});

test("sliceQ8Rows does not alias the source buffers", () => {
  const weight = syntheticQ8(128, 32);
  const slice = sliceQ8Rows(weight, 32, { rowOffset: 64, rowCount: 64 });
  slice.quants[0] = 42;
  assert.notEqual(weight.quants[64 * 32], 42);
});

test("dequantized chunks reproduce the full head matrix element for element", () => {
  const rows = 512;
  const columns = 96;
  const weight = syntheticQ8(rows, columns);
  const reference = dequantizeQ8(weight);

  const chunks = planHeadChunks(rows, columns, 128 * columns);
  assert.ok(chunks.length > 1, "expected this fixture to chunk");

  for (const chunk of chunks) {
    const values = dequantizeQ8(sliceQ8Rows(weight, columns, chunk));
    const base = chunk.rowOffset * columns;
    for (let i = 0; i < values.length; i++) {
      assert.equal(values[i], reference[base + i], `mismatch at element ${base + i}`);
    }
  }
});
