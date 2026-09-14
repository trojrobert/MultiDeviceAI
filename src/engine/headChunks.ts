/*
 * Row-chunking for the language-model head.
 *
 * The head is a `vocab x hidden` Q8_0 matrix, and Qwen3 ties it to the
 * embedding table, so it is by far the largest single tensor in the model:
 * 156 MiB of quants at 0.6B, 311 MiB at 1.7B, 389 MiB at 4B. WebGPU caps a
 * single storage binding at `maxStorageBufferBindingSize`, which defaults to
 * 256 MiB, so anything above 0.6B cannot be bound as one buffer. Splitting the
 * head into row ranges keeps every binding under the device limit without
 * touching the kernels: both Q8 matvec variants bound their writes by
 * `q8Shape.output` and index `q8Output[row]` relative to the dispatch, so a
 * chunk just needs its own weight buffers and its own slice of the logits.
 */

import type { Q8Weight } from "./types.ts";

/** WebGPU's `minStorageBufferOffsetAlignment` default, and its guaranteed maximum. */
export const STORAGE_OFFSET_ALIGNMENT = 256;

/**
 * Rows per chunk are aligned to this so that a chunk's logits slice starts at
 * `rowOffset * 4` bytes, always a multiple of the offset alignment.
 */
const ROW_ALIGNMENT = STORAGE_OFFSET_ALIGNMENT / Float32Array.BYTES_PER_ELEMENT;

export const Q8_BLOCK_SIZE = 32;

export interface HeadChunk {
  /** First output row this chunk covers. */
  rowOffset: number;
  /** Number of output rows in this chunk. */
  rowCount: number;
}

/**
 * Split a `rows x columns` Q8 matrix into row ranges whose quant buffers each
 * stay within `maxBindingBytes`. Chunks tile `[0, rows)` exactly, and every
 * `rowOffset` is a multiple of {@link ROW_ALIGNMENT}.
 */
export function planHeadChunks(rows: number, columns: number, maxBindingBytes: number): HeadChunk[] {
  if (!Number.isInteger(rows) || rows <= 0) throw new RangeError(`head rows must be a positive integer, got ${rows}`);
  if (!Number.isInteger(columns) || columns <= 0) {
    throw new RangeError(`head columns must be a positive integer, got ${columns}`);
  }
  if (columns % Q8_BLOCK_SIZE !== 0) {
    throw new RangeError(`head columns must be a multiple of ${Q8_BLOCK_SIZE}, got ${columns}`);
  }
  if (!Number.isFinite(maxBindingBytes) || maxBindingBytes <= 0) {
    throw new RangeError(`maxBindingBytes must be positive, got ${maxBindingBytes}`);
  }

  // One Q8 quant byte per element, so a chunk's quant buffer is rowCount * columns bytes.
  const unaligned = Math.floor(maxBindingBytes / columns);
  const rowsPerChunk = Math.floor(unaligned / ROW_ALIGNMENT) * ROW_ALIGNMENT;
  if (rowsPerChunk <= 0) {
    throw new RangeError(
      `maxStorageBufferBindingSize (${maxBindingBytes} bytes) is too small to bind ` +
        `${ROW_ALIGNMENT} rows of ${columns} columns for the language-model head`,
    );
  }
  if (rowsPerChunk >= rows) return [{ rowOffset: 0, rowCount: rows }];

  const chunks: HeadChunk[] = [];
  for (let rowOffset = 0; rowOffset < rows; rowOffset += rowsPerChunk) {
    chunks.push({ rowOffset, rowCount: Math.min(rowsPerChunk, rows - rowOffset) });
  }
  return chunks;
}

/**
 * Copy one chunk's rows out of a Q8 weight. Scales are packed two f16 per u32;
 * an aligned `rowOffset` keeps each chunk's block range even, so the scale
 * slice lands on a u32 boundary and needs no repacking.
 */
export function sliceQ8Rows(weight: Q8Weight, columns: number, chunk: HeadChunk): Q8Weight {
  const blocksPerRow = columns / Q8_BLOCK_SIZE;
  const firstBlock = chunk.rowOffset * blocksPerRow;
  const blockCount = chunk.rowCount * blocksPerRow;
  if (firstBlock % 2 !== 0) {
    throw new RangeError(`chunk at row ${chunk.rowOffset} does not start on a packed scale boundary`);
  }

  const quants = weight.quants.slice(chunk.rowOffset * columns, (chunk.rowOffset + chunk.rowCount) * columns);
  const scales16 = new Uint16Array(weight.scales.buffer, weight.scales.byteOffset, weight.scales.length * 2);
  const chunkScales = new Uint32Array(Math.ceil(blockCount / 2));
  new Uint16Array(chunkScales.buffer).set(scales16.subarray(firstBlock, firstBlock + blockCount));

  return { kind: "q8", shape: [chunk.rowCount, columns], quants, scales: chunkScales };
}
