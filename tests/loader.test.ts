import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DEFAULT_FETCH_CONCURRENCY,
  loadModelWeights,
  shardDownloadBytes,
  shardTensorNames,
} from "../src/engine/gguf.ts";
import type { GGUFIndex, LoadProgress, ModelRole } from "../src/engine/types.ts";
import { buildGGUFIndex, type FixtureShape } from "./support/modelFixture.ts";

const SHAPE: FixtureShape = {
  layerCount: 4,
  hiddenSize: 64,
  intermediateSize: 128,
  attentionHeads: 2,
  keyValueHeads: 1,
  headDim: 32,
  vocabSize: 256,
};

const HOST: ModelRole = { layerRange: [0, 2], hasEmbedding: true, hasHead: true };
const WORKER: ModelRole = { layerRange: [2, 4], hasEmbedding: false, hasHead: false };

/** Serves zero-filled bytes for whatever range is asked for, and records concurrency. */
function rangeServer(index: GGUFIndex) {
  let inFlight = 0;
  let peakInFlight = 0;
  const requested: string[] = [];

  const fetchFn: typeof fetch = async (_input, init) => {
    const range = new Headers(init?.headers).get("Range");
    assert.ok(range, "the loader must issue a Range request");
    const [start, end] = range.slice("bytes=".length).split("-").map(Number);
    requested.push(range);

    inFlight++;
    peakInFlight = Math.max(peakInFlight, inFlight);
    // Yield so overlapping requests can actually overlap.
    await new Promise((resolve) => setTimeout(resolve, 1));
    inFlight--;

    return new Response(new Uint8Array(end! - start! + 1), {
      status: 206,
      headers: { "content-range": `bytes ${start}-${end}/${index.dataStart + 1}` },
    });
  };

  return { fetchFn, requested, peak: () => peakInFlight };
}

test("the loader fetches every tensor its role owns, and no others", async () => {
  const index = buildGGUFIndex(SHAPE);
  const server = rangeServer(index);
  const weights = await loadModelWeights(index, HOST, server.fetchFn);

  assert.equal(weights.layers.length, 2);
  assert.ok(weights.embedding, "a host role owns the embedding table");
  assert.ok(weights.finalNorm, "a host role owns the final norm");
  assert.equal(server.requested.length, shardTensorNames(index, HOST).length);

  // Tied embeddings: the head reuses token_embd, so it must not be fetched twice.
  assert.equal(new Set(server.requested).size, server.requested.length, "no tensor may be fetched twice");
});

test("a worker role never fetches the embedding, norm, or head", async () => {
  const index = buildGGUFIndex(SHAPE);
  const server = rangeServer(index);
  const weights = await loadModelWeights(index, WORKER, server.fetchFn);

  assert.equal(weights.layers.length, 2);
  assert.equal(weights.embedding, undefined);
  assert.equal(weights.finalNorm, undefined);
  assert.equal(weights.head, undefined);

  const edges = shardTensorNames(index, { layerRange: [0, 0], hasEmbedding: true, hasHead: true });
  for (const name of edges) {
    const info = index.tensors[name]!;
    assert.ok(
      !server.requested.includes(`bytes=${info.byteOffset}-${info.byteOffset + info.byteLength - 1}`),
      `worker must not fetch ${name}`,
    );
  }
});

test("tensors are fetched concurrently, up to the configured limit", async () => {
  const index = buildGGUFIndex(SHAPE);

  const parallel = rangeServer(index);
  await loadModelWeights(index, HOST, parallel.fetchFn, undefined, 4);
  assert.ok(parallel.peak() > 1, "the loader must overlap requests");
  assert.ok(parallel.peak() <= 4, `concurrency ${parallel.peak()} exceeded the limit of 4`);

  const serial = rangeServer(index);
  await loadModelWeights(index, HOST, serial.fetchFn, undefined, 1);
  assert.equal(serial.peak(), 1, "concurrency 1 must stay sequential");
});

test("progress rises monotonically to exactly the shard size", async () => {
  const index = buildGGUFIndex(SHAPE);
  const server = rangeServer(index);
  const seen: LoadProgress[] = [];
  await loadModelWeights(index, HOST, server.fetchFn, (progress) => seen.push({ ...progress }));

  assert.equal(seen.length, shardTensorNames(index, HOST).length);
  for (let i = 1; i < seen.length; i++) {
    assert.ok(seen[i]!.loadedBytes > seen[i - 1]!.loadedBytes, "loaded bytes must only rise");
  }
  const expected = shardDownloadBytes(index, HOST);
  assert.equal(seen.at(-1)!.loadedBytes, expected);
  assert.equal(seen.at(-1)!.totalBytes, expected);
});

test("an aborting progress callback stops the load", async () => {
  const index = buildGGUFIndex(SHAPE);
  const server = rangeServer(index);
  await assert.rejects(
    loadModelWeights(index, HOST, server.fetchFn, () => {
      throw new Error("aborted");
    }),
    /aborted/,
  );
});

test("the default concurrency is a sane, bounded number", () => {
  assert.ok(Number.isInteger(DEFAULT_FETCH_CONCURRENCY));
  assert.ok(DEFAULT_FETCH_CONCURRENCY > 1 && DEFAULT_FETCH_CONCURRENCY <= 16);
});
