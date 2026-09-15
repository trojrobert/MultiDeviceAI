import assert from "node:assert/strict";
import { test } from "node:test";
import { fetchGGUFIndex, loadModelWeights, shardTensorNames } from "../src/engine/gguf.ts";
import { MemoryWeightCache, type CachedRange, type WeightCache } from "../src/engine/weightCache.ts";
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
const URL_A = "https://example.invalid/model-a.gguf";
const URL_B = "https://example.invalid/model-b.gguf";

/**
 * Serves each range with bytes derived from its offset, so a cache that returns
 * the wrong entry produces different data rather than matching zeros.
 */
function rangeServer(index: GGUFIndex, url = URL_A) {
  const requested: string[] = [];
  const fetchFn: typeof fetch = async (_input, init) => {
    const range = new Headers(init?.headers).get("Range");
    assert.ok(range, "the loader must issue a Range request");
    const [start, end] = range.slice("bytes=".length).split("-").map(Number);
    requested.push(range);
    const bytes = new Uint8Array(end! - start! + 1);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (start! + i) & 0xff;
    return new Response(bytes, {
      status: 206,
      headers: { "content-range": `bytes ${start}-${end}/${index.dataStart + 1}` },
    });
  };
  return { fetchFn, requested, index: { ...index, url } };
}

test("a second load of the same shard fetches nothing", async () => {
  const base = buildGGUFIndex(SHAPE);
  const cache = new MemoryWeightCache();

  const cold = rangeServer(base);
  const first = await loadModelWeights(cold.index, HOST, { fetchFn: cold.fetchFn, cache });
  assert.equal(cold.requested.length, shardTensorNames(base, HOST).length);

  const warm = rangeServer(base);
  const second = await loadModelWeights(warm.index, HOST, { fetchFn: warm.fetchFn, cache });
  assert.equal(warm.requested.length, 0, "a fully cached shard must not touch the network");

  // Same bytes, not merely the same shapes: a cache that mixed up two entries
  // would still produce a structurally valid model.
  assert.equal(first.layers.length, second.layers.length);
  for (let i = 0; i < first.layers.length; i++) {
    assert.deepEqual(first.layers[i]!.inputNorm.data, second.layers[i]!.inputNorm.data);
    assert.deepEqual(first.layers[i]!.query, second.layers[i]!.query);
  }
  assert.deepEqual(first.embedding, second.embedding);
});

test("a warm cache still fetches the tensors a new role adds", async () => {
  const base = buildGGUFIndex(SHAPE);
  const cache = new MemoryWeightCache();

  const cold = rangeServer(base);
  await loadModelWeights(cold.index, HOST, { fetchFn: cold.fetchFn, cache });

  // The worker owns layers the host never fetched, so those must still download
  // while the shared tensors, if any, come back from the cache.
  const warm = rangeServer(base);
  await loadModelWeights(warm.index, WORKER, { fetchFn: warm.fetchFn, cache });
  assert.equal(warm.requested.length, shardTensorNames(base, WORKER).length);

  const again = rangeServer(base);
  await loadModelWeights(again.index, WORKER, { fetchFn: again.fetchFn, cache });
  assert.equal(again.requested.length, 0);
});

test("entries are keyed per model, so two models never share bytes", async () => {
  const base = buildGGUFIndex(SHAPE);
  const cache = new MemoryWeightCache();

  const a = rangeServer(base, URL_A);
  await loadModelWeights(a.index, HOST, { fetchFn: a.fetchFn, cache });

  const b = rangeServer(base, URL_B);
  await loadModelWeights(b.index, HOST, { fetchFn: b.fetchFn, cache });
  assert.equal(
    b.requested.length,
    shardTensorNames(base, HOST).length,
    "a different model URL must miss even at identical offsets",
  );

  const perModel = await cache.stats(URL_A);
  const total = await cache.stats();
  assert.ok(perModel.bytes > 0);
  assert.equal(total.bytes, perModel.bytes * 2, "each model is accounted separately");
});

test("progress separates cached bytes from downloaded bytes", async () => {
  const base = buildGGUFIndex(SHAPE);
  const cache = new MemoryWeightCache();

  const cold = rangeServer(base);
  const coldProgress: LoadProgress[] = [];
  await loadModelWeights(cold.index, HOST, {
    fetchFn: cold.fetchFn,
    cache,
    onProgress: (progress) => coldProgress.push({ ...progress }),
  });
  const coldLast = coldProgress.at(-1)!;
  assert.equal(coldLast.cachedBytes, 0);
  assert.equal(coldLast.downloadedBytes, coldLast.loadedBytes);
  assert.ok(coldProgress.every((entry) => entry.fromCache === false));

  const warm = rangeServer(base);
  const warmProgress: LoadProgress[] = [];
  await loadModelWeights(warm.index, HOST, {
    fetchFn: warm.fetchFn,
    cache,
    onProgress: (progress) => warmProgress.push({ ...progress }),
  });
  const warmLast = warmProgress.at(-1)!;
  assert.equal(warmLast.downloadedBytes, 0);
  assert.equal(warmLast.cachedBytes, warmLast.loadedBytes);
  assert.equal(warmLast.loadedBytes, warmLast.totalBytes);
  assert.ok(warmProgress.every((entry) => entry.fromCache === true));
});

test("a truncated entry is discarded rather than handed to the engine", async () => {
  const base = buildGGUFIndex(SHAPE);
  const name = shardTensorNames(base, WORKER)[0]!;
  const info = base.tensors[name]!;

  // A cache that lost part of a body — an interrupted write, or an eviction
  // mid-entry — must miss, not return a short tensor.
  const damaged: WeightCache = {
    ...new MemoryWeightCache(),
    writable: true,
    async get(_url: string, range: CachedRange) {
      return range.name === name ? new Uint8Array(range.byteLength - 8) : undefined;
    },
    async put() {},
    async getHeader() {
      return undefined;
    },
    async putHeader() {},
    async stats() {
      return { entries: 0, bytes: 0 };
    },
    async clear() {},
  };

  const server = rangeServer(base);
  const weights = await loadModelWeights(server.index, WORKER, {
    fetchFn: server.fetchFn,
    cache: damaged,
  });
  assert.equal(weights.layers.length, 2);
  assert.ok(
    server.requested.includes(`bytes=${info.byteOffset}-${info.byteOffset + info.byteLength - 1}`),
    "the short entry must be re-fetched",
  );
});

test("a cache that cannot store anything still loads the shard", async () => {
  const base = buildGGUFIndex(SHAPE);
  // Mirrors a browser that has run out of quota: every write is dropped.
  const full: WeightCache = {
    writable: false,
    async get() {
      return undefined;
    },
    async put() {},
    async getHeader() {
      return undefined;
    },
    async putHeader() {},
    async stats() {
      return { entries: 0, bytes: 0 };
    },
    async clear() {},
  };

  const first = rangeServer(base);
  await loadModelWeights(first.index, HOST, { fetchFn: first.fetchFn, cache: full });
  const second = rangeServer(base);
  await loadModelWeights(second.index, HOST, { fetchFn: second.fetchFn, cache: full });
  assert.equal(second.requested.length, shardTensorNames(base, HOST).length);
});

test("a cached GGUF header is reused instead of re-probed", async () => {
  const cache = new MemoryWeightCache();
  // A whole file, of which the probe reads only the leading header.
  const file = new Uint8Array(4096);
  file.set(new Uint8Array(minimalGGUFHeader()), 0);

  let requests = 0;
  const fetchFn: typeof fetch = async (_input, init) => {
    requests++;
    const range = new Headers(init?.headers).get("Range")!;
    const [start, end] = range.slice("bytes=".length).split("-").map(Number);
    const slice = file.slice(start!, Math.min(end! + 1, file.length));
    return new Response(slice, {
      status: 206,
      headers: {
        "content-range": `bytes ${start}-${start! + slice.length - 1}/${file.length}`,
      },
    });
  };

  const cold = await fetchGGUFIndex(URL_A, { fetchFn, cache });
  assert.equal(requests, 1);

  const warm = await fetchGGUFIndex(URL_A, { fetchFn, cache });
  assert.equal(requests, 1, "a cached header must not re-probe the file");
  assert.equal(warm.headerBytes, cold.headerBytes);
  assert.equal(warm.dataStart, cold.dataStart);
  assert.equal(warm.url, URL_A);
  assert.deepEqual(warm.tensors, cold.tensors);

  // Only the parsed header is worth storing; the probe over-fetches by design.
  const stored = await cache.getHeader(URL_A);
  assert.ok(stored);
  assert.equal(stored.byteLength, cold.headerBytes);
});

/** One F32 tensor and one metadata key: enough for a real header round trip. */
function minimalGGUFHeader(): ArrayBuffer {
  const bytes = new Uint8Array(256);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  const u32 = (value: number) => {
    view.setUint32(offset, value, true);
    offset += 4;
  };
  const u64 = (value: number) => {
    view.setBigUint64(offset, BigInt(value), true);
    offset += 8;
  };
  const string = (value: string) => {
    const encoded = new TextEncoder().encode(value);
    u64(encoded.length);
    bytes.set(encoded, offset);
    offset += encoded.length;
  };
  u32(0x46554747);
  u32(3);
  u64(1); // tensor count
  u64(1); // metadata count
  string("general.alignment");
  u32(4);
  u32(32);
  string("token_embd.weight");
  u32(2);
  u64(32);
  u64(2);
  u32(8); // Q8_0
  u64(0);
  return bytes.buffer;
}
