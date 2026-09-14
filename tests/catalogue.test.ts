import assert from "node:assert/strict";
import { test } from "node:test";
import { shardDownloadBytes } from "../src/engine/gguf.ts";
import {
  DEFAULT_MODEL_ID,
  MODEL_CATALOGUE,
  QWEN3_06B_Q8_URL,
  findModel,
  profileModel,
  requireModel,
  roleDownloadBytes,
} from "../src/engine/model.ts";
import type { ModelRole } from "../src/engine/types.ts";
import { QWEN3_SHAPES, buildGGUFIndex } from "./support/modelFixture.ts";

test("catalogue entries are unique, Q8_0, and single-file GGUF URLs", () => {
  const ids = new Set<string>();
  for (const entry of MODEL_CATALOGUE) {
    assert.ok(!ids.has(entry.id), `duplicate model id ${entry.id}`);
    ids.add(entry.id);
    assert.equal(entry.quantization, "Q8_0", `${entry.id} must be Q8_0: gguf.ts has no K-quant path`);
    assert.match(entry.url, /^https:\/\/huggingface\.co\/.+\.gguf$/, `${entry.id} url`);
    // The loader range-fetches one file; a sharded "-00001-of-00003.gguf" cannot work.
    assert.doesNotMatch(entry.url, /-\d{5}-of-\d{5}\.gguf$/, `${entry.id} must not be a sharded GGUF`);
    assert.ok(entry.approxBytes > 0, `${entry.id} needs a size`);
  }
});

test("the default model resolves and unknown ids fail loudly", () => {
  assert.ok(findModel(DEFAULT_MODEL_ID), "default model must be in the catalogue");
  assert.equal(findModel("nope"), undefined);
  assert.throws(() => requireModel("nope"), /unknown model nope/);
});

test("the legacy 0.6B url still points at the catalogue entry", () => {
  assert.equal(QWEN3_06B_Q8_URL, requireModel("qwen3-0.6b-q8_0").url);
});

test("profileModel reads real Qwen3 geometry out of a GGUF header", () => {
  const expected = {
    "0.6B": { layerCount: 28, hiddenSize: 1024 },
    "1.7B": { layerCount: 28, hiddenSize: 2048 },
    "4B": { layerCount: 36, hiddenSize: 2560 },
  } as const;

  for (const [label, shape] of Object.entries(QWEN3_SHAPES)) {
    const entry = requireModel(MODEL_CATALOGUE.find((m) => m.params === label)!.id);
    const profile = profileModel(buildGGUFIndex(shape), entry);

    assert.equal(profile.layerCount, expected[label as keyof typeof expected].layerCount);
    assert.equal(profile.hiddenSize, expected[label as keyof typeof expected].hiddenSize);
    assert.equal(profile.vocabSize, 151936);
    assert.equal(profile.layerBytes.length, profile.layerCount);
    assert.equal(
      profile.totalBytes,
      profile.layerBytes.reduce((a, b) => a + b, 0) + profile.edgeBytes,
      "totalBytes must be the sum of its parts",
    );
    // Tensor bytes are the bulk of the file; the rest is header and tokenizer metadata.
    assert.ok(
      profile.totalBytes > entry.approxBytes * 0.95 && profile.totalBytes <= entry.approxBytes,
      `${label}: profiled ${profile.totalBytes} is implausible against the ${entry.approxBytes}-byte file`,
    );
  }
});

test("roleDownloadBytes agrees with shardDownloadBytes for every role", () => {
  const shape = QWEN3_SHAPES["4B"];
  const index = buildGGUFIndex(shape);
  const profile = profileModel(index, requireModel("qwen3-4b-q8_0"));

  const roles: ModelRole[] = [
    { layerRange: [0, 1], hasEmbedding: true, hasHead: true },
    { layerRange: [0, 18], hasEmbedding: true, hasHead: true },
    { layerRange: [18, 36], hasEmbedding: false, hasHead: false },
    { layerRange: [35, 36], hasEmbedding: false, hasHead: false },
    { layerRange: [0, 36], hasEmbedding: true, hasHead: true },
  ];
  for (const role of roles) {
    assert.equal(
      roleDownloadBytes(profile, role),
      shardDownloadBytes(index, role),
      `role [${role.layerRange[0]}, ${role.layerRange[1]}) drifted from the loader`,
    );
  }
});

test("roleDownloadBytes rejects a layer outside the model", () => {
  const profile = profileModel(buildGGUFIndex(QWEN3_SHAPES["0.6B"]), requireModel("qwen3-0.6b-q8_0"));
  assert.throws(
    () => roleDownloadBytes(profile, { layerRange: [0, 99], hasEmbedding: false, hasHead: false }),
    RangeError,
  );
});
