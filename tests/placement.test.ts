import assert from "node:assert/strict";
import { test } from "node:test";
import { defaultBudgetBytes, planPlacement } from "../src/engine/placement.ts";
import { profileModel, requireModel, type ModelProfile } from "../src/engine/model.ts";
import { QWEN3_SHAPES, buildGGUFIndex } from "./support/modelFixture.ts";

const GB = 1024 ** 3;

function profileFor(params: "0.6B" | "1.7B" | "4B"): ModelProfile {
  const entry = requireModel(
    params === "0.6B" ? "qwen3-0.6b-q8_0" : params === "1.7B" ? "qwen3-1.7b-q8_0" : "qwen3-4b-q8_0",
  );
  return profileModel(buildGGUFIndex(QWEN3_SHAPES[params]), entry);
}

test("a small model on a roomy device reports single-device", () => {
  const result = planPlacement({
    profile: profileFor("0.6B"),
    hostBudgetBytes: 4 * GB,
    workerBudgetBytes: 4 * GB,
  });
  assert.equal(result.verdict, "single-device");
  assert.match(result.summary, /fits this device alone/);
});

test("4B on two mid-range devices needs both", () => {
  const profile = profileFor("4B");
  const result = planPlacement({ profile, hostBudgetBytes: 3.6 * GB, workerBudgetBytes: 3.6 * GB });

  assert.equal(result.verdict, "needs-both");
  assert.ok(result.singleDeviceBytes > 3.6 * GB, "4B must not fit one device at this budget");
  assert.ok(result.recommended, "a feasible split must exist");
  assert.ok(result.recommended!.hostFits && result.recommended!.workerFits);
  assert.match(result.summary, /more than either device alone/);
});

test("4B on two small devices is infeasible and names the short side", () => {
  const result = planPlacement({
    profile: profileFor("4B"),
    hostBudgetBytes: 1 * GB,
    workerBudgetBytes: 1 * GB,
  });
  assert.equal(result.verdict, "infeasible");
  assert.equal(result.recommended, undefined);
  assert.ok(result.shortfallBytes! > 0, "an infeasible plan must report how far short it is");

  // The reported shortfall must be the best achievable overage, and the named
  // side must be the one actually at it. At the optimum the two sides land
  // within a few tens of MB of each other, so asserting a fixed side is brittle.
  const overages = result.candidates.map((c) =>
    Math.max(c.hostBytes - 1 * GB, c.workerBytes - 1 * GB),
  );
  assert.equal(result.shortfallBytes, Math.min(...overages));
  const best = result.candidates[overages.indexOf(Math.min(...overages))]!;
  const expectedSide = best.hostBytes - 1 * GB >= best.workerBytes - 1 * GB ? "host" : "worker";
  assert.equal(result.shortfallSide, expectedSide);
  // The summary must name the short side and by how much, not just say "no".
  assert.match(result.summary, new RegExp(`the ${result.shortfallSide} is`));
  assert.match(result.summary, /\d+(\.\d+)? (MB|GB) over/);
});

test("a lopsided pair pushes layers onto the roomier device", () => {
  const profile = profileFor("4B");
  // Both budgets stay under the whole-model requirement so the verdict is
  // "needs-both" in each case and only the split differs.
  const balanced = planPlacement({ profile, hostBudgetBytes: 3.6 * GB, workerBudgetBytes: 3.6 * GB });
  const tinyWorker = planPlacement({ profile, hostBudgetBytes: 3.9 * GB, workerBudgetBytes: 2.0 * GB });

  assert.equal(balanced.verdict, "needs-both");
  assert.equal(tinyWorker.verdict, "needs-both");
  assert.ok(
    tinyWorker.recommended!.split > balanced.recommended!.split,
    "a weaker worker must be given fewer layers",
  );
});

test("candidates cover every legal split and move monotonically", () => {
  const profile = profileFor("1.7B");
  const result = planPlacement({ profile, hostBudgetBytes: 2 * GB, workerBudgetBytes: 2 * GB });

  assert.equal(result.candidates.length, profile.layerCount - 1);
  result.candidates.forEach((candidate, i) => assert.equal(candidate.split, i + 1));

  for (let i = 1; i < result.candidates.length; i++) {
    const previous = result.candidates[i - 1]!;
    const current = result.candidates[i]!;
    assert.ok(current.hostBytes > previous.hostBytes, "host cost must rise with the split");
    assert.ok(current.workerBytes < previous.workerBytes, "worker cost must fall with the split");
  }
});

test("the recommended split maximises the tighter side's headroom", () => {
  const result = planPlacement({
    profile: profileFor("4B"),
    hostBudgetBytes: 3.6 * GB,
    workerBudgetBytes: 3.6 * GB,
  });
  const feasible = result.candidates.filter((c) => c.hostFits && c.workerFits);
  const best = Math.max(...feasible.map((c) => c.minSlack));
  assert.equal(result.recommended!.minSlack, best);
  assert.ok(result.recommended!.minSlack >= 0, "a feasible split has non-negative slack");
});

test("the balanced split evens out the two downloads", () => {
  const result = planPlacement({
    profile: profileFor("4B"),
    hostBudgetBytes: 4 * GB,
    workerBudgetBytes: 4 * GB,
  });
  const delta = (c: { hostDownloadBytes: number; workerDownloadBytes: number }) =>
    Math.abs(c.hostDownloadBytes - c.workerDownloadBytes);
  const feasible = result.candidates.filter((c) => c.hostFits && c.workerFits);
  assert.equal(delta(result.balanced!), Math.min(...feasible.map(delta)));
});

test("a longer context raises the estimate through the KV caches", () => {
  const profile = profileFor("4B");
  const short = planPlacement({ profile, hostBudgetBytes: 4 * GB, workerBudgetBytes: 4 * GB, maxSequenceLength: 256 });
  const long = planPlacement({ profile, hostBudgetBytes: 4 * GB, workerBudgetBytes: 4 * GB, maxSequenceLength: 4096 });
  assert.ok(long.singleDeviceBytes > short.singleDeviceBytes);
});

test("defaultBudgetBytes prefers reported device memory and never returns nothing", () => {
  const reported = defaultBudgetBytes({
    maxBufferSize: 256 * 1024 ** 2,
    maxStorageBufferBindingSize: 256 * 1024 ** 2,
    deviceMemoryGiB: 8,
  });
  assert.equal(reported.source, "device-memory");
  assert.ok(reported.bytes > 3 * GB && reported.bytes < 8 * GB);

  const guessed = defaultBudgetBytes({ maxBufferSize: 128 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2 });
  assert.equal(guessed.source, "heuristic");
  assert.ok(guessed.bytes >= 768 * 1024 ** 2, "the heuristic must not floor below 768 MB");
  assert.ok(guessed.bytes <= 2 * GB, "the heuristic must not invent capacity");

  // More reported memory must never mean a smaller budget.
  let previous = 0;
  for (const deviceMemoryGiB of [0.5, 1, 2, 4, 8]) {
    const { bytes } = defaultBudgetBytes({ maxBufferSize: 1, maxStorageBufferBindingSize: 1, deviceMemoryGiB });
    assert.ok(bytes > previous, "budget must grow with device memory");
    previous = bytes;
  }
});
