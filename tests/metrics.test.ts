import assert from "node:assert/strict";
import { test } from "node:test";
import {
  emptySummary,
  formatCount,
  formatMicros,
  PerfRecorder,
  sampleTotalMicros,
  STAGE_NAMES,
  type TokenSample,
} from "../src/runtime/metrics.ts";

/** A controllable clock, so nothing here depends on real elapsed time. */
function fakeClock(): { now: () => number; advance: (ms: number) => void } {
  let time = 1000;
  return {
    now: () => time,
    advance: (ms: number) => {
      time += ms;
    },
  };
}

function sample(overrides: Partial<TokenSample> = {}): TokenSample {
  return {
    phase: "decode",
    hostMicros: 4000,
    wireMicros: 2000,
    workerMicros: 3000,
    headMicros: 1000,
    bytesOut: 4128,
    bytesIn: 4128,
    ...overrides,
  };
}

test("an empty recorder reports zeroes rather than dividing by zero", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  const summary = recorder.summary();
  assert.equal(summary.totalMicros, 0);
  assert.equal(summary.tokensPerSecond, 0);
  assert.equal(summary.bytesPerToken, 0);
  assert.equal(summary.ttftMicros, undefined);
  assert.equal(summary.dominantStage, undefined);
  for (const stage of STAGE_NAMES) {
    assert.deepEqual(summary.stages[stage], { micros: 0, share: 0, meanMicros: 0 });
  }
  assert.deepEqual(summary.recentMicros, []);
  assert.deepEqual({ ...emptySummary(), running: false }, summary);
});

test("stage shares sum to one and name the dominant stage", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  recorder.record(sample());
  recorder.record(sample());

  const summary = recorder.summary();
  assert.equal(summary.totalMicros, 20_000);
  assert.equal(summary.stages.host.micros, 8000);
  assert.equal(summary.stages.host.meanMicros, 4000);
  assert.equal(summary.stages.wire.micros, 4000);
  assert.equal(summary.stages.worker.micros, 6000);
  assert.equal(summary.stages.head.micros, 2000);

  const shares = STAGE_NAMES.reduce((total, stage) => total + summary.stages[stage].share, 0);
  assert.ok(Math.abs(shares - 1) < 1e-9, `shares summed to ${shares}`);
  assert.equal(summary.dominantStage, "host");
});

test("wire time dominating is reported as such", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  // The interesting failure mode: a fast peer behind a slow link.
  recorder.record(sample({ hostMicros: 1000, wireMicros: 90_000, workerMicros: 2000, headMicros: 500 }));
  const summary = recorder.summary();
  assert.equal(summary.dominantStage, "wire");
  assert.ok(summary.stages.wire.share > 0.9);
});

test("prefill and decode tokens are counted apart", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  for (let i = 0; i < 5; i++) recorder.record(sample({ phase: "prefill", headMicros: 0 }));
  for (let i = 0; i < 3; i++) recorder.record(sample());

  const summary = recorder.summary();
  assert.equal(summary.promptTokens, 5);
  assert.equal(summary.decodeTokens, 3);
});

test("time to first token is wall clock from the run's start", () => {
  const clock = fakeClock();
  const recorder = new PerfRecorder({ now: clock.now });
  recorder.begin();

  clock.advance(120);
  recorder.record(sample({ phase: "prefill" }));
  clock.advance(80);
  recorder.record(sample({ phase: "prefill" }));
  clock.advance(50);
  recorder.record(sample({ phase: "decode" }));
  clock.advance(40);
  recorder.record(sample({ phase: "decode" }));

  // 120 + 80 + 50 ms of prefill before the first decode token landed.
  assert.equal(recorder.summary().ttftMicros, 250_000);
});

test("the decode rate uses wall clock, not summed stage time", () => {
  const clock = fakeClock();
  const recorder = new PerfRecorder({ now: clock.now });
  recorder.begin();

  // Four decode tokens, 250 ms apart: three gaps over 750 ms is 4 tok/s.
  for (let i = 0; i < 4; i++) {
    clock.advance(250);
    recorder.record(sample());
  }
  const summary = recorder.summary();
  assert.ok(Math.abs(summary.tokensPerSecond - 4) < 1e-9, `rate was ${summary.tokensPerSecond}`);
});

test("a single token yields no rate rather than an invented one", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  recorder.record(sample());
  assert.equal(recorder.summary().tokensPerSecond, 0);
});

test("the window bounds the sparkline without losing the run's totals", () => {
  const clock = fakeClock();
  const recorder = new PerfRecorder({ now: clock.now, window: 4 });
  recorder.begin();
  for (let i = 0; i < 10; i++) {
    clock.advance(10);
    recorder.record(sample({ hostMicros: 1000 * (i + 1) }));
  }

  const summary = recorder.summary();
  assert.equal(summary.recentMicros.length, 4, "the sparkline shows only the window");
  assert.equal(summary.decodeTokens, 10, "totals still cover the whole run");
  // The window holds the last four, so the final sample's total is last.
  assert.equal(
    summary.recentMicros.at(-1),
    sampleTotalMicros(sample({ hostMicros: 10_000 })),
  );
});

test("bytes are totalled per direction and per token", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  recorder.record(sample({ bytesOut: 100, bytesIn: 300 }));
  recorder.record(sample({ bytesOut: 100, bytesIn: 300 }));

  const summary = recorder.summary();
  assert.equal(summary.bytesOut, 200);
  assert.equal(summary.bytesIn, 600);
  assert.equal(summary.bytesPerToken, 400);
});

test("begin discards the previous run, end keeps its numbers on screen", () => {
  const recorder = new PerfRecorder({ now: fakeClock().now });
  recorder.begin();
  recorder.record(sample());
  recorder.record(sample());
  recorder.end();

  const finished = recorder.summary();
  assert.equal(finished.running, false);
  assert.equal(finished.decodeTokens, 2, "a finished run's numbers survive");

  recorder.begin();
  const fresh = recorder.summary();
  assert.equal(fresh.running, true);
  assert.equal(fresh.decodeTokens, 0);
  assert.equal(fresh.totalMicros, 0);
});

test("a worker records without ever calling begin", () => {
  const clock = fakeClock();
  const recorder = new PerfRecorder({ now: clock.now });
  // The worker only ever sees `runHidden`; the host owns run boundaries.
  clock.advance(500);
  recorder.record(sample({ hostMicros: 0, wireMicros: 0, headMicros: 0, workerMicros: 7000 }));
  clock.advance(500);
  recorder.record(sample({ hostMicros: 0, wireMicros: 0, headMicros: 0, workerMicros: 7000 }));

  const summary = recorder.summary();
  assert.equal(summary.decodeTokens, 2);
  assert.equal(summary.stages.worker.share, 1);
  assert.equal(summary.stages.wire.share, 0);
});

test("durations and byte counts read as the units a human expects", () => {
  assert.equal(formatMicros(0), "—");
  assert.equal(formatMicros(750), "750µs");
  assert.equal(formatMicros(1500), "1.5ms");
  assert.equal(formatMicros(42_000), "42ms");
  assert.equal(formatMicros(1_500_000), "1.50s");
  assert.equal(formatMicros(65_000_000), "65.0s");

  assert.equal(formatCount(0), "—");
  assert.equal(formatCount(512), "512 B");
  assert.equal(formatCount(4096), "4.0 KB");
  assert.equal(formatCount(5 * 1024 ** 2), "5.0 MB");
  assert.equal(formatCount(2.25 * 1024 ** 3), "2.25 GB");
});
