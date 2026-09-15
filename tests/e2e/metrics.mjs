/*
 * Two-browser smoke test for the performance panel and the weight-cache card.
 *
 * The engine is stubbed at the cluster's own boundary, so this exercises the parts
 * unit tests cannot reach — v3 activation frames crossing a real WebRTC channel,
 * the worker timing its own layers and reporting that time back, and the host
 * splitting a round trip into network and peer compute — without a GPU or a
 * multi-gigabyte download.
 *
 * Needs a running preview server:
 *
 *   npm run preview &
 *   npm run test:e2e:metrics
 */
import assert from "node:assert/strict";
import { chromium } from "playwright";

const baseURL = process.env.BASE_URL ?? "http://127.0.0.1:4173";
const executablePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/** Worker sleep per activation, in ms. Must dominate the loopback wire time. */
const WORKER_COMPUTE_MS = 6;
const PREFILL_TOKENS = 4;
const DECODE_TOKENS = 6;

const STUB = `
window.__LOCAL_CLUSTER_AI_ENGINE_FACTORY__ = async () => {
  const LAYERS = 8;
  const HIDDEN = 64;
  const layerBytes = Array.from({ length: LAYERS }, () => 1024 * 1024);
  const profile = {
    modelId: "stub", label: "Stub", url: "https://example.invalid/stub.gguf",
    layerCount: LAYERS, hiddenSize: HIDDEN, vocabSize: 256,
    layerBytes, edgeBytes: 2 * 1024 * 1024,
    totalBytes: layerBytes.reduce((a, b) => a + b, 0) + 2 * 1024 * 1024,
    config: { hiddenSize: HIDDEN, intermediateSize: 128, layerCount: LAYERS,
      attentionHeads: 2, keyValueHeads: 1, headDim: 32, vocabSize: 256,
      rmsNormEpsilon: 1e-6, ropeTheta: 1e6, contextLength: 4096 },
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let assignment;
  return {
    catalogue: [{ id: "stub", label: "Stub", params: "0.1B",
      url: profile.url, quantization: "Q8_0", approxBytes: profile.totalBytes }],
    activeProfile: profile,
    async probeModel() { return profile; },
    async getCapabilities(label) {
      return { label, userAgent: navigator.userAgent, webgpu: true, gpu: "Stub GPU",
        maxBufferSize: 512 * 1024 ** 2, maxStorageBufferBindingSize: 256 * 1024 ** 2,
        budgetBytes: 4 * 1024 ** 3, budgetSource: "heuristic" };
    },
    async load(options) {
      assignment = options.assignment;
      options.onProgress(0.5, "Restoring layer 0 attn_q…");
      await sleep(20);
      options.onProgress(1, "Model shard ready — 4.00 MB restored from cache");
    },
    async runHidden(hidden) {
      await sleep(${WORKER_COMPUTE_MS});
      return hidden.map((v) => v * 1.01);
    },
    async generate(transcript, options) {
      const hidden = new Float32Array(HIDDEN).fill(0.25);
      const stage = (phase, r, head) => options.onStage({
        phase, hostMicros: 1500,
        wireMicros: Math.max(0, r.roundTripMicros - r.workerMicros),
        workerMicros: r.workerMicros, headMicros: head,
        bytesOut: r.bytesOut, bytesIn: r.bytesIn });
      for (let i = 0; i < ${PREFILL_TOKENS}; i++) {
        stage("prefill", await options.runRemoteHidden(hidden, i, true), 0);
      }
      for (let i = 0; i < ${DECODE_TOKENS}; i++) {
        await sleep(15);
        options.onToken(i, "tok ");
        stage("decode", await options.runRemoteHidden(hidden, ${PREFILL_TOKENS} + i, false), 3200);
      }
    },
    async cacheReport() {
      const bytes = assignment ? assignment.bytes : 0;
      return { available: true, persisted: true,
        shard: { entries: 40, bytes },
        total: { entries: 80, bytes: bytes * 2 },
        storage: { usageBytes: 4 * 1024 ** 3, quotaBytes: 60 * 1024 ** 3 } };
    },
    async clearCache() {},
    reset() {},
    dispose() {},
  };
};
`;

const browser = await chromium.launch({ executablePath, headless: true });
try {
  const hostContext = await browser.newContext();
  const workerContext = await browser.newContext();
  const host = await hostContext.newPage();
  const worker = await workerContext.newPage();
  const errors = [];
  for (const [name, page] of [["host", host], ["worker", worker]]) {
    page.on("pageerror", (error) => errors.push(`${name}: ${error.stack ?? error}`));
    page.on("console", (message) => {
      if (
        message.type() === "error" &&
        !message.text().startsWith("Failed to load resource:")
      ) {
        errors.push(`${name} console: ${message.text()}`);
      }
    });
  }

  // The stub replaces the factory after the bundle has installed the real one;
  // the cluster resolves it when `start` runs, which is the click below.
  await host.goto(baseURL);
  await host.evaluate(STUB);
  await host.locator("#create-cluster").click();
  const clusterCode = (await host.locator("#cluster-code-display").textContent())?.trim();
  assert.match(clusterCode ?? "", /^[A-Z2-9]{6}$/);

  await worker.goto(`${baseURL}/?cluster=${clusterCode}`);
  await worker.evaluate(STUB);
  await worker.locator("#join-cluster").click();
  await host.waitForFunction(
    () => document.querySelector("#remote-phase")?.textContent === "connected",
    undefined,
    { timeout: 30_000 },
  );

  // The cache card reports before any shard is assigned, so a user knows up
  // front whether this session will re-download. It lives in the inspector's
  // model group, which is what a user opens to choose the split.
  await host.locator('[data-inspect="model"]').click();
  await host.locator("#cache-card").waitFor({ state: "visible" });
  assert.equal((await host.locator("#cache-state").textContent())?.trim(), "empty");

  await host.locator("#assign-layers").click();
  await host.waitForFunction(
    () => document.querySelector("#composer-hint")?.textContent?.includes("Enter to send"),
    undefined,
    { timeout: 30_000 },
  );
  // Loading fills the cache, so the card must re-read rather than stay "empty".
  await host.waitForFunction(
    () => document.querySelector("#cache-state")?.textContent?.trim() === "complete",
    undefined,
    { timeout: 30_000 },
  );
  assert.match(
    (await host.locator("#cluster-status").textContent()) ?? "",
    /shards are ready/i,
  );

  await host.locator("#prompt-input").fill("hello");
  await host.locator("#prompt-input").press("Enter");
  // The inspector shows one group at a time, so the panel has to be asked for.
  await host.locator('[data-inspect="perf"]').click();
  await host.locator("#telemetry").waitFor({ state: "visible" });
  await host.waitForFunction(
    () => document.querySelector("#metric-live")?.textContent === "last run",
    undefined,
    { timeout: 30_000 },
  );

  const number = async (page, id) =>
    Number(((await page.locator(`#${id}`).textContent()) ?? "").replace(/[^\d.]/g, ""));

  /** Parse a `formatMicros` reading — "750µs", "7.2ms", "1.50s" — into ms. */
  const millis = async (page, id) => {
    const raw = ((await page.locator(`#${id}`).textContent()) ?? "").trim();
    const value = Number(raw.replace(/[^\d.]/g, ""));
    assert.ok(Number.isFinite(value), `unreadable duration "${raw}" in #${id}`);
    if (raw.endsWith("µs")) return value / 1000;
    if (raw.endsWith("ms")) return value;
    if (raw.endsWith("s")) return value * 1000;
    throw new Error(`unrecognised duration unit in "${raw}"`);
  };

  assert.equal(
    await number(host, "metric-tokens"),
    PREFILL_TOKENS + DECODE_TOKENS,
    "every prompt and generated token must be accounted for",
  );
  assert.ok((await number(host, "metric-ttft")) > 0, "time to first token must be measured");

  const shares = {};
  for (const stage of ["host", "wire", "worker", "head"]) {
    shares[stage] = await number(host, `stage-${stage}-share`);
  }
  const total = Object.values(shares).reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(total - 100) <= 2, `stage shares summed to ${total}%`);

  // A worker never runs `generate`, so its panel is fed purely by the frames it
  // answered: all of its measured time is its own compute. Its last sample may
  // still be inside the publish interval, so wait for the full count.
  await worker.locator('[data-inspect="perf"]').click();
  await worker.locator("#telemetry").waitFor({ state: "visible" });
  await worker.waitForFunction(
    (expected) =>
      Number(document.querySelector("#metric-tokens")?.textContent ?? 0) === expected,
    PREFILL_TOKENS + DECODE_TOKENS,
    { timeout: 30_000 },
  );
  assert.equal((await worker.locator("#stage-worker-share").textContent())?.trim(), "100%");

  // The whole point of the v3 header: the peer's own compute arrives as its own
  // number rather than being lumped into the round trip. Asserted against the
  // peer's own reading rather than against the wire's share, which depends on
  // how fast this particular machine loops WebRTC back to itself.
  const hostViewOfWorker = await millis(host, "stage-worker-time");
  const workerViewOfSelf = await millis(worker, "stage-worker-time");
  assert.ok(
    hostViewOfWorker >= WORKER_COMPUTE_MS * 0.6,
    `the host must see the peer's real compute, got ${hostViewOfWorker}ms ` +
      `for a ${WORKER_COMPUTE_MS}ms peer`,
  );
  assert.ok(
    Math.abs(hostViewOfWorker - workerViewOfSelf) < 3,
    `the peer's compute must survive the wire unchanged: host saw ` +
      `${hostViewOfWorker}ms, peer measured ${workerViewOfSelf}ms`,
  );
  assert.ok(shares.wire > 0, "the network's residual share must be attributed");

  const sparkPath = await host.locator("#spark-line").getAttribute("d");
  assert.ok(sparkPath && sparkPath.length > 10, "the latency sparkline must draw a path");

  const barWidths = await host.evaluate(() =>
    [...document.querySelectorAll("#stage-bar i")].map(
      (element) => element.getBoundingClientRect().width,
    ),
  );
  assert.ok(barWidths.some((width) => width > 1), "the stage bar must render segments");

  assert.ok(
    ((await host.locator("#metric-bytes").textContent()) ?? "").includes("/token"),
    "activation bytes per token must be reported",
  );

  if (errors.length > 0) throw new Error(errors.join("\n"));
  console.log(`metrics smoke passed: ${clusterCode}`);
} finally {
  await browser.close();
}
