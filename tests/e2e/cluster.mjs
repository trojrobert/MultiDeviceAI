import assert from "node:assert/strict";
import { chromium } from "playwright";

const baseURL = process.env.BASE_URL ?? "http://127.0.0.1:4173";
const executablePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
  ],
});

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

  await host.goto(baseURL);
  await host.locator("#create-cluster").click();
  await host.locator("#cluster-view").waitFor({ state: "visible" });
  const clusterCode = (await host.locator("#cluster-code-display").textContent())?.trim();
  assert.match(clusterCode ?? "", /^[A-Z2-9]{6}$/);
  assert.equal(await host.locator("#role-display").textContent(), "Host");

  await worker.goto(`${baseURL}/?cluster=${clusterCode}`);
  await worker.locator("#join-cluster").click();
  await worker.locator("#cluster-view").waitFor({ state: "visible" });
  assert.equal(await worker.locator("#role-display").textContent(), "Worker");

  // The host picks the model; the worker is told what to load.
  // The inspector shows one group at a time; the model picker is in "Model &
  // split", which is where a freshly paired host is sent.
  await host.locator('[data-inspect="model"]').click();
  await host.locator("#model-card").waitFor({ state: "visible" });
  assert.ok(
    (await host.locator("#model-select option").count()) > 1,
    "the host must be offered more than one model",
  );
  assert.equal(await worker.locator("#model-card").isVisible(), false);

  try {
    await host.waitForFunction(
      () => document.querySelector("#remote-name")?.textContent === "Phone" ||
        document.querySelector("#remote-name")?.textContent === "Laptop",
      undefined,
      { timeout: 30_000 },
    );
    await worker.waitForFunction(
      () => document.querySelector("#remote-name")?.textContent === "Laptop",
      undefined,
      { timeout: 30_000 },
    );
  } catch (error) {
    const state = {
      host: {
        status: await host.locator("#cluster-status").textContent(),
        joinStatus: await host.locator("#join-status").textContent(),
        remote: await host.locator("#remote-name").textContent(),
        error: await host.locator("#error-banner").textContent(),
      },
      worker: {
        status: await worker.locator("#cluster-status").textContent(),
        joinStatus: await worker.locator("#join-status").textContent(),
        remote: await worker.locator("#remote-name").textContent(),
        error: await worker.locator("#error-banner").textContent(),
      },
      browserErrors: errors,
    };
    throw new Error(`${error.message}\n${JSON.stringify(state, null, 2)}`);
  }

  const hostStatus = await host.locator("#cluster-status").textContent();
  const workerStatus = await worker.locator("#cluster-status").textContent();
  assert.match(hostStatus ?? "", /Worker connected|Confirm the layer split/i);
  assert.match(workerStatus ?? "", /Connected to host|Waiting for layer assignment/i);

  if (errors.length > 0) throw new Error(errors.join("\n"));
  console.log(`cluster smoke passed: ${clusterCode}`);
} finally {
  await browser.close();
}
