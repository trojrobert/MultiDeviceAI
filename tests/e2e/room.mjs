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
  await host.locator("#create-room").click();
  await host.locator("#room-view").waitFor({ state: "visible" });
  const roomCode = (await host.locator("#room-code-display").textContent())?.trim();
  assert.match(roomCode ?? "", /^[A-Z2-9]{6}$/);
  assert.equal(await host.locator("#role-display").textContent(), "host");

  await worker.goto(`${baseURL}/?room=${roomCode}`);
  await worker.locator("#join-room").click();
  await worker.locator("#room-view").waitFor({ state: "visible" });
  assert.equal(await worker.locator("#role-display").textContent(), "worker");

  try {
    await host.waitForFunction(
      () => document.querySelector("#remote-name")?.textContent === "phone" ||
        document.querySelector("#remote-name")?.textContent === "laptop",
      undefined,
      { timeout: 30_000 },
    );
    await worker.waitForFunction(
      () => document.querySelector("#remote-name")?.textContent === "laptop",
      undefined,
      { timeout: 30_000 },
    );
  } catch (error) {
    const state = {
      host: {
        status: await host.locator("#room-status").textContent(),
        joinStatus: await host.locator("#join-status").textContent(),
        remote: await host.locator("#remote-name").textContent(),
        error: await host.locator("#error-banner").textContent(),
      },
      worker: {
        status: await worker.locator("#room-status").textContent(),
        joinStatus: await worker.locator("#join-status").textContent(),
        remote: await worker.locator("#remote-name").textContent(),
        error: await worker.locator("#error-banner").textContent(),
      },
      browserErrors: errors,
    };
    throw new Error(`${error.message}\n${JSON.stringify(state, null, 2)}`);
  }

  const hostStatus = await host.locator("#room-status").textContent();
  const workerStatus = await worker.locator("#room-status").textContent();
  assert.match(hostStatus ?? "", /Worker connected|Confirm the layer split/i);
  assert.match(workerStatus ?? "", /Connected to host|Waiting for layer assignment/i);

  if (errors.length > 0) throw new Error(errors.join("\n"));
  console.log(`room smoke passed: ${roomCode}`);
} finally {
  await browser.close();
}
