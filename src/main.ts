import { mountApp } from "./ui/app.ts";
import "./ui/styles.css";
import { createQwenRuntimeEngine } from "./engine/runtimeAdapter.ts";

window.__MULTIDEVICE_AI_ENGINE_FACTORY__ = createQwenRuntimeEngine;

const root = document.querySelector<HTMLElement>("#app");
if (!root) throw new Error("Missing #app root");

const dispose = mountApp(root);
window.addEventListener("pagehide", dispose, { once: true });
