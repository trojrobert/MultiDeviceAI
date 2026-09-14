// WebGPU device initialization.
//
// Works in any environment that exposes the WebGPU API on `navigator.gpu`:
// browsers (Chrome/Edge/Firefox), and Deno (`--unstable-webgpu`). Node has no
// built-in WebGPU, so `isWebGPUAvailable()` returns false there and callers can
// skip GPU work gracefully.

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Human-readable adapter description, useful for the demo UI and logs. */
  info: GPUAdapterInfo;
  capabilities: GPUCapabilities;
}

export interface GPUCapabilities {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
}

export class WebGPUInitializationError extends Error {
  constructor(
    message: string,
    readonly causeCode: "unavailable" | "adapter" | "limits" | "device",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "WebGPUInitializationError";
  }
}

/** True when a WebGPU entry point is present in this JS runtime. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
}

/**
 * Request an adapter + device. Throws a descriptive error rather than returning
 * null so failures surface loudly during bring-up.
 */
export async function initWebGPU(
  options: GPURequestAdapterOptions = { powerPreference: "high-performance" },
): Promise<GPUContext> {
  if (!isWebGPUAvailable()) {
    // navigator.gpu is only exposed in a secure context. The most common cause
    // in this app is opening it on a second device over a plain http:// LAN
    // address, where WebGPU is silently withheld — so call that out explicitly.
    const insecureContext = globalThis.isSecureContext === false;
    throw new WebGPUInitializationError(
      insecureContext
        ? "WebGPU is unavailable because this page is not a secure context. " +
            "navigator.gpu is only exposed over HTTPS or on localhost — loading the " +
            "app from a plain http:// LAN address (e.g. http://192.168.x.x:5173) will " +
            "not work. Serve it over HTTPS: deploy it, or run the dev server with TLS."
        : "WebGPU is not available in this runtime (navigator.gpu is undefined). " +
            "Use a WebGPU-capable browser (Chrome/Edge 121+ or Safari 18+), or run " +
            "under Deno with --unstable-webgpu.",
      "unavailable",
    );
  }

  const adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) {
    throw new WebGPUInitializationError(
      "requestAdapter() returned null: no compatible GPU adapter was found.",
      "adapter",
    );
  }

  const requiredLimits: Record<string, number> = {};
  for (const key of ["maxBufferSize", "maxStorageBufferBindingSize"] as const) {
    const supported = adapter.limits[key];
    if (supported >= 256 * 1024 * 1024) requiredLimits[key] = supported;
  }
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredLimits });
  } catch (cause) {
    throw new WebGPUInitializationError(
      "Unable to create a WebGPU device with model-sized buffer limits.",
      "limits",
      { cause },
    );
  }

  // Surface device loss (OOM, driver reset) instead of silently hanging.
  device.lost.then((reason) => {
    console.error(`WebGPU device lost: ${reason.reason} — ${reason.message}`);
  });

  device.addEventListener("uncapturederror", (event) => {
    console.error(`WebGPU validation error: ${event.error.message}`);
  });

  return {
    adapter,
    device,
    info: adapter.info,
    capabilities: {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    },
  };
}
