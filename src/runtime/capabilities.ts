/*
 * Device capability probing, shared by the cluster's fallback path and the real
 * engine adapter so the two cannot report different numbers for one device.
 */

import { defaultBudgetBytes } from "../engine/placement.ts";
import type { DeviceCapabilities } from "./protocol.ts";

/** `navigator.deviceMemory` is Chromium-only and absent from the standard lib types. */
function deviceMemoryGiB(): number | undefined {
  const reported = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof reported === "number" && reported > 0 ? reported : undefined;
}

function withBudget(
  capabilities: Omit<DeviceCapabilities, "budgetBytes" | "budgetSource">,
): DeviceCapabilities {
  const budget = defaultBudgetBytes({
    maxBufferSize: capabilities.maxBufferSize,
    maxStorageBufferBindingSize: capabilities.maxStorageBufferBindingSize,
    deviceMemoryGiB: capabilities.deviceMemoryGiB,
  });
  return { ...capabilities, budgetBytes: budget.bytes, budgetSource: budget.source };
}

/** Capabilities for a device with no usable WebGPU adapter. */
export function unavailableCapabilities(label: string): DeviceCapabilities {
  return withBudget({
    label,
    userAgent: navigator.userAgent,
    webgpu: false,
    maxBufferSize: 0,
    maxStorageBufferBindingSize: 0,
    deviceMemoryGiB: deviceMemoryGiB(),
  });
}

/** Ask the browser for an adapter and describe what it can hold. */
export async function describeAdapter(label: string): Promise<DeviceCapabilities> {
  if (!navigator.gpu) return unavailableCapabilities(label);
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return unavailableCapabilities(label);
    const info = adapter.info;
    return withBudget({
      label,
      userAgent: navigator.userAgent,
      webgpu: true,
      gpu:
        [info?.vendor, info?.architecture, info?.device].filter(Boolean).join(" · ") ||
        "WebGPU adapter",
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      deviceMemoryGiB: deviceMemoryGiB(),
    });
  } catch {
    return unavailableCapabilities(label);
  }
}

/** Apply a user-chosen budget, marking it as deliberate rather than estimated. */
export function withUserBudget(capabilities: DeviceCapabilities, bytes: number): DeviceCapabilities {
  return { ...capabilities, budgetBytes: Math.max(0, Math.round(bytes)), budgetSource: "user" };
}
