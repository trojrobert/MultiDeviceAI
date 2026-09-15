/*
 * Capacity-aware layer placement.
 *
 * Decides whether a model fits on one device, needs both, or fits neither, and
 * where to cut the layer range. Pure: no DOM, no GPU, no network — everything
 * here runs from a `ModelProfile` and two byte budgets, so it is unit-testable.
 */

import { estimateRoleMemoryBytes, roleDownloadBytes, type ModelProfile } from "./model.ts";
import type { ModelRole } from "./types.ts";

export type PlacementVerdict = "single-device" | "needs-both" | "infeasible";

export interface SplitCandidate {
  /** Host owns layers [0, split); the worker owns [split, layerCount). */
  split: number;
  hostBytes: number;
  workerBytes: number;
  hostDownloadBytes: number;
  workerDownloadBytes: number;
  hostFits: boolean;
  workerFits: boolean;
  /** Smaller of the two sides' remaining headXCLUSTERPLACEHOLDERX, as a fraction of budget. Negative when a side overflows. */
  minSlack: number;
}

export interface PlacementResult {
  verdict: PlacementVerdict;
  /** Device memory the whole model would need on the host alone. */
  singleDeviceBytes: number;
  /** One entry per legal split; `candidates[i]` describes `split === i + 1`. */
  candidates: SplitCandidate[];
  /** Best feasible split, or undefined when the verdict is "infeasible". */
  recommended?: SplitCandidate;
  /** Feasible split whose two downloads are closest to equal. */
  balanced?: SplitCandidate;
  /** How far over budget the best split still is. Set only when infeasible. */
  shortfallBytes?: number;
  shortfallSide?: "host" | "worker";
  /** One sentence for the UI. */
  summary: string;
}

export interface PlacementInput {
  profile: ModelProfile;
  hostBudgetBytes: number;
  workerBudgetBytes: number;
  maxSequenceLength?: number;
}

function hostRole(split: number): ModelRole {
  return { layerRange: [0, split], hasEmbedding: true, hasHead: true };
}

function workerRole(split: number, layerCount: number): ModelRole {
  return { layerRange: [split, layerCount], hasEmbedding: false, hasHead: false };
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

export function planPlacement(input: PlacementInput): PlacementResult {
  const { profile, hostBudgetBytes, workerBudgetBytes } = input;
  const maxSequenceLength = input.maxSequenceLength ?? 256;
  const { layerCount } = profile;

  const wholeModel: ModelRole = { layerRange: [0, layerCount], hasEmbedding: true, hasHead: true };
  const singleDeviceBytes = estimateRoleMemoryBytes(profile, wholeModel, maxSequenceLength);

  const candidates: SplitCandidate[] = [];
  for (let split = 1; split < layerCount; split++) {
    const host = hostRole(split);
    const worker = workerRole(split, layerCount);
    const hostBytes = estimateRoleMemoryBytes(profile, host, maxSequenceLength);
    const workerBytes = estimateRoleMemoryBytes(profile, worker, maxSequenceLength);
    candidates.push({
      split,
      hostBytes,
      workerBytes,
      hostDownloadBytes: roleDownloadBytes(profile, host),
      workerDownloadBytes: roleDownloadBytes(profile, worker),
      hostFits: hostBytes <= hostBudgetBytes,
      workerFits: workerBytes <= workerBudgetBytes,
      minSlack: Math.min(1 - hostBytes / hostBudgetBytes, 1 - workerBytes / workerBudgetBytes),
    });
  }

  const feasible = candidates.filter((candidate) => candidate.hostFits && candidate.workerFits);

  // Maximise the tighter side's headXCLUSTERPLACEHOLDERX rather than pushing one device to its
  // ceiling: these budgets are estimates, and the split that only just fits is
  // exactly where a bad estimate costs a lost WebGPU device mid-load.
  const recommended = feasible.reduce<SplitCandidate | undefined>((best, candidate) => {
    if (!best) return candidate;
    if (candidate.minSlack !== best.minSlack) return candidate.minSlack > best.minSlack ? candidate : best;
    return candidate.split > best.split ? candidate : best;
  }, undefined);

  const balanced = feasible.reduce<SplitCandidate | undefined>((best, candidate) => {
    if (!best) return candidate;
    const delta = Math.abs(candidate.hostDownloadBytes - candidate.workerDownloadBytes);
    const bestDelta = Math.abs(best.hostDownloadBytes - best.workerDownloadBytes);
    return delta < bestDelta ? candidate : best;
  }, undefined);

  if (singleDeviceBytes <= hostBudgetBytes) {
    return {
      verdict: "single-device",
      singleDeviceBytes,
      candidates,
      recommended,
      balanced,
      summary:
        `About ${formatBytes(singleDeviceBytes)} — fits this device alone, so splitting is optional.`,
    };
  }

  if (recommended) {
    return {
      verdict: "needs-both",
      singleDeviceBytes,
      candidates,
      recommended,
      balanced,
      summary:
        `About ${formatBytes(singleDeviceBytes)} — more than either device alone. ` +
        `At ${recommended.split}/${layerCount - recommended.split}: ` +
        `${formatBytes(recommended.hostBytes)} here, ${formatBytes(recommended.workerBytes)} on the peer.`,
    };
  }

  // Nothing fits: report the least-bad split and which side falls short.
  let shortfallBytes = Number.POSITIVE_INFINITY;
  let shortfallSide: "host" | "worker" = "host";
  for (const candidate of candidates) {
    const hostOver = candidate.hostBytes - hostBudgetBytes;
    const workerOver = candidate.workerBytes - workerBudgetBytes;
    const worst = Math.max(hostOver, workerOver);
    if (worst < shortfallBytes) {
      shortfallBytes = worst;
      shortfallSide = hostOver >= workerOver ? "host" : "worker";
    }
  }

  return {
    verdict: "infeasible",
    singleDeviceBytes,
    candidates,
    shortfallBytes,
    shortfallSide,
    summary:
      `About ${formatBytes(singleDeviceBytes)} — even split, the ${shortfallSide} is ` +
      `${formatBytes(shortfallBytes)} over. Raise its budget or pick a smaller model.`,
  };
}

export interface BudgetInput {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  /** `navigator.deviceMemory`, in GiB. Chromium only, coarse, capped at 8. */
  deviceMemoryGiB?: number;
}

export interface BudgetEstimate {
  bytes: number;
  source: "device-memory" | "heuristic";
}

/**
 * A usable-memory estimate, which the browser does not actually expose.
 * `maxBufferSize` is a per-allocation limit, not a capacity, so it is only
 * used as a rough class marker. The UI presents the result as editable.
 */
export function defaultBudgetBytes(input: BudgetInput): BudgetEstimate {
  if (input.deviceMemoryGiB && input.deviceMemoryGiB > 0) {
    // Leave the page, the browser, and the compositor their share.
    return { bytes: Math.floor(input.deviceMemoryGiB * 1024 ** 3 * 0.45), source: "device-memory" };
  }
  const floor = 768 * 1024 ** 2;
  return {
    bytes: Math.max(floor, Math.min(input.maxBufferSize * 4, 2 * 1024 ** 3)),
    source: "heuristic",
  };
}
