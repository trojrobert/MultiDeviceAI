/*
 * Adapted from SwarmLLM engine/sampling.js.
 * Copyright (c) 2026 Nehanth Narendrula. MIT License.
 */

export function greedySample(logits: ArrayLike<number>): number {
  if (logits.length === 0) throw new Error("cannot sample empty logits");
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i]! > logits[best]!) best = i;
  }
  return best;
}

export function isStopToken(tokenId: number, stopTokenIds: ReadonlySet<number>): boolean {
  return stopTokenIds.has(tokenId);
}

export interface SampleResult {
  tokenId: number;
  stopped: boolean;
}

export function sampleGreedy(
  logits: ArrayLike<number>,
  stopTokenIds: ReadonlySet<number>,
): SampleResult {
  const tokenId = greedySample(logits);
  return { tokenId, stopped: isStopToken(tokenId, stopTokenIds) };
}
