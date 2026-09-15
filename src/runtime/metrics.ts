/*
 * Per-token performance accounting for a split run.
 *
 * The claim this project makes — that a model too large for either device runs
 * across both — is only interesting if you can see what it costs. Every token
 * is attributed to four stages, so the split's real overhead is visible rather
 * than assumed:
 *
 *   host    embedding and the host's own layer range
 *   wire    round trip minus the worker's reported compute: the network's share
 *   worker  the peer's layer range, as timed on the peer itself
 *   head    final norm, language-model head, and sampling
 *
 * `wire` is a residual, not a direct measurement: it is what the round trip
 * cost beyond the peer's own compute, which is exactly the number a reader
 * wants when deciding whether the split is worth it.
 *
 * Pure: no DOM, no GPU, no network, and the clock is injectable, so the whole
 * module is unit-testable.
 */

export type StageName = "host" | "wire" | "worker" | "head";

export const STAGE_NAMES: readonly StageName[] = ["host", "wire", "worker", "head"];

export interface TokenSample {
  /** Prompt tokens are prefill; generated tokens are decode. */
  phase: "prefill" | "decode";
  hostMicros: number;
  wireMicros: number;
  workerMicros: number;
  headMicros: number;
  bytesOut: number;
  bytesIn: number;
}

export interface StageStat {
  micros: number;
  /** Fraction of all measured time, 0–1. */
  share: number;
  meanMicros: number;
}

export interface PerfSummary {
  /** True while a run is in flight. */
  running: boolean;
  promptTokens: number;
  decodeTokens: number;
  /** Wall clock from run start to the first decode token. */
  ttftMicros?: number;
  /** Decode rate over the recent window; 0 until two tokens have landed. */
  tokensPerSecond: number;
  totalMicros: number;
  stages: Record<StageName, StageStat>;
  /** The stage with the largest share, once anything has been measured. */
  dominantStage?: StageName;
  bytesOut: number;
  bytesIn: number;
  /** Activation bytes crossing the link per token, both directions. */
  bytesPerToken: number;
  /** Total latency of each recent token, oldest first — the sparkline's input. */
  recentMicros: readonly number[];
}

export function sampleTotalMicros(sample: TokenSample): number {
  return sample.hostMicros + sample.wireMicros + sample.workerMicros + sample.headMicros;
}

const EMPTY_STAGE: StageStat = { micros: 0, share: 0, meanMicros: 0 };

export function emptySummary(): PerfSummary {
  return {
    running: false,
    promptTokens: 0,
    decodeTokens: 0,
    tokensPerSecond: 0,
    totalMicros: 0,
    stages: { host: EMPTY_STAGE, wire: EMPTY_STAGE, worker: EMPTY_STAGE, head: EMPTY_STAGE },
    bytesOut: 0,
    bytesIn: 0,
    bytesPerToken: 0,
    recentMicros: [],
  };
}

export interface PerfRecorderOptions {
  /** Milliseconds, monotonic. Injected so tests do not depend on real time. */
  now?: () => number;
  /** How many recent tokens the rate and sparkline consider. */
  window?: number;
}

const DEFAULT_WINDOW = 32;

interface StoredSample extends TokenSample {
  /** Milliseconds from the recorder's clock. */
  at: number;
}

/**
 * Accumulates one run's samples and summarises them. A run is one prompt: its
 * prefill tokens and every token generated from it.
 */
export class PerfRecorder {
  private readonly now: () => number;
  private readonly window: number;
  private samples: StoredSample[] = [];
  private startedAt = 0;
  private ttftMicros?: number;
  private running = false;

  constructor(options: PerfRecorderOptions = {}) {
    this.now = options.now ?? (() => performance.now());
    this.window = Math.max(2, options.window ?? DEFAULT_WINDOW);
  }

  /** Discard the previous run and start timing a new one. */
  begin(): void {
    this.samples = [];
    this.startedAt = this.now();
    this.ttftMicros = undefined;
    this.running = true;
  }

  record(sample: TokenSample): void {
    // A sample arriving outside a run still counts: the worker never calls
    // `begin`, because only the host knows where a prompt starts.
    if (!this.running && this.samples.length === 0) this.startedAt = this.now();
    const at = this.now();
    if (this.ttftMicros === undefined && sample.phase === "decode") {
      this.ttftMicros = Math.max(0, Math.round((at - this.startedAt) * 1000));
    }
    this.samples.push({ ...sample, at });
  }

  end(): void {
    this.running = false;
  }

  reset(): void {
    this.samples = [];
    this.ttftMicros = undefined;
    this.running = false;
    this.startedAt = 0;
  }

  get sampleCount(): number {
    return this.samples.length;
  }

  summary(): PerfSummary {
    if (this.samples.length === 0) {
      return { ...emptySummary(), running: this.running };
    }

    const totals: Record<StageName, number> = { host: 0, wire: 0, worker: 0, head: 0 };
    let bytesOut = 0;
    let bytesIn = 0;
    let promptTokens = 0;
    let decodeTokens = 0;

    for (const sample of this.samples) {
      totals.host += sample.hostMicros;
      totals.wire += sample.wireMicros;
      totals.worker += sample.workerMicros;
      totals.head += sample.headMicros;
      bytesOut += sample.bytesOut;
      bytesIn += sample.bytesIn;
      if (sample.phase === "prefill") promptTokens++;
      else decodeTokens++;
    }

    const totalMicros = totals.host + totals.wire + totals.worker + totals.head;
    const count = this.samples.length;
    const stages = {} as Record<StageName, StageStat>;
    for (const name of STAGE_NAMES) {
      stages[name] = {
        micros: totals[name],
        share: totalMicros > 0 ? totals[name] / totalMicros : 0,
        meanMicros: Math.round(totals[name] / count),
      };
    }

    const dominantStage = totalMicros > 0
      ? STAGE_NAMES.reduce((best, name) => (totals[name] > totals[best] ? name : best))
      : undefined;

    const recent = this.samples.slice(-this.window);
    const decodeRecent = recent.filter((sample) => sample.phase === "decode");
    let tokensPerSecond = 0;
    if (decodeRecent.length >= 2) {
      const span = decodeRecent[decodeRecent.length - 1]!.at - decodeRecent[0]!.at;
      if (span > 0) tokensPerSecond = ((decodeRecent.length - 1) / span) * 1000;
    }

    return {
      running: this.running,
      promptTokens,
      decodeTokens,
      ttftMicros: this.ttftMicros,
      tokensPerSecond,
      totalMicros,
      stages,
      dominantStage,
      bytesOut,
      bytesIn,
      bytesPerToken: Math.round((bytesOut + bytesIn) / count),
      recentMicros: recent.map(sampleTotalMicros),
    };
  }
}

export function formatMicros(micros: number): string {
  if (micros <= 0) return "—";
  if (micros < 1000) return `${micros}µs`;
  if (micros < 10_000) return `${(micros / 1000).toFixed(1)}ms`;
  if (micros < 1_000_000) return `${Math.round(micros / 1000)}ms`;
  return `${(micros / 1_000_000).toFixed(micros < 10_000_000 ? 2 : 1)}s`;
}

export function formatCount(bytes: number): string {
  if (bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
