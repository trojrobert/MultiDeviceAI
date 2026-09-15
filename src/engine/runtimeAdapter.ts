/*
 * Runtime adapter connecting LocalClusterAI's cluster contract to the dense Qwen
 * engine.
 */

import { fetchGGUFIndex } from "./gguf.ts";
import {
  MODEL_CATALOGUE,
  profileModel,
  requireModel,
  type ModelEntry,
  type ModelProfile,
} from "./model.ts";
import {
  initWebGPU,
  loadQwen3Engine,
  type GGUFIndex,
  type LoadedQwen3Engine,
  type ModelRole,
} from "./index.ts";
import { greedySample } from "./sampling.ts";
import {
  estimateStorage,
  openWeightCache,
  requestPersistentStorage,
  type WeightCache,
} from "./weightCache.ts";
import { describeAdapter } from "../runtime/capabilities.ts";
import type {
  DistributedEngine,
  EngineCacheReport,
  EngineLoadOptions,
  GenerationOptions,
} from "../runtime/engine.ts";
import { formatCount } from "../runtime/metrics.ts";
import type {
  DeviceCapabilities,
  LayerAssignment,
  TranscriptEntry,
} from "../runtime/protocol.ts";

const MAX_SEQUENCE_LENGTH = 256;
const MAX_NEW_TOKENS = 64;
const SPECIAL_TOKENS = new Set([
  "<|im_start|>",
  "<|im_end|>",
  "<|endoftext|>",
  "<think>",
  "</think>",
]);

export class QwenRuntimeAdapter implements DistributedEngine {
  readonly catalogue: readonly ModelEntry[] = MODEL_CATALOGUE;

  private loaded?: LoadedQwen3Engine;
  private device?: GPUDevice;
  private assignment?: LayerAssignment;
  private profile?: ModelProfile;
  /** GGUF headers are several MiB; keep them so a load never re-fetches one. */
  private readonly indexCache = new Map<string, GGUFIndex>();
  private weightCache?: WeightCache;
  private weightCacheOpened = false;
  private persisted = false;

  get activeProfile(): ModelProfile | undefined {
    return this.profile;
  }

  /**
   * Opened once and reused. A missing cache is normal — an insecure origin or
   * private browsing — and simply means every tensor is fetched.
   */
  private async cache(): Promise<WeightCache | undefined> {
    if (!this.weightCacheOpened) {
      this.weightCacheOpened = true;
      this.weightCache = await openWeightCache();
      // A shard is far too expensive to lose because a background tab needed
      // room, so ask for storage the browser will not evict.
      if (this.weightCache) this.persisted = await requestPersistentStorage();
    }
    return this.weightCache;
  }

  async probeModel(modelId: string): Promise<ModelProfile> {
    const entry = requireModel(modelId);
    let index = this.indexCache.get(entry.url);
    if (!index) {
      // The host needs tokenizer metadata later, and the header is fetched once
      // per session either way, so parse it now rather than twice.
      index = await fetchGGUFIndex(entry.url, { cache: await this.cache() });
      this.indexCache.set(entry.url, index);
    }
    this.profile = profileModel(index, entry);
    return this.profile;
  }

  async cacheReport(): Promise<EngineCacheReport> {
    const cache = await this.cache();
    if (!cache) {
      return {
        available: false,
        persisted: false,
        shard: { entries: 0, bytes: 0 },
        total: { entries: 0, bytes: 0 },
      };
    }
    const url = this.assignment?.modelUrl ?? this.profile?.url;
    return {
      available: true,
      persisted: this.persisted,
      shard: url ? await cache.stats(url) : { entries: 0, bytes: 0 },
      total: await cache.stats(),
      storage: await estimateStorage(),
    };
  }

  async clearCache(): Promise<void> {
    const cache = await this.cache();
    await cache?.clear();
    // The handle is invalid once the underlying store is deleted.
    this.weightCacheOpened = false;
    this.weightCache = undefined;
  }

  async getCapabilities(label: string): Promise<DeviceCapabilities> {
    return describeAdapter(label);
  }

  async load(options: EngineLoadOptions): Promise<void> {
    this.loaded?.engine.destroy();
    this.device?.destroy();
    this.loaded = undefined;
    this.device = undefined;
    this.assignment = options.assignment;

    options.onProgress(0, "Requesting WebGPU device…");
    if (options.signal.aborted) throw abortError();
    const { device } = await initWebGPU();
    this.device = device;

    // The assignment is authoritative about which model to load: a peer on a
    // stale bundle must fail rather than resolve a different file locally.
    const modelUrl = options.assignment.modelUrl;
    const cache = await this.cache();
    let index = this.indexCache.get(modelUrl);
    if (!index) {
      options.onProgress(0, "Reading model header…");
      index = await fetchGGUFIndex(modelUrl, {
        skipTokenizer: !options.assignment.ownsEmbedding,
        cache,
      });
      this.indexCache.set(modelUrl, index);
    }
    if (options.signal.aborted) throw abortError();

    const role: ModelRole = {
      layerRange: [options.assignment.start, options.assignment.end],
      hasEmbedding: options.assignment.ownsEmbedding,
      hasHead: options.assignment.ownsHead,
    };
    let cachedBytes = 0;
    this.loaded = await loadQwen3Engine({
      device,
      modelUrl,
      index,
      role,
      cache,
      maxSequenceLength: MAX_SEQUENCE_LENGTH,
      onProgress: (progress) => {
        if (options.signal.aborted) throw abortError();
        cachedBytes = progress.cachedBytes ?? 0;
        const fraction =
          progress.totalBytes > 0
            ? progress.loadedBytes / progress.totalBytes
            : 0;
        options.onProgress(fraction, loadDetail(progress));
      },
    });
    if (options.signal.aborted) throw abortError();
    options.onProgress(1, readyDetail(cachedBytes, this.loaded.downloadBytes));
  }

  async runHidden(
    hidden: Float32Array,
    position: number,
  ): Promise<Float32Array> {
    return this.requireLoaded().engine.runHidden(hidden, position);
  }

  async generate(
    transcript: readonly TranscriptEntry[],
    options: GenerationOptions,
  ): Promise<void> {
    const loaded = this.requireLoaded();
    const tokenizer = loaded.tokenizer;
    if (!tokenizer || !this.assignment?.ownsEmbedding || !this.assignment.ownsHead) {
      throw new Error("Only the host shard can generate");
    }

    loaded.engine.reset();
    const messages = transcript
      .filter((entry) => !entry.pending)
      .map((entry) => ({
        role: entry.role,
        content: entry.text,
      }));
    let prompt = tokenizer.applyChatTemplate(messages, true);
    // Qwen3 defaults to a visible thinking block. Pre-closing it gives this
    // first POC concise answers via a deterministic cluster path.
    if (
      tokenizer.tokenId("<think>") !== undefined &&
      tokenizer.tokenId("</think>") !== undefined
    ) {
      prompt += "<think>\n\n</think>\n\n";
    }
    const promptIds = tokenizer.encode(prompt, SPECIAL_TOKENS);
    if (promptIds.length === 0) throw new Error("Tokenizer produced an empty prompt");
    if (promptIds.length + MAX_NEW_TOKENS >= MAX_SEQUENCE_LENGTH) {
      throw new Error(
        `Prompt is too long for the ${MAX_SEQUENCE_LENGTH}-token POC context`,
      );
    }

    // One round trip per token, timed in four parts so the panel can show what
    // the split actually costs. `headMicros` is carried into the next token's
    // sample because the head runs on the hidden state the previous round trip
    // returned, and billing it to the token it produced is what a reader
    // expects.
    let position = 0;
    let returned: Float32Array | undefined;
    let pendingHeadMicros = 0;

    const step = async (
      tokenId: number,
      phase: "prefill" | "decode",
    ): Promise<void> => {
      const hostStart = performance.now();
      const local = await loaded.engine.prefillToken(tokenId, position);
      const hostMicros = micros(performance.now() - hostStart);

      const remote = await options.runRemoteHidden(local, position, phase === "prefill");
      returned = remote.values;
      position++;

      options.onStage?.({
        phase,
        hostMicros,
        // The round trip minus the peer's own compute is the network's share.
        // A peer clock that overshoots must not produce negative wire time.
        wireMicros: Math.max(0, remote.roundTripMicros - remote.workerMicros),
        workerMicros: remote.workerMicros,
        headMicros: pendingHeadMicros,
        bytesOut: remote.bytesOut,
        bytesIn: remote.bytesIn,
      });
      pendingHeadMicros = 0;
    };

    for (const tokenId of promptIds) {
      throwIfAborted(options.signal);
      await step(tokenId, "prefill");
    }

    const stopIds = new Set(
      ["<|im_end|>", "<|endoftext|>"]
        .map((token) => tokenizer.tokenId(token))
        .filter((id): id is number => id !== undefined),
    );
    const generatedIds: number[] = [];
    let emittedText = "";
    const limit = Math.min(options.maxNewTokens ?? MAX_NEW_TOKENS, MAX_NEW_TOKENS);
    for (let i = 0; i < limit; i++) {
      throwIfAborted(options.signal);
      if (!returned) throw new Error("Distributed prefill returned no hidden state");
      const headStart = performance.now();
      const logits = await loaded.engine.headFromHidden(returned);
      const tokenId = greedySample(logits);
      pendingHeadMicros = micros(performance.now() - headStart);
      if (stopIds.has(tokenId)) break;

      generatedIds.push(tokenId);
      const decoded = tokenizer.decode(generatedIds);
      const text = decoded.slice(emittedText.length);
      emittedText = decoded;
      if (text) options.onToken(tokenId, text);

      await step(tokenId, "decode");
    }
  }

  reset(): void {
    this.loaded?.engine.reset();
  }

  dispose(): void {
    this.loaded?.engine.destroy();
    this.device?.destroy();
    this.loaded = undefined;
    this.device = undefined;
    this.assignment = undefined;
  }

  private requireLoaded(): LoadedQwen3Engine {
    if (!this.loaded) throw new Error("Model shard is not loaded");
    return this.loaded;
  }
}

export async function createQwenRuntimeEngine(): Promise<DistributedEngine> {
  return new QwenRuntimeAdapter();
}

function shortTensorName(name: string): string {
  const match = name.match(/^blk\.(\d+)\.(.+)$/);
  return match ? `layer ${match[1]} ${match[2]}` : name;
}

function micros(milliseconds: number): number {
  return Math.max(0, Math.round(milliseconds * 1000));
}

/**
 * "Restoring" rather than "Downloading" when a tensor came from the cache: on a
 * second session that is the whole difference, and the number beside the
 * progress bar is otherwise indistinguishable from a very fast network.
 */
function loadDetail(progress: { tensor?: string; fromCache?: boolean }): string {
  if (!progress.tensor) return "Loading model shard…";
  const verb = progress.fromCache ? "Restoring" : "Downloading";
  return `${verb} ${shortTensorName(progress.tensor)}…`;
}

function readyDetail(cachedBytes: number, totalBytes: number): string {
  if (cachedBytes <= 0 || totalBytes <= 0) return "Model shard ready";
  if (cachedBytes >= totalBytes) {
    return `Model shard ready — ${formatCount(cachedBytes)} restored from cache`;
  }
  return `Model shard ready — ${formatCount(cachedBytes)} of ${formatCount(totalBytes)} from cache`;
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

function abortError(): DOMException {
  return new DOMException("Operation aborted", "AbortError");
}
