/*
 * Runtime adapter connecting MultiDeviceAI's room contract to the dense Qwen
 * engine.
 */

import {
  initWebGPU,
  loadQwen3Engine,
  QWEN3_06B_LAYER_COUNT,
  type LoadedQwen3Engine,
  type ModelRole,
} from "./index.ts";
import { greedySample } from "./sampling.ts";
import type {
  DistributedEngine,
  EngineLoadOptions,
  GenerationOptions,
} from "../runtime/engine.ts";
import type {
  DeviceCapabilities,
  LayerAssignment,
  TranscriptEntry,
} from "../runtime/protocol.ts";

const MODEL_ID = "qwen3-0.6b-q8_0";
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
  readonly layerCount = QWEN3_06B_LAYER_COUNT;
  readonly modelId = MODEL_ID;

  private loaded?: LoadedQwen3Engine;
  private device?: GPUDevice;
  private assignment?: LayerAssignment;

  async getCapabilities(label: string): Promise<DeviceCapabilities> {
    if (!navigator.gpu) {
      return {
        label,
        userAgent: navigator.userAgent,
        webgpu: false,
        maxBufferSize: 0,
      };
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      return {
        label,
        userAgent: navigator.userAgent,
        webgpu: false,
        maxBufferSize: 0,
      };
    }
    const info = adapter.info;
    return {
      label,
      userAgent: navigator.userAgent,
      webgpu: true,
      gpu:
        [info.vendor, info.architecture, info.device]
          .filter(Boolean)
          .join(" · ") || "WebGPU adapter",
      maxBufferSize: adapter.limits.maxBufferSize,
      estimatedMemoryBytes: adapter.limits.maxBufferSize,
    };
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

    const role: ModelRole = {
      layerRange: [options.assignment.start, options.assignment.end],
      hasEmbedding: options.assignment.ownsEmbedding,
      hasHead: options.assignment.ownsHead,
    };
    this.loaded = await loadQwen3Engine({
      device,
      role,
      maxSequenceLength: MAX_SEQUENCE_LENGTH,
      onProgress: (progress) => {
        if (options.signal.aborted) throw abortError();
        const fraction =
          progress.totalBytes > 0
            ? progress.loadedBytes / progress.totalBytes
            : 0;
        options.onProgress(
          fraction,
          progress.tensor
            ? `Downloading ${shortTensorName(progress.tensor)}…`
            : "Downloading model shard…",
        );
      },
    });
    if (options.signal.aborted) throw abortError();
    options.onProgress(1, "Model shard ready");
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
    // first POC concise answers via a deterministic room path.
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

    let position = 0;
    let returned: Float32Array | undefined;
    for (const tokenId of promptIds) {
      throwIfAborted(options.signal);
      const local = await loaded.engine.prefillToken(tokenId, position);
      returned = await options.runRemoteHidden(local, position);
      position++;
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
      const logits = await loaded.engine.headFromHidden(returned);
      const tokenId = greedySample(logits);
      if (stopIds.has(tokenId)) break;

      generatedIds.push(tokenId);
      const decoded = tokenizer.decode(generatedIds);
      const text = decoded.slice(emittedText.length);
      emittedText = decoded;
      if (text) options.onToken(tokenId, text);

      const local = await loaded.engine.prefillToken(tokenId, position);
      returned = await options.runRemoteHidden(local, position);
      position++;
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

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

function abortError(): DOMException {
  return new DOMException("Operation aborted", "AbortError");
}
