/*
 * Adapted from SwarmLLM engine/dense.js.
 * Copyright (c) 2026 Nehanth Narendrula. MIT License.
 */

import { createInitializedBuffer, destroyBuffers, readFloatBuffer } from "./buffers.ts";
import { dequantizeQ8Row } from "./quant.ts";
import type {
  F32Weight,
  ModelRole,
  ModelWeights,
  Qwen3Config,
  SplitEngine,
  Weight,
} from "./types.ts";
import { BASE_WGSL } from "./wgsl/base.ts";
import { cooperativeQ8WGSL } from "./wgsl/cooperative.ts";

type BufferKind = "uniform" | "read-only-storage" | "storage";
type PipelineName =
  | "matvec"
  | "matvec_q8"
  | "matvec_q8_cooperative"
  | "rms_norm"
  | "head_norm"
  | "rope"
  | "attention_scores"
  | "attention_softmax"
  | "attention_output"
  | "silu_multiply"
  | "add_residual";

interface GPUF32Weight {
  kind: "f32";
  data: GPUBuffer;
}

interface GPUQ8Weight {
  kind: "q8";
  quants: GPUBuffer;
  scales: GPUBuffer;
}

type GPUWeight = GPUF32Weight | GPUQ8Weight;

interface MatvecOperation {
  pipeline: PipelineName;
  bindGroup: GPUBindGroup;
  workgroups: number;
}

interface GPULayer {
  inputNorm: GPUF32Weight;
  query: GPUWeight;
  key: GPUWeight;
  value: GPUWeight;
  output: GPUWeight;
  queryNorm: GPUF32Weight;
  keyNorm: GPUF32Weight;
  postAttentionNorm: GPUF32Weight;
  gate: GPUWeight;
  up: GPUWeight;
  down: GPUWeight;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
}

interface LayerBindings {
  inputNorm: GPUBindGroup;
  query: MatvecOperation;
  key: MatvecOperation;
  value: MatvecOperation;
  output: MatvecOperation;
  queryNorm: GPUBindGroup;
  keyNorm: GPUBindGroup;
  scores: GPUBindGroup;
  softmax: GPUBindGroup;
  attentionOutput: GPUBindGroup;
  postAttentionNorm: GPUBindGroup;
  gate: MatvecOperation;
  up: MatvecOperation;
  down: MatvecOperation;
}

const PIPELINE_LAYOUTS: Record<PipelineName, BufferKind[]> = {
  matvec: ["read-only-storage", "read-only-storage", "storage", "uniform"],
  matvec_q8: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"],
  matvec_q8_cooperative: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"],
  rms_norm: ["read-only-storage", "read-only-storage", "storage", "uniform"],
  head_norm: ["storage", "read-only-storage", "uniform"],
  rope: ["storage", "uniform"],
  attention_scores: ["read-only-storage", "read-only-storage", "storage"],
  attention_softmax: ["storage"],
  attention_output: ["read-only-storage", "read-only-storage", "storage"],
  silu_multiply: ["storage", "read-only-storage"],
  add_residual: ["storage", "read-only-storage"],
};

export interface DenseEngineOptions {
  device: GPUDevice;
  config: Qwen3Config;
  weights: ModelWeights;
  role: ModelRole;
  maxSequenceLength?: number;
}

export class DenseQwen3Engine implements SplitEngine {
  readonly config: Qwen3Config;
  readonly role: ModelRole;
  private readonly device: GPUDevice;
  private readonly maxSequenceLength: number;
  private readonly pipelines = new Map<PipelineName, GPUComputePipeline>();
  private readonly commonGroups = new Map<PipelineName, GPUBindGroup>();
  private readonly buffers = new Set<GPUBuffer>();
  private readonly shapeBuffers = new Map<string, GPUBuffer>();
  private readonly layers: GPULayer[] = [];
  private readonly layerBindings: LayerBindings[] = [];
  private readonly embedding: Weight | undefined;
  private currentPosition = 0;

  private configBuffer!: GPUBuffer;
  private frameBuffer!: GPUBuffer;
  private dimensionBuffer!: GPUBuffer;
  private attentionHeadsBuffer!: GPUBuffer;
  private keyValueHeadsBuffer!: GPUBuffer;
  private hidden!: GPUBuffer;
  private normalized!: GPUBuffer;
  private query!: GPUBuffer;
  private key!: GPUBuffer;
  private value!: GPUBuffer;
  private attention!: GPUBuffer;
  private temporary!: GPUBuffer;
  private gate!: GPUBuffer;
  private up!: GPUBuffer;
  private scores!: GPUBuffer;
  private ropeQueryGroup!: GPUBindGroup;
  private ropeKeyGroup!: GPUBindGroup;
  private siluGroup!: GPUBindGroup;
  private residualGroup!: GPUBindGroup;
  private finalNormGroup?: GPUBindGroup;
  private headOperation?: MatvecOperation;
  private logits?: GPUBuffer;

  private constructor(options: DenseEngineOptions) {
    this.device = options.device;
    this.config = options.config;
    this.role = options.role;
    this.embedding = options.weights.embedding;
    this.maxSequenceLength = options.maxSequenceLength ?? Math.min(512, options.config.contextLength);
  }

  static async create(options: DenseEngineOptions): Promise<DenseQwen3Engine> {
    const engine = new DenseQwen3Engine(options);
    await engine.initialize(options.weights);
    return engine;
  }

  get position(): number {
    return this.currentPosition;
  }

  reset(): void {
    this.currentPosition = 0;
  }

  private track(buffer: GPUBuffer): GPUBuffer {
    this.buffers.add(buffer);
    return buffer;
  }

  private initialized(source: ArrayBuffer | ArrayBufferView, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
    return this.track(createInitializedBuffer(this.device, source, usage, label));
  }

  private empty(size: number, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
    return this.track(this.device.createBuffer({ size, usage, label }));
  }

  private async initialize(weights: ModelWeights): Promise<void> {
    const { hiddenSize, attentionHeads, keyValueHeads, headDim, intermediateSize, vocabSize } = this.config;
    const queryDimension = attentionHeads * headDim;
    const keyValueDimension = keyValueHeads * headDim;
    const module = this.device.createShaderModule({
      label: "Qwen3 dense kernels",
      code: BASE_WGSL + cooperativeQ8WGSL(),
    });
    const compilation = await module.getCompilationInfo();
    const shaderErrors = compilation.messages.filter((message) => message.type === "error");
    if (shaderErrors.length) throw new Error(`Qwen3 WGSL compilation failed: ${shaderErrors.map((e) => e.message).join("; ")}`);

    const commonLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    await Promise.all(Object.entries(PIPELINE_LAYOUTS).map(async ([name, kinds]) => {
      const operationLayout = this.device.createBindGroupLayout({
        entries: kinds.map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type },
        })),
      });
      const pipeline = await this.device.createComputePipelineAsync({
        label: name,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [commonLayout, operationLayout] }),
        compute: { module, entryPoint: name },
      });
      this.pipelines.set(name as PipelineName, pipeline);
    }));

    const configData = new ArrayBuffer(48);
    const configUint = new Uint32Array(configData);
    const configFloat = new Float32Array(configData);
    configUint.set([
      hiddenSize,
      keyValueDimension,
      attentionHeads,
      keyValueHeads,
      headDim,
      intermediateSize,
      vocabSize,
      this.maxSequenceLength,
    ]);
    configFloat[8] = this.config.rmsNormEpsilon;
    configFloat[9] = this.config.ropeTheta;
    configUint[10] = queryDimension;
    this.configBuffer = this.initialized(configData, GPUBufferUsage.UNIFORM, "model config");
    this.frameBuffer = this.empty(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "decode frame");
    this.dimensionBuffer = this.initialized(new Uint32Array([hiddenSize]), GPUBufferUsage.UNIFORM);
    this.attentionHeadsBuffer = this.initialized(new Uint32Array([attentionHeads]), GPUBufferUsage.UNIFORM);
    this.keyValueHeadsBuffer = this.initialized(new Uint32Array([keyValueHeads]), GPUBufferUsage.UNIFORM);

    for (const [name, pipeline] of this.pipelines) {
      this.commonGroups.set(name, this.bind(pipeline, 0, [this.configBuffer, this.frameBuffer]));
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.hidden = this.empty(hiddenSize * 4, storage, "hidden");
    this.normalized = this.empty(hiddenSize * 4, storage, "normalized");
    this.query = this.empty(queryDimension * 4, storage, "query");
    this.key = this.empty(keyValueDimension * 4, storage, "key");
    this.value = this.empty(keyValueDimension * 4, storage, "value");
    this.attention = this.empty(queryDimension * 4, storage, "attention output");
    this.temporary = this.empty(hiddenSize * 4, storage, "residual update");
    this.gate = this.empty(intermediateSize * 4, storage, "gate");
    this.up = this.empty(intermediateSize * 4, storage, "up");
    this.scores = this.empty(attentionHeads * this.maxSequenceLength * 4, storage, "attention scores");

    this.ropeQueryGroup = this.bind(this.pipeline("rope"), 1, [this.query, this.attentionHeadsBuffer]);
    this.ropeKeyGroup = this.bind(this.pipeline("rope"), 1, [this.key, this.keyValueHeadsBuffer]);
    this.siluGroup = this.bind(this.pipeline("silu_multiply"), 1, [this.gate, this.up]);
    this.residualGroup = this.bind(this.pipeline("add_residual"), 1, [this.hidden, this.temporary]);

    for (const layerWeights of weights.layers) {
      const layer: GPULayer = {
        inputNorm: this.uploadF32(layerWeights.inputNorm),
        query: this.upload(layerWeights.query),
        key: this.upload(layerWeights.key),
        value: this.upload(layerWeights.value),
        output: this.upload(layerWeights.output),
        queryNorm: this.uploadF32(layerWeights.queryNorm),
        keyNorm: this.uploadF32(layerWeights.keyNorm),
        postAttentionNorm: this.uploadF32(layerWeights.postAttentionNorm),
        gate: this.upload(layerWeights.gate),
        up: this.upload(layerWeights.up),
        down: this.upload(layerWeights.down),
        keyCache: this.empty(this.maxSequenceLength * keyValueDimension * 4, storage),
        valueCache: this.empty(this.maxSequenceLength * keyValueDimension * 4, storage),
      };
      this.layers.push(layer);
      this.layerBindings.push({
        inputNorm: this.normGroup(layer.inputNorm),
        query: this.matvec(layer.query, this.normalized, this.query, queryDimension, hiddenSize),
        key: this.matvec(layer.key, this.normalized, this.key, keyValueDimension, hiddenSize),
        value: this.matvec(layer.value, this.normalized, this.value, keyValueDimension, hiddenSize),
        output: this.matvec(layer.output, this.attention, this.temporary, hiddenSize, queryDimension),
        queryNorm: this.bind(this.pipeline("head_norm"), 1, [this.query, layer.queryNorm.data, this.attentionHeadsBuffer]),
        keyNorm: this.bind(this.pipeline("head_norm"), 1, [this.key, layer.keyNorm.data, this.keyValueHeadsBuffer]),
        scores: this.bind(this.pipeline("attention_scores"), 1, [this.query, layer.keyCache, this.scores]),
        softmax: this.bind(this.pipeline("attention_softmax"), 1, [this.scores]),
        attentionOutput: this.bind(this.pipeline("attention_output"), 1, [this.scores, layer.valueCache, this.attention]),
        postAttentionNorm: this.normGroup(layer.postAttentionNorm),
        gate: this.matvec(layer.gate, this.normalized, this.gate, intermediateSize, hiddenSize),
        up: this.matvec(layer.up, this.normalized, this.up, intermediateSize, hiddenSize),
        down: this.matvec(layer.down, this.gate, this.temporary, hiddenSize, intermediateSize),
      });
    }

    if (this.role.hasHead) {
      if (!weights.finalNorm) throw new Error("host/head role is missing final norm");
      const finalNorm = this.uploadF32(weights.finalNorm);
      this.finalNormGroup = this.normGroup(finalNorm);
      const head = this.upload(weights.head ?? this.requireEmbedding());
      this.logits = this.empty(vocabSize * 4, storage, "logits");
      this.headOperation = this.matvec(head, this.normalized, this.logits, vocabSize, hiddenSize);
    }
  }

  private requireEmbedding(): Weight {
    if (!this.embedding) throw new Error("embedding/head weight is unavailable");
    return this.embedding;
  }

  private upload(weight: Weight): GPUWeight {
    if (weight.kind === "f32") return this.uploadF32(weight);
    return {
      kind: "q8",
      quants: this.initialized(weight.quants, GPUBufferUsage.STORAGE),
      scales: this.initialized(weight.scales, GPUBufferUsage.STORAGE),
    };
  }

  private uploadF32(weight: F32Weight): GPUF32Weight {
    return { kind: "f32", data: this.initialized(weight.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
  }

  private pipeline(name: PipelineName): GPUComputePipeline {
    const pipeline = this.pipelines.get(name);
    if (!pipeline) throw new Error(`pipeline ${name} was not initialized`);
    return pipeline;
  }

  private bind(pipeline: GPUComputePipeline, group: number, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(group),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  private shape(output: number, input: number): GPUBuffer {
    const key = `${output},${input}`;
    let buffer = this.shapeBuffers.get(key);
    if (!buffer) {
      buffer = this.initialized(new Uint32Array([output, input, 0, 0]), GPUBufferUsage.UNIFORM);
      this.shapeBuffers.set(key, buffer);
    }
    return buffer;
  }

  private matvec(weight: GPUWeight, input: GPUBuffer, output: GPUBuffer, rows: number, columns: number): MatvecOperation {
    if (weight.kind === "f32") {
      const pipeline = this.pipeline("matvec");
      return {
        pipeline: "matvec",
        bindGroup: this.bind(pipeline, 1, [weight.data, input, output, this.shape(rows, columns)]),
        workgroups: Math.ceil(rows / 64),
      };
    }
    const cooperative = this.device.limits.maxComputeInvocationsPerWorkgroup >= 256
      && this.device.limits.maxComputeWorkgroupStorageSize >= 4096;
    const name: PipelineName = cooperative ? "matvec_q8_cooperative" : "matvec_q8";
    return {
      pipeline: name,
      bindGroup: this.bind(this.pipeline(name), 1, [weight.quants, weight.scales, input, output, this.shape(rows, columns)]),
      workgroups: cooperative ? Math.ceil(rows / 4) : Math.ceil(rows / 64),
    };
  }

  private normGroup(weight: GPUF32Weight): GPUBindGroup {
    return this.bind(this.pipeline("rms_norm"), 1, [this.hidden, weight.data, this.normalized, this.dimensionBuffer]);
  }

  private setFrame(position: number): void {
    if (!Number.isInteger(position) || position < 0 || position >= this.maxSequenceLength) {
      throw new RangeError(`position ${position} is outside the ${this.maxSequenceLength}-token KV cache`);
    }
    this.device.queue.writeBuffer(this.frameBuffer, 0, new Uint32Array([position, position + 1, 0, 0]));
  }

  private dispatch(pass: GPUComputePassEncoder, name: PipelineName, bindGroup: GPUBindGroup, workgroups: number): void {
    pass.setPipeline(this.pipeline(name));
    pass.setBindGroup(0, this.commonGroups.get(name)!);
    pass.setBindGroup(1, bindGroup);
    pass.dispatchWorkgroups(workgroups);
  }

  private dispatchMatvec(pass: GPUComputePassEncoder, operation: MatvecOperation): void {
    this.dispatch(pass, operation.pipeline, operation.bindGroup, operation.workgroups);
  }

  private encodeLayer(encoder: GPUCommandEncoder, index: number, position: number): void {
    const bindings = this.layerBindings[index]!;
    const layer = this.layers[index]!;
    const { attentionHeads, keyValueHeads, headDim, hiddenSize, intermediateSize } = this.config;
    const queryDimension = attentionHeads * headDim;
    const keyValueDimension = keyValueHeads * headDim;
    {
      const pass = encoder.beginComputePass();
      this.dispatch(pass, "rms_norm", bindings.inputNorm, 1);
      this.dispatchMatvec(pass, bindings.query);
      this.dispatchMatvec(pass, bindings.key);
      this.dispatchMatvec(pass, bindings.value);
      this.dispatch(pass, "head_norm", bindings.queryNorm, Math.ceil(attentionHeads / 32));
      this.dispatch(pass, "head_norm", bindings.keyNorm, Math.ceil(keyValueHeads / 32));
      this.dispatch(pass, "rope", this.ropeQueryGroup, Math.ceil(queryDimension / 2 / 64));
      this.dispatch(pass, "rope", this.ropeKeyGroup, Math.ceil(keyValueDimension / 2 / 64));
      pass.end();
    }
    encoder.copyBufferToBuffer(this.key, 0, layer.keyCache, position * keyValueDimension * 4, keyValueDimension * 4);
    encoder.copyBufferToBuffer(this.value, 0, layer.valueCache, position * keyValueDimension * 4, keyValueDimension * 4);
    {
      const pass = encoder.beginComputePass();
      this.dispatch(pass, "attention_scores", bindings.scores, Math.ceil(attentionHeads * (position + 1) / 64));
      this.dispatch(pass, "attention_softmax", bindings.softmax, attentionHeads);
      this.dispatch(pass, "attention_output", bindings.attentionOutput, Math.ceil(queryDimension / 64));
      this.dispatchMatvec(pass, bindings.output);
      this.dispatch(pass, "add_residual", this.residualGroup, Math.ceil(hiddenSize / 64));
      this.dispatch(pass, "rms_norm", bindings.postAttentionNorm, 1);
      this.dispatchMatvec(pass, bindings.gate);
      this.dispatchMatvec(pass, bindings.up);
      this.dispatch(pass, "silu_multiply", this.siluGroup, Math.ceil(intermediateSize / 64));
      this.dispatchMatvec(pass, bindings.down);
      this.dispatch(pass, "add_residual", this.residualGroup, Math.ceil(hiddenSize / 64));
      pass.end();
    }
  }

  private embeddingRow(tokenId: number): Float32Array {
    const embedding = this.requireEmbedding();
    const [vocab, width] = embedding.shape;
    if (embedding.shape.length !== 2 || vocab === undefined || width !== this.config.hiddenSize) {
      throw new Error("embedding tensor has an unexpected shape");
    }
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocab) throw new RangeError(`token ${tokenId} is out of range`);
    return embedding.kind === "f32"
      ? Float32Array.from(embedding.data.subarray(tokenId * width, (tokenId + 1) * width))
      : dequantizeQ8Row(embedding, tokenId);
  }

  private writeFloats(buffer: GPUBuffer, values: Float32Array): void {
    const copy = new Float32Array(values);
    this.device.queue.writeBuffer(buffer, 0, copy);
  }

  async prefillToken(tokenId: number, position = this.currentPosition): Promise<Float32Array> {
    if (!this.role.hasEmbedding) throw new Error("prefillToken requires the embedding/first-layer role");
    this.setFrame(position);
    this.writeFloats(this.hidden, this.embeddingRow(tokenId));
    const encoder = this.device.createCommandEncoder();
    for (let layer = 0; layer < this.layers.length; layer++) this.encodeLayer(encoder, layer, position);
    this.device.queue.submit([encoder.finish()]);
    const hidden = await readFloatBuffer(this.device, this.hidden, this.config.hiddenSize);
    this.currentPosition = position + 1;
    return hidden;
  }

  async runHidden(hidden: Float32Array, position: number): Promise<Float32Array> {
    if (hidden.length !== this.config.hiddenSize) throw new Error(`hidden state must contain ${this.config.hiddenSize} floats`);
    this.setFrame(position);
    this.writeFloats(this.hidden, hidden);
    const encoder = this.device.createCommandEncoder();
    for (let layer = 0; layer < this.layers.length; layer++) this.encodeLayer(encoder, layer, position);
    this.device.queue.submit([encoder.finish()]);
    const output = await readFloatBuffer(this.device, this.hidden, this.config.hiddenSize);
    this.currentPosition = position + 1;
    return output;
  }

  async headFromHidden(hidden: Float32Array): Promise<Float32Array> {
    if (!this.role.hasHead || !this.finalNormGroup || !this.headOperation || !this.logits) {
      throw new Error("headFromHidden requires the host/head role");
    }
    if (hidden.length !== this.config.hiddenSize) throw new Error(`hidden state must contain ${this.config.hiddenSize} floats`);
    this.writeFloats(this.hidden, hidden);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.dispatch(pass, "rms_norm", this.finalNormGroup, 1);
    this.dispatchMatvec(pass, this.headOperation);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return readFloatBuffer(this.device, this.logits, this.config.vocabSize);
  }

  destroy(): void {
    destroyBuffers(this.buffers);
    this.buffers.clear();
  }
}
