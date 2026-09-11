/*
 * Adapted from SwarmLLM engine/wgsl/base.js.
 * Copyright (c) 2026 Nehanth Narendrula. MIT License.
 */

export const BASE_WGSL = /* wgsl */ `
struct Config {
  dim: u32, kvDim: u32, nHeads: u32, nKvHeads: u32,
  headDim: u32, intermediate: u32, vocab: u32, maxSequence: u32,
  epsilon: f32, ropeTheta: f32, queryDim: u32, _padding: u32,
};
struct Frame { position: u32, sequenceLength: u32, _p0: u32, _p1: u32 };
struct Shape { output: u32, input: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<uniform> frame: Frame;

@group(1) @binding(0) var<storage, read> mvWeight: array<f32>;
@group(1) @binding(1) var<storage, read> mvInput: array<f32>;
@group(1) @binding(2) var<storage, read_write> mvOutput: array<f32>;
@group(1) @binding(3) var<uniform> mvShape: Shape;
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= mvShape.output) { return; }
  var sum = 0.0;
  for (var column = 0u; column < mvShape.input; column++) {
    sum += mvWeight[row * mvShape.input + column] * mvInput[column];
  }
  mvOutput[row] = sum;
}

@group(1) @binding(0) var<storage, read> q8Quants: array<u32>;
@group(1) @binding(1) var<storage, read> q8Scales: array<u32>;
@group(1) @binding(2) var<storage, read> q8Input: array<f32>;
@group(1) @binding(3) var<storage, read_write> q8Output: array<f32>;
@group(1) @binding(4) var<uniform> q8Shape: Shape;
fn q8Scale(index: u32) -> f32 {
  return unpack2x16float(q8Scales[index >> 1u])[index & 1u];
}
@compute @workgroup_size(64)
fn matvec_q8(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= q8Shape.output) { return; }
  let blocks = q8Shape.input / 32u;
  let rowWords = row * q8Shape.input / 4u;
  var sum = 0.0;
  for (var block = 0u; block < blocks; block++) {
    var blockSum = 0.0;
    for (var wordIndex = 0u; wordIndex < 8u; wordIndex++) {
      let word = bitcast<i32>(q8Quants[rowWords + block * 8u + wordIndex]);
      let column = block * 32u + wordIndex * 4u;
      blockSum += f32(extractBits(word, 0u, 8u)) * q8Input[column]
        + f32(extractBits(word, 8u, 8u)) * q8Input[column + 1u]
        + f32(extractBits(word, 16u, 8u)) * q8Input[column + 2u]
        + f32(extractBits(word, 24u, 8u)) * q8Input[column + 3u];
    }
    sum += q8Scale(row * blocks + block) * blockSum;
  }
  q8Output[row] = sum;
}

@group(1) @binding(0) var<storage, read> normInput: array<f32>;
@group(1) @binding(1) var<storage, read> normWeight: array<f32>;
@group(1) @binding(2) var<storage, read_write> normOutput: array<f32>;
@group(1) @binding(3) var<uniform> normLength: u32;
var<workgroup> normPartial: array<f32, 256>;
@compute @workgroup_size(256)
fn rms_norm(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var squared = 0.0;
  for (var i = lane; i < normLength; i += 256u) {
    squared += normInput[i] * normInput[i];
  }
  normPartial[lane] = squared;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lane < stride) { normPartial[lane] += normPartial[lane + stride]; }
    workgroupBarrier();
    stride >>= 1u;
  }
  let inverse = inverseSqrt(normPartial[0] / f32(normLength) + config.epsilon);
  for (var i = lane; i < normLength; i += 256u) {
    normOutput[i] = normInput[i] * inverse * normWeight[i];
  }
}

@group(1) @binding(0) var<storage, read_write> headNormVector: array<f32>;
@group(1) @binding(1) var<storage, read> headNormWeight: array<f32>;
@group(1) @binding(2) var<uniform> headNormCount: u32;
@compute @workgroup_size(32)
fn head_norm(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  if (head >= headNormCount) { return; }
  let offset = head * config.headDim;
  var squared = 0.0;
  for (var i = 0u; i < config.headDim; i++) {
    squared += headNormVector[offset + i] * headNormVector[offset + i];
  }
  let inverse = inverseSqrt(squared / f32(config.headDim) + config.epsilon);
  for (var i = 0u; i < config.headDim; i++) {
    headNormVector[offset + i] *= inverse * headNormWeight[i];
  }
}

@group(1) @binding(0) var<storage, read_write> ropeVector: array<f32>;
@group(1) @binding(1) var<uniform> ropeHeads: u32;
@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) id: vec3<u32>) {
  let half = config.headDim / 2u;
  if (id.x >= ropeHeads * half) { return; }
  let head = id.x / half;
  let i = id.x % half;
  let offset = head * config.headDim;
  let angle = f32(frame.position) * pow(config.ropeTheta, -f32(2u * i) / f32(config.headDim));
  let cosine = cos(angle);
  let sine = sin(angle);
  let first = ropeVector[offset + i];
  let second = ropeVector[offset + i + half];
  ropeVector[offset + i] = first * cosine - second * sine;
  ropeVector[offset + i + half] = second * cosine + first * sine;
}

@group(1) @binding(0) var<storage, read> scoreQuery: array<f32>;
@group(1) @binding(1) var<storage, read> scoreKeys: array<f32>;
@group(1) @binding(2) var<storage, read_write> scores: array<f32>;
@compute @workgroup_size(64)
fn attention_scores(@builtin(global_invocation_id) id: vec3<u32>) {
  let total = config.nHeads * frame.sequenceLength;
  if (id.x >= total) { return; }
  let head = id.x / frame.sequenceLength;
  let token = id.x % frame.sequenceLength;
  let kvHead = head / (config.nHeads / config.nKvHeads);
  var sum = 0.0;
  for (var i = 0u; i < config.headDim; i++) {
    sum += scoreQuery[head * config.headDim + i]
      * scoreKeys[token * config.kvDim + kvHead * config.headDim + i];
  }
  scores[head * config.maxSequence + token] = sum / sqrt(f32(config.headDim));
}

@group(1) @binding(0) var<storage, read_write> softmaxScores: array<f32>;
@compute @workgroup_size(1)
fn attention_softmax(@builtin(global_invocation_id) id: vec3<u32>) {
  let head = id.x;
  if (head >= config.nHeads) { return; }
  let offset = head * config.maxSequence;
  var maximum = -3.0e38;
  for (var token = 0u; token < frame.sequenceLength; token++) {
    maximum = max(maximum, softmaxScores[offset + token]);
  }
  var sum = 0.0;
  for (var token = 0u; token < frame.sequenceLength; token++) {
    let value = exp(softmaxScores[offset + token] - maximum);
    softmaxScores[offset + token] = value;
    sum += value;
  }
  for (var token = 0u; token < frame.sequenceLength; token++) {
    softmaxScores[offset + token] /= sum;
  }
}

@group(1) @binding(0) var<storage, read> outputScores: array<f32>;
@group(1) @binding(1) var<storage, read> outputValues: array<f32>;
@group(1) @binding(2) var<storage, read_write> attentionOutput: array<f32>;
@compute @workgroup_size(64)
fn attention_output(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= config.queryDim) { return; }
  let head = id.x / config.headDim;
  let index = id.x % config.headDim;
  let kvHead = head / (config.nHeads / config.nKvHeads);
  var sum = 0.0;
  for (var token = 0u; token < frame.sequenceLength; token++) {
    sum += outputScores[head * config.maxSequence + token]
      * outputValues[token * config.kvDim + kvHead * config.headDim + index];
  }
  attentionOutput[id.x] = sum;
}

@group(1) @binding(0) var<storage, read_write> gate: array<f32>;
@group(1) @binding(1) var<storage, read> up: array<f32>;
@compute @workgroup_size(64)
fn silu_multiply(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= config.intermediate) { return; }
  let value = gate[id.x];
  gate[id.x] = value / (1.0 + exp(-value)) * up[id.x];
}

@group(1) @binding(0) var<storage, read_write> residual: array<f32>;
@group(1) @binding(1) var<storage, read> residualUpdate: array<f32>;
@compute @workgroup_size(64)
fn add_residual(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < config.dim) { residual[id.x] += residualUpdate[id.x]; }
}
`;
