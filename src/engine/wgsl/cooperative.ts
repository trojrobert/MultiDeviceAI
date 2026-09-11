/*
 * Cooperative Q8 matmul WGSL kernel.
 */

export function cooperativeQ8WGSL(workgroupSize = 256, rowsPerGroup = 4): string {
  if (workgroupSize % 4 !== 0 || rowsPerGroup < 1) {
    throw new Error("cooperative Q8 kernel requires a workgroup divisible by four");
  }
  const lanes = workgroupSize / 4;
  const accumulators = Array.from({ length: rowsPerGroup }, (_, row) => `var a${row} = 0.0;`).join("\n  ");
  const accumulate = Array.from(
    { length: rowsPerGroup },
    (_, row) => `if (full || row0 + ${row}u < q8Shape.output) {
      a${row} += coop_q8_row(
        wordBase + ${row}u * rowWords,
        q8Scale(scaleBase + ${row}u * blocks),
        first,
        second
      );
    }`,
  ).join("\n    ");
  const store = Array.from(
    { length: rowsPerGroup },
    (_, row) => `coopPartial[${row * workgroupSize}u + lane] = a${row};`,
  ).join("\n  ");
  const reduce = Array.from(
    { length: rowsPerGroup },
    (_, row) => `coopPartial[${row * workgroupSize}u + lane] += coopPartial[${row * workgroupSize}u + lane + stride];`,
  ).join("\n      ");

  return /* wgsl */ `
@group(1) @binding(2) var<storage, read> coopQ8Input: array<vec4<f32>>;
var<workgroup> coopPartial: array<f32, ${workgroupSize * rowsPerGroup}>;

fn coop_q8_row(wordBase: u32, scale: f32, first: vec4<f32>, second: vec4<f32>) -> f32 {
  let word0 = bitcast<i32>(q8Quants[wordBase]);
  let word1 = bitcast<i32>(q8Quants[wordBase + 1u]);
  let quant0 = vec4<f32>(
    f32((word0 << 24u) >> 24u), f32((word0 << 16u) >> 24u),
    f32((word0 << 8u) >> 24u), f32(word0 >> 24u)
  );
  let quant1 = vec4<f32>(
    f32((word1 << 24u) >> 24u), f32((word1 << 16u) >> 24u),
    f32((word1 << 8u) >> 24u), f32(word1 >> 24u)
  );
  return scale * (dot(quant0, first) + dot(quant1, second));
}

@compute @workgroup_size(${workgroupSize})
fn matvec_q8_cooperative(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>
) {
  let lane = local.x;
  let quarter = lane & 3u;
  let blockLane = lane >> 2u;
  let blocks = q8Shape.input / 32u;
  let rowWords = q8Shape.input / 4u;
  let row0 = group.x * ${rowsPerGroup}u;
  let full = row0 + ${rowsPerGroup - 1}u < q8Shape.output;
  ${accumulators}
  for (var block = blockLane; block < blocks; block += ${lanes}u) {
    let vectorIndex = block * 8u + quarter * 2u;
    let first = coopQ8Input[vectorIndex];
    let second = coopQ8Input[vectorIndex + 1u];
    let wordBase = row0 * rowWords + block * 8u + quarter * 2u;
    let scaleBase = row0 * blocks + block;
    ${accumulate}
  }
  ${store}
  workgroupBarrier();
  var stride = ${workgroupSize / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      ${reduce}
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  if (lane < ${rowsPerGroup}u) {
    let row = row0 + lane;
    if (row < q8Shape.output) {
      q8Output[row] = coopPartial[lane * ${workgroupSize}u];
    }
  }
}
`;
}
