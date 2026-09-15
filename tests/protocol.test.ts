import assert from "node:assert/strict";
import { test } from "node:test";
import {
  control,
  decodeHiddenFrame,
  encodeHiddenFrame,
  HIDDEN_FRAME_HEADER_BYTES,
  isControlMessage,
  modelFingerprint,
  PROTOCOL_VERSION,
  type LayerAssignment,
} from "../src/runtime/protocol.ts";

const FINGERPRINT = modelFingerprint({ modelId: "qwen3-4b-q8_0", layerCount: 36, hiddenSize: 2560 });

test("hidden-state request preserves identifiers and every f32 value", () => {
  const values = new Float32Array([
    0,
    -0,
    1,
    -12.5,
    Math.PI,
    Number.MIN_VALUE,
    Number.MAX_VALUE,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NaN,
  ]);
  const encoded = encodeHiddenFrame({
    kind: "request",
    requestId: 0xfedcba,
    position: 4095,
    fingerprint: FINGERPRINT,
    values,
  });
  assert.equal(encoded.byteLength, HIDDEN_FRAME_HEADER_BYTES + values.byteLength);

  const decoded = decodeHiddenFrame(encoded);
  assert.equal(decoded.kind, "request");
  assert.equal(decoded.requestId, 0xfedcba);
  assert.equal(decoded.position, 4095);
  assert.equal(decoded.fingerprint, FINGERPRINT);
  assert.deepEqual(
    new Uint32Array(decoded.values.buffer),
    new Uint32Array(values.buffer),
  );
});

test("hidden-state response decodes from an independently owned buffer", () => {
  const source = new Float32Array([1.25, -2.5, 3.75]);
  const encoded = encodeHiddenFrame({
    kind: "response",
    requestId: 7,
    position: 11,
    fingerprint: FINGERPRINT,
    values: source.subarray(1),
  });
  source.fill(0);
  const decoded = decodeHiddenFrame(encoded);
  assert.deepEqual(Array.from(decoded.values), [-2.5, 3.75]);
});

test("a response carries the peer's compute time and the request's phase", () => {
  const encoded = encodeHiddenFrame({
    kind: "response",
    requestId: 12,
    position: 3,
    fingerprint: FINGERPRINT,
    computeMicros: 18_430,
    prefill: true,
    values: new Float32Array([1, 2]),
  });
  const decoded = decodeHiddenFrame(encoded);
  assert.equal(decoded.computeMicros, 18_430);
  assert.equal(decoded.prefill, true);

  // Without a reported compute time the host can only see a round trip, so the
  // absent case must read as zero rather than as undefined arithmetic.
  const bare = decodeHiddenFrame(
    encodeHiddenFrame({
      kind: "request",
      requestId: 13,
      position: 4,
      fingerprint: FINGERPRINT,
      values: new Float32Array([1]),
    }),
  );
  assert.equal(bare.computeMicros, 0);
  assert.equal(bare.prefill, false);
});

test("an absurd compute time clamps rather than wrapping around the u32", () => {
  // A wrapped value would report a stalled peer as instantaneous, which is the
  // one reading the panel must never show.
  const decoded = decodeHiddenFrame(
    encodeHiddenFrame({
      kind: "response",
      requestId: 1,
      position: 0,
      fingerprint: FINGERPRINT,
      computeMicros: 8e9,
      values: new Float32Array([1]),
    }),
  );
  assert.equal(decoded.computeMicros, 0xffffffff);

  for (const bogus of [Number.NaN, Number.POSITIVE_INFINITY, -5]) {
    const frame = decodeHiddenFrame(
      encodeHiddenFrame({
        kind: "response",
        requestId: 1,
        position: 0,
        fingerprint: FINGERPRINT,
        computeMicros: bogus,
        values: new Float32Array([1]),
      }),
    );
    assert.equal(frame.computeMicros, 0, `computeMicros ${bogus}`);
  }
});

test("the activation payload stays 8-byte aligned behind the header", () => {
  // f32 views over the frame buffer depend on it, and the header grew in v3.
  assert.equal(HIDDEN_FRAME_HEADER_BYTES % 8, 0);
});

test("hidden-state decoder rejects corrupt and truncated frames", () => {
  assert.throws(() => decodeHiddenFrame(new ArrayBuffer(4)), /shorter/);
  const frame = encodeHiddenFrame({
    kind: "request",
    requestId: 1,
    position: 2,
    fingerprint: 0,
    values: new Float32Array([3]),
  });
  new DataView(frame).setUint32(16, 2, true);
  assert.throws(() => decodeHiddenFrame(frame), /length/);
});

test("versioned control messages are accepted and unknown versions rejected", () => {
  const assignment: LayerAssignment = {
    start: 18,
    end: 36,
    layerCount: 36,
    modelId: "qwen3-4b-q8_0",
    modelUrl: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    hiddenSize: 2560,
    fingerprint: FINGERPRINT,
    bytes: 1_931_000_000,
    ownsEmbedding: false,
    ownsHead: false,
  };
  const message = control({ type: "assignment", assignment });

  assert.equal(message.v, PROTOCOL_VERSION);
  assert.equal(isControlMessage(message), true);
  assert.deepEqual(JSON.parse(JSON.stringify(message)).assignment, assignment);

  // A peer on a previous protocol must be rejected outright: v1 would ignore
  // modelUrl and silently load a different model, and v2 frames are eight bytes
  // short of a v3 header.
  assert.equal(isControlMessage({ ...message, v: 1 }), false);
  assert.equal(isControlMessage({ ...message, v: 2 }), false);
  assert.equal(isControlMessage({ v: PROTOCOL_VERSION, type: "surprise" }), false);
});

test("model fingerprints are stable, distinct, and survive the u32 frame field", () => {
  const model = { modelId: "qwen3-4b-q8_0", layerCount: 36, hiddenSize: 2560 };
  assert.equal(modelFingerprint(model), modelFingerprint({ ...model }));

  const others = [
    { ...model, modelId: "qwen3-1.7b-q8_0" },
    { ...model, layerCount: 28 },
    { ...model, hiddenSize: 1024 },
  ];
  for (const other of others) {
    assert.notEqual(modelFingerprint(other), modelFingerprint(model), JSON.stringify(other));
  }

  for (const candidate of [model, ...others]) {
    const fingerprint = modelFingerprint(candidate);
    assert.ok(Number.isInteger(fingerprint) && fingerprint >= 0 && fingerprint <= 0xffffffff);
    const frame = encodeHiddenFrame({
      kind: "request",
      requestId: 1,
      position: 0,
      fingerprint,
      values: new Float32Array([1]),
    });
    assert.equal(decodeHiddenFrame(frame).fingerprint, fingerprint);
  }
});
