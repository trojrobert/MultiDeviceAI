import assert from "node:assert/strict";
import { test } from "node:test";
import {
  control,
  decodeHiddenFrame,
  encodeHiddenFrame,
  HIDDEN_FRAME_HEADER_BYTES,
  isControlMessage,
} from "../src/runtime/protocol.ts";

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
    values,
  });
  assert.equal(encoded.byteLength, HIDDEN_FRAME_HEADER_BYTES + values.byteLength);

  const decoded = decodeHiddenFrame(encoded);
  assert.equal(decoded.kind, "request");
  assert.equal(decoded.requestId, 0xfedcba);
  assert.equal(decoded.position, 4095);
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
    values: source.subarray(1),
  });
  source.fill(0);
  const decoded = decodeHiddenFrame(encoded);
  assert.deepEqual(Array.from(decoded.values), [-2.5, 3.75]);
});

test("hidden-state decoder rejects corrupt and truncated frames", () => {
  assert.throws(() => decodeHiddenFrame(new ArrayBuffer(4)), /shorter/);
  const frame = encodeHiddenFrame({
    kind: "request",
    requestId: 1,
    position: 2,
    values: new Float32Array([3]),
  });
  new DataView(frame).setUint32(16, 2, true);
  assert.throws(() => decodeHiddenFrame(frame), /length/);
});

test("versioned control messages are accepted and unknown versions rejected", () => {
  const message = control({
    type: "assignment",
    assignment: {
      start: 14,
      end: 28,
      layerCount: 28,
      modelId: "qwen3-0.6b-q8_0",
      ownsEmbedding: false,
      ownsHead: false,
    },
  });
  assert.equal(message.v, 1);
  assert.equal(isControlMessage(message), true);
  assert.equal(isControlMessage({ ...message, v: 2 }), false);
  assert.equal(isControlMessage({ v: 1, type: "surprise" }), false);
});
