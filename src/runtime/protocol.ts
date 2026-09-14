/**
 * Room control and activation protocol for the two-peer POC.
 *
 * Version 2 carries the model with the assignment. A v1 peer would ignore
 * those fields and silently load a different model from the host, so the
 * version gate below turns that into a clean failure instead of garbage
 * output.
 *
 * The binary framing is original to MultiDeviceAI, with a separation between
 * control messages and activation frames.
 */

export const PROTOCOL_VERSION = 2 as const;
export const HIDDEN_FRAME_HEADER_BYTES = 24;

const MAGIC = 0x4d444c4d; // "MDLM"

export type RoomRole = "host" | "worker";
export type PeerPhase =
  | "connecting"
  | "connected"
  | "loading"
  | "ready"
  | "generating"
  | "error";

export interface DeviceCapabilities {
  label: string;
  userAgent: string;
  webgpu: boolean;
  gpu?: string;
  /** Largest single allocation. A limit, not a capacity. */
  maxBufferSize: number;
  /** Largest single storage binding; drives language-model head chunking. */
  maxStorageBufferBindingSize: number;
  /** `navigator.deviceMemory` in GiB. Chromium only, coarse, capped at 8. */
  deviceMemoryGiB?: number;
  /** Memory this device is willing to spend on a shard. An estimate the user can correct. */
  budgetBytes: number;
  budgetSource: "device-memory" | "heuristic" | "user";
}

export interface LayerAssignment {
  start: number;
  end: number;
  layerCount: number;
  modelId: string;
  /** Authoritative source for the weights. Sent rather than resolved from the
   *  peer's own catalogue, so a peer on a stale bundle fails instead of
   *  quietly loading a different file. */
  modelUrl: string;
  hiddenSize: number;
  /** Identifies the exact model both peers must agree on; see `modelFingerprint`. */
  fingerprint: number;
  /** Exact download size for this role, so the UI never has to estimate. */
  bytes: number;
  ownsEmbedding: boolean;
  ownsHead: boolean;
}

/**
 * Stable u32 identity for a model, carried on every activation frame so a
 * mismatch is caught at the first token. A width check alone is not enough:
 * two different models can share a hidden size.
 */
export function modelFingerprint(model: {
  modelId: string;
  layerCount: number;
  hiddenSize: number;
}): number {
  const text = `${model.modelId}|${model.layerCount}|${model.hiddenSize}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export interface TranscriptEntry {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  createdAt: number;
  pending?: boolean;
}

interface MessageBase {
  v: typeof PROTOCOL_VERSION;
  type: string;
}

export type ControlMessage =
  | (MessageBase & {
      type: "hello";
      role: RoomRole;
      capabilities: DeviceCapabilities;
    })
  | (MessageBase & { type: "assignment"; assignment: LayerAssignment })
  | (MessageBase & {
      type: "load-progress";
      progress: number;
      detail?: string;
    })
  | (MessageBase & { type: "ready"; assignment: LayerAssignment })
  | (MessageBase & { type: "prompt"; id: string; text: string })
  | (MessageBase & { type: "transcript"; entries: TranscriptEntry[] })
  | (MessageBase & {
      type: "token";
      entryId: string;
      tokenId: number;
      text: string;
      done: boolean;
    })
  | (MessageBase & {
      type: "status";
      phase: PeerPhase;
      detail: string;
    })
  | (MessageBase & { type: "stop" })
  | (MessageBase & { type: "reset" })
  | (MessageBase & { type: "error"; code: string; message: string });

export type ControlMessageInput = ControlMessage extends infer M
  ? M extends MessageBase
    ? Omit<M, "v">
    : never
  : never;

export interface HiddenStateFrame {
  kind: "request" | "response";
  requestId: number;
  position: number;
  /** Model identity, checked by the receiver before it runs any layers. */
  fingerprint: number;
  values: Float32Array;
}

export function control<T extends ControlMessageInput>(
  message: T,
): T & { v: typeof PROTOCOL_VERSION } {
  return { ...message, v: PROTOCOL_VERSION };
}

export function isControlMessage(value: unknown): value is ControlMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Record<string, unknown>;
  return (
    message.v === PROTOCOL_VERSION &&
    typeof message.type === "string" &&
    CONTROL_TYPES.has(message.type as ControlMessage["type"])
  );
}

const CONTROL_TYPES = new Set<ControlMessage["type"]>([
  "hello",
  "assignment",
  "load-progress",
  "ready",
  "prompt",
  "transcript",
  "token",
  "status",
  "stop",
  "reset",
  "error",
]);

export function encodeHiddenFrame(frame: HiddenStateFrame): ArrayBuffer {
  if (!Number.isInteger(frame.requestId) || frame.requestId < 0) {
    throw new RangeError("requestId must be a non-negative integer");
  }
  if (!Number.isInteger(frame.position) || frame.position < 0) {
    throw new RangeError("position must be a non-negative integer");
  }

  const buffer = new ArrayBuffer(
    HIDDEN_FRAME_HEADER_BYTES + frame.values.byteLength,
  );
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, false);
  view.setUint8(4, PROTOCOL_VERSION);
  view.setUint8(5, frame.kind === "request" ? 1 : 2);
  view.setUint16(6, HIDDEN_FRAME_HEADER_BYTES, true);
  view.setUint32(8, frame.requestId, true);
  view.setUint32(12, frame.position, true);
  view.setUint32(16, frame.values.length, true);
  view.setUint32(20, frame.fingerprint >>> 0, true);
  new Uint8Array(buffer, HIDDEN_FRAME_HEADER_BYTES).set(
    new Uint8Array(
      frame.values.buffer,
      frame.values.byteOffset,
      frame.values.byteLength,
    ),
  );
  return buffer;
}

export function decodeHiddenFrame(buffer: ArrayBuffer): HiddenStateFrame {
  if (buffer.byteLength < HIDDEN_FRAME_HEADER_BYTES) {
    throw new Error("Hidden-state frame is shorter than its header");
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== MAGIC) {
    throw new Error("Invalid hidden-state frame magic");
  }
  if (view.getUint8(4) !== PROTOCOL_VERSION) {
    throw new Error(`Unsupported protocol version ${view.getUint8(4)}`);
  }
  const kindByte = view.getUint8(5);
  if (kindByte !== 1 && kindByte !== 2) {
    throw new Error(`Unknown hidden-state frame kind ${kindByte}`);
  }
  const headerBytes = view.getUint16(6, true);
  const count = view.getUint32(16, true);
  if (
    headerBytes !== HIDDEN_FRAME_HEADER_BYTES ||
    buffer.byteLength !== headerBytes + count * Float32Array.BYTES_PER_ELEMENT
  ) {
    throw new Error("Hidden-state frame length does not match its header");
  }
  const bytes = new Uint8Array(buffer, headerBytes, count * 4);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return {
    kind: kindByte === 1 ? "request" : "response",
    requestId: view.getUint32(8, true),
    position: view.getUint32(12, true),
    fingerprint: view.getUint32(20, true),
    values: new Float32Array(copy.buffer),
  };
}
