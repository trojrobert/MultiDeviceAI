/**
 * Two-peer PeerJS transport.
 *
 * Provides cluster-code addressing and reliable PeerJS connections.
 */
import Peer, {
  type DataConnection,
  type PeerError,
  type PeerOptions,
} from "peerjs";
import type { ControlMessage } from "./protocol.ts";
import {
  decodeHiddenFrame,
  encodeHiddenFrame,
  isControlMessage,
  PROTOCOL_VERSION,
  type HiddenStateFrame,
  type ClusterRole,
} from "./protocol.ts";

const CLUSTER_PREFIX = "lcai-poc-";
const CLUSTER_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

export interface PeerTransportEvents {
  onOpen(peerId: string): void;
  onControl(message: ControlMessage): void;
  /** `byteLength` is the framed size on the wire, which the metrics panel bills. */
  onHidden(frame: HiddenStateFrame, byteLength: number): void;
  onStatus(status: string): void;
  onError(error: Error): void;
  onClose(): void;
}

export interface PeerTransportOptions {
  role: ClusterRole;
  clusterCode: string;
  events: PeerTransportEvents;
  signal?: string;
}

export function normalizeClusterCode(value: string): string {
  return value
    .toUpperCase()
    .replace(/[^A-Z2-9]/g, "")
    .slice(0, 8);
}

export function createClusterCode(length = 6): string {
  const random = crypto.getRandomValues(new Uint8Array(length));
  return Array.from(
    random,
    (byte) => CLUSTER_ALPHABET[byte % CLUSTER_ALPHABET.length],
  ).join("");
}

export class PeerTransport {
  private peer?: Peer;
  private connection?: DataConnection;
  private closed = false;
  private readonly options: PeerTransportOptions;

  constructor(options: PeerTransportOptions) {
    this.options = options;
  }

  async connect(): Promise<void> {
    const clusterCode = normalizeClusterCode(this.options.clusterCode);
    if (!clusterCode) throw new Error("A cluster code is required");
    const hostId = `${CLUSTER_PREFIX}${clusterCode}`;
    const peerOptions = this.peerOptions();
    this.options.events.onStatus("Connecting to PeerJS signaling…");
    this.peer =
      this.options.role === "host"
        ? new Peer(hostId, peerOptions)
        : new Peer(peerOptions);
    this.peer.on("error", (error) => this.handlePeerError(error));
    this.peer.on("disconnected", () => {
      if (!this.closed) this.options.events.onStatus("Signaling disconnected");
    });
    this.peer.on("close", () => this.options.events.onClose());

    await new Promise<void>((resolve, reject) => {
      const onError = (error: PeerError<string>) => reject(error);
      this.peer!.once("error", onError);
      this.peer!.once("open", () => {
        this.peer!.off("error", onError);
        resolve();
      });
    });

    if (this.options.role === "host") {
      this.options.events.onStatus("Cluster open; waiting for worker…");
      this.peer.on("connection", (connection) => {
        if (this.connection?.open) {
          connection.close();
          return;
        }
        void this.acceptConnection(connection);
      });
      return;
    }

    this.options.events.onStatus(`Joining cluster ${clusterCode}…`);
    const connection = this.peer.connect(hostId, {
      reliable: true,
      serialization: "binary",
    });
    await this.acceptConnection(connection);
  }

  sendControl(message: ControlMessage): void {
    this.assertOpen().send(message);
  }

  /** Returns the framed byte count actually sent. */
  sendHidden(frame: HiddenStateFrame): number {
    const encoded = encodeHiddenFrame(frame);
    this.assertOpen().send(encoded);
    return encoded.byteLength;
  }

  close(): void {
    this.closed = true;
    this.connection?.close();
    this.peer?.destroy();
  }

  private async acceptConnection(connection: DataConnection): Promise<void> {
    this.connection = connection;
    connection.on("data", (data) => this.handleData(data));
    connection.on("error", (error) =>
      this.options.events.onError(asError(error)),
    );
    connection.on("close", () => {
      this.connection = undefined;
      this.options.events.onClose();
    });
    if (!connection.open) {
      await new Promise<void>((resolve, reject) => {
        connection.once("open", resolve);
        connection.once("error", reject);
      });
    }
    const channel = connection.dataChannel;
    if (channel) {
      channel.binaryType = "arraybuffer";
      if (!channel.ordered) {
        connection.close();
        throw new Error("The activation data channel must be ordered");
      }
    }
    this.options.events.onStatus("Direct WebRTC channel connected");
    this.options.events.onOpen(connection.peer);
  }

  private handleData(data: unknown): void {
    if (isControlMessage(data)) {
      this.options.events.onControl(data);
      return;
    }
    const buffer = toArrayBuffer(data);
    if (buffer) {
      try {
        this.options.events.onHidden(decodeHiddenFrame(buffer), buffer.byteLength);
      } catch (error) {
        this.options.events.onError(asError(error));
      }
      return;
    }
    // A peer on a cached bundle speaks an older protocol. Say so, rather than
    // reporting a generic unknown payload.
    if (data && typeof data === "object" && typeof (data as { v?: unknown }).v === "number") {
      const version = (data as { v: number }).v;
      if (version !== PROTOCOL_VERSION) {
        this.options.events.onError(
          new Error(
            `Peer is running protocol v${version}; this device speaks v${PROTOCOL_VERSION}. ` +
              "Reload the app on both devices.",
          ),
        );
        return;
      }
    }
    this.options.events.onError(new Error("Received an unknown cluster payload"));
  }

  private assertOpen(): DataConnection {
    if (!this.connection?.open) {
      throw new Error("The worker WebRTC channel is not connected");
    }
    return this.connection;
  }

  private peerOptions(): PeerOptions {
    const signal = this.options.signal?.trim();
    const custom = signal
      ? (() => {
          const url = new URL(
            signal.includes("://") ? signal : `https://${signal}`,
          );
          return {
            host: url.hostname,
            port: url.port ? Number(url.port) : url.protocol === "https:" ? 443 : 80,
            path: url.pathname || "/",
            secure: url.protocol === "https:",
          };
        })()
      : {};
    return {
      debug: 1,
      ...custom,
      config: {
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun1.l.google.com:19302" },
        ],
      },
    };
  }

  private handlePeerError(error: PeerError<string>): void {
    const messages: Record<string, string> = {
      "unavailable-id": "That cluster code is already being hosted",
      "peer-unavailable": "No open host was found for that cluster code",
      network: "PeerJS signaling is unreachable",
      "webrtc": "The direct WebRTC connection failed",
    };
    this.options.events.onError(
      new Error(messages[error.type] ?? error.message ?? error.type),
    );
  }
}

function toArrayBuffer(data: unknown): ArrayBuffer | undefined {
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) {
    const copy = new Uint8Array(data.byteLength);
    copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    return copy.buffer;
  }
  return undefined;
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
