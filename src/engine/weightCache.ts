/*
 * Persistent tensor cache.
 *
 * A shard is 0.4–2.2 GB of range requests, and without this every session
 * re-downloads all of it. Entries hold raw GGUF bytes keyed by
 * (model URL, tensor name, byte range), so the cache is independent of how a
 * tensor is later repacked or dequantized, and a re-published GGUF that moves a
 * tensor misses rather than silently returning the wrong bytes.
 *
 * Cache Storage is used rather than OPFS because it is the one large-binary
 * store available in every browser this app targets — Safari's `createWritable`
 * arrived far later than its `caches` support — and because it stores multi-
 * hundred-megabyte bodies without holding them in memory.
 *
 * Nothing here touches the GPU or the DOM, and the interface is small enough to
 * implement in memory, so the loader's cache behaviour is unit-testable.
 */

/** Bumped whenever the stored byte format changes, so old entries are ignored. */
const CACHE_NAME = "lcai-weights-v1";

/** Keys are synthetic: this origin is never contacted. */
const KEY_ORIGIN = "https://weights.localclusterai.local";

const HEADER_ENTRY = "header";

/** The byte range of one tensor inside a GGUF file. */
export interface CachedRange {
  name: string;
  byteOffset: number;
  byteLength: number;
}

export interface WeightCacheStats {
  entries: number;
  bytes: number;
}

export interface WeightCache {
  /** False once a write has failed, so a full disk stops costing a retry per tensor. */
  readonly writable: boolean;
  get(url: string, range: CachedRange): Promise<Uint8Array | undefined>;
  put(url: string, range: CachedRange, bytes: Uint8Array): Promise<void>;
  /** Parsed GGUF header bytes for a model, if they were stored. */
  getHeader(url: string): Promise<Uint8Array | undefined>;
  putHeader(url: string, bytes: Uint8Array): Promise<void>;
  /** Totals across every model, plus this model alone when a URL is given. */
  stats(url?: string): Promise<WeightCacheStats>;
  clear(): Promise<void>;
}

/** FNV-1a over the model URL: keeps keys short without risking a collision in practice. */
function hashUrl(url: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < url.length; i++) {
    hash ^= url.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36).padStart(7, "0");
}

function modelPrefix(url: string): string {
  return `${KEY_ORIGIN}/${hashUrl(url)}`;
}

/**
 * The byte length lives in the query string so `stats()` can total the cache
 * from `keys()` alone, without reading a single body.
 */
function rangeKey(url: string, range: CachedRange): string {
  return (
    `${modelPrefix(url)}/${encodeURIComponent(range.name)}` +
    `?o=${range.byteOffset}&l=${range.byteLength}`
  );
}

function headerKey(url: string, byteLength: number): string {
  return `${modelPrefix(url)}/${HEADER_ENTRY}?l=${byteLength}`;
}

function keyBytes(key: string): number {
  const length = Number(new URL(key).searchParams.get("l"));
  return Number.isFinite(length) && length > 0 ? length : 0;
}

/**
 * A plain `ArrayBuffer` body, copying only when the view does not already own
 * its whole buffer. The head tensor alone is 371 MiB at 4B, so an unconditional
 * copy here would be a real memory spike.
 */
function bodyOf(bytes: Uint8Array): ArrayBuffer {
  return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? (bytes.buffer as ArrayBuffer)
    : (bytes.slice().buffer as ArrayBuffer);
}

class CacheStorageWeightCache implements WeightCache {
  writable = true;
  /** Spelled out rather than a parameter property: the test runner strips types only. */
  private readonly cache: Cache;

  constructor(cache: Cache) {
    this.cache = cache;
  }

  async get(url: string, range: CachedRange): Promise<Uint8Array | undefined> {
    try {
      const response = await this.cache.match(rangeKey(url, range));
      if (!response) return undefined;
      const bytes = new Uint8Array(await response.arrayBuffer());
      // A truncated body means an interrupted write or an evicted partial
      // entry. Drop it and let the caller re-fetch rather than hand the engine
      // a short tensor.
      if (bytes.byteLength !== range.byteLength) {
        await this.cache.delete(rangeKey(url, range)).catch(() => {});
        return undefined;
      }
      return bytes;
    } catch {
      return undefined;
    }
  }

  async put(url: string, range: CachedRange, bytes: Uint8Array): Promise<void> {
    if (!this.writable || bytes.byteLength !== range.byteLength) return;
    await this.write(rangeKey(url, range), bytes);
  }

  async getHeader(url: string): Promise<Uint8Array | undefined> {
    try {
      // The stored length is unknown up front, so match the path and ignore the
      // query that carries it.
      const response = await this.cache.match(`${modelPrefix(url)}/${HEADER_ENTRY}`, {
        ignoreSearch: true,
      });
      if (!response) return undefined;
      const bytes = new Uint8Array(await response.arrayBuffer());
      return bytes.byteLength > 0 ? bytes : undefined;
    } catch {
      return undefined;
    }
  }

  async putHeader(url: string, bytes: Uint8Array): Promise<void> {
    if (!this.writable || bytes.byteLength === 0) return;
    // A header re-read at a different probe size must not leave two entries.
    await this.cache
      .delete(`${modelPrefix(url)}/${HEADER_ENTRY}`, { ignoreSearch: true })
      .catch(() => {});
    await this.write(headerKey(url, bytes.byteLength), bytes);
  }

  async stats(url?: string): Promise<WeightCacheStats> {
    try {
      const prefix = url ? `${modelPrefix(url)}/` : `${KEY_ORIGIN}/`;
      let entries = 0;
      let bytes = 0;
      for (const request of await this.cache.keys()) {
        if (!request.url.startsWith(prefix)) continue;
        entries++;
        bytes += keyBytes(request.url);
      }
      return { entries, bytes };
    } catch {
      return { entries: 0, bytes: 0 };
    }
  }

  async clear(): Promise<void> {
    try {
      await caches.delete(CACHE_NAME);
      this.writable = true;
    } catch {
      /* nothing usable to clear */
    }
  }

  private async write(key: string, bytes: Uint8Array): Promise<void> {
    try {
      await this.cache.put(
        key,
        new Response(bodyOf(bytes), {
          headers: {
            "content-type": "application/octet-stream",
            "content-length": String(bytes.byteLength),
          },
        }),
      );
    } catch {
      // Almost always a quota failure. One shard is hundreds of writes, so stop
      // after the first rather than pay a rejected promise per tensor; the load
      // itself continues uncached.
      this.writable = false;
    }
  }
}

/** In-memory cache, for tests and for callers that want caching without storage. */
export class MemoryWeightCache implements WeightCache {
  readonly writable = true;
  private readonly entries = new Map<string, Uint8Array>();

  async get(url: string, range: CachedRange): Promise<Uint8Array | undefined> {
    const bytes = this.entries.get(rangeKey(url, range));
    return bytes && bytes.byteLength === range.byteLength ? bytes : undefined;
  }

  async put(url: string, range: CachedRange, bytes: Uint8Array): Promise<void> {
    if (bytes.byteLength !== range.byteLength) return;
    this.entries.set(rangeKey(url, range), bytes.slice());
  }

  async getHeader(url: string): Promise<Uint8Array | undefined> {
    const prefix = `${modelPrefix(url)}/${HEADER_ENTRY}?`;
    for (const [key, bytes] of this.entries) {
      if (key.startsWith(prefix)) return bytes;
    }
    return undefined;
  }

  async putHeader(url: string, bytes: Uint8Array): Promise<void> {
    if (bytes.byteLength === 0) return;
    const prefix = `${modelPrefix(url)}/${HEADER_ENTRY}?`;
    for (const key of [...this.entries.keys()]) {
      if (key.startsWith(prefix)) this.entries.delete(key);
    }
    this.entries.set(headerKey(url, bytes.byteLength), bytes.slice());
  }

  async stats(url?: string): Promise<WeightCacheStats> {
    const prefix = url ? `${modelPrefix(url)}/` : `${KEY_ORIGIN}/`;
    let entries = 0;
    let bytes = 0;
    for (const [key, value] of this.entries) {
      if (!key.startsWith(prefix)) continue;
      entries++;
      bytes += value.byteLength;
    }
    return { entries, bytes };
  }

  async clear(): Promise<void> {
    this.entries.clear();
  }
}

/**
 * Cache Storage is unavailable outside secure contexts and can throw outright
 * in private browsing, so a missing cache is a normal outcome, not an error.
 */
export async function openWeightCache(): Promise<WeightCache | undefined> {
  if (typeof caches === "undefined") return undefined;
  try {
    return new CacheStorageWeightCache(await caches.open(CACHE_NAME));
  } catch {
    return undefined;
  }
}

/**
 * Ask for storage the browser will not evict under pressure. A shard is far too
 * expensive to re-download because a background tab needed room.
 */
export async function requestPersistentStorage(): Promise<boolean> {
  try {
    if (!navigator.storage?.persist) return false;
    if (await navigator.storage.persisted()) return true;
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

export interface StorageEstimate {
  usageBytes: number;
  quotaBytes: number;
}

/** Whole-origin storage figures, for the UI's cache readout. */
export async function estimateStorage(): Promise<StorageEstimate | undefined> {
  try {
    if (!navigator.storage?.estimate) return undefined;
    const estimate = await navigator.storage.estimate();
    return { usageBytes: estimate.usage ?? 0, quotaBytes: estimate.quota ?? 0 };
  } catch {
    return undefined;
  }
}
