/**
 * Minimal QR encoder: byte mode, error-correction level L, versions 1–10.
 *
 * Written from scratch to keep MultiDeviceAI dependency-free. Capacity tops
 * out at 271 bytes, which comfortably covers any room invite URL.
 */

interface VersionSpec {
  /** Total codewords (data + error correction). */
  total: number;
  /** Error-correction codewords per block. */
  ecPerBlock: number;
  /** Data codewords for each block. */
  blocks: number[];
}

const VERSIONS: Record<number, VersionSpec> = {
  1: { total: 26, ecPerBlock: 7, blocks: [19] },
  2: { total: 44, ecPerBlock: 10, blocks: [34] },
  3: { total: 70, ecPerBlock: 15, blocks: [55] },
  4: { total: 100, ecPerBlock: 20, blocks: [80] },
  5: { total: 134, ecPerBlock: 26, blocks: [108] },
  6: { total: 172, ecPerBlock: 18, blocks: [68, 68] },
  7: { total: 196, ecPerBlock: 20, blocks: [78, 78] },
  8: { total: 242, ecPerBlock: 24, blocks: [97, 97] },
  9: { total: 292, ecPerBlock: 30, blocks: [116, 116] },
  10: { total: 346, ecPerBlock: 18, blocks: [68, 68, 69, 69] },
};

const ALIGNMENT: Record<number, number[]> = {
  1: [],
  2: [6, 18],
  3: [6, 22],
  4: [6, 26],
  5: [6, 30],
  6: [6, 34],
  7: [6, 22, 38],
  8: [6, 24, 42],
  9: [6, 26, 46],
  10: [6, 28, 50],
};

/** Format-info bits for error-correction level L. */
const EC_LEVEL_L = 1;

// ── GF(256) arithmetic over the QR primitive polynomial 0x11d ──────────────
const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
{
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
}

function mul(a: number, b: number): number {
  return a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]];
}

function generatorPoly(degree: number): number[] {
  let poly = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array<number>(poly.length + 1).fill(0);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= poly[j];
      next[j + 1] ^= mul(poly[j], EXP[i]);
    }
    poly = next;
  }
  return poly;
}

function errorCorrection(data: number[], ecLen: number): number[] {
  const gen = generatorPoly(ecLen);
  const rem = new Array<number>(ecLen).fill(0);
  for (const byte of data) {
    const factor = byte ^ rem[0];
    rem.shift();
    rem.push(0);
    for (let i = 0; i < ecLen; i++) rem[i] ^= mul(gen[i + 1], factor);
  }
  return rem;
}

/** BCH(15,5) format information, already XOR-masked per the spec. */
export function formatInfo(ecLevel: number, mask: number): number {
  const data = (ecLevel << 3) | mask;
  let rem = data << 10;
  for (let i = 14; i >= 10; i--) {
    if ((rem >> i) & 1) rem ^= 0x537 << (i - 10);
  }
  return ((data << 10) | (rem & 0x3ff)) ^ 0x5412;
}

/** BCH(18,6) version information, used for versions 7 and above. */
export function versionInfo(version: number): number {
  let rem = version << 12;
  for (let i = 17; i >= 12; i--) {
    if ((rem >> i) & 1) rem ^= 0x1f25 << (i - 12);
  }
  return (version << 12) | (rem & 0xfff);
}

function maskBit(mask: number, r: number, c: number): boolean {
  switch (mask) {
    case 0: return (r + c) % 2 === 0;
    case 1: return r % 2 === 0;
    case 2: return c % 3 === 0;
    case 3: return (r + c) % 3 === 0;
    case 4: return (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0;
    case 5: return ((r * c) % 2) + ((r * c) % 3) === 0;
    case 6: return (((r * c) % 2) + ((r * c) % 3)) % 2 === 0;
    default: return (((r + c) % 2) + ((r * c) % 3)) % 2 === 0;
  }
}

export interface QRCode {
  version: number;
  size: number;
  /** Row-major modules; 1 is dark. */
  modules: Uint8Array;
}

function byteCapacity(version: number): number {
  const spec = VERSIONS[version];
  const dataCodewords = spec.blocks.reduce((sum, n) => sum + n, 0);
  const headerBits = 4 + (version < 10 ? 8 : 16);
  return Math.floor((dataCodewords * 8 - headerBits) / 8);
}

function chooseVersion(length: number): number {
  for (let version = 1; version <= 10; version++) {
    if (length <= byteCapacity(version)) return version;
  }
  throw new Error(`${length} bytes exceeds the supported QR capacity`);
}

function buildCodewords(bytes: Uint8Array, version: number): number[] {
  const spec = VERSIONS[version];
  const dataCodewords = spec.blocks.reduce((sum, n) => sum + n, 0);

  // Bit stream: mode indicator, character count, payload, terminator.
  const bits: number[] = [];
  const push = (value: number, width: number) => {
    for (let i = width - 1; i >= 0; i--) bits.push((value >> i) & 1);
  };
  push(0b0100, 4);
  push(bytes.length, version < 10 ? 8 : 16);
  for (const byte of bytes) push(byte, 8);

  const capacityBits = dataCodewords * 8;
  push(0, Math.min(4, capacityBits - bits.length));
  while (bits.length % 8 !== 0) bits.push(0);

  const data: number[] = [];
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0;
    for (let j = 0; j < 8; j++) byte = (byte << 1) | bits[i + j];
    data.push(byte);
  }
  const padBytes = [0xec, 0x11];
  while (data.length < dataCodewords) {
    data.push(padBytes[(data.length - bits.length / 8) % 2]);
  }

  // Split into blocks, then interleave data and error-correction codewords.
  const dataBlocks: number[][] = [];
  const ecBlocks: number[][] = [];
  let offset = 0;
  for (const blockLength of spec.blocks) {
    const block = data.slice(offset, offset + blockLength);
    offset += blockLength;
    dataBlocks.push(block);
    ecBlocks.push(errorCorrection(block, spec.ecPerBlock));
  }

  const result: number[] = [];
  const longestBlock = Math.max(...spec.blocks);
  for (let i = 0; i < longestBlock; i++) {
    for (const block of dataBlocks) {
      if (i < block.length) result.push(block[i]);
    }
  }
  for (let i = 0; i < spec.ecPerBlock; i++) {
    for (const block of ecBlocks) result.push(block[i]);
  }
  return result;
}

function penalty(modules: Uint8Array, size: number): number {
  const at = (r: number, c: number) => modules[r * size + c];
  let score = 0;

  // Rule 1: runs of five or more same-coloured modules.
  for (let i = 0; i < size; i++) {
    for (const horizontal of [true, false]) {
      let run = 1;
      for (let j = 1; j < size; j++) {
        const prev = horizontal ? at(i, j - 1) : at(j - 1, i);
        const cur = horizontal ? at(i, j) : at(j, i);
        if (cur === prev) {
          run++;
        } else {
          if (run >= 5) score += 3 + (run - 5);
          run = 1;
        }
      }
      if (run >= 5) score += 3 + (run - 5);
    }
  }

  // Rule 2: 2x2 blocks of the same colour.
  for (let r = 0; r < size - 1; r++) {
    for (let c = 0; c < size - 1; c++) {
      const v = at(r, c);
      if (v === at(r, c + 1) && v === at(r + 1, c) && v === at(r + 1, c + 1)) {
        score += 3;
      }
    }
  }

  // Rule 3: finder-like 1:1:3:1:1 patterns with four light modules alongside.
  const target = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0];
  const reversed = [...target].reverse();
  const matches = (get: (k: number) => number, start: number, pattern: number[]) => {
    for (let k = 0; k < pattern.length; k++) {
      if (get(start + k) !== pattern[k]) return false;
    }
    return true;
  };
  for (let i = 0; i < size; i++) {
    for (let j = 0; j + target.length <= size; j++) {
      const row = (k: number) => at(i, k);
      const col = (k: number) => at(k, i);
      if (matches(row, j, target) || matches(row, j, reversed)) score += 40;
      if (matches(col, j, target) || matches(col, j, reversed)) score += 40;
    }
  }

  // Rule 4: deviation from an even balance of dark and light modules.
  let dark = 0;
  for (let i = 0; i < modules.length; i++) dark += modules[i];
  const percent = (dark * 100) / (size * size);
  score += Math.floor(Math.abs(percent - 50) / 5) * 10;

  return score;
}

export function encodeQR(text: string): QRCode {
  const bytes = new TextEncoder().encode(text);
  const version = chooseVersion(bytes.length);
  const codewords = buildCodewords(bytes, version);
  const size = version * 4 + 17;

  const base = new Uint8Array(size * size);
  const reserved = new Uint8Array(size * size);
  const setFunction = (r: number, c: number, value: number) => {
    if (r < 0 || r >= size || c < 0 || c >= size) return;
    base[r * size + c] = value;
    reserved[r * size + c] = 1;
  };

  // Finder patterns with their separators.
  for (const [r0, c0] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let dr = -1; dr <= 7; dr++) {
      for (let dc = -1; dc <= 7; dc++) {
        let dark = 0;
        if (dr >= 0 && dr <= 6 && dc >= 0 && dc <= 6) {
          const ring = Math.max(Math.abs(dr - 3), Math.abs(dc - 3));
          dark = ring === 3 || ring <= 1 ? 1 : 0;
        }
        setFunction(r0 + dr, c0 + dc, dark);
      }
    }
  }

  // Timing patterns.
  for (let i = 8; i < size - 8; i++) {
    const value = i % 2 === 0 ? 1 : 0;
    setFunction(6, i, value);
    setFunction(i, 6, value);
  }

  // Alignment patterns, skipping the finder corners.
  const centers = ALIGNMENT[version];
  for (const r of centers) {
    for (const c of centers) {
      const nearFinder =
        (r <= 8 && c <= 8) ||
        (r <= 8 && c >= size - 9) ||
        (r >= size - 9 && c <= 8);
      if (nearFinder) continue;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const ring = Math.max(Math.abs(dr), Math.abs(dc));
          setFunction(r + dr, c + dc, ring === 1 ? 0 : 1);
        }
      }
    }
  }

  // Reserve the format and version areas, then place the always-dark module.
  for (let i = 0; i <= 8; i++) {
    if (i === 6) continue; // row and column 6 belong to the timing patterns
    setFunction(8, i, 0);
    setFunction(i, 8, 0);
  }
  for (let i = 0; i < 8; i++) {
    setFunction(8, size - 1 - i, 0);
    setFunction(size - 1 - i, 8, 0);
  }
  setFunction(size - 8, 8, 1);
  if (version >= 7) {
    for (let i = 0; i < 18; i++) {
      const a = size - 11 + (i % 3);
      const b = Math.floor(i / 3);
      setFunction(a, b, 0);
      setFunction(b, a, 0);
    }
  }

  // Zigzag data placement, two columns at a time, right to left.
  const totalBits = codewords.length * 8;
  let bitIndex = 0;
  let upward = true;
  for (let col = size - 1; col >= 0; col -= 2) {
    if (col === 6) col--;
    for (let i = 0; i < size; i++) {
      const row = upward ? size - 1 - i : i;
      for (const c of [col, col - 1]) {
        if (c < 0 || reserved[row * size + c]) continue;
        let bit = 0;
        if (bitIndex < totalBits) {
          bit = (codewords[bitIndex >> 3] >> (7 - (bitIndex & 7))) & 1;
        }
        base[row * size + c] = bit;
        bitIndex++;
      }
    }
    upward = !upward;
  }

  // Try every mask and keep the lowest-penalty result.
  let best: Uint8Array | undefined;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let mask = 0; mask < 8; mask++) {
    const candidate = Uint8Array.from(base);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (reserved[r * size + c]) continue;
        if (maskBit(mask, r, c)) candidate[r * size + c] ^= 1;
      }
    }

    const format = formatInfo(EC_LEVEL_L, mask);
    const bit = (i: number) => (format >> i) & 1;
    const put = (r: number, c: number, value: number) => {
      candidate[r * size + c] = value;
    };
    for (let i = 0; i <= 5; i++) put(i, 8, bit(i));
    put(7, 8, bit(6));
    put(8, 8, bit(7));
    put(8, 7, bit(8));
    for (let i = 9; i < 15; i++) put(8, 14 - i, bit(i));
    for (let i = 0; i < 8; i++) put(8, size - 1 - i, bit(i));
    for (let i = 8; i < 15; i++) put(size - 15 + i, 8, bit(i));

    if (version >= 7) {
      const info = versionInfo(version);
      for (let i = 0; i < 18; i++) {
        const value = (info >> i) & 1;
        const a = size - 11 + (i % 3);
        const b = Math.floor(i / 3);
        put(a, b, value);
        put(b, a, value);
      }
    }

    const score = penalty(candidate, size);
    if (score < bestScore) {
      bestScore = score;
      best = candidate;
    }
  }

  return { version, size, modules: best ?? base };
}

/** Render a QR code as a self-contained, theme-agnostic SVG string. */
export function qrToSvg(code: QRCode, margin = 2): string {
  const span = code.size + margin * 2;
  let path = "";
  for (let r = 0; r < code.size; r++) {
    for (let c = 0; c < code.size; c++) {
      if (code.modules[r * code.size + c]) {
        path += `M${c + margin} ${r + margin}h1v1h-1z`;
      }
    }
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${span} ${span}" ` +
    `shape-rendering="crispEdges" role="img" aria-label="Room invite QR code">` +
    `<rect width="${span}" height="${span}" fill="#fff"/>` +
    `<path d="${path}" fill="#000"/>` +
    `</svg>`
  );
}
