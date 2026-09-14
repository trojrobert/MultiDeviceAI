import assert from "node:assert/strict";
import { test } from "node:test";
import { encodeQR, formatInfo, qrToSvg, versionInfo } from "../src/ui/qr.ts";

test("format information matches the specification's published values", () => {
  // Level L with mask 0 is documented as 111011111000100.
  assert.equal(formatInfo(1, 0), 0b111011111000100);
  // Level L with mask 7 is documented as 110100101110110.
  assert.equal(formatInfo(1, 7), 0b110100101110110);
});

test("version information matches the specification's published values", () => {
  assert.equal(versionInfo(7), 0b000111110010010100);
  assert.equal(versionInfo(8), 0b001000010110111100);
  assert.equal(versionInfo(10), 0b001010010011010011);
});

test("short payloads produce a version 1 symbol", () => {
  const code = encodeQR("HELLO");
  assert.equal(code.version, 1);
  assert.equal(code.size, 21);
  assert.equal(code.modules.length, 21 * 21);
});

test("finder, timing and dark modules are placed correctly", () => {
  const code = encodeQR("https://example.com/?room=ABC123");
  const { size, modules } = code;
  const at = (r: number, c: number) => modules[r * size + c];

  for (const [r0, c0] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let dr = 0; dr <= 6; dr++) {
      for (let dc = 0; dc <= 6; dc++) {
        const ring = Math.max(Math.abs(dr - 3), Math.abs(dc - 3));
        const expected = ring === 3 || ring <= 1 ? 1 : 0;
        assert.equal(at(r0 + dr, c0 + dc), expected, `finder at ${r0 + dr},${c0 + dc}`);
      }
    }
  }

  for (let i = 8; i < size - 8; i++) {
    assert.equal(at(6, i), i % 2 === 0 ? 1 : 0);
    assert.equal(at(i, 6), i % 2 === 0 ? 1 : 0);
  }

  assert.equal(at(size - 8, 8), 1, "the fixed dark module must be set");
});

test("payload length selects an appropriate version and stays scannable", () => {
  assert.equal(encodeQR("a".repeat(17)).version, 1);
  assert.equal(encodeQR("a".repeat(18)).version, 2);
  assert.equal(encodeQR("a".repeat(271)).version, 10);
  assert.throws(() => encodeQR("a".repeat(272)), /exceeds the supported QR capacity/);
});

test("both format-information copies agree and decode to level L", () => {
  const code = encodeQR("https://example.com/?room=ZXCVB");
  const { size, modules } = code;
  const at = (r: number, c: number) => modules[r * size + c];

  const first: number[] = [];
  for (let i = 0; i <= 5; i++) first.push(at(i, 8));
  first.push(at(7, 8), at(8, 8), at(8, 7));
  for (let i = 9; i < 15; i++) first.push(at(8, 14 - i));

  const second: number[] = [];
  for (let i = 0; i < 8; i++) second.push(at(8, size - 1 - i));
  for (let i = 8; i < 15; i++) second.push(at(size - 15 + i, 8));

  assert.deepEqual(first, second, "the two format copies must be identical");

  let value = 0;
  for (let i = 0; i < 15; i++) value |= first[i] << i;

  const masks = [0, 1, 2, 3, 4, 5, 6, 7].map((mask) => formatInfo(1, mask));
  assert.ok(masks.includes(value), `format ${value.toString(2)} is not a level-L format`);
});

test("symbols use a mix of dark and light modules", () => {
  const code = encodeQR("https://multideviceai.example/?room=QWERTY");
  let dark = 0;
  for (const module of code.modules) dark += module;
  const ratio = dark / code.modules.length;
  assert.ok(ratio > 0.3 && ratio < 0.7, `unbalanced module ratio ${ratio}`);
});

/**
 * Independent reimplementation of the function-module map, written from the
 * specification rather than from the encoder, so a shared mistake is unlikely
 * to cancel itself out in the round-trip test below.
 */
function functionModules(version: number): Uint8Array {
  const size = version * 4 + 17;
  const mask = new Uint8Array(size * size);
  const mark = (r: number, c: number) => {
    if (r >= 0 && r < size && c >= 0 && c < size) mask[r * size + c] = 1;
  };

  for (const [r0, c0] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let dr = -1; dr <= 7; dr++) {
      for (let dc = -1; dc <= 7; dc++) mark(r0 + dr, c0 + dc);
    }
  }
  for (let i = 0; i < size; i++) {
    mark(6, i);
    mark(i, 6);
  }

  const centers: Record<number, number[]> = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
  };
  for (const r of centers[version]) {
    for (const c of centers[version]) {
      const nearFinder =
        (r <= 8 && c <= 8) || (r <= 8 && c >= size - 9) || (r >= size - 9 && c <= 8);
      if (nearFinder) continue;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) mark(r + dr, c + dc);
      }
    }
  }

  for (let i = 0; i <= 8; i++) {
    mark(i, 8);
    mark(8, i);
  }
  for (let i = 0; i < 8; i++) {
    mark(8, size - 1 - i);
    mark(size - 1 - i, 8);
  }
  return mask;
}

const SINGLE_BLOCK_DATA_CODEWORDS: Record<number, number> = {
  1: 19, 2: 34, 3: 55, 4: 80, 5: 108,
};

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

/** Decode a single-block, byte-mode symbol back into its original string. */
function decodeQR(code: { version: number; size: number; modules: Uint8Array }): string {
  const { version, size, modules } = code;
  const fn = functionModules(version);
  const at = (r: number, c: number) => modules[r * size + c];

  // Recover the applied mask from the format information.
  let format = 0;
  for (let i = 0; i <= 5; i++) format |= at(i, 8) << i;
  format |= at(7, 8) << 6;
  format |= at(8, 8) << 7;
  format |= at(8, 7) << 8;
  for (let i = 9; i < 15; i++) format |= at(8, 14 - i) << i;

  let mask = -1;
  for (let candidate = 0; candidate < 8; candidate++) {
    if (formatInfo(1, candidate) === format) mask = candidate;
  }
  assert.notEqual(mask, -1, "could not recover a level-L mask from the symbol");

  const bits: number[] = [];
  let upward = true;
  for (let col = size - 1; col >= 0; col -= 2) {
    if (col === 6) col--;
    for (let i = 0; i < size; i++) {
      const row = upward ? size - 1 - i : i;
      for (const c of [col, col - 1]) {
        if (c < 0 || fn[row * size + c]) continue;
        const raw = at(row, c);
        bits.push(maskBit(mask, row, c) ? raw ^ 1 : raw);
      }
    }
    upward = !upward;
  }

  const codewords: number[] = [];
  for (let i = 0; i + 8 <= bits.length; i += 8) {
    let byte = 0;
    for (let j = 0; j < 8; j++) byte = (byte << 1) | bits[i + j];
    codewords.push(byte);
  }

  const data = codewords.slice(0, SINGLE_BLOCK_DATA_CODEWORDS[version]);
  let bitCursor = 0;
  const take = (width: number) => {
    let value = 0;
    for (let i = 0; i < width; i++) {
      const index = bitCursor + i;
      value = (value << 1) | ((data[index >> 3] >> (7 - (index & 7))) & 1);
    }
    bitCursor += width;
    return value;
  };

  assert.equal(take(4), 0b0100, "expected a byte-mode segment");
  const length = take(version < 10 ? 8 : 16);
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) bytes[i] = take(8);
  return new TextDecoder().decode(bytes);
}

test("encoded symbols decode back to the original payload", () => {
  const payloads = [
    "room",
    "HELLO WORLD",
    "https://example.com/?room=ABC123",
    "https://dist-l91bcjgzm-trojroberts-projects.vercel.app/?room=QWERTY",
    "https://multideviceai.example.com/invite?room=ZXCVBN&host=laptop&v=1",
  ];
  for (const payload of payloads) {
    const code = encodeQR(payload);
    assert.ok(code.version <= 5, `${payload} used version ${code.version}`);
    assert.equal(decodeQR(code), payload);
  }
});

test("SVG rendering covers the module grid with a quiet zone", () => {
  const code = encodeQR("room");
  const svg = qrToSvg(code, 2);
  assert.match(svg, /^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  assert.match(svg, new RegExp(`viewBox="0 0 ${code.size + 4} ${code.size + 4}"`));
  assert.ok(svg.includes("<path d=\"M"));
});
