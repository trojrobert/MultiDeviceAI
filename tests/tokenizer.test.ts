import assert from "node:assert/strict";
import { test } from "node:test";
import { createQwenTokenizer, type TokenizerDefinition } from "../src/engine/tokenizer.ts";

function fixtureDefinition(): TokenizerDefinition {
  const vocab: Record<string, number> = {
    h: 0,
    e: 1,
    l: 2,
    o: 3,
    he: 4,
    hel: 5,
    hello: 6,
    "Ġ": 7,
    w: 8,
    r: 9,
    d: 10,
    wo: 11,
    wor: 12,
    worl: 13,
    world: 14,
    "Ċ": 15,
    u: 16,
    s: 17,
    er: 18,
    user: 19,
    us: 25,
    a: 20,
    i: 21,
    t: 22,
    n: 23,
    assistant: 24,
  };
  return {
    model: {
      vocab,
      merges: [
        "h e", "he l", "hel l", "hell o",
        "w o", "wo r", "wor l", "worl d",
        "u s", "s er", "us er",
        "a s", "as s", "ass i", "assi s", "assis t", "assist a", "assista n", "assistan t",
      ],
    },
    added_tokens: [
      { id: 100, content: "<|im_start|>" },
      { id: 101, content: "<|im_end|>" },
    ],
  };
}

test("byte BPE encodes known merges and decodes UTF-8 bytes", () => {
  const tokenizer = createQwenTokenizer(fixtureDefinition());
  const ids = tokenizer.encode("hello world");
  assert.deepEqual(ids, [6, 7, 14]);
  assert.equal(tokenizer.decode(ids), "hello world");
});

test("Qwen chat template and allowed special tokens are deterministic", () => {
  const tokenizer = createQwenTokenizer(fixtureDefinition());
  const prompt = tokenizer.applyChatTemplate([{ role: "user", content: "hello" }]);
  assert.equal(prompt, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
  const ids = tokenizer.encode(prompt, new Set(["<|im_start|>", "<|im_end|>"]));
  assert.equal(ids[0], 100);
  assert.ok(ids.includes(101));
  assert.equal(ids.filter((id) => id === 100).length, 2);
});
