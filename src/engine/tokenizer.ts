/*
 * Adapted from SwarmLLM engine/tokenizer.js.
 * Copyright (c) 2026 Nehanth Narendrula. MIT License.
 */

export interface TokenizerDefinition {
  model: {
    vocab: Record<string, number>;
    merges: Array<string | [string, string]>;
  };
  added_tokens?: Array<{ id: number; content: string }>;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface QwenTokenizer {
  readonly vocab: Readonly<Record<string, number>>;
  tokenId(token: string): number | undefined;
  encode(text: string, allowedSpecial?: ReadonlySet<string>): number[];
  decode(ids: Iterable<number>): string;
  applyChatTemplate(messages: readonly ChatMessage[], addGenerationPrompt?: boolean): string;
}

const TOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteAlphabet(): { byteToCharacter: string[]; characterToByte: Map<string, number> } {
  const bytes: number[] = [];
  for (let i = 33; i <= 126; i++) bytes.push(i);
  for (let i = 161; i <= 172; i++) bytes.push(i);
  for (let i = 174; i <= 255; i++) bytes.push(i);
  const characters = bytes.slice();
  let extra = 0;
  for (let byte = 0; byte < 256; byte++) {
    if (!bytes.includes(byte)) {
      bytes.push(byte);
      characters.push(256 + extra++);
    }
  }
  const byteToCharacter: string[] = [];
  const characterToByte = new Map<string, number>();
  bytes.forEach((byte, index) => {
    const character = String.fromCodePoint(characters[index]!);
    byteToCharacter[byte] = character;
    characterToByte.set(character, byte);
  });
  return { byteToCharacter, characterToByte };
}

export function createQwenTokenizer(definition: TokenizerDefinition): QwenTokenizer {
  const vocab: Record<string, number> = { ...definition.model.vocab };
  for (const token of definition.added_tokens ?? []) vocab[token.content] = token.id;
  const idToToken = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) idToToken.set(id, token);
  const ranks = new Map<string, number>();
  definition.model.merges.forEach((merge, index) => {
    ranks.set(Array.isArray(merge) ? merge.join(" ") : merge, index);
  });
  const { byteToCharacter, characterToByte } = byteAlphabet();
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  function bpe(word: string): string[] {
    let parts = [...word];
    while (parts.length > 1) {
      let bestIndex = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      parts.splice(bestIndex, 2, parts[bestIndex]! + parts[bestIndex + 1]!);
    }
    return parts;
  }

  function encodeOrdinary(text: string): number[] {
    const ids: number[] = [];
    for (const piece of text.match(TOKEN_PATTERN) ?? []) {
      let encoded = "";
      for (const byte of encoder.encode(piece)) encoded += byteToCharacter[byte]!;
      for (const token of bpe(encoded)) {
        const id = vocab[token];
        if (id === undefined) throw new Error(`tokenizer vocabulary is missing byte-BPE token ${JSON.stringify(token)}`);
        ids.push(id);
      }
    }
    return ids;
  }

  return {
    vocab,
    tokenId: (token) => vocab[token],
    encode(text, allowedSpecial = new Set()) {
      if (allowedSpecial.size === 0) return encodeOrdinary(text);
      const specials = [...allowedSpecial]
        .filter((token) => vocab[token] !== undefined)
        .sort((a, b) => b.length - a.length);
      if (specials.length === 0) return encodeOrdinary(text);
      const ids: number[] = [];
      let cursor = 0;
      while (cursor < text.length) {
        let matchIndex = -1;
        let match = "";
        for (const special of specials) {
          const index = text.indexOf(special, cursor);
          if (index >= 0 && (matchIndex < 0 || index < matchIndex || (index === matchIndex && special.length > match.length))) {
            matchIndex = index;
            match = special;
          }
        }
        if (matchIndex < 0) {
          ids.push(...encodeOrdinary(text.slice(cursor)));
          break;
        }
        ids.push(...encodeOrdinary(text.slice(cursor, matchIndex)));
        ids.push(vocab[match]!);
        cursor = matchIndex + match.length;
      }
      return ids;
    },
    decode(ids) {
      const bytes: number[] = [];
      for (const id of ids) {
        const token = idToToken.get(id);
        if (token === undefined) continue;
        for (const character of token) {
          const byte = characterToByte.get(character);
          if (byte !== undefined) bytes.push(byte);
        }
      }
      return decoder.decode(new Uint8Array(bytes), { stream: false });
    },
    applyChatTemplate(messages, addGenerationPrompt = true) {
      let prompt = "";
      for (const message of messages) {
        prompt += `<|im_start|>${message.role}\n${message.content}<|im_end|>\n`;
      }
      if (addGenerationPrompt) prompt += "<|im_start|>assistant\n";
      return prompt;
    },
  };
}

export function tokenizerDefinitionFromGGUF(
  metadata: Record<string, unknown>,
): TokenizerDefinition {
  const tokens = metadata["tokenizer.ggml.tokens"];
  const merges = metadata["tokenizer.ggml.merges"];
  if (!Array.isArray(tokens) || !Array.isArray(merges)) {
    throw new Error("GGUF does not contain tokenizer.ggml.tokens and tokenizer.ggml.merges");
  }
  return {
    model: {
      vocab: Object.fromEntries(tokens.map((token, id) => [String(token), id])),
      merges: merges.map(String),
    },
  };
}
