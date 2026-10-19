// A Turing machine table -> tokenizer.json, in the browser. A port of turing() in tokcpu/small.py, the
// Machine model in tokcpu/machine.py and spec() in tokcpu/compile.py; tests/js_compile.mjs checks that
// it writes the same file as Python, byte for byte.

export const LEFT = "⊢", RIGHT = "⊣";
const CIRCLED = [];   // readable heads: an R-marker is a circled small letter, an L-marker the circled capital
for (let i = 0; i < 26; i++) CIRCLED.push(String.fromCharCode(0x24d0 + i), String.fromCharCode(0x24b6 + i));
const TAG0 = 0xe000, TAGS = 64;
const TAG_CLASS = `[${String.fromCharCode(TAG0)}-${String.fromCharCode(TAG0 + TAGS - 1)}]`;

const byCode = (a, b) => (a < b ? -1 : a > b ? 1 : 0);   // BMP only, so this is code-point order

function tag(i) {
  let out = "";
  while (i) { out = String.fromCharCode(TAG0 + (i % TAGS)) + out; i = Math.floor(i / TAGS); }
  return out;
}

/** Parse "state read write move next" lines (// starts a comment). The first rule's state starts. */
export function parseTable(text) {
  const table = [];
  text.split("\n").forEach((line, k) => {
    const words = line.replace(/\/\/.*/, "").trim().split(/\s+/).filter(Boolean);
    if (!words.length) return;
    if (words.length !== 5) throw new Error(`line ${k + 1}: expected "state read write move next"`);
    const [q, a, b, d, n] = words;
    for (const c of [a, b]) {
      if (Array.from(c).length !== 1 || c.codePointAt(0) > 0xffff) throw new Error(`line ${k + 1}: a symbol is one character, not "${c}"`);
      if (c === LEFT || c === RIGHT || CIRCLED.includes(c) || (c.charCodeAt(0) >= TAG0 && c.charCodeAt(0) < TAG0 + TAGS))
        throw new Error(`line ${k + 1}: "${c}" is reserved`);
    }
    if (!"RLS".includes(d) || d.length !== 1) throw new Error(`line ${k + 1}: move is R, L or S, not "${d}"`);
    if (table.some(([q2, a2]) => q2 === q && a2 === a)) throw new Error(`line ${k + 1}: a second rule for state ${q} reading ${a}`);
    table.push([q, a, b, d, n]);
  });
  if (!table.length) throw new Error("no rules");
  return table;
}

/** Like turing() in small.py: { chars, rules, markerOf, start }. */
export function turing(table, blank = "0") {
  const start = table[0][0];
  const states = [...new Set([...table.map((r) => r[0]), ...table.map((r) => r[4])])]
    .sort((p, q) => (p !== start) - (q !== start) || byCode(p, q));
  if (states.length > 26) throw new Error("at most 26 states");
  const markers = new Map(), markerOf = {}, symbols = new Set([blank]), rules = new Map();
  states.forEach((q, i) => {
    const [r, l] = [CIRCLED[2 * i], CIRCLED[2 * i + 1]];
    markers.set(q, [r, l]); markerOf[r] = [q, "R"]; markerOf[l] = [q, "L"];
  });
  const R = (q) => markers.get(q)[0], L = (q) => markers.get(q)[1];
  const rule = (w, o) => {
    if (rules.has(w) && rules.get(w) !== o) throw new Error(`conflicting rules for ${w}`);
    rules.set(w, o);
  };
  for (const [q, a, b, d, n] of table) {          // Machine.on, for both sides
    symbols.add(a); symbols.add(b);
    for (const side of "RL") {
      const here = side === "R" ? R(q) : L(q);
      const out = d === "R" ? b + R(n) : d === "L" ? L(n) + b : side === "R" ? R(n) + b : b + L(n);
      if (side === "R") {
        rule(here + a, out);
        if (a === blank) rule(here + RIGHT, out + RIGHT);
      } else {
        rule(a + here, out);
        if (a === blank) rule(LEFT + here, LEFT + out);
      }
    }
  }
  const chars = [...new Set([...symbols, LEFT, RIGHT, ...Object.keys(markerOf)])].sort(byCode);
  return { chars, rules, markerOf, start: LEFT + R(start) + RIGHT };
}

/** Like spec() in compile.py. */
export function spec(chars, rules) {
  const windows = [...rules.keys()].sort(byCode);
  const C = chars.length, M = windows.length;
  const vocab = {}, seen = new Map(), added = [];
  chars.forEach((c, i) => { vocab[c] = i; });
  windows.forEach((w, k) => {
    const out = rules.get(w);
    seen.set(out, (seen.get(out) || 0) + 1);
    vocab[w] = C + M + k;
    added.push({ id: C + M + k, content: out + tag(seen.get(out)), single_word: false, lstrip: false,
                 rstrip: false, normalized: false, special: false });
  });
  return {
    version: "1.0", truncation: null, padding: null,
    added_tokens: added,
    normalizer: null, pre_tokenizer: null, post_processor: null,
    decoder: { type: "Sequence", decoders: [
      { type: "Replace", pattern: { Regex: TAG_CLASS }, content: "" },
      { type: "Fuse" }] },
    model: { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: null,
             end_of_word_suffix: null, fuse_unk: false, byte_fallback: false, ignore_merges: false,
             vocab, merges: windows.map((w) => [w[0], w[1]]) },
  };
}

/** The file's text, as Python writes it. JavaScript lists integer-like keys ("0", "1") first in any
 *  object, so the vocab is written by hand, in id order. */
export function stringify(sp) {
  const vocab = Object.entries(sp.model.vocab).sort((a, b) => a[1] - b[1]);
  const hole = JSON.stringify({ ...sp, model: { ...sp.model, vocab: "\u0000" } });
  return hole.replace('"\\u0000"', () => "{" + vocab.map(([k, v]) => JSON.stringify(k) + ":" + v).join(",") + "}");
}

export const CONFIG = { tokenizer_class: "PreTrainedTokenizerFast", clean_up_tokenization_spaces: false,
                        model_max_length: 1000000000 };
