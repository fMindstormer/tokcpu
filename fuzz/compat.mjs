// The JavaScript side of `python -m tokcpu check` / `compat`, as one long-lived process.
// One JSON request per stdin line, one JSON answer per stdout line:
//   {"op": "load", "dir": ...}                          load tokenizer.json + tokenizer_config.json in each runtime
//   {"op": "encode", "texts": [...]}                     ids of each text, special tokens added (the default)
//   {"op": "decode", "ids": [[...], ...]}                text of each id list, special tokens skipped, no clean-up
//   {"op": "versions"}
import fs from "fs";
import readline from "readline";
import * as tj3 from "tj3";
import * as tj4 from "tj4";
import { Tokenizer } from "@huggingface/tokenizers";

const MAKERS = {
  "transformers.js@3.7.6": (j, c) => new tj3.PreTrainedTokenizer(j, c),
  "transformers.js@4.3.1": (j, c) => new tj4.PreTrainedTokenizer(j, c),
  "@huggingface/tokenizers@0.2.0": (j, c) => new Tokenizer(j, c),
};
const message = (e) => String((e && e.message) || e).slice(0, 300);
let loaded = {};

const answer = {
  load({ dir }) {
    const j = fs.readFileSync(dir + "/tokenizer.json", "utf8");
    const c = fs.existsSync(dir + "/tokenizer_config.json") ? JSON.parse(fs.readFileSync(dir + "/tokenizer_config.json", "utf8")) : {};
    loaded = {};
    const errors = {};
    for (const [name, make] of Object.entries(MAKERS)) {
      try { loaded[name] = make(JSON.parse(j), c); } catch (e) { errors[name] = message(e); }
    }
    return { errors };
  },
  encode({ texts }) {
    const out = {};
    for (const [name, tok] of Object.entries(loaded)) {
      out[name] = texts.map((t) => {
        try { const e = tok.encode(t); return Array.isArray(e) ? e : e.ids; } catch (e) { return { error: message(e) }; }
      });
    }
    return out;
  },
  decode({ ids }) {
    const out = {};
    for (const [name, tok] of Object.entries(loaded)) {
      out[name] = ids.map((x) => {
        try { return tok.decode(x, { skip_special_tokens: true, clean_up_tokenization_spaces: false }); } catch (e) { return { error: message(e) }; }
      });
    }
    return out;
  },
  versions() {
    // V8's JSON.parse can misread an escaped key after an earlier parse (nodejs/node#63785): parse two
    // same-shaped objects whose first keys are \\ and \" and see whether the second comes back right.
    const body = (k) => `{"${k}":0,${Array.from({ length: 8 }, (_, i) => `"selftest${i}":${i}`).join(",")}}`;
    JSON.parse(body("\\\\"));
    const jsonParseBug = Object.keys(JSON.parse(body('\\"')))[0] !== '"';
    return { node: process.version, v8: process.versions.v8, jsonParseBug };
  },
};

readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const req = JSON.parse(line);
  let res;
  try { res = answer[req.op](req); } catch (e) { res = { error: message(e) }; }
  process.stdout.write(JSON.stringify(res) + "\n");
});
