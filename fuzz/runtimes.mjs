// The JavaScript tokenizer runtimes for `python -m tokcpu fuzz`, as one long-lived process.
// One JSON request per stdin line, one JSON answer per stdout line:
//   {"op": "load", "dir": ...}                    load tokenizer.json + tokenizer_config.json everywhere
//   {"op": "run", "pairs": [[input, expected], ...]}   one tick per pair, in each runtime; first mismatch
//   {"op": "tick", "runtime": name, "tape": ...}  one tick
import fs from "fs";
import readline from "readline";
import * as tj3 from "tj3";
import * as tj4 from "tj4";
import { Tokenizer } from "@huggingface/tokenizers";

const ENC = { add_special_tokens: false };
const DEC = { skip_special_tokens: false, clean_up_tokenization_spaces: false };
const MAKERS = {
  "transformers.js@3.7.6": (j, c) => new tj3.PreTrainedTokenizer(j, c),
  "transformers.js@4.3.1": (j, c) => new tj4.PreTrainedTokenizer(j, c),
  "@huggingface/tokenizers@0.2.0": (j, c) => new Tokenizer(j, c),
};
const ids = (enc) => (Array.isArray(enc) ? enc : enc.ids);   // transformers.js returns ids, tokenizers.js an Encoding

let loaded = {};   // runtime -> tokenizer, or {error}
function tick(name, tape) {
  const tok = loaded[name];
  return tok.decode(ids(tok.encode(tape, ENC)), DEC);
}
const safe = (f) => { try { return { out: f() }; } catch (e) { return { error: String(e && e.message || e) }; } };

const answer = {
  load({ dir }) {
    const j = fs.readFileSync(dir + "/tokenizer.json", "utf8"), c = JSON.parse(fs.readFileSync(dir + "/tokenizer_config.json", "utf8"));
    loaded = {};
    const errors = {};
    for (const [name, make] of Object.entries(MAKERS)) {
      try { loaded[name] = make(JSON.parse(j), c); } catch (e) { errors[name] = String(e && e.message || e); }
    }
    return { runtimes: Object.keys(loaded), errors };
  },
  run({ pairs }) {
    const res = {};
    for (const name of Object.keys(loaded)) {
      res[name] = { ok: true };
      for (let n = 0; n < pairs.length; n++) {
        const r = safe(() => tick(name, pairs[n][0]));
        if (r.error !== undefined || r.out !== pairs[n][1]) { res[name] = { ok: false, tick: n, ...r }; break; }
      }
    }
    return res;
  },
  tick({ runtime, tape }) { return safe(() => tick(runtime, tape)); },
  versions() { return { node: process.version, v8: process.versions.v8 }; },
};

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const req = JSON.parse(line);
  let res;
  try { res = answer[req.op](req); } catch (e) { res = { error: String(e && e.message || e) }; }
  process.stdout.write(JSON.stringify(res) + "\n");
});
