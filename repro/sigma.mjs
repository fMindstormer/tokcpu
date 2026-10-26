// The same word and the same tokenizer.json in JavaScript, with tokenizers.js (@huggingface/tokenizers),
// which lowercases with String.prototype.toLowerCase. Run sigma.py first: it downloads the files.
// transformers.js 3.7.6 and 4.3.1 give the same ids.
import fs from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";

const read = (name) => JSON.parse(fs.readFileSync(new URL(`./mbert/${name}`, import.meta.url), "utf8"));
const tok = new Tokenizer(read("tokenizer.json"), read("tokenizer_config.json"));
const word = "ΟΔΟΣ", expected = [295, 42501];
const enc = tok.encode(word, { add_special_tokens: false });
const show = (label, value) => console.log(label.padEnd(40) + value);
show("JavaScript's toLowerCase():", word.toLowerCase());
show(`tokenizers.js 0.2.0, ${word}:`, `${enc.tokens.join(" ").padEnd(12)} [${enc.ids.join(", ")}]`);
if (enc.ids.join() !== expected.join()) {
  console.error(`expected [${expected.join(", ")}]`);
  process.exit(1);
}
