// Run a tokenizer folder with transformers.js against a reference trace (one tape per line).
import { PreTrainedTokenizer } from "@huggingface/transformers";
import fs from "fs";
const [dir, tracePath] = process.argv.slice(2);
const tok = new PreTrainedTokenizer(JSON.parse(fs.readFileSync(dir + "/tokenizer.json", "utf8")),
                                    JSON.parse(fs.readFileSync(dir + "/tokenizer_config.json", "utf8")));
const want = fs.readFileSync(tracePath, "utf8").split("\n");
let tape = want[0], n = 0, bad = -1;
for (;;) {
  const next = tok.decode(tok.encode(tape, { add_special_tokens: false }),
                          { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  if (next === tape) break;
  if (bad < 0 && next !== want[n + 1]) bad = n;
  tape = next; n++;
}
console.log(JSON.stringify({ ticks: n, firstMismatch: bad }));
