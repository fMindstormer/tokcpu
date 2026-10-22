// Compile with the page's compiler (web/tokcpu.js): prints the tokenizer.json text, then the starting
// tape on the last line. Either a Turing machine table and its blank, or --ca RULE CELLS LEFT RIGHT.
import { parseTable, turing, automaton, spec, stringify } from "../web/tokcpu.js";
import fs from "fs";
const args = process.argv.slice(2);
if (args[0] === "--ca") {
  const m = automaton(Number(args[1]), args[2], args[3], args[4]);
  console.log(stringify(spec(m.chars, m.rules, m.preTokenizer)));
  console.log(m.start);
} else {
  const [tablePath, blank] = args;
  const m = turing(parseTable(fs.readFileSync(tablePath, "utf8")), blank);
  console.log(stringify(spec(m.chars, m.rules)));
  console.log(m.start);
}
