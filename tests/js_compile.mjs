// Compile a Turing machine table with the page's compiler (web/tokcpu.js): prints the tokenizer.json
// text, then the starting tape on the last line.
import { parseTable, turing, spec, stringify } from "../web/tokcpu.js";
import fs from "fs";
const [tablePath, blank] = process.argv.slice(2);
const m = turing(parseTable(fs.readFileSync(tablePath, "utf8")), blank);
console.log(stringify(spec(m.chars, m.rules)));
console.log(m.start);
