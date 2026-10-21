# Tokenizer CPU

[![test](https://github.com/fMindstormer/tokcpu/actions/workflows/test.yml/badge.svg)](https://github.com/fMindstormer/tokcpu/actions/workflows/test.yml)

A stock Hugging Face tokenizer that runs Brainfuck. It is a plain `tokenizer.json`, with no custom
code and no model. Every clock tick is one encode and one decode:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizers/brainfuck")
while (nxt := tok.decode(tok.encode(tape))) != tape:
    tape = nxt
```

Start the tape with a Brainfuck program written on it and the loop ends with the program's output
written there too:

```
⊢Hello World!␊┃++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.▼가║⠀⠀⡈⡤⡗⠡▲⠊⊣
 └── output ──┘ └──────────────────────────────── the program ───────────────────────────────────────┘  └─ memory ─┘
```

**On the Hugging Face Hub:** [fMindstormer/tokcpu](https://huggingface.co/fMindstormer/tokcpu), so
`AutoTokenizer.from_pretrained("fMindstormer/tokcpu")` loads the universal tokenizer, with Hello World's
tape in its `start.txt`.

**Interactive page:** https://claude.ai/artifact/GXyttZQHii8GFLEjKhMZe9 (private until you share it from
its Share menu). It runs these same files in the browser with transformers.js, and compiles Turing
machines you type into new tokenizers you can download.

## How one tick works

The tape is a string. Exactly one character on it is the *head*: a marker that stands for a machine
state and reads the character beside it, on its right or its left depending on the marker.

1. **Encode finds the head.** The BPE model has one merge rule for each pair the head can form with a
   cell, and no other merges. So the tape splits into single characters plus one two-character token:
   the head and the cell it reads.
2. **The id is the rule.** Each such pair has its own id, and that id stands for one Turing-machine
   rule: what to write, which way to move, which state comes next.
3. **Decode applies it.** The same id is also an *added token* whose text is the rule's output: the new
   symbol with the next head placed beside it. Decoding prints that output where the pair was. Every
   other character decodes to itself.
4. **Halting.** A halting state has no merges, so its tape decodes to itself. The loop stops when the
   tape stops changing.

A head at either end of the tape reads a blank, and its rules can grow the tape there, so the tape is
as long as the program needs.

### The catch: steps would have to be reversible

A token id decodes to exactly one string, and when the library saves a tokenizer (as `transformers`
does internally when it loads one) it keeps one string per id. If two rules wrote the same output, they
would need two added tokens with the same text, which a tokenizer cannot hold. Taken plainly, the
trick only runs machines whose every step can be undone: reversible computers.

Ordinary programs throw information away constantly. So each added token's text is the output plus a
short tag of private-use characters (U+E000 to U+E03F) that makes it unique, and the decoder has one
generic rule: a `Replace` that deletes those characters. That deletion is the only place the machine
forgets anything. The tags never reach the tape, so an added token can never match tape text when
encoding.

The ids line up because the library numbers added tokens after the model's vocabulary, in the order
they are listed. With C single characters and M windows, window k gets id C + M + k, and so does the
k-th added token. Ids C to C + M − 1 are unused; the files still save, reload and run correctly
(the tests check this).

### One pass versus the loop

A single encode is a weak computer. Cognetta and Okazaki showed that BPE tokenization is a
finite-state transduction ([Tokenization as Finite-State Transduction](https://arxiv.org/abs/2410.15696)):
one pass can only look a bounded distance ahead. One tick here rewrites just the two characters around
the head. Repeating the tick, on a tape that can grow, is what makes it Turing complete. The loop is
three lines of Python, the same way "CSS is Turing complete" demos need someone clicking.

## The ladder

| Rung | Tokenizer | Merge rules | File | Program | Ticks |
|---|---|---:|---:|---|---:|
| 1 | binary counter | 7 | 1.5 KB | counts forever | |
| 1 | 4-state busy beaver | 24 | 4 KB | itself | 107 |
| 2 | Hello World, compiled | 38,265 | 5.7 MB | Hello World | 1,194 |
| 2 | alphabet, compiled | 28,301 | 4.2 MB | alphabet | 556 |
| 2 | grid, compiled | 25,474 | 3.8 MB | grid | 477 |
| 2 | jumps, compiled | 24,173 | 3.6 MB | jumps | 531 |
| 3 | universal Brainfuck | 17,210 | 2.6 MB | Hello World | 154,137 |
| 3 | | | | alphabet | 27,053 |
| 3 | | | | grid (nested loops) | 17,474 |
| 3 | | | | jumps (skipped and nested) | 21,415 |
| 3 | | | | reverse, input `stressed` | 1,904 |

- **Rung 1, small machines** (`tokcpu/small.py`). Any Turing machine given as a table. The 4-state busy
  beaver halts after exactly 107 ticks with 13 ones on the tape, its known record.
- **Rung 2, one program per tokenizer** (`compile_program` in `tokcpu/bf.py`). A Brainfuck program
  compiles into a machine whose states are the program: each instruction is one tick. Printing leaves
  a mark at the data pointer, carries the byte left to the output and walks back to the mark.
- **Rung 3, one tokenizer for every program** (`universal` in `tokcpu/bf.py`). The program is written
  on the tape. The head carries each instruction from `▼` (the instruction pointer) to `▲` (the data
  pointer), executes it there, and walks back. Jumps scan for the matching bracket with the loop depth
  held in the state, up to 64 levels. Input waits at the right end of the tape after `¦`, and `,` takes
  its first character.

Both rungs print a byte in two trips, one per nibble. The high nibble goes left and waits before `┃`
as a circled hex digit (`⓪` to `ⓕ`); the head goes back, reads the cell again, and the low nibble
joins it. Carrying the whole byte at once would take 256 carrying states, each with a rule for every
symbol it walks over: that was 92% of the universal tokenizer's rules. Two trips take 32 states, which
cut it from 73,968 rules (11.1 MB) to 15,179 (2.3 MB) for 2% more ticks. Input (`,`) reuses the same
trips the other way: the cell becomes a hole `◌`, the high nibble of the input's first character comes
back and waits in it, and a second trip takes the character and brings the low nibble. It costs 2,031
rules.

Brainfuck details: cells are bytes that wrap; the data tape grows to the right as needed; moving left
of cell 0 stops the machine; `,` reads 0 once the input is used up, and needs the universal tokenizer.
Data cells are Braille characters (U+2800 + value), so the dots spell the byte. Printed and input bytes
appear as themselves when printable, as control pictures below 32 (`␊` is a newline), and as
U+0100 + (byte − 128) above 127.

### Clock speed

The universal tokenizer running Hello World (154,137 ticks), best of three runs on an Apple M4:

| Library | Ticks a second |
|---|---:|
| `tokenizers` 0.23 (Rust, from Python) | 9,200 |
| `transformers` 5.20 AutoTokenizer | 5,900 |
| transformers.js 3.7.6 in Node | 11,700 |
| transformers.js in headless Chromium (the page) | 17,300 |

Speed does not depend on the number of rules; it falls slowly as the tape gets longer.

## Parallel ticks

One encode reads the whole tape, but a single head uses two characters of it. Two ways to use the rest:

**Several heads.** Heads that never meet each get their rule applied in the same tick, with no change to
the compiler. Two busy beavers on one tape finish together in 107 ticks.

**A block cellular automaton** (`tokcpu/blocks.py`). A stock `Split` pre-tokenizer cuts the tape into
two-character blocks before BPE runs, and every block is a merge, so one tick rewrites every block at
once. A phase mark `◦` after the left edge shifts the blocks by one character, and the edge block adds
or drops it, so the blocks alternate between two alignments (the Margolus neighbourhood) and
information can cross block boundaries.

The demo, `tokenizers/sort`, is odd-even transposition sort: each block puts its two letters in order,
and n letters are sorted after at most n ticks. It has 704 rules.

```
⊢◦thequickbrownfoxjumpsoverthelazydog⊣     tick 0
⊢◦abcdeeefghhijklmnoooopqrrsttuuvwxyz⊣     tick 30
```

The mark flips every tick, so the tape never stops changing; a run stops when the tape repeats every
two ticks. A tick costs time in proportion to the tape's length either way, so this buys throughput:
sorting 1,000 letters does about 320,000 block rewrites a second, where one head does about 10,000
rules a second on the same machine. Block cellular automata can be universal too, but this one only
sorts.

## Fuzzing tokenizer runtimes

The same `tokenizer.json` is re-implemented in many runtimes, and they can quietly disagree. Most
cross-checks compare two implementations on sample text, so they can tell that two disagree but not
which one is wrong. Here the machine is an independent oracle: `Machine.step` says what every tick
must give without asking any tokenizer library.

`python -m tokcpu fuzz` (`tokcpu/fuzz.py`) compiles random machines over awkward alphabets (spaces and
controls, quotes and backslashes, combining marks, characters outside the Basic Multilingual Plane)
and writes each one as several tokenizers, one per feature of the format:

- **plain**: variants the format says are equivalent (merges as `"a b"` strings, added tokens marked
  special or normalized, shuffled ids, no Fuse decoder);
- **normalizers**: Lowercase, NFC, NFD, NFKC, NFKD, Strip, Replace and a Sequence, with input noise the
  normalizer must undo (capitals, decomposed or full-width characters, padding);
- **Split pre-tokenizers**: every behaviour, with the pattern written six ways, and separators
  sprinkled on the tape; a rule may only fire if its window lies in one piece;
- **WhitespaceSplit**, with Unicode whitespace and look-alikes that are not whitespace;
- **byte fallback**: characters outside the vocab must survive as byte tokens;
- **added tokens that occur in the input**, with `lstrip` and `rstrip`.

For each feature a small model in `fuzz.py` says what a tick must give, written from the format's
documented behaviour; a test checks every model against the Rust library first. Every tokenizer then
runs in every runtime it finds: `tokenizers`, `transformers`, and, after `cd fuzz && npm install`,
transformers.js 3.7.6 and 4.3.1 and `@huggingface/tokenizers` 0.2.0 (`--node` picks the Node binary).
A disagreement is checked again in a fresh process, shrunk to a one-tick input, and kept with its
tokenizer in `build/fuzz/`.

### What it found

On Node 22, 200 machines and about 372,000 ticks: the two Python runtimes agreed with the models
everywhere. The three JavaScript runtimes disagreed in the same ways, which small probes then pinned
down (all agree with each other, so these are disagreements with the Rust library, the reference):

| | Disagreement | Runtimes |
|---|---|---|
| A | Whitespace means JavaScript's `\s`, not Unicode White_Space: U+FEFF (byte order mark) is stripped or split on, U+0085 (next line) is not. Affects WhitespaceSplit, Strip, and `lstrip`/`rstrip` of added tokens | all three |
| B | A Split with a `String` pattern and behaviour `Removed` keeps the separator (the same pattern as a `Regex` works) | all three |
| C | A Split with `invert: true` is wrong for every behaviour but `Removed` | all three |
| D | A character missing from the vocab, with no `unk_token`: Rust drops it before merging, so `aßb` becomes `ab`; JavaScript merges around it and gives `a`, `b` | 4.3.1, tokenizers.js; 3.7.6 emits a `null` id and crashes in decode |
| E | An added token's `id`: Rust ignores it and numbers added tokens after the vocab; JavaScript uses it, so the same file gives the model different ids | all three |
| F | The Oniguruma escape `\x{B6}` in a regex fails to load | 3.7.6 (fixed in 4.3.1) |

C, D and E came from probes rather than the fuzzer's own features; adding them as features is next.

On Node 24 and 25 and in Chrome 139 there is one more, below all of them: V8's `JSON.parse` can misread
an escaped key, such as a vocab entry `"`, after an earlier parse in the same process, so the token
silently disappears. It is a known V8 bug ([nodejs/node#63785](https://github.com/nodejs/node/issues/63785)),
fixed in V8 and awaiting a Node 24 backport ([nodejs/node#66542](https://github.com/nodejs/node/pull/66542)).
Six popular Hub tokenizers loaded one after another in Node 24 came through intact.

### Real tokenizers

The fuzzer's tokenizers are built to be awkward. To see what reaches real models, `python -m tokcpu
check <repo>...` (`tokcpu/compat.py`) runs Hub tokenizers through the same three JavaScript runtimes on 42
texts (everyday text, 15 languages and scripts, Unicode edge cases, stress cases), and compares their ids
and decoded text with Python. Python has two references: `tokenizers` reading the file as written, and
`transformers`' AutoTokenizer, which adds model-specific behaviour (CLIP's decode turns `</w>` into
spaces). A difference counts only if JavaScript differs from both, or from the only one that loads the
model; where they disagree with each other, JavaScript always sides with one of them.

`python -m tokcpu compat --top 1000` does this for the most-downloaded models, once per distinct
`tokenizer.json`. In October 2026, on Node 22 against `transformers` 5.20 and `tokenizers` 0.23, 500 of
the top 1,000 models had a tokenizer.json (220 distinct files). In transformers.js 4.3.1 and
tokenizers.js 0.2.0, 375 of the 500 matched Python on every text; transformers.js 3.7.6 matched on 362.
Weighted by downloads, the share that matched everywhere was 48%, because the popular BERT-style models
differ on one rare character:

| Cause | Models (transformers.js 4) | What happens |
|---|---|---|
| Final sigma | 84 | Lowercasing a capital Σ at the end of a word gives ς in JavaScript (`toLowerCase` is context-sensitive) and σ in Python. 69 BERT-style WordPiece models (`bert-base-uncased`, `all-MiniLM-L6-v2`, the BGE and E5 families) and 15 CLIP models |
| Whitespace (A above) | 44 | U+FEFF and U+0085 |
| Other | 8 | Arabic-Indic digits in a ByteLevel split, runs of spaces in Metaspace, a zero-width joiner in a Precompiled normalizer |
| Byte fallback ignored | 3 | A Unigram model's `byte_fallback` is not applied, so characters outside the vocab become `<unk>` ([tokenizers.js#26](https://github.com/huggingface/tokenizers.js/issues/26)); SigLIP and ruri-v3 lose German, Chinese and emoji this way |
| Throws | 3 | Decoding an empty list of ids throws instead of returning "" |

On everyday text, only those three byte-fallback models and five others differ. transformers.js 3.7.6
also drops the leading `▁` of Metaspace tokenizers with `prepend_scheme: "first"` (11 models), fixed in 4.

The results are on a [dashboard](https://claude.ai/artifact/CnCFwFeqHXPsNTKdyuod3d), and `--out` keeps them
as JSON. The run takes about an hour, mostly waiting on the Hub's rate limit, and resumes where it stopped.

**Your own tokenizer.** Before you ship a tokenizer to the browser, check it (after `cd fuzz && npm
install`, on Node 22):

```
python -m tokcpu check your-name/your-model          # a Hub repo; private ones need `hf auth login`
python -m tokcpu check path/to/folder                # or a folder holding tokenizer.json
```

For each runtime, it lists every text where the ids or the decoded text differ from Python, with the
tokens around the first difference and a guess at the cause. It exits with 1 when a runtime differs from
both Python references or cannot load the tokenizer, so it can run in CI.

**New releases.** [`.github/workflows/compat.yml`](.github/workflows/compat.yml) checks every day for a
new release of transformers.js, tokenizers.js, `transformers` or `tokenizers`. When there is one, or the
last run is four weeks old, it runs the top 1,000 again with the latest of each. The results go on the
[`compat-results`](../../tree/compat-results) branch, and its README lists the models that changed since
the run before. To get the same summary locally, run `python -m tokcpu report results.json --previous
old.json`.

## Checking it

`python -m pytest tests` (about a minute; CI runs it on every push):

- every tick of every run is compared with the machine's own rules applied directly (`Machine.step`);
- every Brainfuck output is compared with a plain interpreter (`interpret` in `tokcpu/bf.py`), for each
  program on both rung 2 and rung 3;
- the files survive the library's own save and reload, and run through `transformers`' AutoTokenizer;
- with transformers.js installed (`cd tests && npm install`), the compiled Hello World runs tick for
  tick the same in JavaScript;
- the sorter and two heads on one tape are checked tick for tick against a direct reference step;
- the page's own compiler (`web/tokcpu.js`) writes the same `tokenizer.json` as Python, byte for byte,
  for the busy beaver and 20 random machines.

Tested with `tokenizers` 0.22 and 0.23, `transformers` 4.57 and 5.20, and transformers.js 3.7.6, on
Python 3.13. CI also runs Python 3.10.

## Run it

```
pip install -r requirements.txt
python -m tokcpu build                                    # writes every tokenizer into tokenizers/
python -m tokcpu run tokenizers/brainfuck programs/hello.bf
python -m tokcpu run tokenizers/brainfuck programs/grid.bf --auto      # through AutoTokenizer
python -m tokcpu run tokenizers/brainfuck programs/reverse.bf --input stressed
python -m tokcpu run tokenizers/4-state-busy-beaver --show 1           # print every tick
python -m tokcpu run tokenizers/hello                     # the compiled one; start.txt holds its tape
python -m tokcpu run tokenizers/sort --show 2             # the block automaton, sorting a pangram
```

Each folder in `tokenizers/` holds `tokenizer.json` and `tokenizer_config.json`, so it loads with
`AutoTokenizer.from_pretrained` and can go on the Hugging Face Hub like any other tokenizer. The build
is deterministic and takes under a second, so git keeps only the small machines' files.

`python -m tokcpu hub your-name/tokcpu` stages all of them as one Hub repo in `build/hub`: the universal
tokenizer at the root, with Hello World's tape as `start.txt`, and the others in subfolders. `--push`
uploads it.

## Files

```
tokcpu/machine.py     the machine model: one head on a string tape, two-character windows, rules
tokcpu/compile.py     machine -> tokenizer.json (BPE merges, added tokens, the tag-deleting decoder)
tokcpu/small.py       rung 1: a binary counter, the 4-state busy beaver, any Turing machine table
tokcpu/bf.py          rungs 2 and 3: the Brainfuck compiler, the universal interpreter, a reference interpreter
tokcpu/blocks.py      parallel ticks: a block cellular automaton, and the sorter
tokcpu/fuzz.py        differential fuzzing of tokenizer runtimes, with the machine as oracle
tokcpu/compat.py      the same runtimes against Python on real Hub tokenizers, and the report
.github/workflows/    the tests on every push; compat.yml, the top 1,000 again on each new release
fuzz/                 the JavaScript runtimes for the fuzzer (runtimes.mjs) and compat (compat.mjs), pinned
tokcpu/__main__.py    build, run, and stage the Hub repo
tokcpu/hub_card.md    the Hub repo's model card
programs/             hello, alphabet, grid, jumps, and reverse (reads input)
tests/                the checks above, and the transformers.js runner
web/                  the interactive page; `build` copies its tokenizers into web/tok/
web/tokcpu.js         the page's compiler: a Turing machine table -> tokenizer.json, in JavaScript
```

## License

MIT, see [LICENSE](LICENSE).
