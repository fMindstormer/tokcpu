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

**Interactive page:** https://claude.ai/artifact/GXyttZQHii8GFLEjKhMZe9 (private until you share it from
its Share menu). It runs these same files in the browser with transformers.js. The version in `web/`
also compiles Turing machines you type into new tokenizers you can download.

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

The universal tokenizer running Hello World, on one core of a cloud machine:

| Library | Ticks a second |
|---|---:|
| `tokenizers` (Rust, from Python) | 9,600 |
| `transformers` AutoTokenizer | 5,900 |
| transformers.js 3.7.6 in Node | 10,000 |
| transformers.js in Chromium (the page) | 8,000 |

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
tokcpu/__main__.py    build, run, and stage the Hub repo
tokcpu/hub_card.md    the Hub repo's model card
programs/             hello, alphabet, grid, jumps, and reverse (reads input)
tests/                the checks above, and the transformers.js runner
web/                  the interactive page; `build` copies its tokenizers into web/tok/
web/tokcpu.js         the page's compiler: a Turing machine table -> tokenizer.json, in JavaScript
```

## License

MIT, see [LICENSE](LICENSE).
