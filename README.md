# Tokenizer CPU

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
its Share menu). It runs these same files in the browser with transformers.js.

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
| 2 | Hello World, compiled | 94,843 | 14.3 MB | Hello World | 1,024 |
| 3 | universal Brainfuck | 73,968 | 11.1 MB | Hello World | 151,159 |
| 3 | | | | alphabet | 23,433 |
| 3 | | | | grid (nested loops) | 15,428 |
| 3 | | | | jumps (skipped and nested) | 21,313 |

- **Rung 1, small machines** (`tokcpu/small.py`). Any Turing machine given as a table. The 4-state busy
  beaver halts after exactly 107 ticks with 13 ones on the tape, its known record.
- **Rung 2, one program per tokenizer** (`compile_program` in `tokcpu/bf.py`). A Brainfuck program
  compiles into a machine whose states are the program: each instruction is one tick. Printing leaves
  a mark at the data pointer, carries the byte left to the output (one state per byte value) and walks
  back to the mark.
- **Rung 3, one tokenizer for every program** (`universal` in `tokcpu/bf.py`). The program is written
  on the tape. The head carries each instruction from `▼` (the instruction pointer) to `▲` (the data
  pointer), executes it there, and walks back. Jumps scan for the matching bracket with the loop depth
  held in the state, up to 64 levels.

Brainfuck details: cells are bytes that wrap; the data tape grows to the right as needed; moving left
of cell 0 stops the machine; input (`,`) is not supported. Data cells are Braille characters (U+2800 +
value), so the dots spell the byte. Printed bytes appear as themselves when printable, as control
pictures below 32 (`␊` is a newline), and as U+0100 + (byte − 128) above 127.

### Clock speed

The universal tokenizer running Hello World, on one core of a cloud machine:

| Library | Ticks a second |
|---|---:|
| `tokenizers` (Rust, from Python) | 9,600 |
| `transformers` AutoTokenizer | 5,900 |
| transformers.js 3.7.6 in Node | 10,000 |
| transformers.js in Chromium (the page) | 8,000 |

Speed does not depend on the number of rules; it falls slowly as the tape gets longer.

## Checking it

`python -m pytest tests` (about 40 seconds):

- every tick of every run is compared with the machine's own rules applied directly (`Machine.step`);
- every Brainfuck output is compared with a plain interpreter (`interpret` in `tokcpu/bf.py`), for each
  program on both rung 2 and rung 3;
- the files survive the library's own save and reload, and run through `transformers`' AutoTokenizer;
- with transformers.js installed (`cd tests && npm install @huggingface/transformers@3.7.6`), the
  compiled Hello World runs tick for tick the same in JavaScript.

## Run it

```
pip install tokenizers transformers
python -m tokcpu build                                    # writes every tokenizer into tokenizers/
python -m tokcpu run tokenizers/brainfuck programs/hello.bf
python -m tokcpu run tokenizers/brainfuck programs/grid.bf --auto      # through AutoTokenizer
python -m tokcpu run tokenizers/4-state-busy-beaver --show 1           # print every tick
python -m tokcpu run tokenizers/hello                     # the compiled one; start.txt holds its tape
```

Each folder in `tokenizers/` holds `tokenizer.json` and `tokenizer_config.json`, so it loads with
`AutoTokenizer.from_pretrained` and can go on the Hugging Face Hub like any other tokenizer.

## Files

```
tokcpu/machine.py     the machine model: one head on a string tape, two-character windows, rules
tokcpu/compile.py     machine -> tokenizer.json (BPE merges, added tokens, the tag-deleting decoder)
tokcpu/small.py       rung 1: a binary counter, the 4-state busy beaver, any Turing machine table
tokcpu/bf.py          rungs 2 and 3: the Brainfuck compiler, the universal interpreter, a reference interpreter
tokcpu/__main__.py    build and run
programs/             hello, alphabet, grid, jumps
tests/                the checks above, and the transformers.js runner
web/                  the interactive page; `build` copies its tokenizers into web/tok/
```
