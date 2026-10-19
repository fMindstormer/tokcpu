---
license: mit
library_name: transformers
tags:
  - tokenizer
  - brainfuck
  - turing-machine
  - esoteric
---

# Tokenizer CPU

A stock Hugging Face tokenizer that runs Brainfuck. It is a plain `tokenizer.json`, with no custom
code and no model (no `trust_remote_code`). Every clock tick is one encode and one decode:

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
tape = open(hf_hub_download("{repo}", "start.txt"), encoding="utf-8").read()
while (nxt := tok.decode(tok.encode(tape))) != tape:   # 154,137 ticks
    tape = nxt
print(tape)
```

`start.txt` holds Hello World. The loop ends with the program's output written on the tape:

```
⊢Hello World!␊┃++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.▼가║⠀⠀⡈⡤⡗⠡▲⠊⊣
 └── output ──┘ └──────────────────────────────── the program ───────────────────────────────────────┘  └─ memory ─┘
```

To run your own program, start from `"⊢┃▼가" + program + "║▲⠀¦" + input + "⊣"`. `,` reads the input's
first character, and 0 once it is used up. Loops can nest up to 64 deep.

## How it works

Exactly one character on the tape is the *head*. The BPE model has one merge for each pair the head can
form with a cell, so encoding splits the tape into single characters plus one two-character token. That
token's id is a Turing-machine rule, and the same id is an added token whose text is the rule's output,
so decoding applies the rule. A halted head has no merges, and the loop stops when the tape stops
changing.

The full explanation, the compiler and the tests are on GitHub: https://github.com/fMindstormer/tokcpu

## What's in this repo

| Folder | Machine | Merge rules |
|---|---|---:|
| (root) | universal Brainfuck interpreter: runs any program written on the tape | 17,210 |
| `hello`, `alphabet`, `grid`, `jumps` | one Brainfuck program compiled into each tokenizer | 24,173 – 38,265 |
| `4-state-busy-beaver` | the 4-state busy beaver: halts after 107 ticks with 13 ones | 24 |
| `binary-counter` | counts in binary forever | 7 |
| `sort` | a block cellular automaton: every two-letter block of the tape is rewritten in one tick, sorting it | 704 |

Load a subfolder with `AutoTokenizer.from_pretrained("{repo}", subfolder="hello")`. Each one has its
own `start.txt`. `programs/` holds the Brainfuck sources. The sorter's tape never stops changing (a phase
mark flips every tick), so stop it when the tape repeats every two ticks.
