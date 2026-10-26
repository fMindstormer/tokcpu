"""Run a tokenizer as a computer with the stock `tokenizers` library: decode(encode(tape)) until the tape
stops changing, or until the step limit.

    python run.py increment/tokenizer.json increment/start.txt --show
    python run.py brainfuck/tokenizer.json brainfuck/reverse.txt
    python run.py --check          every example in examples.json, against its expected final tape
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent


def run(tok: Tokenizer, tape: str, limit: int, show: bool = False) -> tuple[str, int, bool]:
    """(final tape, ticks that changed it, whether it halted within the limit)."""
    if show:
        print(f"{0:6d}  {tape}")
    for n in range(limit):
        nxt = tok.decode(tok.encode(tape).ids, skip_special_tokens=False)
        if nxt == tape:
            return tape, n, True
        tape = nxt
        if show:
            print(f"{n + 1:6d}  {tape}")
    return tape, limit, False


def reading(tape: str) -> str:
    """What the tape says: a Brainfuck program's output (left of ┃), else the binary number on it."""
    if "┃" in tape:
        return "output: " + tape[1:tape.index("┃")]
    bits = "".join(c for c in tape if c in "01")
    return f"number: {bits} = {int(bits, 2)}" if bits else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tokenizer", nargs="?", help="a tokenizer.json")
    ap.add_argument("start", nargs="?", help="a file holding the starting tape")
    ap.add_argument("--limit", type=int, default=1_000_000, help="stop after this many ticks (default 1,000,000)")
    ap.add_argument("--show", action="store_true", help="print the tape after every tick")
    ap.add_argument("--check", action="store_true", help="run every example in examples.json")
    a = ap.parse_args()
    if not a.check:
        if not a.start:
            ap.error("give a tokenizer.json and a starting tape, or --check")
        tape = Path(a.start).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        end, n, halted = run(Tokenizer.from_file(a.tokenizer), tape, a.limit, a.show)
        print(f"{'halted' if halted else 'stopped at the step limit'} after {n:,} ticks ({time.perf_counter() - t0:.1f} s)")
        print(f"tape: {end}")
        print(reading(end))
        return 0 if halted else 1
    failed = 0
    for ex in json.loads((HERE / "examples.json").read_text(encoding="utf-8")):
        tok = Tokenizer.from_file(str(HERE / ex["tokenizer"]))
        t0 = time.perf_counter()
        end, n, halted = run(tok, (HERE / ex["start"]).read_text(encoding="utf-8"), a.limit)
        ok = halted and n == ex["ticks"] and end == (HERE / ex["expected"]).read_text(encoding="utf-8")
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {ex['name']:10s} {n:9,} ticks  {time.perf_counter() - t0:5.1f} s  {reading(end)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
