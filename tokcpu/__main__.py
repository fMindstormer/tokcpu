"""Command line.

    python -m tokcpu build                         build every tokenizer into tokenizers/
    python -m tokcpu run tokenizers/brainfuck programs/hello.bf
    python -m tokcpu run tokenizers/4-state-busy-beaver --show 1
    python -m tokcpu run tokenizers/hello --auto   (through transformers' AutoTokenizer)

A run ticks `decode(encode(tape))` until the tape stops changing. Nothing else touches the tape.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import bf, small
from .compile import save

ROOT = Path(__file__).resolve().parent.parent
PAGE = ("brainfuck", "hello", "4-state-busy-beaver", "binary-counter")     # machines on the web page


def build(out: Path) -> None:
    jobs = [("binary-counter", *small.counter()), ("4-state-busy-beaver", *small.busy_beaver_4())]
    for prog in sorted((ROOT / "programs").glob("*.bf")):
        m, tape = bf.compile_program(prog.read_text())
        jobs.append((prog.stem, m, tape))
    m = bf.universal()
    jobs.append(("brainfuck", m, None))
    for name, m, tape in jobs:
        path = save(m, out / name, tape)
        print(f"{name:22s} {len(m.rules):6d} rules  {path.stat().st_size / 1e6:6.2f} MB")
        if name in PAGE:                 # the interactive page's copy, with the head labels it shows
            d = ROOT / "web" / "tok" / name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, d / "tokenizer.json")
            meta = {"markers": {c: [q, s] for c, (q, s) in m.marker_of.items()}, "rules": len(m.rules),
                    "bytes": path.stat().st_size, "config": json.loads((out / name / "tokenizer_config.json").read_text())}
            if tape is not None:
                meta["start"] = tape
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tokcpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(ROOT / "tokenizers"))
    r = sub.add_parser("run")
    r.add_argument("tokenizer")
    r.add_argument("program", nargs="?", help="a .bf file, for the universal brainfuck tokenizer")
    r.add_argument("--auto", action="store_true", help="load with transformers' AutoTokenizer")
    r.add_argument("--show", type=int, default=0, help="print the tape every N ticks")
    r.add_argument("--max", type=int, default=10 ** 9)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        build(Path(a.out))
        return 0

    d = Path(a.tokenizer)
    if a.program:
        tape = bf.universal_tape(bf.universal(), Path(a.program).read_text())
    else:
        tape = (d / "start.txt").read_text()
    if a.auto:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(d))
        enc, dec = (lambda t: tok.encode(t)), (lambda ids: tok.decode(ids))
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        enc, dec = (lambda t: tok.encode(t).ids), tok.decode

    n, t0 = 0, time.time()
    if a.show:
        print(f"{n:>9} {tape}")
    while n < a.max:
        nxt = dec(enc(tape))
        if nxt == tape:
            break
        tape, n = nxt, n + 1
        if a.show and n % a.show == 0:
            print(f"{n:>9} {tape}")
    dt = time.time() - t0
    print(f"\n{'halted' if n < a.max else 'stopped'} after {n:,} ticks in {dt:.2f} s ({n / max(dt, 1e-9):,.0f} ticks/s)")
    if bf.OUT_END in tape:
        print("output:", bf.read_output(tape).decode("latin-1"), end="" if bf.read_output(tape).endswith(b"\n") else "\n")
    print("tape:  ", tape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
