"""Command line.

    python -m tokcpu build                         build every tokenizer into tokenizers/
    python -m tokcpu run tokenizers/brainfuck programs/hello.bf
    python -m tokcpu run tokenizers/brainfuck programs/reverse.bf --input stressed
    python -m tokcpu run tokenizers/4-state-busy-beaver --show 1
    python -m tokcpu run tokenizers/hello --auto   (through transformers' AutoTokenizer)
    python -m tokcpu hub USER/tokcpu [--push]      stage (and upload) the Hugging Face Hub repo
    python -m tokcpu fuzz --cases 200              differential fuzzing of tokenizer runtimes

A run ticks `decode(encode(tape))` until the tape stops changing, or until it repeats every two ticks
(a block automaton that has settled, see blocks.py). Nothing else touches the tape.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import bf, blocks, small
from .compile import save

ROOT = Path(__file__).resolve().parent.parent
PAGE = ("brainfuck", "hello", "4-state-busy-beaver", "binary-counter")     # machines on the web page


def build(out: Path, web: bool = True) -> None:
    jobs = [("binary-counter", *small.counter()), ("4-state-busy-beaver", *small.busy_beaver_4())]
    for prog in sorted((ROOT / "programs").glob("*.bf")):
        if "," in prog.read_text():          # input needs the universal tokenizer
            continue
        m, tape = bf.compile_program(prog.read_text())
        jobs.append((prog.stem, m, tape))
    m = bf.universal()
    jobs.append(("brainfuck", m, None))
    for name, m, tape in jobs:
        path = save(m, out / name, tape)
        print(f"{name:22s} {len(m.rules):6d} rules  {path.stat().st_size / 1e6:6.2f} MB")
        if web and name in PAGE:         # the interactive page's copy, with the head labels it shows
            d = ROOT / "web" / "tok" / name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, d / "tokenizer.json")
            meta = {"markers": {c: [q, s] for c, (q, s) in m.marker_of.items()}, "rules": len(m.rules),
                    "bytes": path.stat().st_size, "config": json.loads((out / name / "tokenizer_config.json").read_text())}
            if tape is not None:
                meta["start"] = tape
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    spec, rules, tape = blocks.sorter()
    path = save(spec, out / "sort", tape)
    print(f"{'sort':22s} {len(rules):6d} rules  {path.stat().st_size / 1e6:6.2f} MB")


def hub(repo: str, out: Path, push: bool) -> None:
    """One Hub repo: the universal tokenizer at the root (so `from_pretrained(repo)` loads it, with
    Hello World's tape as start.txt), every other machine in a subfolder, and the programs."""
    if out.exists():
        shutil.rmtree(out)
    build(out, web=False)
    for f in (out / "brainfuck").iterdir():
        f.rename(out / f.name)
    (out / "brainfuck").rmdir()
    (out / "start.txt").write_text(bf.universal_tape(bf.universal(), (ROOT / "programs" / "hello.bf").read_text()))
    shutil.copytree(ROOT / "programs", out / "programs")
    (out / "README.md").write_text((Path(__file__).parent / "hub_card.md").read_text().replace("{repo}", repo))
    print(f"\nstaged {repo} in {out}")
    if push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(out), commit_message="Tokenizer CPU")
        print(f"uploaded: https://huggingface.co/{repo}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tokcpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(ROOT / "tokenizers"))
    r = sub.add_parser("run")
    r.add_argument("tokenizer")
    r.add_argument("program", nargs="?", help="a .bf file, for the universal brainfuck tokenizer")
    r.add_argument("--input", default="", help="what the program's , reads (as UTF-8)")
    r.add_argument("--auto", action="store_true", help="load with transformers' AutoTokenizer")
    r.add_argument("--show", type=int, default=0, help="print the tape every N ticks")
    r.add_argument("--max", type=int, default=10 ** 9)
    h = sub.add_parser("hub")
    h.add_argument("repo", help="the Hub repo id, e.g. your-name/tokcpu")
    h.add_argument("--out", default=str(ROOT / "build" / "hub"))
    h.add_argument("--push", action="store_true", help="upload it (needs `hf auth login`)")
    f = sub.add_parser("fuzz", help="differential fuzzing of tokenizer runtimes, with the machine as oracle")
    f.add_argument("--cases", type=int, default=100, help="random machines to try")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--limit", type=int, default=300, help="most ticks per machine")
    f.add_argument("--out", default=str(ROOT / "build" / "fuzz"))
    f.add_argument("--node", default="node", help="the node binary for the JavaScript runtimes")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        build(Path(a.out))
        return 0
    if a.cmd == "hub":
        hub(a.repo, Path(a.out), a.push)
        return 0
    if a.cmd == "fuzz":
        from .fuzz import fuzz
        found = fuzz(a.cases, a.seed, Path(a.out), a.limit, a.node)
        print(f"\n{len(found)} distinct disagreements" + (f", kept in {a.out}" if found else ""))
        for f in found:
            what = f.error or f"expected {f.want!r}, got {f.got!r}"
            when = "" if f.fresh in (None, True) else ", only after earlier tokenizers in the same process"
            print(f"- {f.runtime} [{f.feature}] x{f.seen}{when}: input {f.tape!r}: {what[:200]}")
        return 1 if found else 0

    d = Path(a.tokenizer)
    if not (d / "tokenizer.json").exists():
        ap.error(f"no {d / 'tokenizer.json'}; run `python -m tokcpu build` first")
    if a.program:
        if not Path(a.program).exists():
            ap.error(f"no such program: {a.program}")
        try:
            tape = bf.universal_tape(bf.universal(), Path(a.program).read_text(), a.input.encode())
        except ValueError as e:
            ap.error(f"{a.program}: {e}")
    elif (d / "start.txt").exists():
        tape = (d / "start.txt").read_text()
    else:
        ap.error(f"{d} has no start.txt; give it a .bf program to run")
    if a.auto:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(d))
        enc, dec = (lambda t: tok.encode(t)), (lambda ids: tok.decode(ids))
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        enc, dec = (lambda t: tok.encode(t).ids), tok.decode

    n, t0, prev, how = 0, time.time(), None, "stopped"
    if a.show:
        print(f"{n:>9} {tape}")
    while n < a.max:
        nxt = dec(enc(tape))
        if nxt == tape:
            how = "halted"
            break
        if nxt == prev:                          # settled one tick ago
            tape, n, how = prev, n - 1, "settled (the tape repeats every 2 ticks)"
            break
        prev, tape, n = tape, nxt, n + 1
        if a.show and n % a.show == 0:
            print(f"{n:>9} {tape}")
    dt = time.time() - t0
    print(f"\n{how} after {n:,} ticks in {dt:.2f} s ({n / max(dt, 1e-9):,.0f} ticks/s)")
    if bf.OUT_END in tape:
        print("output:", bf.read_output(tape).decode("latin-1"), end="" if bf.read_output(tape).endswith(b"\n") else "\n")
    print("tape:  ", tape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
