"""Command line.

    python -m tokcpu build                         build every tokenizer into tokenizers/
    python -m tokcpu run tokenizers/brainfuck programs/hello.bf
    python -m tokcpu run tokenizers/brainfuck programs/reverse.bf --input stressed
    python -m tokcpu run tokenizers/4-state-busy-beaver --show 1
    python -m tokcpu run tokenizers/hello --auto   (through transformers' AutoTokenizer)
    python -m tokcpu ca tokenizers/rule110 --gens 24   a cellular automaton's generations, one row each
    python -m tokcpu hub USER/tokcpu [--push]      stage (and upload) the Hugging Face Hub repo
    python -m tokcpu fuzz --cases 200              differential fuzzing of tokenizer runtimes
    python -m tokcpu check MODEL [MODEL...]        does a Hub tokenizer (or a local folder) give the same ids in JavaScript?
    python -m tokcpu compat --top 1000             ...for the most-downloaded models, as one JSON
    python -m tokcpu report RESULTS [--previous OLD]   that JSON in Markdown, and what changed since OLD
    python -m tokcpu gguf --top 300                does llama.cpp tokenize Hub GGUFs like their base models?
    python -m tokcpu effect                        does a disagreement (final sigma) change embeddings and retrieval?

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

from . import bf, blocks, elementary, small
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
    for name, (spec, rules, tape) in [("sort", blocks.sorter()), ("rule110", elementary.automaton(110, "01")),
                                      ("rule110-ether", elementary.automaton(110, elementary.COLLISION,
                                                                             elementary.ETHER, elementary.ETHER))]:
        path = save(spec, out / name, tape)
        print(f"{name:22s} {len(rules):6d} rules  {path.stat().st_size / 1e6:6.2f} MB")


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
    ca = sub.add_parser("ca", help="run an elementary cellular automaton's tokenizer, one row a generation")
    ca.add_argument("tokenizer")
    ca.add_argument("--gens", type=int, default=24)
    ca.add_argument("--png", help="draw the rows into this PNG instead of printing them")
    ca.add_argument("--shade", action="store_true", help="draw Rule 110's ether pale, so its gliders stand out")
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
    c = sub.add_parser("check", help="compare a Hub model's tokenizer in Python and in JavaScript")
    c.add_argument("repos", nargs="+", help="Hub repos, or folders holding a tokenizer.json")
    c.add_argument("--node", default="node", help="the node binary for the JavaScript runtimes")
    k = sub.add_parser("compat", help="check the most-downloaded models' tokenizers")
    k.add_argument("--top", type=int, default=1000)
    k.add_argument("--out", default=str(ROOT / "build" / "compat" / "results.json"))
    k.add_argument("--node", default="node", help="the node binary for the JavaScript runtimes")
    g = sub.add_parser("gguf", help="compare Hub GGUFs' tokenizers in llama.cpp with their base models'")
    g.add_argument("repos", nargs="*", help="GGUF repos to check and print (default: the --top most downloaded)")
    g.add_argument("--top", type=int, default=300)
    g.add_argument("--out", default=str(ROOT / "build" / "gguf" / "results.json"))
    e = sub.add_parser("effect", help="feed Python's and transformers.js's ids for Greek text to the same models")
    e.add_argument("--out", default=str(ROOT / "effect"))
    e.add_argument("--node", default="node", help="the node binary, if the pairs need building")
    p = sub.add_parser("report", help="summarise compat's results in Markdown")
    p.add_argument("results")
    p.add_argument("--previous", help="an earlier results file: list the models that changed since")
    p.add_argument("--tables", help="also write the dashboard's tables (models, differences, texts) here as JSON")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        build(Path(a.out))
        return 0
    if a.cmd == "hub":
        hub(a.repo, Path(a.out), a.push)
        return 0
    if a.cmd == "ca":
        from tokenizers import Tokenizer
        d = Path(a.tokenizer)
        t0 = time.time()
        rows = elementary.run(Tokenizer.from_file(str(d / "tokenizer.json")), (d / "start.txt").read_text(), a.gens)
        dt = time.time() - t0
        if a.png:
            elementary.png(rows, a.png, elementary.ether_windows() if a.shade else None)
        else:
            for g, row in enumerate(rows):
                print(" " * (a.gens - g) + row.replace("0", "·").replace("1", "█"))
        print(f"\n{a.gens:,} generations ({2 * a.gens:,} ticks) in {dt:.2f} s", file=sys.stderr)
        return 0
    if a.cmd == "check":
        from .compat import check
        return check(a.repos, a.node)
    if a.cmd == "compat":
        from .compat import compat
        compat(a.top, Path(a.out), a.node)
        return 0
    if a.cmd == "gguf":
        from . import gguf
        if a.repos:
            return gguf.check(a.repos)
        gguf.campaign(a.top, Path(a.out))
        return 0
    if a.cmd == "effect":
        from .effect import effect
        effect(Path(a.out), a.node)
        print((Path(a.out) / "summary.md").read_text(), end="")
        return 0
    if a.cmd == "report":
        from .compat import summary, tables
        data = json.loads(Path(a.results).read_text())
        old = json.loads(Path(a.previous).read_text()) if a.previous and Path(a.previous).exists() else None
        print(summary(data, old), end="")
        if a.tables:
            Path(a.tables).mkdir(parents=True, exist_ok=True)
            for name, rows in tables(data).items():
                (Path(a.tables) / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
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
