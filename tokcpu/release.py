"""A frozen release, as one folder and a zip of it: the guided incrementer and the universal Brainfuck
tokenizer with their starting tapes and expected results, a minimal runner (repro/run.py), the
final-sigma reproduction (repro/sigma.py and sigma.mjs), a dated snapshot of the compatibility results,
and SHA-256 checksums of every file.

The expected results come from the machines' own step function (Machine.step), not from any tokenizer
library, so `run.py --check` tests the library against them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path

from . import bf, small
from .compile import save

ROOT = Path(__file__).resolve().parent.parent
HUB = "fMindstormer/tokcpu"
RESULTS = ("78f8532", "2026-10-19")   # the compat-results commit of the run the page and README describe


def expected(m, tape: str) -> tuple[str, int]:
    n = 0
    while (nxt := m.step(tape)) != tape:
        tape, n = nxt, n + 1
    return tape, n


def reading(tape: str) -> str:
    if "┃" in tape:
        return "output: " + tape[1:tape.index("┃")]
    bits = "".join(c for c in tape if c in "01")
    return f"number: {bits} = {int(bits, 2)}"


def release(version: str, hub_revision: str, out: Path) -> Path:
    d = out / f"tokcpu-{version}"
    if d.exists():
        shutil.rmtree(d)
    examples = []

    def example(name: str, m, folder: str, start_file: str, tape: str) -> None:
        (d / folder / start_file).write_text(tape, encoding="utf-8")
        end, n = expected(m, tape)
        (d / "expected").mkdir(parents=True, exist_ok=True)
        (d / "expected" / f"{name}.txt").write_text(end, encoding="utf-8")
        examples.append({"name": name, "tokenizer": f"{folder}/tokenizer.json", "start": f"{folder}/{start_file}",
                         "expected": f"expected/{name}.txt", "ticks": n, "reads": reading(end)})

    m, start = small.turing(small.INCREMENT, "increment", blank="_")
    tape = start[:-1] + "1011" + start[-1]
    save(m, d / "increment", tape)
    example("increment", m, "increment", "start.txt", tape)
    u = bf.universal()
    save(u, d / "brainfuck")
    for name, data in (("hello", b""), ("reverse", b"stressed")):
        example(name, u, "brainfuck", f"{name}.txt", bf.universal_tape(u, (ROOT / "programs" / f"{name}.bf").read_text(), data))
    (d / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    shutil.copy(ROOT / "repro" / "run.py", d / "run.py")
    shutil.copy(ROOT / "repro" / "requirements.txt", d / "requirements.txt")
    (d / "sigma").mkdir()
    for f in ("sigma.py", "sigma.mjs", "package.json", "package-lock.json"):
        shutil.copy(ROOT / "repro" / f, d / "sigma" / f)
    commit, day = RESULTS
    snap = d / f"compat-results-{day}"
    snap.mkdir()
    for f in ("README.md", "results.json"):
        (snap / f).write_bytes(subprocess.run(["git", "show", f"{commit}:{f}"], cwd=ROOT, capture_output=True, check=True).stdout)

    by = {e["name"]: e for e in examples}
    (d / "README.md").write_text(README.format(
        version=version, day=date.today().isoformat(), hub=HUB, rev=hub_revision, results=snap.name, commit=commit,
        inc=by["increment"], hello=by["hello"], rev_ex=by["reverse"],
        inc_end=(d / "expected" / "increment.txt").read_text(encoding="utf-8")), encoding="utf-8")

    files = sorted(p for p in d.rglob("*") if p.is_file())
    (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(d).as_posix()}\n" for p in files))
    z = out / f"tokcpu-{version}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(out).as_posix())
    return z


README = """# Tokenizer CPU {version}

A frozen copy of [Tokenizer CPU](https://github.com/fMindstormer/tokcpu), made on {day}: a stock Hugging
Face tokenizer that executes programs, and the tokenizer disagreement the project found in real models.
The same tokenizer files are on the Hugging Face Hub at tag `{version}`, revision
[`{rev}`](https://huggingface.co/{hub}/tree/{rev}): the universal Brainfuck tokenizer at the root and
the incrementer in `increment/`. The live [dashboard](https://fmindstormer.github.io/tokcpu/compat/)
keeps updating; this folder does not.

## A computation

Python 3.10 or later:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run.py increment/tokenizer.json increment/start.txt --show
python run.py --check
```

`run.py` uses only the `tokenizers` library: it loads a `tokenizer.json`, then runs
`tape = tok.decode(tok.encode(tape).ids)` until the tape stops changing, or until a step limit
(`--limit`, 1,000,000 ticks by default). Each tick applies one rule; the loop is the only thing that repeats.

| Example | Tokenizer | Starting tape | Ticks | Result |
|---|---|---|---:|---|
| Add one to 1011 | `increment/` (16 rules) | `{inc[start]}` | {inc[ticks]:,} | `{inc_end}`: 1100, so 11 becomes 12 |
| Hello World | `brainfuck/` (17,210 rules) | `{hello[start]}` | {hello[ticks]:,} | {hello[reads]} |
| Reverse, input `stressed` | `brainfuck/` | `{rev_ex[start]}` | {rev_ex[ticks]:,} | {rev_ex[reads]} |

The expected final tapes, in `expected/`, come from the machines' own rules, not from a tokenizer
library. `--check` runs all three and prints `ok` for each, in a few seconds. In the
incrementer's tape, `ⓐ` is the head in state A, reading to its right, `Ⓑ` state B, reading to its left,
`Ⓒ` the halted state, and `_` a blank cell. In Brainfuck's output, `␊` is a newline.

## A runtime disagreement

The same `tokenizer.json` gives the Greek word `ΟΔΟΣ` different ids in Python and in JavaScript. This
needs only mBERT's tokenizer (1.7 MB, at a pinned revision), no model weights, and Node 22 or later:

```
cd sigma
python sigma.py
npm ci && node sigma.mjs
```

Expected:

```
Python's own str.lower():               οδος
tokenizers 0.23.3 lowercasing (Rust):   οδοσ
tokenizers 0.23.3, ΟΔΟΣ:                ο ##δο ##σ   [295, 37187, 62591]
tokenizers 0.23.3, οδος (lowercase):    ο ##δος      [295, 42501]
JavaScript's toLowerCase():             οδος
tokenizers.js 0.2.0, ΟΔΟΣ:              ο ##δος      [295, 42501]
```

Python's own `str.lower()` and JavaScript's `toLowerCase` both follow Unicode: a capital Σ that ends a
word becomes ς. The Rust `tokenizers` library, which Python's `transformers` uses, lowercases one
character at a time and gives σ. So the browser's ids are those of the word typed in lowercase, and
Python's are not. transformers.js 3.7.6 and 4.3.1 give the same ids as tokenizers.js.

In the project's Greek retrieval experiment (900 Belebele questions typed in capitals, 488 passages,
the same weights and code on both sides), changing only these ids changed mBERT's top-1 accuracy from
23.3% with Python's ids to 31.9% with the browser's. Which lowercasing mBERT was trained with is not
certain: Google's original code gives ς on Python 3 but σ on Python 2. To rerun that experiment, see
"Does a disagreement change what the model does?" in the repository's README (`python -m tokcpu effect`,
about 15 minutes on a laptop CPU).

## Compatibility results, {day_results}

`{results}/` is the run of {day_results} (commit `{commit}` of the repository's `compat-results` branch): the
500 most-downloaded Hub models with a `tokenizer.json`, on 42 texts, in `transformers` 5.20.2 and
`tokenizers` 0.23.3 against transformers.js 3.7.6 and 4.3.1 and tokenizers.js 0.2.0 on Node 22.23.3. Its
README summarises it; `results.json` holds every model and difference.

## Checksums

```
shasum -a 256 -c SHA256SUMS
```
""".replace("{day_results}", RESULTS[1])
