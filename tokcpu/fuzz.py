"""Differential fuzzing of tokenizer runtimes, with the machine as the oracle.

Every random machine compiles to a tokenizer.json, and `Machine.step` says what each tick must give
without asking any tokenizer library. So when a runtime disagrees, we know which one is wrong, at which
tick, on which tape, and can shrink that tape to a one-tick reproduction.

Each machine is written out in several variants that the tokenizer.json format says are equivalent
(merges as "a b" strings, added tokens marked special or normalized, shuffled character ids, no Fuse
decoder), and every variant runs in every runtime available:

    tokenizers                       the Rust library, from Python
    transformers                     AutoTokenizer around it (its own added-token and decode handling)
    transformers.js, tokenizers.js   independent JavaScript implementations, through fuzz/runtimes.mjs

    python -m tokcpu fuzz --cases 200 --seed 1
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .compile import save, tokenizer_spec
from .machine import LEFT, RIGHT, Machine

ROOT = Path(__file__).resolve().parent.parent

# Awkward characters for the tape: spaces and controls, quotes and backslashes, combining and joining
# marks, characters other tokenizers treat specially (Ġ, ▁), and ones outside the Basic Multilingual
# Plane, which JavaScript stores as two UTF-16 units.
SYMBOLS = (list("01ab#<>/.+") + [" ", "\t", "\n", '"', "\\", "é", "́", "‍", " ", "漢", "א",
           "Ġ", "▁", "😀", "𝔸"])
MARKER_SETS = {
    "circled": [c for i in range(26) for c in (chr(0x24D0 + i), chr(0x24B6 + i))],
    "hangul": [chr(0xAC00 + i) for i in range(52)],
    "astral": [c for i in range(26) for c in (chr(0x1F130 + i), chr(0x1F150 + i))],   # squared, negative circled
}


def random_machine(rng: random.Random) -> tuple[Machine, str]:
    """A random Turing machine over a random awkward alphabet, and a random starting tape."""
    syms = rng.sample(SYMBOLS, rng.randint(2, 5))
    states = "ABCD"[:rng.randint(1, 4)]
    kind = rng.choice(sorted(MARKER_SETS))
    m = Machine(f"fuzz-{kind}", blank=syms[0], marker_chars=MARKER_SETS[kind])
    for q in states + "H":
        m.state(q)
    m.symbol(*syms)
    for q in states:
        for a in syms:
            if rng.random() < 0.85:
                m.on(q, a, rng.choice(syms), rng.choice("RLS"), rng.choice(states + "H"))
    left = "".join(rng.choice(syms) for _ in range(rng.randint(0, 8)))
    right = "".join(rng.choice(syms) for _ in range(rng.randint(0, 8)))
    return m, LEFT + left + m.R("A") + right + RIGHT


def trace(m: Machine, tape: str, limit: int) -> list[str]:
    """The oracle: the tapes the machine goes through, by its own rules, until it halts or `limit`."""
    out = [tape]
    while len(out) <= limit:
        nxt = m.step(out[-1])
        if nxt == out[-1]:
            break
        out.append(nxt)
    return out


# ---- equivalent variants of one tokenizer.json ------------------------------------------------

def variants(m: Machine, rng: random.Random) -> dict[str, dict]:
    base = tokenizer_spec(m)
    out = {"plain": base}
    if not any(" " in w for w in m.rules):                           # the old format splits on spaces
        v = json.loads(json.dumps(base))
        v["model"]["merges"] = [f"{a} {b}" for a, b in v["model"]["merges"]]
        out["merges-as-strings"] = v
    for flag in ("special", "normalized"):
        v = json.loads(json.dumps(base))
        for t in v["added_tokens"]:
            t[flag] = True
        out[f"added-{flag}"] = v
    v = json.loads(json.dumps(base))                                  # permute the single characters' ids
    chars = [c for c in v["model"]["vocab"] if len(c) == 1]
    ids = [v["model"]["vocab"][c] for c in chars]
    rng.shuffle(ids)
    v["model"]["vocab"].update(zip(chars, ids))
    out["shuffled-char-ids"] = v
    v = json.loads(json.dumps(base))
    v["decoder"] = v["decoder"]["decoders"][0]                       # the Replace alone, no Fuse
    out["no-fuse"] = v
    return out


# ---- runtimes ---------------------------------------------------------------------------------

class PyRuntimes:
    """tokenizers, and transformers if it imports."""

    def __init__(self):
        self.makers = {}
        from tokenizers import Tokenizer
        self.makers["tokenizers"] = lambda d: (lambda t, tok=Tokenizer.from_file(str(d / "tokenizer.json")):
                                               tok.decode(tok.encode(t).ids, skip_special_tokens=False))
        try:
            from transformers import AutoTokenizer, logging
            logging.set_verbosity_error()
            self.makers["transformers"] = lambda d: (lambda t, tok=AutoTokenizer.from_pretrained(str(d)):
                                                     tok.decode(tok.encode(t, add_special_tokens=False),
                                                                skip_special_tokens=False,
                                                                clean_up_tokenization_spaces=False))
        except Exception:
            pass
        self.ticks = {}

    def load(self, d: Path) -> dict[str, str]:
        self.ticks, errors = {}, {}
        for name, make in self.makers.items():
            try:
                self.ticks[name] = make(d)
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"
        return errors

    def run(self, tapes: list[str]) -> dict:
        res = {}
        for name, tick in self.ticks.items():
            res[name] = {"ok": True}
            for n in range(len(tapes) - 1):
                r = self.tick(name, tapes[n])
                if "error" in r or r["out"] != tapes[n + 1]:
                    res[name] = {"ok": False, "tick": n, **r}
                    break
        return res

    def tick(self, name: str, tape: str) -> dict:
        try:
            return {"out": self.ticks[name](tape)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


class JsRuntimes:
    """fuzz/runtimes.mjs: transformers.js 3 and 4, and tokenizers.js."""

    def __init__(self, log: Path):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(["node", str(ROOT / "fuzz" / "runtimes.mjs")], cwd=ROOT / "fuzz", text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log.open("w"),
                                  encoding="utf-8")

    def ask(self, **req) -> dict:
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def load(self, d: Path) -> dict[str, str]:
        return self.ask(op="load", dir=str(d.resolve()))["errors"]

    def run(self, tapes: list[str]) -> dict:
        return self.ask(op="run", trace=tapes)

    def tick(self, name: str, tape: str) -> dict:
        return self.ask(op="tick", runtime=name, tape=tape)

    def close(self):
        self.p.stdin.close()
        self.p.wait()

    def versions(self) -> str:
        v = self.ask(op="versions")
        return f"node {v['node']}, V8 {v['v8']}"


# ---- shrinking --------------------------------------------------------------------------------

def shrink(m: Machine, tape: str, wrong) -> str:
    """Delete characters from a tape while `wrong(tape)` still holds. The edges and the head stay."""
    changed = True
    while changed:
        changed = False
        for i in range(len(tape)):
            c = tape[i]
            if c in (LEFT, RIGHT) or c in m.marker_of:
                continue
            t = tape[:i] + tape[i + 1:]
            if wrong(t):
                tape, changed = t, True
                break
    return tape


# ---- the campaign -----------------------------------------------------------------------------

@dataclass
class Finding:
    runtime: str
    variant: str
    case: int
    tape: str
    want: str
    got: str
    error: str = ""
    dir: str = ""
    seen: int = 1
    examples: list = field(default_factory=list)
    fresh: bool | None = None      # does it still go wrong in a fresh process, with nothing loaded before?

    def signature(self):
        return (self.runtime, self.error.split(":")[0] if self.error else "wrong output", self.fresh)


def fuzz(cases: int, seed: int, out: Path, limit: int = 300, log=print) -> list[Finding]:
    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    py = PyRuntimes()
    js = JsRuntimes(out / "node-stderr.log") if (ROOT / "fuzz" / "node_modules").exists() else None
    names = list(py.makers) + (["transformers.js@3.7.6", "transformers.js@4.3.1", "@huggingface/tokenizers@0.2.0"] if js else [])
    log(f"runtimes: {', '.join(names)}" + (f" ({js.versions()})" if js else ""))
    findings: dict[tuple, Finding] = {}
    load_errors: dict[tuple, Finding] = {}
    checked = ticks = 0
    for case in range(cases):
        m, start = random_machine(rng)
        tapes = trace(m, start, limit)
        for vname, spec in variants(m, rng).items():
            d = out / "work"
            if d.exists():
                shutil.rmtree(d)
            save(spec, d)
            for rt in filter(None, (py, js)):
                for name, err in rt.load(d).items():
                    f = Finding(name, vname, case, start, "", "", error=err)
                    key = f.signature()
                    if key in load_errors:
                        load_errors[key].seen += 1
                    else:
                        load_errors[key] = f
                        f.dir = str(keep(d, out, f"load-{len(load_errors)}", {"runtime": name, "error": err}))
                for name, r in rt.run(tapes).items():
                    checked += 1
                    ticks += r.get("tick", len(tapes) - 1)
                    if r["ok"]:
                        continue
                    n = r["tick"]
                    want = tapes[n + 1]

                    # A runtime can carry state from earlier tokenizers in the same process. Check the
                    # failing tick again in a fresh one; shrink there if it still fails, else keep the tape.
                    fresh_rt = JsRuntimes(out / "node-stderr-fresh.log") if rt is js else PyRuntimes()
                    fresh_rt.load(d)

                    def wrong_fresh(t, name=name, rt=fresh_rt):
                        got = rt.tick(name, t)
                        return "error" in got or got["out"] != m.step(t)
                    fresh = wrong_fresh(tapes[n])
                    small = shrink(m, tapes[n], wrong_fresh) if fresh else tapes[n]
                    got = (fresh_rt if fresh else rt).tick(name, small)
                    if isinstance(fresh_rt, JsRuntimes):
                        fresh_rt.close()
                    f = Finding(name, vname, case, small, m.step(small), got.get("out", ""), got.get("error", ""),
                                fresh=fresh)
                    key = f.signature()
                    if key in findings:
                        findings[key].seen += 1
                        if len(findings[key].examples) < 3:
                            findings[key].examples.append(small)
                        continue
                    findings[key] = f
                    f.dir = str(keep(d, out, f"finding-{len(findings)}", {
                        "runtime": name, "variant": vname, "tape": small, "expected": f.want, "got": f.got,
                        "error": f.error, "full_trace_tick": n, "fails_in_fresh_process": fresh}))
                    log(f"  case {case} {vname}: {name} disagrees at tick {n}"
                        + ("" if fresh else " (only after earlier tokenizers in this process)"))
        if (case + 1) % 25 == 0:
            log(f"{case + 1} machines, {checked:,} runs, {ticks:,} ticks checked, {len(findings)} distinct disagreements")
    if js:
        js.close()
    shutil.rmtree(out / "work", ignore_errors=True)
    return list(load_errors.values()) + list(findings.values())


def keep(d: Path, out: Path, name: str, info: dict) -> Path:
    """Copy a tokenizer folder next to a repro.json describing what went wrong with it."""
    k = out / name
    shutil.copytree(d, k)
    (k / "repro.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    return k
