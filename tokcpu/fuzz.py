"""Differential fuzzing of tokenizer runtimes, with the machine as the oracle.

Every random machine compiles to a tokenizer.json, and `Machine.step` says what each tick must give
without asking any tokenizer library. So when a runtime disagrees, we know which one is wrong, at which
tick, on which tape, and can shrink that tape to a one-tick reproduction.

Each machine is written out as several tokenizers, one per *feature* of the format, and every one runs
in every runtime available:

    plain/...           variants the format says are equivalent: merges as "a b" strings, added tokens
                        marked special or normalized, shuffled character ids, no Fuse decoder
    normalizer/...      Lowercase, NFC, NFKC, NFD, NFKD, Strip, Replace, a Sequence; the input gets noise
                        the normalizer must undo (capitals, decomposed or full-width characters, padding)
    split/...           a Split pre-tokenizer, each behaviour, several ways to write the pattern; separators
                        are sprinkled on the tape, and a rule only fires if its window is in one piece
    whitespace-split    WhitespaceSplit, with Unicode whitespace and look-alikes that are not whitespace
    byte-fallback       characters outside the vocab must survive as byte tokens
    added-literals      added tokens that do occur in the input, with lstrip and rstrip

For each feature a small model here says what one tick must give, written from the format's
documented behaviour, not from any library; tests check the models against the Rust library.

    tokenizers                       the Rust library, from Python
    transformers                     AutoTokenizer around it (its own added-token and decode handling)
    transformers.js, tokenizers.js   independent JavaScript implementations, through fuzz/runtimes.mjs

    python -m tokcpu fuzz --cases 200 --seed 1 [--node /path/to/node]
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .compile import spec as make_spec, save, tokenizer_spec
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
# Unicode White_Space: what the Rust library's char::is_whitespace calls whitespace (Strip, lstrip and
# rstrip, WhitespaceSplit). Some libraries also count the look-alikes below, which Unicode does not.
WHITESPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0          "
                       "      　")
WS_NOISE = " \t\n\r\x0b\x0c\x85\xa0      　"
NOT_WHITESPACE = "﻿\x1c\x1d\x1e\x1f​᠎"
SEPARATORS = "¶§"
FOREIGN = "Ωß€漢😀\x00́ﬁ"
LITERALS = ["⟦x⟧", "⟦yy⟧", "⟦a b⟧", "⟦⟧"]


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


# ---- features: a tokenizer.json, noise for the input, and what one tick must give ----------------

@dataclass
class Feature:
    name: str
    spec: dict
    noise: Callable[[random.Random, str], str]
    expect: Callable[[str], str | None]          # None: this input is outside the model; skip it

    def pairs(self, rng: random.Random, tapes: list[str]) -> list[list[str]]:
        out = []
        for t in tapes:
            x = self.noise(rng, t)
            want = self.expect(x)
            if want is not None:
                out.append([x, want])
        return out


def one_head(m: Machine, t: str) -> bool:
    return sum(c in m.marker_of for c in t) == 1


def step_pieces(m: Machine, pieces: list[tuple[str, bool]]) -> str:
    """One tick when the model only sees `pieces` one at a time: (text, is_tape) pairs, where the other
    pieces (separators, matched added tokens) come out as given. The head's rule applies only if its
    whole window lies inside one piece."""
    out = []
    for text, is_tape in pieces:
        if is_tape:
            for i, c in enumerate(text):
                if c in m.marker_of:
                    at = i if m.marker_of[c][1] == "R" else i - 1
                    w = text[at:at + 2] if at >= 0 else ""
                    if len(w) == 2 and w in m.rules:
                        text = text[:at] + m.rules[w] + text[at + 2:]
                    break
        out.append(text)
    return "".join(out)


def insert(rng: random.Random, t: str, chars: str, k: int) -> str:
    """Insert k characters drawn from `chars` inside the tape, between its edges."""
    for _ in range(k):
        i = rng.randint(1, len(t) - 1)
        t = t[:i] + rng.choice(chars) + t[i:]
    return t


def with_vocab(s: dict, extra: list[str]) -> dict:
    """Add vocab entries without breaking the ids' alignment. The Rust library ignores an added token's
    "id" and numbers added tokens after the model's vocab, by count; so the extra entries go in below
    the windows, and the windows and their added tokens all move up."""
    v, added = s["model"]["vocab"], s["added_tokens"]
    chars = len(v) - len(added)                  # a fresh spec: C characters, then M windows and M added
    for k in v:
        if v[k] >= chars:
            v[k] += len(extra)
    for a in added:
        a["id"] += len(extra)
    for k, e in enumerate(extra):
        v[e] = chars + k
    return s


def next_added_id(s: dict) -> int:
    """The id the Rust library gives the next added token: after the vocab and the added tokens, by count."""
    return len(s["model"]["vocab"]) + len(s["added_tokens"])


def base(m: Machine, extra_chars: str = "", pre_tokenizer: dict | None = None) -> dict:
    m.validate()
    return make_spec(sorted(set(m.alphabet()) | set(extra_chars)), m.rules, pre_tokenizer)


def plain_features(m: Machine, rng: random.Random) -> list[Feature]:
    b = tokenizer_spec(m)
    specs = {"plain": b}
    if not any(" " in w for w in m.rules):                           # the old format splits on spaces
        v = json.loads(json.dumps(b))
        v["model"]["merges"] = [f"{a} {c}" for a, c in v["model"]["merges"]]
        specs["plain/merges-as-strings"] = v
    for flag in ("special", "normalized"):
        v = json.loads(json.dumps(b))
        for t in v["added_tokens"]:
            t[flag] = True
        specs[f"plain/added-{flag}"] = v
    v = json.loads(json.dumps(b))                                     # permute the single characters' ids
    chars = [c for c in v["model"]["vocab"] if len(c) == 1]
    ids = [v["model"]["vocab"][c] for c in chars]
    rng.shuffle(ids)
    v["model"]["vocab"].update(zip(chars, ids))
    specs["plain/shuffled-char-ids"] = v
    v = json.loads(json.dumps(b))
    v["decoder"] = v["decoder"]["decoders"][0]                       # the Replace alone, no Fuse
    specs["plain/no-fuse"] = v
    same = lambda rng, t: t
    return [Feature(n, s, same, m.step) for n, s in specs.items()]


def ws_strip(s: str) -> str:
    i, j = 0, len(s)
    while i < j and s[i] in WHITESPACE:
        i += 1
    while j > i and s[j - 1] in WHITESPACE:
        j -= 1
    return s[i:j]


def normalizer_feature(m: Machine, rng: random.Random) -> Feature:
    alphabet = set(m.alphabet())
    target = rng.choice(sorted(m.symbols))
    kinds = {
        "lowercase": ({"type": "Lowercase"}, lambda s: "".join(c.lower() for c in s)),
        "nfc": ({"type": "NFC"}, lambda s: unicodedata.normalize("NFC", s)),
        "nfd": ({"type": "NFD"}, lambda s: unicodedata.normalize("NFD", s)),
        "nfkc": ({"type": "NFKC"}, lambda s: unicodedata.normalize("NFKC", s)),
        "nfkd": ({"type": "NFKD"}, lambda s: unicodedata.normalize("NFKD", s)),
        "strip": ({"type": "Strip", "strip_left": True, "strip_right": True}, ws_strip),
        "replace": ({"type": "Replace", "pattern": {"String": "@@"}, "content": target},
                    lambda s: s.replace("@@", target)),
        "nfkc+lowercase": ({"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"}]},
                           lambda s: "".join(c.lower() for c in unicodedata.normalize("NFKC", s))),
    }
    kind = rng.choice(sorted(kinds))
    config, norm = kinds[kind]
    # inputs that normalize to each tape character: capitals, decomposed, full-width, compatibility forms
    pool = set()
    for c in alphabet:
        pool |= {c.upper(), c.title(), unicodedata.normalize("NFD", c), unicodedata.normalize("NFKD", c)}
        if " " < c <= "~":
            pool.add(chr(ord(c) + 0xFEE0))
    pool |= set(WS_NOISE) | set("ａªﬁ①")
    pre = {}
    for p in pool:
        n = norm(p)
        if n in alphabet and n != p:
            pre.setdefault(n, []).append(p)
    if kind == "replace":
        pre = {target: ["@@"]}

    def noise(rng, t):
        t = "".join(rng.choice(pre[c]) if c in pre and rng.random() < 0.4 else c for c in t)
        if kind == "strip":
            t = "".join(rng.choices(WS_NOISE, k=rng.randint(0, 3))) + t + "".join(rng.choices(WS_NOISE, k=rng.randint(0, 3)))
        return t

    def expect(x):
        n = norm(x)
        return m.step(n) if set(n) <= alphabet and one_head(m, n) else None

    s = base(m)
    s["normalizer"] = config
    return Feature(f"normalizer/{kind}", s, noise, expect)


SPLIT_PATTERNS = [("string", {"String": "¶"}, "¶"), ("regex-literal", {"Regex": "¶"}, "¶"),
                  ("regex-class", {"Regex": "[¶§]"}, "¶§"), ("regex-hex", {"Regex": "\\x{B6}|\\x{A7}"}, "¶§"),
                  ("regex-plus", {"Regex": "(?:¶|§)+"}, "¶§"), ("regex-u", {"Regex": "\\u00B6"}, "¶")]
SPLIT_BEHAVIOURS = ["Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous"]


def split_feature(m: Machine, rng: random.Random) -> Feature:
    label, pattern, seps = rng.choice(SPLIT_PATTERNS)
    behaviour = rng.choice(SPLIT_BEHAVIOURS)
    pre = {"type": "Split", "pattern": pattern, "behavior": behaviour, "invert": False}
    noise = lambda rng, t: insert(rng, t, seps, rng.randint(1, 4))

    def expect(x):
        if not one_head(m, x):
            return None
        pieces, cur = [], ""
        for c in x:
            if c in seps:
                pieces += [(cur, True), ("" if behaviour == "Removed" else c, False)]
                cur = ""
            else:
                cur += c
        return step_pieces(m, pieces + [(cur, True)])

    return Feature(f"split/{behaviour}/{label}", base(m, seps, pre), noise, expect)


def whitespace_split_feature(m: Machine, rng: random.Random) -> Feature:
    extra = WS_NOISE + NOT_WHITESPACE
    noise = lambda rng, t: insert(rng, t, extra, rng.randint(1, 4))

    def expect(x):
        if not one_head(m, x):
            return None
        pieces, cur = [], ""
        for c in x:
            if c in WHITESPACE:
                pieces.append((cur, True))
                cur = ""
            else:
                cur += c
        return step_pieces(m, pieces + [(cur, True)])

    return Feature("whitespace-split", base(m, extra, {"type": "WhitespaceSplit"}), noise, expect)


def byte_fallback_feature(m: Machine, rng: random.Random) -> Feature:
    s = with_vocab(base(m), [f"<0x{b:02X}>" for b in range(256)])
    s["model"]["byte_fallback"] = True
    s["decoder"]["decoders"].insert(1, {"type": "ByteFallback"})
    foreign = "".join(c for c in FOREIGN if c not in m.alphabet())
    noise = lambda rng, t: insert(rng, t, foreign, rng.randint(1, 3))
    return Feature("byte-fallback", s, noise, lambda x: m.step(x) if one_head(m, x) else None)


def literals_feature(m: Machine, rng: random.Random) -> Feature:
    extra = WS_NOISE + NOT_WHITESPACE
    s = base(m, extra)
    lits = rng.sample(LITERALS, rng.randint(1, 3))
    nxt = next_added_id(s)
    vocab = set(s["model"]["vocab"])
    flags = {}
    for k, lit in enumerate(lits):
        f = {"lstrip": rng.random() < 0.6, "rstrip": rng.random() < 0.6, "special": rng.random() < 0.5}
        flags[lit] = f
        s["added_tokens"].append({"id": nxt + k, "content": lit, "single_word": False, "normalized": False, **f})
    pad = WS_NOISE + NOT_WHITESPACE

    def noise(rng, t):
        for _ in range(rng.randint(1, 3)):
            i = rng.randint(1, len(t) - 1)
            ins = ("".join(rng.choices(pad, k=rng.randint(0, 2))) + rng.choice(lits)
                   + "".join(rng.choices(pad, k=rng.randint(0, 2))))
            t = t[:i] + ins + t[i:]
        return t

    def expect(x):
        """Find the added tokens leftmost-longest, widen each over whitespace for lstrip (never past the
        previous match) and rstrip, then tick each stretch of tape between them separately."""
        if not one_head(m, x):
            return None
        pieces, done, i = [], 0, 0
        by_len = sorted(lits, key=len, reverse=True)
        while i < len(x):
            lit = next((l for l in by_len if x.startswith(l, i)), None)
            if lit is None:
                i += 1
                continue
            start, stop = i, i + len(lit)
            if flags[lit]["lstrip"]:
                while start > done and x[start - 1] in WHITESPACE:
                    start -= 1
            if flags[lit]["rstrip"]:
                while stop < len(x) and x[stop] in WHITESPACE:
                    stop += 1
            pieces += [(x[done:start], True), (lit, False)]
            done = i = stop
        pieces.append((x[done:], True))
        if any(c not in vocab for text, is_tape in pieces if is_tape for c in text):
            return None                     # a broken literal left characters the vocab doesn't have
        return step_pieces(m, pieces)

    return Feature("added-literals", s, noise, expect)


def features(m: Machine, rng: random.Random) -> list[Feature]:
    return [*plain_features(m, rng), normalizer_feature(m, rng), split_feature(m, rng),
            whitespace_split_feature(m, rng), byte_fallback_feature(m, rng), literals_feature(m, rng)]


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

    def run(self, pairs: list[list[str]]) -> dict:
        res = {}
        for name in self.ticks:
            res[name] = {"ok": True}
            for n, (x, want) in enumerate(pairs):
                r = self.tick(name, x)
                if "error" in r or r["out"] != want:
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

    def __init__(self, log: Path, node: str = "node"):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.node = node
        self.p = subprocess.Popen([node, str(ROOT / "fuzz" / "runtimes.mjs")], cwd=ROOT / "fuzz", text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log.open("w"),
                                  encoding="utf-8")

    def ask(self, **req) -> dict:
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def load(self, d: Path) -> dict[str, str]:
        r = self.ask(op="load", dir=str(d.resolve()))
        if "errors" not in r:
            raise RuntimeError(f"{self.node} could not load {d}: {r.get('error')}")
        return r["errors"]

    def run(self, pairs: list[list[str]]) -> dict:
        return self.ask(op="run", pairs=pairs)

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
    """Delete characters from an input while `wrong(input)` still holds. The edges and the head stay."""
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
    feature: str
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
        family = "plain" if self.feature.startswith("plain") else self.feature
        return (self.runtime, family, self.error.split(":")[0] if self.error else "wrong output", self.fresh)


def fuzz(cases: int, seed: int, out: Path, limit: int = 300, node: str = "node", log=print) -> list[Finding]:
    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    py = PyRuntimes()
    js = JsRuntimes(out / "node-stderr.log", node) if (ROOT / "fuzz" / "node_modules").exists() else None
    names = list(py.makers) + (["transformers.js@3.7.6", "transformers.js@4.3.1", "@huggingface/tokenizers@0.2.0"] if js else [])
    log(f"runtimes: {', '.join(names)}" + (f" ({js.versions()})" if js else ""))
    findings: dict[tuple, Finding] = {}
    load_errors: dict[tuple, Finding] = {}
    checked = ticks = 0
    for case in range(cases):
        m, start = random_machine(rng)
        tapes = trace(m, start, limit)
        for feat in features(m, rng):
            pairs = feat.pairs(rng, tapes)
            d = out / "work"
            if d.exists():
                shutil.rmtree(d)
            save(feat.spec, d)
            for rt in filter(None, (py, js)):
                for name, err in rt.load(d).items():
                    f = Finding(name, feat.name, case, start, "", "", error=err)
                    key = f.signature()
                    if key in load_errors:
                        load_errors[key].seen += 1
                    else:
                        load_errors[key] = f
                        f.dir = str(keep(d, out, f"load-{len(load_errors)}", {"runtime": name, "feature": feat.name, "error": err}))
                        log(f"  case {case} {feat.name}: {name} cannot load it: {err[:120]}")
                for name, r in rt.run(pairs).items():
                    checked += 1
                    ticks += r.get("tick", len(pairs))
                    if r["ok"]:
                        continue
                    n = r["tick"]

                    # A runtime can carry state from earlier tokenizers in the same process. Check the
                    # failing tick again in a fresh one; shrink there if it still fails, else keep the input.
                    fresh_rt = JsRuntimes(out / "node-stderr-fresh.log", node) if rt is js else PyRuntimes()
                    fresh_rt.load(d)

                    def wrong_fresh(x, name=name, rt=fresh_rt, feat=feat):
                        want = feat.expect(x)
                        if want is None:
                            return False
                        got = rt.tick(name, x)
                        return "error" in got or got["out"] != want
                    x = pairs[n][0]
                    fresh = wrong_fresh(x)
                    small = shrink(m, x, wrong_fresh) if fresh else x
                    got = (fresh_rt if fresh else rt).tick(name, small)
                    if isinstance(fresh_rt, JsRuntimes):
                        fresh_rt.close()
                    f = Finding(name, feat.name, case, small, feat.expect(small), got.get("out", ""),
                                got.get("error", ""), fresh=fresh)
                    key = f.signature()
                    if key in findings:
                        findings[key].seen += 1
                        if len(findings[key].examples) < 3:
                            findings[key].examples.append(small)
                        continue
                    findings[key] = f
                    f.dir = str(keep(d, out, f"finding-{len(findings)}", {
                        "runtime": name, "feature": feat.name, "input": small, "expected": f.want, "got": f.got,
                        "error": f.error, "fails_in_fresh_process": fresh}))
                    log(f"  case {case} {feat.name}: {name} disagrees"
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
