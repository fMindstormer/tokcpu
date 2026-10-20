"""Every tick of every run is checked against the machine's own rules (Machine.step), and every
Brainfuck output against a plain interpreter."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokcpu import bf, blocks, small
from tokcpu.compile import save, tokenizer_spec

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = sorted((ROOT / "programs").glob("*.bf"))
COMPILED = [p for p in PROGRAMS if "," not in p.read_text()]     # rung 2 has no input
INPUT = b"stressed"


def tok_of(m):
    return Tokenizer.from_str(json.dumps(tokenizer_spec(m), ensure_ascii=False))


def run_checked(m, tok, tape, limit=10 ** 7, tick=None):
    tick = tick or (lambda t: tok.decode(tok.encode(t).ids))
    n = 0
    while n < limit:
        want, got = m.step(tape), tick(tape)
        assert got == want, f"tick {n}: {tape!r} -> {got!r}, expected {want!r}"
        if got == tape:
            return tape, n
        tape, n = got, n + 1
    return tape, n


def test_busy_beaver_halts_with_its_record():
    m, tape = small.busy_beaver_4()
    end, n = run_checked(m, tok_of(m), tape)
    assert n == 107 and end.count("1") == 13


def test_two_heads_tick_together():
    """Heads that never meet each get their rule applied in the same tick."""
    m, _ = small.busy_beaver_4()
    tok = tok_of(m)
    half = "0" * 30 + m.R("A") + "0" * 30
    tape, n = "⊢" + half + half + "⊣", 0
    while (nxt := tok.decode(tok.encode(tape).ids)) != tape:
        heads = [i for i, c in enumerate(tape) if c in m.marker_of]
        want = tape
        for i in reversed(heads):                    # every head steps at once
            at = i if m.marker_of[tape[i]][1] == "R" else i - 1
            want = want[:at] + m.rules.get(tape[at:at + 2], tape[at:at + 2]) + want[at + 2:]
        assert nxt == want, n
        tape, n = nxt, n + 1
    assert n == 107 and tape.count("1") == 26


def run_blocks(tok, rules, tape):
    """Tick until the tape repeats every two ticks; returns (tape, ticks)."""
    n = 0
    while True:
        a = tok.decode(tok.encode(tape).ids)
        assert a == blocks.step(rules, tape), n
        b = tok.decode(tok.encode(a).ids)
        assert b == blocks.step(rules, a), n
        if b == tape:
            return tape, n
        tape, n = b, n + 2


def test_block_sort():
    spec, rules, start = blocks.sorter()
    tok = Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))
    for s in ["", "a", "ba", blocks.LETTERS[::-1], blocks.cells(start)] + [
            "".join(sorted(blocks.LETTERS * 2, key=lambda c, k=k: hash((c, k)))) for k in range(5)]:
        end, n = run_blocks(tok, rules, blocks.tape(s))
        assert blocks.cells(end) == "".join(sorted(s)) and n <= len(s) + 1, (s, n)


def test_counter_counts():
    m, tape = small.counter()
    tok = tok_of(m)
    seen = []
    for _ in range(3000):
        nxt = tok.decode(tok.encode(tape).ids)
        assert nxt == m.step(tape)
        if m.where(nxt)[0] == "ret" and m.where(tape)[0] == "inc":
            seen.append(small.counter_value(nxt))
        tape = nxt
    assert seen == list(range(1, len(seen) + 1)) and len(seen) > 300


def test_survives_save_and_reload():
    m = bf.universal()
    tok = tok_of(m)
    again = Tokenizer.from_str(tok.to_str())             # the library's own serialization
    tape = bf.universal_tape(m, (ROOT / "programs" / "alphabet.bf").read_text())
    for _ in range(2000):
        a, b = tok.decode(tok.encode(tape).ids), again.decode(again.encode(tape).ids)
        assert a == b == m.step(tape)
        tape = a


def test_autotokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    m, tape = small.busy_beaver_4()
    save(m, tmp_path / "bb")
    at = transformers.AutoTokenizer.from_pretrained(str(tmp_path / "bb"))
    end, n = run_checked(m, None, tape, tick=lambda t: at.decode(at.encode(t)))
    assert n == 107


@pytest.mark.parametrize("prog", COMPILED, ids=lambda p: p.stem)
def test_compiled_program(prog):
    src = prog.read_text()
    m, tape = bf.compile_program(src)
    end, n = run_checked(m, tok_of(m), tape)
    assert bf.read_output(end) == bf.interpret(src)[0]


@pytest.fixture(scope="module")
def universal():
    m = bf.universal()
    return m, tok_of(m)


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.stem)
def test_universal(universal, prog):
    m, tok = universal
    src = prog.read_text()
    end, n = run_checked(m, tok, bf.universal_tape(m, src, INPUT))     # every program gets input
    assert bf.read_output(end) == bf.interpret(src, INPUT)[0]


@pytest.mark.parametrize("src, data", [
    (",[.,]", b"cat me"),                      # cat: the end of the input reads 0 and stops it
    (",.,.,.", b"a"),                          # reading past the end keeps giving 0
    (",.", b""),                               # no input at all
    (">>>,.", b"z"),                           # new cells grow in front of the input
    (",+.,-.", bytes([255, 0])),               # every byte value survives the two trips
    (",[.,]", bytes(range(256))),
])
def test_universal_input(universal, src, data):
    m, tok = universal
    end, n = run_checked(m, tok, bf.universal_tape(m, src, data))
    assert bf.read_output(end) == bf.interpret(src, data)[0]


def test_errors_halt(universal):
    m, tok = universal
    end, n = run_checked(m, tok, bf.universal_tape(m, "+.<+."))     # moves left of cell 0: stops there
    assert bf.read_output(end) == b"\x01"
    with pytest.raises(ValueError):
        bf.compile_program(",")
    with pytest.raises(ValueError):
        bf.compile_program("[")


def test_page_compiler_matches(tmp_path):
    """The page's compiler (web/tokcpu.js) writes the same tokenizer.json as Python, byte for byte."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    rng = random.Random(1)
    tables = [(small.BUSY_BEAVER_4, "0")]
    for _ in range(20):
        syms, states = rng.sample('01#$x"\\ab', 4), "ABCD"
        table = {(q, a): (rng.choice(syms), rng.choice("RLS"), rng.choice(states + "H"))
                 for q in states for a in syms if rng.random() < 0.8}
        table[("A", syms[0])] = (syms[1], "R", "B")
        tables.append((dict(sorted(table.items(), key=lambda kv: kv[0][0] != "A")), syms[0]))
    for table, blank in tables:
        m, tape = small.turing(table, "t", blank=blank)
        (tmp_path / "t.txt").write_text("".join(f"{q} {a} {b} {d} {n}\n" for (q, a), (b, d, n) in table.items()))
        out = subprocess.run(["node", str(ROOT / "tests" / "js_compile.mjs"), str(tmp_path / "t.txt"), blank],
                             capture_output=True, text=True, check=True).stdout.split("\n")
        assert out[0] == json.dumps(tokenizer_spec(m), ensure_ascii=False, separators=(",", ":"))
        assert out[1] == tape


def test_fuzz_models_agree_with_rust():
    """The fuzzer's models of each feature (equivalent variants, normalizers, pre-tokenizers, byte
    fallback, added tokens that match the input) agree with the Rust library, so a runtime that
    disagrees with them is the one that is wrong."""
    from tokcpu import fuzz
    rng = random.Random(2)
    seen = set()
    for _ in range(60):
        m, start = fuzz.random_machine(rng)
        tapes = fuzz.trace(m, start, 100)
        for feat in fuzz.features(m, rng):
            tok = Tokenizer.from_str(json.dumps(feat.spec, ensure_ascii=False))
            for x, want in feat.pairs(rng, tapes):
                assert tok.decode(tok.encode(x).ids, skip_special_tokens=False) == want, (feat.name, x)
                seen.add(feat.name.split("/")[0])
    assert seen == {"plain", "normalizer", "split", "whitespace-split", "byte-fallback", "added-literals"}


def test_transformers_js(tmp_path):
    """The same tokenizer in transformers.js gives the same ticks (needs node and
    `npm install @huggingface/transformers` in tests/)."""
    if not shutil.which("node") or not (ROOT / "tests" / "node_modules" / "@huggingface" / "transformers").exists():
        pytest.skip("transformers.js not installed")
    m, tape = bf.compile_program((ROOT / "programs" / "hello.bf").read_text())
    save(m, tmp_path / "hello", tape)
    trace = [tape]
    while (nxt := m.step(trace[-1])) != trace[-1]:
        trace.append(nxt)
    (tmp_path / "trace.txt").write_text("\n".join(trace))
    out = subprocess.run(["node", str(ROOT / "tests" / "js_ticks.mjs"), str(tmp_path / "hello"), str(tmp_path / "trace.txt")],
                         capture_output=True, text=True, check=True, cwd=ROOT / "tests").stdout
    res = json.loads(out.strip().splitlines()[-1])
    assert res["firstMismatch"] == -1 and res["ticks"] == len(trace) - 1


def test_compat_finds_final_sigma(tmp_path):
    """`check` flags a real JavaScript difference against both Python references, and nothing on plain
    text: a WordPiece tokenizer that lowercases, where JavaScript's toLowerCase turns a word-final Σ into ς
    and Python gives σ (needs node and `npm install` in fuzz/)."""
    if not shutil.which("node") or not (ROOT / "fuzz" / "node_modules" / "tj4").exists():
        pytest.skip("fuzz/ runtimes not installed")
    from tokenizers import models, normalizers, pre_tokenizers
    from tokcpu import compat
    vocab = {w: i for i, w in enumerate(["[UNK]", "[CLS]", "[SEP]", "σοφοσ", "σοφος", "the", "dog", "."])}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}))
    old = compat.TEXTS
    compat.TEXTS = {"english": ("everyday", "The dog."), "greek": ("languages", "ΣΟΦΟΣ")}
    js = compat.Js()
    try:
        r = compat.check_dir(tmp_path, js)
    finally:
        js.close()
        compat.TEXTS = old
    assert r["references"] == {"tokenizers": "ok", "transformers": "ok"}
    for rt in compat.RUNTIMES:
        diffs = r["runtimes"][rt]["differences"]
        assert set(diffs) == {"greek"}, rt
        assert diffs["greek"]["verdict"] == "both" and diffs["greek"]["cause"] == "final sigma (toLowerCase)"
        assert diffs["greek"]["python"] == ["σοφοσ"] and diffs["greek"]["js"] == ["σοφος"]
