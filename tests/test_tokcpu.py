"""Every tick of every run is checked against the machine's own rules (Machine.step), and every
Brainfuck output against a plain interpreter."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokcpu import bf, small
from tokcpu.compile import save, tokenizer_spec

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = sorted((ROOT / "programs").glob("*.bf"))


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


@pytest.mark.parametrize("prog", PROGRAMS, ids=lambda p: p.stem)
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
    end, n = run_checked(m, tok, bf.universal_tape(m, src))
    assert bf.read_output(end) == bf.interpret(src)[0]


def test_errors_halt(universal):
    m, tok = universal
    end, n = run_checked(m, tok, bf.universal_tape(m, "+.<+."))     # moves left of cell 0: stops there
    assert bf.read_output(end) == b"\x01"
    with pytest.raises(ValueError):
        bf.universal_tape(m, ",")
    with pytest.raises(ValueError):
        bf.compile_program("[")


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
