"""Compile a Machine into a stock Hugging Face tokenizer.json, and run it.

One tick is `decode(encode(tape))`:

- encode: the BPE model has one merge per rule window ("head" + the cell it reads), so the only
  multi-character token that can form is the window around the head. Every other character is a
  token of its own.
- decode: each window's id is also the id of an *added token* whose text is that rule's output.
  Decoding prints the output where the window was, and every other character as itself.

Two rules can have the same output, but two added tokens cannot have the same text. So each added
token ends in a short tag of private-use characters that makes it unique, and the decoder's one
generic rule deletes those characters. The tags never reach the tape, so an added token can never
match tape text when encoding.

The ids line up because the library numbers added tokens after the model's vocabulary, in the order
they are listed: with C single characters and M windows, window k gets id C + M + k, and so does the
k-th added token. Ids C .. C + M - 1 are left unused.
"""

from __future__ import annotations

import json
from pathlib import Path

from .machine import Machine

TAG0, TAGS = 0xE000, 64                      # tag characters: U+E000 .. U+E03F
TAG_CLASS = f"[{chr(TAG0)}-{chr(TAG0 + TAGS - 1)}]"


def tag(i: int) -> str:
    """A unique, non-empty tag for the i-th output that shares its text (i >= 1)."""
    assert i >= 1
    out = ""
    while i:
        i, d = divmod(i, TAGS)
        out = chr(TAG0 + d) + out
    return out


def tokenizer_spec(m: Machine) -> dict:
    m.validate()
    return spec(m.alphabet(), m.rules)


def spec(chars: list[str], rules: dict[str, str], pre_tokenizer: dict | None = None) -> dict:
    """tokenizer.json for any table of two-character windows -> outputs over these characters."""
    assert not any(TAG0 <= ord(c) < TAG0 + TAGS for c in chars), "tape characters may not be tag characters"
    windows = sorted(rules)
    C, M = len(chars), len(windows)
    vocab = {c: i for i, c in enumerate(chars)}
    seen: dict[str, int] = {}
    added = []
    for k, w in enumerate(windows):
        out = rules[w]
        seen[out] = seen.get(out, 0) + 1
        vocab[w] = C + M + k
        added.append({"id": C + M + k, "content": out + tag(seen[out]), "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False, "special": False})
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added,
        "normalizer": None, "pre_tokenizer": pre_tokenizer, "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"Regex": TAG_CLASS}, "content": ""},
            {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [[w[0], w[1]] for w in windows]},
    }


def save(m: Machine | dict, directory: str | Path, start: str | None = None) -> Path:
    """Write tokenizer.json (from a Machine, or a ready spec), a tokenizer_config.json (so AutoTokenizer
    can load the folder) and, if given, the machine's starting tape as start.txt."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if start is not None:
        (d / "start.txt").write_text(start)
    s = m if isinstance(m, dict) else tokenizer_spec(m)
    (d / "tokenizer.json").write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "clean_up_tokenization_spaces": False,
        "model_max_length": 1_000_000_000}))
    return d / "tokenizer.json"


def load(path: str | Path):
    from tokenizers import Tokenizer
    p = Path(path)
    return Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))


def tick(tok, tape: str) -> str:
    return tok.decode(tok.encode(tape).ids)


def run(tok, tape: str, max_ticks: int = 10 ** 9, every: int = 0, show=None):
    """Tick until the tape stops changing (the machine halted). Returns (tape, ticks)."""
    n = 0
    while n < max_ticks:
        nxt = tok.decode(tok.encode(tape).ids)
        if nxt == tape:
            return tape, n
        tape, n = nxt, n + 1
        if every and show and n % every == 0:
            show(n, tape)
    return tape, n
