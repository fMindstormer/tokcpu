"""Parallel ticks: a block cellular automaton, where one tick rewrites the whole tape at once.

A stock `Split` pre-tokenizer cuts the tape into two-character blocks before BPE runs, and every block
is a merge. So one encode turns the whole tape into blocks, and one decode rewrites every block.

The blocks alternate between two alignments (the Margolus neighbourhood), so information can cross
block boundaries. The alignment is set by a phase mark ◦ after the left edge:

    ⊢◦ x0 x1 x2 x3 ...  ⊣      blocks (⊢◦)(x0 x1)(x2 x3)...
    ⊢  x0 x1 x2 x3 ...  ⊣      blocks (⊢x0)(x1 x2)(x3 x4)...

The left edge block adds or drops the mark, so the phase flips every tick. The right edge needs no rule:
⊣ ends up alone or after a cell, and either way decodes to itself.

Because the mark flips every tick, the tape never stops changing. A run stops when the tape comes back
the same after two ticks: one tick in each alignment, which changed nothing but the mark.

The demo is odd-even transposition sort: each block puts its two letters in order, and n letters are
sorted after at most n ticks, with every comparison of a tick done in the same encode and decode.
"""

from __future__ import annotations

from typing import Callable

from .compile import spec
from .machine import LEFT, RIGHT

MARK = "◦"
SPLIT = {"type": "Split", "pattern": {"Regex": ".."}, "behavior": "Isolated", "invert": False}


def block_rules(alphabet: str, f: Callable[[str, str], str]) -> dict[str, str]:
    """Every pair of cells -> f(pair), plus the left edge's rules that flip the phase."""
    rules = {x + y: f(x, y) for x in alphabet for y in alphabet}
    for x in alphabet + RIGHT:                  # RIGHT: an empty tape flips too
        rules[LEFT + x] = LEFT + MARK + x
    rules[LEFT + MARK] = LEFT
    return rules


def tokenizer_spec(alphabet: str, rules: dict[str, str]) -> dict:
    return spec(sorted(set(alphabet) | {LEFT, RIGHT, MARK}), rules, pre_tokenizer=SPLIT)


def step(rules: dict[str, str], tape: str) -> str:
    """One tick, computed directly: cut into blocks from the left, rewrite each one that has a rule."""
    return "".join(rules.get(tape[i:i + 2], tape[i:i + 2]) for i in range(0, len(tape), 2))


LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sorter() -> tuple[dict, dict[str, str], str]:
    """Odd-even transposition sort over a-z: (tokenizer spec, rules, a starting tape)."""
    rules = block_rules(LETTERS, lambda x, y: min(x, y) + max(x, y))
    return tokenizer_spec(LETTERS, rules), rules, tape("thequickbrownfoxjumpsoverthelazydog")


def tape(cells: str) -> str:
    return LEFT + MARK + cells + RIGHT


def cells(t: str) -> str:
    return t.replace(LEFT, "").replace(MARK, "").replace(RIGHT, "")
