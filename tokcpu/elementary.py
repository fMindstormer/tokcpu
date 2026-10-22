"""Elementary cellular automata, Rule 110 among them, as block automata: every cell updates in the same
tick.

An elementary automaton is a row of cells, each 0 or 1. Every generation, each cell's new value is a
fixed function of itself and its two neighbours, given by the rule's number: bit 4l + 2v + r of the
number is the new value of a cell v between l and r. Rule 110 is Turing complete (Cook, 2004).

A cell needs both neighbours, but a block holds two cells, so a generation takes two ticks, one in each
alignment of the blocks (see blocks.py):

    tick 1, blocks (x0 x1)(x2 x3)...   each cell learns the value of the other cell in its block
    tick 2, blocks (x1 x2)(x3 x4)...   each cell now knows its other neighbour too, and updates

Between the two ticks a cell is one of eight in-between symbols: a-d for a cell that knows its left
neighbour, e-h for one that knows its right one.

The tape grows by one cell at each edge every generation, as fast as anything can spread, so it always
holds every cell that can differ from the background. The background is a pattern that repeats forever
on each side: all zeros by default. Cook's construction needs other patterns: a finite middle between a
pattern repeating to the left and one repeating to the right. A spatially periodic row stays periodic,
and comes back to itself after some generations, so each edge has finitely many states: which
generation of the background it is in, and where it is along the pattern. An edge with one state is
the plain ⊢ or ⊣; more states use ① .. ⑳ on the left and ⑴ .. ⒇ on the right. The edges do their work
in tick 2, where ⊢ is blocked with the first cell and ⊣ with the last: the tape always holds an even
number of cells, so those blocks always line up.
"""

from __future__ import annotations

from typing import Callable

from .blocks import MARK, SPLIT
from .compile import spec
from .machine import LEFT, RIGHT

PLAIN = "·█"              # a cell between generations, 0 and 1
KNOWS_LEFT = "abcd"       # after tick 1: (left, value) -> KNOWS_LEFT[2 * left + value]
KNOWS_RIGHT = "efgh"      #               (value, right) -> KNOWS_RIGHT[2 * value + right]
LEFTS = [chr(0x2460 + k) for k in range(20)]       # ① .. ⑳
RIGHTS = [chr(0x2474 + k) for k in range(20)]      # ⑴ .. ⒇
ETHER = "11111000100110"  # Rule 110's ether: the background its gliders travel on
COLLISION = "00110011001110001000010111111010001011111110101010011001"   # random cells; on the ether they
                                                                        # settle into two gliders

Rule = Callable[[int, int, int], int]


def table(number: int) -> Rule:
    assert 0 <= number < 256
    return lambda l, v, r: number >> (4 * l + 2 * v + r) & 1


def edge(f: Rule, word: str, step: int) -> list[tuple[int, int, int]]:
    """The states of one edge, starting state first: (the background's value just outside the tape,
    the value one further out, the next state). `step` is -1 for the left edge, +1 for the right.

    The background repeats `word` forever: to the left of the tape its last character is next to the
    first cell, to the right its first character is next to the last cell. A state is (which generation
    of the background, the position of the cell at the edge, modulo the pattern's length)."""
    p = len(word)
    gens, seen = [], {}
    row = tuple(int(c) for c in word)
    while row not in seen:
        seen[row] = len(gens)
        gens.append(row)
        row = tuple(f(row[i - 1], row[i], row[(i + 1) % p]) for i in range(p))
    nxt = lambda t, x: (t + 1 if t + 1 < len(gens) else seen[row], (x + step) % p)
    order, index = [], {}
    s = (0, 0 if step < 0 else -1 % p)
    while s not in index:
        index[s] = len(order)
        order.append(s)
        s = nxt(*s)
    return [(gens[t][(x + step) % p], gens[t][(x + 2 * step) % p], index[nxt(t, x)]) for t, x in order]


def edges(number: int, left: str, right: str) -> tuple[list[str], list[str]]:
    f = table(number)
    n, m = len(edge(f, left, -1)), len(edge(f, right, +1))
    if max(n, m) > len(LEFTS):
        raise ValueError(f"the background needs {max(n, m)} states at an edge; at most {len(LEFTS)} fit")
    return [LEFT] if n == 1 else LEFTS[:n], [RIGHT] if m == 1 else RIGHTS[:m]


def rules(number: int, left: str = "0", right: str = "0") -> dict[str, str]:
    f = table(number)
    lefts, rights = edges(number, left, right)
    out = {}
    for a in (0, 1):
        for b in (0, 1):
            out[PLAIN[a] + PLAIN[b]] = KNOWS_RIGHT[2 * a + b] + KNOWS_LEFT[2 * a + b]   # tick 1
            for c in (0, 1):
                for d in (0, 1):                                                        # tick 2
                    out[KNOWS_LEFT[2 * a + b] + KNOWS_RIGHT[2 * c + d]] = PLAIN[f(a, b, c)] + PLAIN[f(b, c, d)]
    for k, (near, far, nxt) in enumerate(edge(f, left, -1)):
        out[lefts[k] + MARK] = lefts[k]
        for v in (0, 1):
            for r in (0, 1):     # the first cell updates, and a new one appears to its left
                out[lefts[k] + KNOWS_RIGHT[2 * v + r]] = lefts[nxt] + MARK + PLAIN[f(far, near, v)] + PLAIN[f(near, v, r)]
    for k, (near, far, nxt) in enumerate(edge(f, right, +1)):
        for l in (0, 1):
            for v in (0, 1):     # the last cell updates, and a new one appears to its right
                out[KNOWS_LEFT[2 * l + v] + rights[k]] = PLAIN[f(l, v, near)] + PLAIN[f(v, near, far)] + rights[nxt]
    return out


def automaton(number: int, cells: str, left: str = "0", right: str = "0") -> tuple[dict, dict[str, str], str]:
    """(tokenizer spec, rules, starting tape). `cells`, `left` and `right` are strings of 0 and 1; the
    middle needs an even number of cells, at least two."""
    if not cells or len(cells) % 2 or set(cells + left + right) - {"0", "1"} or not left or not right:
        raise ValueError("the cells are an even number of 0s and 1s, at least two, and each background at least one")
    r = rules(number, left, right)
    lefts, rights = edges(number, left, right)
    chars = sorted(set(PLAIN + KNOWS_LEFT + KNOWS_RIGHT + MARK) | set(lefts) | set(rights))
    tape = lefts[0] + MARK + "".join(PLAIN[int(c)] for c in cells) + rights[0]
    return spec(chars, r, pre_tokenizer=SPLIT), r, tape


def generation(tape: str) -> str | None:
    """The row of 0s and 1s on a tape between generations (after tick 2), else None."""
    if len(tape) < 2 or tape[1] != MARK:
        return None
    return "".join(str(PLAIN.index(c)) for c in tape[2:-1])


def run(tok, tape: str, gens: int) -> list[str]:
    """Tick a tokenizer (from the `tokenizers` library) for `gens` generations; every generation's row."""
    rows = [generation(tape)]
    for _ in range(2 * gens):
        tape = tok.decode(tok.encode(tape).ids)
        if (row := generation(tape)) is not None:
            rows.append(row)
    return rows


def ether_windows(number: int = 110, ether: str = ETHER, width: int = 6) -> set[str]:
    """Every run of `width` cells found in the ether, at any generation. For pictures only: a cell
    inside such a run is drawn as background, so the gliders stand out."""
    f, row, seen, out = table(number), ether, set(), set()
    while row not in seen:
        seen.add(row)
        out |= {(row * 2)[i:i + width] for i in range(len(row))}
        row = "".join(str(f(int(row[i - 1]), int(row[i]), int(row[(i + 1) % len(row)]))) for i in range(len(row)))
    return out


def png(rows: list[str], path, shade: set[str] | None = None, scale: int = 2) -> None:
    """A picture of the rows, generation 0 at the top, each row one cell wider on each side than the
    last. With `shade`, cells inside those runs (see ether_windows) are pale and the rest orange."""
    import struct
    import zlib
    ink, paper, out = (0x13, 0x1C, 0x33), (0xFF, 0xFF, 0xFF), (0xED, 0xF1, 0xF7)
    pale, hot, warm = (0xCF, 0xD5, 0xE0), (0xE0, 0x54, 0x0D), (0xFB, 0xD5, 0xC0)
    gens, width = len(rows) - 1, len(rows[-1])
    k = len(next(iter(shade))) if shade else 0
    lines = []
    for g, row in enumerate(rows):
        bg = [False] * len(row)
        for i in range(len(row) - k + 1 if shade else 0):
            if row[i:i + k] in shade:
                bg[i:i + k] = [True] * k
        px = [out] * (gens - g) + [
            ((pale if c == "1" else paper) if bg[i] else (hot if c == "1" else warm)) if shade else (ink if c == "1" else paper)
            for i, c in enumerate(row)] + [out] * (gens - g)
        line = b"\0" + b"".join(bytes(p) * scale for p in px)
        lines += [line] * scale
    raw = zlib.compress(b"".join(lines), 9)
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    head = struct.pack(">IIBBBBB", width * scale, len(rows) * scale, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def evolve(number: int, cells: str, left: str, right: str, gens: int) -> list[str]:
    """The reference: the same automaton computed directly, on a row wide enough that its ends can't
    reach the cells the tape holds. Row g covers the tape's cells after g generations."""
    f, w, n = table(number), 2 * gens, len(cells)
    row = [int(left[x % len(left)]) for x in range(-w, 0)] + [int(c) for c in cells] + [
        int(right[x % len(right)]) for x in range(w)]
    out = []
    for g in range(gens + 1):        # the row covers [g - w, n + w - g); the tape covers [-g, n + g)
        out.append("".join(map(str, row[w - 2 * g: w + n])))
        row = [f(row[i - 1], row[i], row[i + 1]) for i in range(1, len(row) - 1)]
    return out
