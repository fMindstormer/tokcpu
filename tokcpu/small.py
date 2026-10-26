"""Rung 1: small Turing machines."""

from __future__ import annotations

from .machine import LEFT, RIGHT, Machine

# readable heads: an R-marker is a circled small letter, an L-marker the circled capital
CIRCLED = [c for i in range(26) for c in (chr(0x24D0 + i), chr(0x24B6 + i))]


def counter() -> tuple[Machine, str]:
    """A binary counter that counts forever. `ret` walks right to the '#' sentinel and turns;
    `inc` walks left adding one, carrying through 1s, and grows the number at the left edge."""
    m = Machine("binary counter", blank="0", marker_chars=CIRCLED)
    m.symbol("0", "1", "#")
    m.state("inc"), m.state("ret")
    m.on("inc", "1", "0", "L", "inc", sides="L")
    m.on("inc", "0", "1", "R", "ret", sides="L")        # also past the left edge: the number grows
    for d in "01":
        m.on("ret", d, d, "R", "ret", sides="R")
    m.on("ret", "#", "#", "L", "inc", sides="R")
    return m, LEFT + m.R("ret") + "0#" + RIGHT


BUSY_BEAVER_4 = {   # Brady's 4-state, 2-symbol champion: 107 steps, 13 ones
    ("A", "0"): ("1", "R", "B"), ("A", "1"): ("1", "L", "B"),
    ("B", "0"): ("1", "L", "A"), ("B", "1"): ("0", "L", "C"),
    ("C", "0"): ("1", "R", "H"), ("C", "1"): ("1", "L", "D"),
    ("D", "0"): ("1", "R", "D"), ("D", "1"): ("0", "R", "A"),
}

INCREMENT = {       # adds one to a binary number, with blank _: the page's guided example
    ("A", "0"): ("0", "R", "A"), ("A", "1"): ("1", "R", "A"),     # A walks to the end of the number
    ("A", "_"): ("_", "L", "B"),                                  # and turns back as B
    ("B", "1"): ("0", "L", "B"),                                  # B carries: 1 becomes 0
    ("B", "0"): ("1", "S", "C"), ("B", "_"): ("1", "S", "C"),     # 0 or a blank becomes 1, and C halts
}


def turing(table: dict, name: str, start: str = "A", blank: str = "0") -> tuple[Machine, str]:
    """Any Turing machine given as {(state, read): (write, move, next)}; a state without rules halts."""
    m = Machine(name, blank=blank, marker_chars=CIRCLED)
    states = sorted({q for q, _ in table} | {n for _, _, n in table.values()}, key=lambda q: (q != start, q))
    for q in states:
        m.state(q)
    for (q, a), (b, d, n) in table.items():
        m.symbol(a, b)
        m.on(q, a, b, d, n)
    m.symbol(blank)
    return m, LEFT + m.R(start) + RIGHT


def busy_beaver_4() -> tuple[Machine, str]:
    return turing(BUSY_BEAVER_4, "4-state busy beaver")


def counter_value(tape: str) -> int:
    digits = "".join(c for c in tape if c in "01")
    return int(digits, 2) if digits else 0
