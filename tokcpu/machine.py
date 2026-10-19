"""A machine the tokenizer can run: one head on a string tape, rewritten locally, one step per tick.

The tape is a string framed by two edge characters, LEFT and RIGHT. Exactly one character on it is a
*head*: a marker that stands for a machine state and reads one neighbour. Each state has two markers:

    an R-marker reads the character to its right:   window = marker + cell
    an L-marker reads the character to its left:    window = cell + marker

A rule maps a two-character window to any string that contains exactly one marker. That is all a
step is: find the window around the head, replace it. A head at an edge reads a blank cell, so a rule
can grow the tape there.

For ordinary Turing-machine rules, `on()` writes the window rewrites for you:

    on(state, read, write, move, next)    move: "R", "L" or "S" (stay)

placing the next head so it reads the right cell:

    head was an R-marker (window "q a"):  R -> "b q'R"   L -> "q'L b"   S -> "q'R b"
    head was an L-marker (window "a q"):  R -> "b q'R"   L -> "q'L b"   S -> "b q'L"
"""

from __future__ import annotations

from dataclasses import dataclass, field

LEFT, RIGHT = "⊢", "⊣"          # the tape's edges


@dataclass
class Machine:
    name: str
    blank: str                               # what a head reads past an edge
    symbols: set[str] = field(default_factory=set)
    rules: dict[str, str] = field(default_factory=dict)
    markers: dict[str, tuple[str, str]] = field(default_factory=dict)   # state -> (R-marker, L-marker)
    marker_of: dict[str, tuple[str, str]] = field(default_factory=dict)  # marker -> (state, side)
    marker_chars: list[str] | None = None    # optional readable characters for markers, in order
    marker_base: int = 0xAC00                # otherwise markers are allocated from here

    # ---- alphabet -------------------------------------------------------------------------
    def symbol(self, *chars: str) -> None:
        for c in chars:
            assert len(c) == 1, c
            self.symbols.add(c)

    def state(self, name: str) -> tuple[str, str]:
        if name not in self.markers:
            i = len(self.markers)
            if self.marker_chars is not None:
                r, l = self.marker_chars[2 * i], self.marker_chars[2 * i + 1]
            else:
                r, l = chr(self.marker_base + 2 * i), chr(self.marker_base + 2 * i + 1)
            self.markers[name] = (r, l)
            self.marker_of[r] = (name, "R")
            self.marker_of[l] = (name, "L")
        return self.markers[name]

    def R(self, name: str) -> str:
        return self.state(name)[0]

    def L(self, name: str) -> str:
        return self.state(name)[1]

    def alphabet(self) -> list[str]:
        chars = set(self.symbols) | {LEFT, RIGHT} | set(self.marker_of)
        return sorted(chars)

    # ---- rules ----------------------------------------------------------------------------
    def rule(self, window: str, output: str) -> None:
        """A raw rewrite: window (two characters, one of them a head) -> output (one head)."""
        assert len(window) == 2, window
        if window in self.rules and self.rules[window] != output:
            raise ValueError(f"conflicting rules for {window!r}: {self.rules[window]!r} and {output!r}")
        self.rules[window] = output

    def on(self, state: str, read: str, write: str, move: str, nxt: str, sides: str = "RL") -> None:
        """A Turing-machine rule, for each side the head may be on. `read` may be the blank: then it
        also applies at an edge, where the tape grows by one cell."""
        assert move in "RLS", move
        for side in sides:
            here = self.R(state) if side == "R" else self.L(state)
            if move == "R":
                out = write + self.R(nxt)
            elif move == "L":
                out = self.L(nxt) + write
            else:
                out = (self.R(nxt) + write) if side == "R" else (write + self.L(nxt))
            if side == "R":
                self.rule(here + read, out)
                if read == self.blank:
                    self.rule(here + RIGHT, out + RIGHT)
            else:
                self.rule(read + here, out)
                if read == self.blank:
                    self.rule(LEFT + here, LEFT + out)

    # ---- checks and the reference step ----------------------------------------------------
    def validate(self) -> None:
        alpha = set(self.alphabet())
        for w, o in self.rules.items():
            a, b = w
            if a in self.marker_of:
                assert self.marker_of[a][1] == "R" and b not in self.marker_of, f"bad window {w!r}"
            else:
                assert b in self.marker_of and self.marker_of[b][1] == "L", f"bad window {w!r}"
            heads = [c for c in o if c in self.marker_of]
            assert len(heads) == 1, f"output {o!r} of {w!r} must hold exactly one head"
            assert set(o) <= alpha and set(w) <= alpha, f"unknown character in {w!r} -> {o!r}"

    def head(self, tape: str) -> int:
        pos = [i for i, c in enumerate(tape) if c in self.marker_of]
        assert len(pos) == 1, f"tape must hold exactly one head, found {len(pos)}"
        return pos[0]

    def step(self, tape: str) -> str:
        """One tick, computed directly from the rules (the reference the tokenizer is checked against)."""
        i = self.head(tape)
        if self.marker_of[tape[i]][1] == "R":
            w, at = tape[i:i + 2], i
        else:
            w, at = tape[i - 1:i + 1], i - 1
        out = self.rules.get(w)
        return tape if out is None else tape[:at] + out + tape[at + 2:]

    def where(self, tape: str) -> tuple[str, str]:
        """The head's state and side."""
        return self.marker_of[tape[self.head(tape)]]
