"""Brainfuck in a tokenizer.

Rung 2, `compile_program`: one program becomes one tokenizer. The program counter lives in the
machine's state, so each Brainfuck instruction is one tick; printing walks the byte to the output.

    tape:  ⊢ OUTPUT ┃ DATA ⊣        the head sits at the data pointer

Rung 3, `universal`: one tokenizer for every program. The program is written on the tape and the head
shuttles between the instruction pointer (▼, before the current instruction) and the data pointer
(▲, before the current cell).

    tape:  ⊢ OUTPUT ┃ PROGRAM ║ DATA ⊣

Data cells are Braille characters: U+2800 + value, so the dots show the byte's bits. Output bytes are
written as themselves when printable ASCII, as control pictures (␊ for newline) below 32, ␡ for 127,
and as U+0100 + (byte - 128) above.
"""

from __future__ import annotations

from .machine import LEFT, RIGHT, Machine

OPS = "+-<>[].,"
OUT_END, PROG_END = "┃", "║"       # ┃ after the output, ║ after the program
IP, DP = "▼", "▲"                  # ▼ instruction pointer, ▲ data pointer (rung 3)


def cell(v: int) -> str:
    return chr(0x2800 + (v % 256))


def cell_value(c: str) -> int:
    return ord(c) - 0x2800


def out_char(v: int) -> str:
    if 32 <= v < 127:
        return chr(v)
    if v < 32:
        return chr(0x2400 + v)
    if v == 127:
        return "␡"
    return chr(0x0100 + v - 128)


def out_byte(c: str) -> int:
    o = ord(c)
    if 32 <= o < 127:
        return o
    if 0x2400 <= o < 0x2420:
        return o - 0x2400
    if o == 0x2421:
        return 127
    return o - 0x0100 + 128


def read_output(tape: str) -> bytes:
    """The bytes printed so far: everything between the left edge and ┃."""
    return bytes(out_byte(c) for c in tape[tape.index(LEFT) + 1:tape.index(OUT_END)])


def clean(program: str) -> str:
    return "".join(c for c in program if c in OPS)


def brackets(program: str) -> dict[int, int]:
    stack, pair = [], {}
    for i, c in enumerate(program):
        if c == "[":
            stack.append(i)
        elif c == "]":
            if not stack:
                raise ValueError(f"unmatched ] at {i}")
            j = stack.pop()
            pair[i], pair[j] = j, i
    if stack:
        raise ValueError(f"unmatched [ at {stack[-1]}")
    return pair


def interpret(program: str, max_steps: int = 10 ** 8) -> tuple[bytes, int]:
    """The reference: a plain Brainfuck interpreter. Returns (output, instructions executed)."""
    p = clean(program)
    pair, mem, dp, ip, out, n = brackets(p), [0], 0, 0, bytearray(), 0
    while ip < len(p) and n < max_steps:
        c = p[ip]
        if c == "+":
            mem[dp] = (mem[dp] + 1) % 256
        elif c == "-":
            mem[dp] = (mem[dp] - 1) % 256
        elif c == ">":
            dp += 1
            if dp == len(mem):
                mem.append(0)
        elif c == "<":
            if dp == 0:
                raise ValueError("moved left of cell 0")
            dp -= 1
        elif c == ".":
            out.append(mem[dp])
        elif c == ",":
            raise ValueError("input (,) is not supported")
        elif c == "[" and mem[dp] == 0:
            ip = pair[ip]
        elif c == "]" and mem[dp] != 0:
            ip = pair[ip]
        ip += 1
        n += 1
    return bytes(out), n


# ---------------------------------------------------------------------------------------------
# Rung 2: a program compiled into the machine's states
# ---------------------------------------------------------------------------------------------

def compile_program(program: str) -> tuple[Machine, str]:
    p = clean(program)
    if "," in p:
        raise ValueError("input (,) is not supported")
    pair = brackets(p)
    n = len(p)
    m = Machine(f"brainfuck program ({n} instructions)", blank=cell(0))
    m.symbol(OUT_END, *(cell(v) for v in range(256)), *(out_char(v) for v in range(256)))
    P = [f"P{i}" for i in range(n + 1)]                 # P{n}: past the end, no rules: halt
    for q in P:
        m.state(q)
    prints = [i for i, c in enumerate(p) if c == "."]
    mark = {i: chr(0xA000 + k) for k, i in enumerate(prints)}   # print site i's return mark
    m.symbol(*mark.values())

    # which side the head can be on at each instruction (it starts as an R-marker)
    sides = [set() for _ in range(n + 1)]
    sides[0].add("R")
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(p):
            nxt = []
            if c in "+-":
                nxt = [(i + 1, s) for s in sides[i]]
            elif c == ">":
                nxt = [(i + 1, "R")] if sides[i] else []
            elif c == "<":
                nxt = [(i + 1, "L")] if sides[i] else []
            elif c == "[":
                nxt = [(t, s) for s in sides[i] for t in (i + 1, pair[i] + 1)]
            elif c == "]":
                nxt = [(t, s) for s in sides[i] for t in (i + 1, pair[i] + 1)]
            elif c == ".":
                nxt = [(i + 1, "R")] if sides[i] else []
            for t, s in nxt:
                if s not in sides[t]:
                    sides[t].add(s)
                    changed = True

    for i, c in enumerate(p):
        sd = "".join(sorted(sides[i]))
        if not sd:
            continue
        for v in range(256):
            a = cell(v)
            if c == "+":
                m.on(P[i], a, cell(v + 1), "S", P[i + 1], sides=sd)
            elif c == "-":
                m.on(P[i], a, cell(v - 1), "S", P[i + 1], sides=sd)
            elif c == ">":
                m.on(P[i], a, a, "R", P[i + 1], sides=sd)
            elif c == "<":
                m.on(P[i], a, a, "L", P[i + 1], sides=sd)
            elif c == "[":
                m.on(P[i], a, a, "S", P[pair[i] + 1] if v == 0 else P[i + 1], sides=sd)
            elif c == "]":
                m.on(P[i], a, a, "S", P[pair[i] + 1] if v != 0 else P[i + 1], sides=sd)
            elif c == ".":
                # leave this print site's mark before the cell and carry the byte left
                out = m.L(f"C{v}") + mark[i] + a
                for s in sd:
                    if s == "R":
                        m.rule(m.R(P[i]) + a, out)
                        if v == 0:
                            m.rule(m.R(P[i]) + RIGHT, out + RIGHT)
                    else:
                        m.rule(a + m.L(P[i]), out)
    if prints:
        for v in range(256):                            # carry byte v left over the data
            C = m.L(f"C{v}")
            for u in range(256):
                m.rule(cell(u) + C, C + cell(u))
            m.rule(OUT_END + C, out_char(v) + OUT_END + m.R("B"))     # print it, walk back
        for u in range(256):
            m.rule(m.R("B") + cell(u), cell(u) + m.R("B"))
        for i in prints:
            m.rule(m.R("B") + mark[i], m.R(P[i + 1]))   # found the mark: remove it, carry on
    return m, LEFT + OUT_END + m.R(P[0]) + RIGHT


# ---------------------------------------------------------------------------------------------
# Rung 3: one universal interpreter; the program is on the tape
# ---------------------------------------------------------------------------------------------

UOPS = "+-<>[]."            # input (,) is not supported
DEPTH = 64                   # deepest loop nesting the interpreter can match


def universal() -> Machine:
    """Fetch: F reads the instruction after ▼ and G carries it right to ▲. Execute: E changes the cell,
    moves ▲, or starts a print or a jump. Return: BK walks back to ▼ and ADV moves ▼ one instruction on.
    Jumps scan with the loop depth held in the state (D going right, U going left). A print carries the
    byte (C) left to the output, then FD walks back to ▼."""
    m = Machine("universal brainfuck interpreter", blank=cell(0))
    cells = [cell(v) for v in range(256)]
    m.symbol(OUT_END, PROG_END, IP, DP, *cells, *(out_char(v) for v in range(256)), *UOPS)
    R, L = m.R, m.L
    m.state("F")
    for X in UOPS:                                       # fetch
        m.rule(R("F") + X, X + R(f"G{X}"))
        G = R(f"G{X}")
        for c in list(UOPS) + [PROG_END] + cells:        # carry the instruction right to ▲
            m.rule(G + c, c + G)
        m.rule(G + DP, R(f"E{X}"))                       # drop ▲ and execute on the cell after it
    for v in range(256):                                 # execute
        a = cells[v]
        m.rule(R("E+") + a, L("BK") + DP + cell(v + 1))
        m.rule(R("E-") + a, L("BK") + DP + cell(v - 1))
        m.rule(R("E>") + a, a + R("N"))
        m.rule(R("E<") + a, L("P") + a)
        m.rule(R("E.") + a, L(f"C{v}") + DP + a)
        m.rule(R("E[") + a, (L("BKJ") if v == 0 else L("BK")) + DP + a)
        m.rule(R("E]") + a, (L("BKB") if v != 0 else L("BK")) + DP + a)
        m.rule(R("N") + a, L("BK") + DP + a)             # > : mark the next cell...
        m.rule(a + L("P"), L("BK") + DP + a)             # < : ...or the previous one
    m.rule(R("N") + RIGHT, L("BK") + DP + cells[0] + RIGHT)   # > past the end: a new zero cell
    for back in ("BK", "BKJ", "BKB"):                    # walk back to ▼
        for c in list(UOPS) + [PROG_END] + cells:
            m.rule(c + L(back), L(back) + c)
    m.rule(IP + L("BK"), R("ADV"))                       # next instruction: lift ▼ ...
    for X in UOPS:
        m.rule(R("ADV") + X, X + IP + R("F"))            # ... and set it down one further on
    m.rule(IP + L("BKJ"), R("JF"))                       # [ on zero: find the matching ]
    m.rule(R("JF") + "[", "[" + R("D1"))
    for d in range(1, DEPTH + 1):
        D = R(f"D{d}")
        for X in UOPS:
            if X == "[":
                if d < DEPTH:
                    m.rule(D + X, X + R(f"D{d + 1}"))
            elif X == "]":
                m.rule(D + X, X + (IP + R("F") if d == 1 else R(f"D{d - 1}")))
            else:
                m.rule(D + X, X + D)
    m.rule(IP + L("BKB"), L("U1"))                       # ] on non-zero: back to the matching [
    for d in range(1, DEPTH + 1):
        U = L(f"U{d}")
        for X in UOPS:
            if X == "]":
                if d < DEPTH:
                    m.rule(X + U, L(f"U{d + 1}") + X)
            elif X == "[":
                m.rule(X + U, X + IP + R("F") if d == 1 else L(f"U{d - 1}") + X)
            else:
                m.rule(X + U, U + X)
    for v in range(256):                                 # print: carry the byte left to ┃
        C = L(f"C{v}")
        for c in list(UOPS) + [PROG_END, IP] + cells:
            m.rule(c + C, C + c)
        m.rule(OUT_END + C, out_char(v) + OUT_END + R("FD"))
    for X in UOPS:                                       # and walk back to ▼
        m.rule(R("FD") + X, X + R("FD"))
    m.rule(R("FD") + IP, R("ADV"))
    return m


def universal_tape(m: Machine, program: str) -> str:
    p = clean(program)
    if "," in p:
        raise ValueError("input (,) is not supported")
    brackets(p)
    depth = max((p[:i + 1].count("[") - p[:i + 1].count("]") for i in range(len(p))), default=0)
    if depth > DEPTH:
        raise ValueError(f"loops nest {depth} deep; the interpreter matches up to {DEPTH}")
    return LEFT + OUT_END + IP + m.R("F") + p + PROG_END + DP + cell(0) + RIGHT
