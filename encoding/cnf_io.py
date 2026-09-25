#!/usr/bin/env python3
"""CNF 檔案讀寫與追加 blocking clause。"""

from __future__ import annotations
from pathlib import Path


def read_cnf_header(path: Path) -> tuple[int, int]:
    with open(path, encoding="ascii") as f:
        for line in f:
            if line.startswith("p cnf "):
                parts = line.split()
                return int(parts[2]), int(parts[3])
    raise ValueError(f"no cnf header in {path}")


def append_blocking_clauses(path: Path, blocks: list[list[int]]) -> None:
    if not blocks:
        return
    num_vars, num_clauses = read_cnf_header(path)
    with open(path, "a", encoding="ascii") as f:
        for bv in blocks:
            clause = " ".join(str(-abs(v)) if v > 0 else str(v) for v in bv)
            f.write(clause + " 0\n")
    text = path.read_text(encoding="ascii")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("p cnf "):
            lines[i] = f"p cnf {num_vars} {num_clauses + len(blocks)}\n"
            break
    path.write_text("".join(lines), encoding="ascii")
