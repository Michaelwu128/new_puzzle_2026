#!/usr/bin/env python3
"""解的對稱封鎖：D4 旋轉／鏡射後各加一條 blocking clause。"""

from __future__ import annotations

from typing import Callable

BOARD_H = BOARD_W = 12

D4_TRANSFORMS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    "identity": lambda r, c: (r, c),
    "90CCW": lambda r, c: (BOARD_W - 1 - c, r),
    "180": lambda r, c: (BOARD_H - 1 - r, BOARD_W - 1 - c),
    "270CCW": lambda r, c: (c, BOARD_H - 1 - r),
    "H_flip": lambda r, c: (r, BOARD_W - 1 - c),
    "V_flip": lambda r, c: (BOARD_H - 1 - r, c),
    "main_diag": lambda r, c: (c, r),
    "anti_diag": lambda r, c: (BOARD_H - 1 - c, BOARD_W - 1 - r),
}


def build_placement_index(by_piece: dict) -> dict[tuple[int, frozenset], int]:
    idx: dict[tuple[int, frozenset], int] = {}
    for p_idx, plcs in by_piece.items():
        for plc in plcs:
            key = (plc["piece"], frozenset(plc["cells"]))
            idx[key] = plc["var"]
    return idx


def piece_cells_from_known(
    known: list[list[tuple[int, int]]],
) -> dict[int, frozenset]:
    return {i: frozenset(cells) for i, cells in enumerate(known)}


def parse_kissat_true_vars(text: str) -> set[int]:
    pos: set[int] = set()
    for line in text.splitlines():
        if line.startswith("v "):
            for w in line[2:].split():
                if w == "0":
                    continue
                v = int(w)
                if v > 0:
                    pos.add(v)
    return pos


def true_vars_to_piece_cells(
    true_vars: set[int],
    var_to_plc: dict[int, dict],
) -> dict[int, frozenset]:
    result: dict[int, frozenset] = {}
    for v in true_vars:
        plc = var_to_plc.get(v)
        if plc is None:
            continue
        p = plc["piece"]
        result[p] = frozenset(plc["cells"])
    return result


def solution_to_block_vars(
    piece_cells: dict[int, frozenset],
    plc_index: dict[tuple[int, frozenset], int],
) -> list[int] | None:
    vars_out: list[int] = []
    for pid in range(len(piece_cells) if piece_cells else 11):
        if pid not in piece_cells:
            return None
        key = (pid, piece_cells[pid])
        if key not in plc_index:
            return None
        vars_out.append(plc_index[key])
    if len(vars_out) != 11:
        return None
    return sorted(vars_out)


def all_symmetry_block_clauses(
    piece_cells: dict[int, frozenset],
    plc_index: dict[tuple[int, frozenset], int],
) -> list[tuple[str, list[int]]]:
    """回傳 (變換名, blocking vars)；無法對應到 placement 的變換略過。"""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[str, list[int]]] = []
    for tname, tfn in D4_TRANSFORMS.items():
        transformed: dict[int, frozenset] = {
            pid: frozenset(tfn(r, c) for r, c in cells)
            for pid, cells in piece_cells.items()
        }
        bv = solution_to_block_vars(transformed, plc_index)
        if bv is None:
            continue
        key = tuple(bv)
        if key in seen:
            continue
        seen.add(key)
        out.append((tname, bv))
    return out


def blocking_clause_literal(vars_out: list[int]) -> list[int]:
    return [-v for v in vars_out]
