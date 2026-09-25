#!/usr/bin/env python3
"""
shape'（shape_prime）：公平、不針對特定 piece 的幾何剪枝。

相對於 shape（舊版）：
  1. Singleton — 內部封閉死區：只放一塊時，若出現「不貼棋盤邊」且 <6 格的
     4-連通空腔 → 刪除該 placement（不會誤刪 v2 已知解）
  2. 全盤配對死區：pairwise dead-pocket 在**整個 12×12** 上檢查
     （shape 只在邊角 margin=4 內檢查，會漏掉中央組合）
"""

from __future__ import annotations

import sys
from collections import defaultdict

from generate_cnf import (
    BOARD_H,
    BOARD_W,
    exclusion_zone,
    compute_extra_shape_clauses,
)

MIN_COMPONENT = 6


def filter_singleton_placements(
    placements: list[dict],
) -> tuple[list[dict], defaultdict, defaultdict, int]:
    """刪除會造成內部封閉小空腔的 placement，並重新編號。"""
    kept = [
        p
        for p in placements
        if not placement_has_enclosed_dead_pocket(p["cells"])
    ]
    removed = len(placements) - len(kept)

    by_piece: defaultdict = defaultdict(list)
    covering: defaultdict = defaultdict(lambda: defaultdict(list))
    var_id = 1
    renumbered: list[dict] = []

    for plc in kept:
        new_plc = {**plc, "var": var_id}
        renumbered.append(new_plc)
        by_piece[plc["piece"]].append(new_plc)
        for r, c in new_plc["cells"]:
            covering[(r, c)][plc["piece"]].append(var_id)
        var_id += 1

    return renumbered, by_piece, covering, removed


def placement_has_enclosed_dead_pocket(cells: list[tuple[int, int]]) -> bool:
    forbidden = exclusion_zone(cells, BOARD_H, BOARD_W)
    free = {
        (r, c)
        for r in range(BOARD_H)
        for c in range(BOARD_W)
        if (r, c) not in forbidden
    }
    visited: set[tuple[int, int]] = set()
    for start in free:
        if start in visited:
            continue
        comp: list[tuple[int, int]] = []
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in visited or cell not in free:
                continue
            visited.add(cell)
            comp.append(cell)
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in free and (nr, nc) not in visited:
                    stack.append((nr, nc))
        if 0 < len(comp) < MIN_COMPONENT:
            on_edge = any(
                r == 0 or r == BOARD_H - 1 or c == 0 or c == BOARD_W - 1
                for r, c in comp
            )
            if not on_edge:
                return True
    return False


def compute_shape_prime_pairwise_clauses(
    placements: list[dict],
    by_piece: defaultdict,
) -> list[list[int]]:
    """全盤 dead-pocket（edge_margin 覆蓋整個棋盤）。"""
    print("  [shape'] 全盤配對死區檢測（edge_margin=6）…", file=sys.stderr)
    extra, _ = compute_extra_shape_clauses(
        placements, by_piece, edge_margin=6
    )
    return extra
