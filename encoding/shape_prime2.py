#!/usr/bin/env python3
"""
shape''（shape_prime2）：在 shape' 上再加強。

  1. 強化 singleton：只放一塊時，若有**任何** <6 格 4-連通空腔（含貼邊）→ 刪除 placement
  2. 全盤二元死區（同 shape'）
  3. 三元死區：三塊不同 piece、中心距 ≤5、兩兩不衝突，聯合產生 <6 空腔
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import defaultdict

from generate_cnf import (
    BOARD_H,
    BOARD_W,
    exclusion_zone,
    compute_extra_shape_clauses,
)

MIN_COMPONENT = 6
CENTER_DIST = 5
BOARD_CELLS = BOARD_H * BOARD_W
ALL_MASK = (1 << BOARD_CELLS) - 1

# 4-鄰居（位元索引）
_NEIGH: list[int] = []
for _r in range(BOARD_H):
    for _c in range(BOARD_W):
        i = _r * BOARD_W + _c
        m = 0
        if _r:
            m |= 1 << (i - BOARD_W)
        if _r + 1 < BOARD_H:
            m |= 1 << (i + BOARD_W)
        if _c:
            m |= 1 << (i - 1)
        if _c + 1 < BOARD_W:
            m |= 1 << (i + 1)
        _NEIGH.append(m)


def _cells_mask(cells: list[tuple[int, int]]) -> int:
    m = 0
    for r, c in cells:
        m |= 1 << (r * BOARD_W + c)
    return m


def _exc_mask(cells: list[tuple[int, int]]) -> int:
    return _cells_mask(
        list(exclusion_zone(cells, BOARD_H, BOARD_W))
    )


def _smallest_component_size(free_mask: int) -> int | None:
    """free_mask 上最小 4-連通分量大小；無空格子則 None。"""
    if not free_mask:
        return None
    visited = 0
    smallest: int | None = None
    remaining = free_mask
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        if bit & visited:
            continue
        comp = bit
        stack = [bit]
        visited |= bit
        size = 0
        while stack:
            b = stack.pop()
            size += 1
            frontier = b
            while frontier:
                cell = frontier & -frontier
                frontier ^= cell
                nb = _NEIGH[cell.bit_length() - 1] & free_mask & ~visited
                if nb:
                    visited |= nb
                    comp |= nb
                    x = nb
                    while x:
                        stack.append(x & -x)
                        x ^= x & -x
        if size and (smallest is None or size < smallest):
            smallest = size
    return smallest


def _component_masks(free_mask: int) -> list[tuple[int, int]]:
    """回傳 [(component_mask, size), ...]。"""
    out: list[tuple[int, int]] = []
    visited = 0
    remaining = free_mask
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        if bit & visited:
            continue
        comp = bit
        stack = [bit]
        visited |= bit
        size = 0
        while stack:
            b = stack.pop()
            size += 1
            frontier = b
            while frontier:
                cell = frontier & -frontier
                frontier ^= cell
                nb = _NEIGH[cell.bit_length() - 1] & free_mask & ~visited
                if nb:
                    visited |= nb
                    comp |= nb
                    x = nb
                    while x:
                        stack.append(x & -x)
                        x ^= x & -x
        out.append((comp, size))
    return out


def _dilate_mask(mask: int, radius: int = 2) -> int:
    m = mask
    for _ in range(radius):
        grown = 0
        x = m
        while x:
            b = x & -x
            x ^= b
            grown |= _NEIGH[b.bit_length() - 1]
        m |= grown
    return m


def placement_has_small_pocket_when_alone(cells: list[tuple[int, int]]) -> bool:
    forbidden = _exc_mask(cells)
    free = ALL_MASK & ~forbidden
    sm = _smallest_component_size(free)
    return sm is not None and sm < MIN_COMPONENT


def filter_shape_prime2_placements(
    placements: list[dict],
) -> tuple[list[dict], defaultdict, defaultdict, int]:
    kept = [p for p in placements if not placement_has_small_pocket_when_alone(p["cells"])]
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


def _center(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return (
        sum(r for r, _ in cells) // len(cells),
        sum(c for _, c in cells) // len(cells),
    )


def _centers_close(ca: tuple[int, int], cb: tuple[int, int]) -> bool:
    return max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) <= CENTER_DIST


def _index_by_center(plcs: list[dict]) -> dict[tuple[int, int], list[dict]]:
    idx: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for plc in plcs:
        idx[_center(plc["cells"])].append(plc)
    return idx


def _nearby_from_index(
    idx: dict[tuple[int, int], list[dict]],
    center: tuple[int, int],
) -> list[dict]:
    cr, cc = center
    out: list[dict] = []
    for dr in range(-CENTER_DIST, CENTER_DIST + 1):
        for dc in range(-CENTER_DIST, CENTER_DIST + 1):
            out.extend(idx.get((cr + dr, cc + dc), ()))
    return out


def _build_placement_masks(
    placements: list[dict],
) -> tuple[dict[int, int], dict[int, int], dict[int, tuple[int, int]], dict[int, int]]:
    exc: dict[int, int] = {}
    cells: dict[int, int] = {}
    centers: dict[int, tuple[int, int]] = {}
    piece_of: dict[int, int] = {}
    for plc in placements:
        v = plc["var"]
        exc[v] = _exc_mask(plc["cells"])
        cells[v] = _cells_mask(plc["cells"])
        centers[v] = _center(plc["cells"])
        piece_of[v] = plc["piece"]
    return exc, cells, centers, piece_of


def _build_exc_index(placements: list[dict], exc: dict[int, int]) -> list[list[int]]:
    """每個棋格 → 其 exc 覆蓋該格的 placement var 列表。"""
    cell_vars: list[list[int]] = [[] for _ in range(BOARD_CELLS)]
    for plc in placements:
        v = plc["var"]
        m = exc[v]
        while m:
            b = m & -m
            m ^= b
            cell_vars[b.bit_length() - 1].append(v)
    return cell_vars


def _vars_near_mask(region: int, cell_vars: list[list[int]]) -> set[int]:
    seen: set[int] = set()
    m = region
    while m:
        b = m & -m
        m ^= b
        for v in cell_vars[b.bit_length() - 1]:
            seen.add(v)
    return seen


def compute_triple_dead_pocket_clauses(
    placements: list[dict],
    by_piece: defaultdict,
    dead_pairs: set[tuple[int, int]],
) -> list[list[int]]:
    exc, cell_mask, centers, piece_of = _build_placement_masks(placements)
    cell_vars = _build_exc_index(placements, exc)
    center_idx = {p: _index_by_center(by_piece[p]) for p in by_piece}
    piece_ids = sorted(by_piece.keys())

    extra: list[list[int]] = []
    seen_clause: set[tuple[int, int, int]] = set()
    checked = 0
    t0 = time.perf_counter()
    t_report = t0
    pair_loops = 0

    for p_i, p_j, p_k in itertools.combinations(piece_ids, 3):
        plcs_i = by_piece[p_i]
        idx_j = center_idx[p_j]
        for plc_a in plcs_i:
            va = plc_a["var"]
            ca = centers[va]
            exc_a = exc[va]
            cm_a = cell_mask[va]
            for plc_b in _nearby_from_index(idx_j, ca):
                vb = plc_b["var"]
                if piece_of[vb] == piece_of[va]:
                    continue
                pair = (min(va, vb), max(va, vb))
                if pair in dead_pairs:
                    continue
                exc_b = exc[vb]
                cm_b = cell_mask[vb]
                if exc_a & cm_b or exc_b & cm_a:
                    continue

                fab = exc_a | exc_b
                free_ab = ALL_MASK & ~fab
                comps = _component_masks(free_ab)
                if any(0 < sz < MIN_COMPONENT for _, sz in comps):
                    continue

                # 只有「夠小」的空腔才可能被第三塊壓成 <6
                max_cut = 40
                region = 0
                for comp_m, sz in comps:
                    if sz <= MIN_COMPONENT + max_cut:
                        region |= comp_m
                if not region:
                    continue
                region = _dilate_mask(region, 2)
                cand_vars = _vars_near_mask(region, cell_vars)

                cb = centers[vb]
                for vc in cand_vars:
                    if piece_of[vc] != p_k:
                        continue
                    if not _centers_close(ca, centers[vc]) or not _centers_close(cb, centers[vc]):
                        continue
                    exc_c = exc[vc]
                    cm_c = cell_mask[vc]
                    if exc_a & cm_c or exc_c & cm_a or exc_b & cm_c or exc_c & cm_b:
                        continue
                    if not (exc_c & region):
                        continue

                    checked += 1
                    fabc = fab | exc_c
                    sm = _smallest_component_size(ALL_MASK & ~fabc)
                    if sm is not None and sm < MIN_COMPONENT:
                        key = tuple(sorted((va, vb, vc)))
                        if key in seen_clause:
                            continue
                        seen_clause.add(key)
                        extra.append([-va, -vb, -vc])

                pair_loops += 1
                now = time.perf_counter()
                if now - t_report >= 10.0:
                    print(
                        f"  [shape''] 三元進度：pair {pair_loops}，"
                        f"檢查 {checked}，子句 {len(extra)}，{now - t0:.0f}s",
                        file=sys.stderr,
                    )
                    t_report = now

    print(
        f"  [shape''] 三元配對檢查 {checked} 組，死區子句 {len(extra)}，"
        f"耗時 {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return extra


def compute_shape_prime2_clauses(
    placements: list[dict],
    by_piece: defaultdict,
) -> list[list[int]]:
    print("  [shape''] 全盤二元死區…", file=sys.stderr)
    extra, dead_pairs = compute_extra_shape_clauses(
        placements, by_piece, edge_margin=6, collect_dead_pairs=True
    )
    print("  [shape''] 局部三元死區…", file=sys.stderr)
    extra.extend(compute_triple_dead_pocket_clauses(placements, by_piece, dead_pairs))
    print(f"  [shape''] 共 {len(extra)} 條額外子句", file=sys.stderr)
    return extra
