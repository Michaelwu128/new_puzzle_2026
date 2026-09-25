#!/usr/bin/env python3
"""Conflict-free placement pair 枚舉（與 shape' 全盤 dead-pocket 一致）。"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterator

from generate_cnf import BOARD_H, BOARD_W, NUM_PIECES, build_placements, exclusion_zone


def build_exc_cache(placements: list[dict]) -> dict[int, set[tuple[int, int]]]:
    return {plc["var"]: exclusion_zone(plc["cells"], BOARD_H, BOARD_W) for plc in placements}


def pair_passes_filters(
    plc_a: dict,
    plc_b: dict,
    exc_cache: dict[int, set[tuple[int, int]]],
    *,
    edge_margin: int = 6,
) -> bool:
    if plc_a["piece"] == plc_b["piece"]:
        return False
    cells_a = plc_a["cells"]
    cells_b = plc_b["cells"]
    if edge_margin < BOARD_H:
        corner_region = {
            (r, c)
            for r in range(BOARD_H)
            for c in range(BOARD_W)
            if r < edge_margin
            or r >= BOARD_H - edge_margin
            or c < edge_margin
            or c >= BOARD_W - edge_margin
        }
        near = any(c in corner_region for c in cells_a) or any(
            c in corner_region for c in cells_b
        )
        if not near:
            return False
    exc_a = exc_cache[plc_a["var"]]
    if any(c in exc_a for c in cells_b):
        return False
    return True


def iter_candidate_pairs(
    placements: list[dict] | None = None,
    by_piece: defaultdict | None = None,
    *,
    edge_margin: int = 6,
) -> Iterator[tuple[dict, dict]]:
    if placements is None or by_piece is None:
        placements, by_piece, _ = build_placements("baseline")
    exc_cache = build_exc_cache(placements)

    for p_i in range(NUM_PIECES):
        for p_j in range(p_i + 1, NUM_PIECES):
            for plc_a in by_piece.get(p_i, []):
                for plc_b in by_piece.get(p_j, []):
                    if pair_passes_filters(plc_a, plc_b, exc_cache, edge_margin=edge_margin):
                        yield plc_a, plc_b


def stratify_bucket(plc_a: dict, plc_b: dict) -> str:
    """random / near_edge / band_overlap"""
    cells_a = plc_a["cells"]
    cells_b = plc_b["cells"]
    r0a, r1a, c0a, c1a = (
        min(r for r, _ in cells_a),
        max(r for r, _ in cells_a),
        min(c for _, c in cells_a),
        max(c for _, c in cells_a),
    )
    r0b, r1b, c0b, c1b = (
        min(r for r, _ in cells_b),
        max(r for r, _ in cells_b),
        min(c for _, c in cells_b),
        max(c for _, c in cells_b),
    )
    margin = 4
    near = False
    for r, c in cells_a + cells_b:
        if r < margin or r >= BOARD_H - margin or c < margin or c >= BOARD_W - margin:
            near = True
            break
    row_overlap = r0a <= r1b and r0b <= r1a
    col_overlap = c0a <= c1b and c0b <= c1a
    if near:
        return "near_edge"
    if row_overlap or col_overlap:
        return "band_overlap"
    return "random"


def count_candidate_pairs(
    placements: list[dict] | None = None,
    by_piece: defaultdict | None = None,
    *,
    edge_margin: int = 6,
) -> int:
    if placements is None or by_piece is None:
        placements, by_piece, _ = build_placements("baseline")
    exc_cache = build_exc_cache(placements)
    n = 0
    for plc_a, plc_b in itertools.combinations(placements, 2):
        if pair_passes_filters(plc_a, plc_b, exc_cache, edge_margin=edge_margin):
            n += 1
    return n
