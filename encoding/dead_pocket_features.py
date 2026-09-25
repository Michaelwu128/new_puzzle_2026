#!/usr/bin/env python3
"""Placement-pair 幾何特徵（不呼叫 has_dead_pocket BFS）。"""

from __future__ import annotations

from typing import Any

from generate_cnf import BOARD_H, BOARD_W, exclusion_zone

FEATURE_NAMES: list[str] = [
    "piece_a",
    "piece_b",
    "piece_dist",
    "bbox_h_a",
    "bbox_w_a",
    "bbox_h_b",
    "bbox_w_b",
    "area_a",
    "area_b",
    "center_r_a",
    "center_c_a",
    "center_r_b",
    "center_c_b",
    "min_edge_dist_a",
    "min_edge_dist_b",
    "max_edge_dist_a",
    "max_edge_dist_b",
    "mean_edge_dist_a",
    "mean_edge_dist_b",
    "row_overlap",
    "col_overlap",
    "row_gap",
    "col_gap",
    "bbox_row_gap",
    "bbox_col_gap",
    "exc_overlap_count",
    "free_proxy",
    "near_edge_a",
    "near_edge_b",
    "near_edge_either",
    "near_margin4_a",
    "near_margin4_b",
    "near_margin4_either",
]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _edge_distances(cells: list[tuple[int, int]]) -> list[int]:
    return [
        min(r, BOARD_H - 1 - r, c, BOARD_W - 1 - c)
        for r, c in cells
    ]


def _near_margin(cells: list[tuple[int, int]], margin: int) -> bool:
    for r, c in cells:
        if r < margin or r >= BOARD_H - margin or c < margin or c >= BOARD_W - margin:
            return True
    return False


def extract_pair_features(
    plc_a: dict[str, Any],
    plc_b: dict[str, Any],
    *,
    exc_a: set[tuple[int, int]] | None = None,
    exc_b: set[tuple[int, int]] | None = None,
) -> dict[str, float]:
    cells_a = plc_a["cells"]
    cells_b = plc_b["cells"]
    pa, pb = plc_a["piece"], plc_b["piece"]

    r0a, r1a, c0a, c1a = _bbox(cells_a)
    r0b, r1b, c0b, c1b = _bbox(cells_b)

    ed_a = _edge_distances(cells_a)
    ed_b = _edge_distances(cells_b)

    row_overlap = 1.0 if r0a <= r1b and r0b <= r1a else 0.0
    col_overlap = 1.0 if c0a <= c1b and c0b <= c1a else 0.0

    row_gap = max(r0a - r1b - 1, r0b - r1a - 1, 0)
    col_gap = max(c0a - c1b - 1, c0b - c1a - 1, 0)

    if exc_a is None:
        exc_a = exclusion_zone(cells_a, BOARD_H, BOARD_W)
    if exc_b is None:
        exc_b = exclusion_zone(cells_b, BOARD_H, BOARD_W)
    exc_overlap = len(exc_a & exc_b)

    board_cells = BOARD_H * BOARD_W
    free_proxy = board_cells - len(exc_a) - len(exc_b) + exc_overlap

    cra = sum(r for r, _ in cells_a) / len(cells_a)
    cca = sum(c for _, c in cells_a) / len(cells_a)
    crb = sum(r for r, _ in cells_b) / len(cells_b)
    ccb = sum(c for _, c in cells_b) / len(cells_b)

    near4_a = _near_margin(cells_a, 4)
    near4_b = _near_margin(cells_b, 4)
    near6_a = _near_margin(cells_a, 6)
    near6_b = _near_margin(cells_b, 6)

    return {
        "piece_a": float(pa),
        "piece_b": float(pb),
        "piece_dist": float(abs(pa - pb)),
        "bbox_h_a": float(r1a - r0a + 1),
        "bbox_w_a": float(c1a - c0a + 1),
        "bbox_h_b": float(r1b - r0b + 1),
        "bbox_w_b": float(c1b - c0b + 1),
        "area_a": float(len(cells_a)),
        "area_b": float(len(cells_b)),
        "center_r_a": cra,
        "center_c_a": cca,
        "center_r_b": crb,
        "center_c_b": ccb,
        "min_edge_dist_a": float(min(ed_a)),
        "min_edge_dist_b": float(min(ed_b)),
        "max_edge_dist_a": float(max(ed_a)),
        "max_edge_dist_b": float(max(ed_b)),
        "mean_edge_dist_a": float(sum(ed_a) / len(ed_a)),
        "mean_edge_dist_b": float(sum(ed_b) / len(ed_b)),
        "row_overlap": row_overlap,
        "col_overlap": col_overlap,
        "row_gap": float(row_gap),
        "col_gap": float(col_gap),
        "bbox_row_gap": float(row_gap),
        "bbox_col_gap": float(col_gap),
        "exc_overlap_count": float(exc_overlap),
        "free_proxy": float(free_proxy),
        "near_edge_a": 1.0 if near6_a else 0.0,
        "near_edge_b": 1.0 if near6_b else 0.0,
        "near_edge_either": 1.0 if (near6_a or near6_b) else 0.0,
        "near_margin4_a": 1.0 if near4_a else 0.0,
        "near_margin4_b": 1.0 if near4_b else 0.0,
        "near_margin4_either": 1.0 if (near4_a or near4_b) else 0.0,
    }


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [feat[name] for name in FEATURE_NAMES]
