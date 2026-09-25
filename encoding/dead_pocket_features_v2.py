#!/usr/bin/env python3
"""Puzzle-invariant placement-pair 特徵（v2）：不含 piece id。"""

from __future__ import annotations

from typing import Any

from generate_cnf import BOARD_H, BOARD_W, exclusion_zone

# 不含 piece_a/b/dist
FEATURE_NAMES: list[str] = [
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
    "exc_overlap_count",
    "free_proxy",
    "near_edge_a",
    "near_edge_b",
    "near_edge_either",
    "near_margin4_a",
    "near_margin4_b",
    "near_margin4_either",
    # relative placement
    "delta_center_r",
    "delta_center_c",
    "min_cell_dist",
    "bbox_iou",
    "row_band_overlap_len",
    "col_band_overlap_len",
    # shape signature (canonical 6 cells, sorted coords flattened)
    "shape_sig_0_r",
    "shape_sig_0_c",
    "shape_sig_1_r",
    "shape_sig_1_c",
    "shape_sig_2_r",
    "shape_sig_2_c",
    "shape_sig_3_r",
    "shape_sig_3_c",
    "shape_sig_4_r",
    "shape_sig_4_c",
    "shape_sig_5_r",
    "shape_sig_5_c",
    "shape_sig_b_0_r",
    "shape_sig_b_0_c",
    "shape_sig_b_1_r",
    "shape_sig_b_1_c",
    "shape_sig_b_2_r",
    "shape_sig_b_2_c",
    "shape_sig_b_3_r",
    "shape_sig_b_3_c",
    "shape_sig_b_4_r",
    "shape_sig_b_4_c",
    "shape_sig_b_5_r",
    "shape_sig_b_5_c",
    # 4x4 coarse grid histogram (16)
    "grid_0_0",
    "grid_0_1",
    "grid_0_2",
    "grid_0_3",
    "grid_1_0",
    "grid_1_1",
    "grid_1_2",
    "grid_1_3",
    "grid_2_0",
    "grid_2_1",
    "grid_2_2",
    "grid_2_3",
    "grid_3_0",
    "grid_3_1",
    "grid_3_2",
    "grid_3_3",
    # heuristic prior components
    "heur_free_inv",
    "heur_near4",
    "heur_row_overlap",
    "heur_score",
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


def _canonical_shape_sig(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Normalize to origin, lex-min over 8 dihedral variants (sorted cell list)."""
    variants: list[list[tuple[int, int]]] = []
    base = sorted(cells)
    mr = min(r for r, _ in base)
    mc = min(c for _, c in base)
    norm = sorted((r - mr, c - mc) for r, c in base)

    def rot90(cs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted((c, -r) for r, c in cs)

    def flip_h(cs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted((r, -c) for r, c in cs)

    cur = norm
    for _ in range(2):
        for _ in range(4):
            variants.append(cur)
            cur = rot90(cur)
        cur = flip_h(norm)

    return min(variants)


def _shape_sig_flat(cells: list[tuple[int, int]], prefix: str) -> dict[str, float]:
    sig = _canonical_shape_sig(cells)
    out: dict[str, float] = {}
    for i in range(6):
        if i < len(sig):
            r, c = sig[i]
            out[f"{prefix}_{i}_r"] = float(r)
            out[f"{prefix}_{i}_c"] = float(c)
        else:
            out[f"{prefix}_{i}_r"] = 0.0
            out[f"{prefix}_{i}_c"] = 0.0
    return out


def _min_cell_dist(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> float:
    best = float(BOARD_H + BOARD_W)
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            d = abs(ra - rb) + abs(ca - cb)
            if d < best:
                best = float(d)
    return best


def _bbox_iou(r0a, r1a, c0a, c1a, r0b, r1b, c0b, c1b) -> float:
    ro0 = max(r0a, r0b)
    ro1 = min(r1a, r1b)
    co0 = max(c0a, c0b)
    co1 = min(c1a, c1b)
    if ro0 > ro1 or co0 > co1:
        return 0.0
    inter = (ro1 - ro0 + 1) * (co1 - co0 + 1)
    area_a = (r1a - r0a + 1) * (c1a - c0a + 1)
    area_b = (r1b - r0b + 1) * (c1b - c0b + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _grid_histogram(
    cells_a: list[tuple[int, int]],
    cells_b: list[tuple[int, int]],
    exc_a: set[tuple[int, int]],
    exc_b: set[tuple[int, int]],
    grid: int = 4,
) -> dict[str, float]:
    """4x4 occupancy: 0=empty, 1=piece, 2=exclusion-only."""
    gh = BOARD_H // grid
    gw = BOARD_W // grid
    if gh == 0 or gw == 0:
        gh, gw = 3, 3
    occ = [[0.0 for _ in range(grid)] for _ in range(grid)]

    def mark(cells, exc, piece_val: float, exc_val: float) -> None:
        for r, c in exc:
            gr, gc = min(r // gh, grid - 1), min(c // gw, grid - 1)
            if (r, c) in cells:
                occ[gr][gc] = max(occ[gr][gc], piece_val)
            else:
                occ[gr][gc] = max(occ[gr][gc], exc_val)

    mark(cells_a, exc_a, 1.0, 0.5)
    mark(cells_b, exc_b, 1.0, 0.5)

    out: dict[str, float] = {}
    for gr in range(grid):
        for gc in range(grid):
            out[f"grid_{gr}_{gc}"] = occ[gr][gc]
    return out


def cheap_heur_score(
    plc_a: dict[str, Any],
    plc_b: dict[str, Any],
    *,
    exc_a: set[tuple[int, int]] | None = None,
    exc_b: set[tuple[int, int]] | None = None,
) -> float:
    """L0 超便宜分數（無 shape sig / grid histogram），供 cascade 預篩。"""
    cells_a = plc_a["cells"]
    cells_b = plc_b["cells"]
    r0a, r1a, c0a, c1a = _bbox(cells_a)
    r0b, r1b, c0b, c1b = _bbox(cells_b)
    row_overlap = 1.0 if r0a <= r1b and r0b <= r1a else 0.0
    if exc_a is None:
        exc_a = exclusion_zone(cells_a, BOARD_H, BOARD_W)
    if exc_b is None:
        exc_b = exclusion_zone(cells_b, BOARD_H, BOARD_W)
    exc_overlap = len(exc_a & exc_b)
    free_proxy = BOARD_H * BOARD_W - len(exc_a) - len(exc_b) + exc_overlap
    near4 = _near_margin(cells_a, 4) or _near_margin(cells_b, 4)
    return 0.4 / max(free_proxy, 1.0) + (0.35 if near4 else 0.0) + 0.25 * row_overlap


def extract_pair_features(
    plc_a: dict[str, Any],
    plc_b: dict[str, Any],
    *,
    exc_a: set[tuple[int, int]] | None = None,
    exc_b: set[tuple[int, int]] | None = None,
) -> dict[str, float]:
    cells_a = plc_a["cells"]
    cells_b = plc_b["cells"]

    r0a, r1a, c0a, c1a = _bbox(cells_a)
    r0b, r1b, c0b, c1b = _bbox(cells_b)

    ed_a = _edge_distances(cells_a)
    ed_b = _edge_distances(cells_b)

    row_overlap = 1.0 if r0a <= r1b and r0b <= r1a else 0.0
    col_overlap = 1.0 if c0a <= c1b and c0b <= c1a else 0.0
    row_gap = float(max(r0a - r1b - 1, r0b - r1a - 1, 0))
    col_gap = float(max(c0a - c1b - 1, c0b - c1a - 1, 0))

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

    row_band_overlap_len = float(max(0, min(r1a, r1b) - max(r0a, r0b) + 1)) if row_overlap else 0.0
    col_band_overlap_len = float(max(0, min(c1a, c1b) - max(c0a, c0b) + 1)) if col_overlap else 0.0

    heur_free_inv = 1.0 / max(free_proxy, 1.0)
    heur_near4 = 1.0 if (near4_a or near4_b) else 0.0
    heur_row_overlap = row_overlap
    heur_score = 0.4 * heur_free_inv + 0.35 * heur_near4 + 0.25 * heur_row_overlap

    feat: dict[str, float] = {
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
        "row_gap": row_gap,
        "col_gap": col_gap,
        "exc_overlap_count": float(exc_overlap),
        "free_proxy": float(free_proxy),
        "near_edge_a": 1.0 if near6_a else 0.0,
        "near_edge_b": 1.0 if near6_b else 0.0,
        "near_edge_either": 1.0 if (near6_a or near6_b) else 0.0,
        "near_margin4_a": 1.0 if near4_a else 0.0,
        "near_margin4_b": 1.0 if near4_b else 0.0,
        "near_margin4_either": 1.0 if (near4_a or near4_b) else 0.0,
        "delta_center_r": crb - cra,
        "delta_center_c": ccb - cca,
        "min_cell_dist": _min_cell_dist(cells_a, cells_b),
        "bbox_iou": _bbox_iou(r0a, r1a, c0a, c1a, r0b, r1b, c0b, c1b),
        "row_band_overlap_len": row_band_overlap_len,
        "col_band_overlap_len": col_band_overlap_len,
        "heur_free_inv": heur_free_inv,
        "heur_near4": heur_near4,
        "heur_row_overlap": heur_row_overlap,
        "heur_score": heur_score,
    }
    feat.update(_shape_sig_flat(cells_a, "shape_sig"))
    feat.update(_shape_sig_flat(cells_b, "shape_sig_b"))
    feat.update(_grid_histogram(cells_a, cells_b, exc_a, exc_b))
    return feat


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [feat[n] for n in FEATURE_NAMES]
