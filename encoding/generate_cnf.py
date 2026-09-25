#!/usr/bin/env python3
"""
拼圖 CNF 生成器（多 puzzle、多剪枝方法）

  baseline      : 純 no-touch + each-piece-once
  shape         : 與 baseline 相同放置 + 邊角區配對死區子句（舊 shape_prune）
  shape_prime   : singleton 刪除 + 全盤配對死區（shape'，見 shape_prime.py）
  shape_prime2  : shape''（見 shape_prime2.py）
  learned_shape : AI 預篩 + formal dead-pocket oracle（見 learned_shape.py）

使用方式：
  python3 generate_cnf.py --puzzle v6 --method baseline --out cnf/v6/baseline.cnf
  python3 generate_cnf.py --puzzle v6 --method learned_shape --out cnf/v6/learned.cnf
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import deque, defaultdict
from pathlib import Path

from puzzle_defs import PUZZLES, PuzzleDef

BOARD_H = 12
BOARD_W = 12
NUM_PIECES = 11

# 由 apply_puzzle() 填入（預設 v1）
PIECES_RAW: list[list[str]] = []
PIECE_NAMES: list[str] = []
KNOWN_SOLUTION: list[list[tuple[int, int]]] = []
SOFT_PRUNE_SKIP: set[int] = set()


def apply_puzzle(name: str) -> PuzzleDef:
    """載入拼圖定義並更新模組全域變數。"""
    global PIECES_RAW, PIECE_NAMES, KNOWN_SOLUTION, SOFT_PRUNE_SKIP
    p = PUZZLES[name]
    PIECES_RAW = p.pieces_raw
    PIECE_NAMES = p.piece_names
    KNOWN_SOLUTION = p.known_solution
    SOFT_PRUNE_SKIP = set(getattr(p, "soft_prune_skip", ()))
    return p


apply_puzzle("v1")


def str_rows_to_cells(rows: list[str]) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "1"]


def normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def rotate90(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(-c, r) for r, c in cells]


def flip_h(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, -c) for r, c in cells]


def all_orientations(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    seen: set[tuple] = set()
    result: list[list[tuple[int, int]]] = []
    cur = list(cells)
    for _ in range(4):
        for variant in [cur, flip_h(cur)]:
            key = normalize(variant)
            if key not in seen:
                seen.add(key)
                mr = min(r for r, _ in variant)
                mc = min(c for _, c in variant)
                result.append([(r - mr, c - mc) for r, c in variant])
        cur = rotate90(cur)
    return result


def touches_edge(cells: list[tuple[int, int]], bh: int, bw: int) -> bool:
    return any(r == 0 or r == bh - 1 or c == 0 or c == bw - 1 for r, c in cells)


def exclusion_zone(cells: list[tuple[int, int]], bh: int, bw: int) -> set[tuple[int, int]]:
    exc: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < bh and 0 <= nc < bw:
                    exc.add((nr, nc))
    return exc


def has_dead_pocket(
    cells_a: list[tuple[int, int]],
    cells_b: list[tuple[int, int]],
    bh: int,
    bw: int,
    min_size: int = 6,
) -> bool:
    forbidden = exclusion_zone(cells_a, bh, bw) | exclusion_zone(cells_b, bh, bw)
    free = {(r, c) for r in range(bh) for c in range(bw) if (r, c) not in forbidden}

    visited: set[tuple[int, int]] = set()
    for start in free:
        if start in visited:
            continue
        comp: list[tuple[int, int]] = []
        q = deque([start])
        while q:
            cell = q.popleft()
            if cell in visited or cell not in free:
                continue
            visited.add(cell)
            comp.append(cell)
            r, c = cell
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if (nr, nc) not in visited and (nr, nc) in free:
                    q.append((nr, nc))
        if 0 < len(comp) < min_size:
            return True
    return False


def build_placements(method: str) -> tuple[list[dict], defaultdict, defaultdict]:
    """回傳 (placements, by_piece, covering)。"""
    raw_pieces = [str_rows_to_cells(r) for r in PIECES_RAW]
    all_transforms = [all_orientations(p) for p in raw_pieces]

    placements: list[dict] = []
    by_piece: defaultdict = defaultdict(list)
    covering: defaultdict = defaultdict(lambda: defaultdict(list))
    var_id = 1

    for p_idx, transforms in enumerate(all_transforms):
        for t_idx, t in enumerate(transforms):
            h = max(r for r, _ in t) + 1
            w = max(c for _, c in t) + 1
            for x in range(BOARD_H - h + 1):
                for y in range(BOARD_W - w + 1):
                    cells = [(x + r, y + c) for r, c in t]

                    if method == "soft":
                        if p_idx not in SOFT_PRUNE_SKIP:
                            if not touches_edge(cells, BOARD_H, BOARD_W):
                                continue

                    plc = {
                        "var": var_id,
                        "piece": p_idx,
                        "t_idx": t_idx,
                        "x": x,
                        "y": y,
                        "cells": cells,
                    }
                    placements.append(plc)
                    by_piece[p_idx].append(plc)
                    for r, c in cells:
                        covering[(r, c)][p_idx].append(var_id)
                    var_id += 1

    return placements, by_piece, covering


def compute_extra_shape_clauses(
    placements: list[dict],
    by_piece: defaultdict,
    *,
    edge_margin: int = 4,
    collect_dead_pairs: bool = False,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    extra: list[list[int]] = []
    dead_pairs: set[tuple[int, int]] = set()
    all_plc = placements

    exc_cache: dict[int, set] = {}
    for plc in all_plc:
        exc_cache[plc["var"]] = exclusion_zone(plc["cells"], BOARD_H, BOARD_W)

    corner_region = set()
    margin = edge_margin
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            if r < margin or r >= BOARD_H - margin or c < margin or c >= BOARD_W - margin:
                corner_region.add((r, c))

    dead_pocket_count = 0
    for plc_a, plc_b in itertools.combinations(all_plc, 2):
        if plc_a["piece"] == plc_b["piece"]:
            continue
        cells_a = plc_a["cells"]
        cells_b = plc_b["cells"]
        near_corner = any(c in corner_region for c in cells_a) or any(
            c in corner_region for c in cells_b
        )
        if not near_corner:
            continue
        exc_a = exc_cache[plc_a["var"]]
        if any(c in exc_a for c in cells_b):
            continue
        if has_dead_pocket(cells_a, cells_b, BOARD_H, BOARD_W):
            va, vb = plc_a["var"], plc_b["var"]
            extra.append([-va, -vb])
            if collect_dead_pairs:
                dead_pairs.add((min(va, vb), max(va, vb)))
            dead_pocket_count += 1

    band_count = 0
    for p_i in range(NUM_PIECES):
        for p_j in range(p_i + 1, NUM_PIECES):
            plcs_i = by_piece.get(p_i, [])
            plcs_j = by_piece.get(p_j, [])
            for pi in plcs_i:
                cells_i = pi["cells"]
                ri_min = min(r for r, _ in cells_i)
                ri_max = max(r for r, _ in cells_i)
                ci_min = min(c for _, c in cells_i)
                ci_max = max(c for _, c in cells_i)
                wi = ci_max - ci_min + 1

                for pj in plcs_j:
                    cells_j = pj["cells"]
                    rj_min = min(r for r, _ in cells_j)
                    rj_max = max(r for r, _ in cells_j)
                    cj_min = min(c for _, c in cells_j)
                    cj_max = max(c for _, c in cells_j)
                    wj = cj_max - cj_min + 1

                    row_overlap = ri_min <= rj_max and rj_min <= ri_max
                    if not row_overlap:
                        continue
                    col_gap = max(ci_min - cj_max - 1, cj_min - ci_max - 1, 0)
                    needed = wi + wj + 2
                    if needed > BOARD_W and col_gap < 2:
                        if not any(c in exc_cache.get(pi["var"], set()) for c in cells_j):
                            extra.append([-pi["var"], -pj["var"]])
                            band_count += 1

    print(f"  [shape_prune] Dead-pocket 子句: {dead_pocket_count}", file=sys.stderr)
    print(f"  [shape_prune] Band-exclusion 子句: {band_count}", file=sys.stderr)
    print(f"  [shape_prune] 共新增 {len(extra)} 條額外子句", file=sys.stderr)
    return extra, dead_pairs


def build_cnf(
    method: str,
    *,
    extra_blocks: list[list[int]] | None = None,
    block_known_solution: bool = False,
) -> tuple[int, list[list[int]], list[dict], defaultdict]:
    plc_method = "baseline" if method != "soft" else "soft"
    placements, by_piece, covering = build_placements(plc_method)

    if method == "shape_prime":
        from shape_prime import filter_singleton_placements

        placements, by_piece, covering, n_removed = filter_singleton_placements(placements)
        print(f"  [shape'] singleton 刪除 {n_removed} 個 placement", file=sys.stderr)
    elif method == "shape_prime2":
        from shape_prime2 import filter_shape_prime2_placements

        placements, by_piece, covering, n_removed = filter_shape_prime2_placements(
            placements
        )
        print(f"  [shape''] 強化 singleton 刪除 {n_removed} 個 placement", file=sys.stderr)

    for p_idx in range(NUM_PIECES):
        cnt = len(by_piece.get(p_idx, []))
        print(f"  piece {p_idx:2d} ({PIECE_NAMES[p_idx]:6s}): {cnt} placements", file=sys.stderr)

    num_plc_vars = max(p["var"] for p in placements)
    total_placements = len(placements)
    print(f"  總共 {total_placements} 個 placements（{num_plc_vars} 個變數）", file=sys.stderr)

    aux_start = num_plc_vars + 1
    next_aux = [aux_start]
    clauses: list[list[int]] = []

    for p_idx in range(NUM_PIECES):
        plcs = by_piece.get(p_idx, [])
        if not plcs:
            print(f"[ERROR] piece {p_idx} has no placements!", file=sys.stderr)
            sys.exit(1)
        v_vars = [p["var"] for p in plcs]
        clauses.append(v_vars)
        k = len(v_vars)
        if k <= 1:
            continue
        s_vars = list(range(next_aux[0], next_aux[0] + k - 1))
        next_aux[0] += k - 1
        clauses.append([-v_vars[0], s_vars[0]])
        clauses.append([-v_vars[-1], -s_vars[-1]])
        for i in range(1, k - 1):
            clauses.append([-v_vars[i], s_vars[i]])
            clauses.append([-s_vars[i - 1], s_vars[i]])
            clauses.append([-v_vars[i], -s_vars[i - 1]])

    for r, c in itertools.product(range(BOARD_H), range(BOARD_W)):
        for p_a, p_b in itertools.combinations(range(NUM_PIECES), 2):
            va_list = covering[(r, c)].get(p_a, [])
            vb_list = covering[(r, c)].get(p_b, [])
            for va in va_list:
                for vb in vb_list:
                    clauses.append([-va, -vb])

    for r, c in itertools.product(range(BOARD_H), range(BOARD_W)):
        for dr, dc in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_H and 0 <= nc < BOARD_W):
                continue
            for p_a, p_b in itertools.product(range(NUM_PIECES), range(NUM_PIECES)):
                if p_a == p_b:
                    continue
                for va in covering[(r, c)].get(p_a, []):
                    for vb in covering[(nr, nc)].get(p_b, []):
                        clauses.append([-va, -vb])

    if method == "shape":
        extra, _ = compute_extra_shape_clauses(placements, by_piece)
        clauses.extend(extra)
    elif method == "shape_prime":
        from shape_prime import compute_shape_prime_pairwise_clauses

        extra = compute_shape_prime_pairwise_clauses(placements, by_piece)
        clauses.extend(extra)
    elif method == "shape_prime2":
        from shape_prime2 import compute_shape_prime2_clauses

        extra = compute_shape_prime2_clauses(placements, by_piece)
        clauses.extend(extra)
    elif method == "learned_shape":
        import os

        from learned_shape import compute_learned_shape_clauses

        root = Path(__file__).resolve().parent.parent
        default_ranker = root / "baselines" / "learned_shape" / "ranker_loo_v1_train.joblib"
        default_rf = root / "baselines" / "learned_shape" / "model_splitB.joblib"
        model_env = os.environ.get("LEARNED_SHAPE_MODEL")
        if model_env:
            model_path = Path(model_env)
        elif default_ranker.is_file():
            model_path = default_ranker
        else:
            model_path = default_rf
        top_k = float(os.environ.get("LEARNED_SHAPE_TOP_K", "100"))
        threshold_env = os.environ.get("LEARNED_SHAPE_THRESHOLD")
        threshold = float(threshold_env) if threshold_env else None
        cascade_env = os.environ.get("LEARNED_SHAPE_CASCADE_L0")
        cascade_l0 = float(cascade_env) if cascade_env else None
        feat_v = os.environ.get("DEAD_POCKET_FEATURES", "v2")
        extra, _stats = compute_learned_shape_clauses(
            placements,
            by_piece,
            model_path=model_path,
            top_k_percent=top_k if threshold is None else None,
            score_threshold=threshold,
            edge_margin=6,
            full_dead_pairs=None,
            features_version=feat_v,
            cascade_l0_frac=cascade_l0,
        )
        clauses.extend(extra)

    if block_known_solution:
        known_sol_cells = [frozenset(cells) for cells in KNOWN_SOLUTION]
        sol_vars: list[int] = []
        for p_idx in range(NUM_PIECES):
            target = known_sol_cells[p_idx]
            found = None
            for plc in by_piece.get(p_idx, []):
                if frozenset(plc["cells"]) == target:
                    found = plc["var"]
                    break
            if found is None:
                print(f"[ERROR] 已知解 piece {p_idx} 無對應 placement", file=sys.stderr)
                sys.exit(1)
            sol_vars.append(found)
        clauses.append([-v for v in sol_vars])
        print(f"  Blocking 已知解: {[-v for v in sol_vars]}", file=sys.stderr)

    if extra_blocks:
        for bv in extra_blocks:
            clauses.append([-v for v in bv])
        print(f"  額外 blocking clauses: {len(extra_blocks)}", file=sys.stderr)

    for plc in by_piece.get(0, []):
        max_r = max(r for r, _ in plc["cells"])
        if max_r > BOARD_H // 2:
            clauses.append([-plc["var"]])

    total_vars = next_aux[0] - 1
    return total_vars, clauses, placements, by_piece


def write_cnf(out_path: Path, total_vars: int, clauses: list[list[int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"p cnf {total_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
    print(f"  寫出 {out_path}  ({total_vars} vars, {len(clauses)} clauses)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="v1")
    ap.add_argument(
        "--method",
        choices=["baseline", "shape", "shape_prime", "shape_prime2", "learned_shape"],
        required=True,
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--block-known",
        action="store_true",
        help="封鎖 puzzle_defs 中的已知解（單條，不含對稱）",
    )
    args = ap.parse_args()

    p = apply_puzzle(args.puzzle)
    print(f"\n=== 拼圖: {p.name} | 方法: {args.method} ===", file=sys.stderr)
    total_vars, clauses, _, _ = build_cnf(
        args.method, block_known_solution=args.block_known
    )
    write_cnf(args.out, total_vars, clauses)
    print(f"  完成。{total_vars} vars, {len(clauses)} clauses\n")


if __name__ == "__main__":
    main()
