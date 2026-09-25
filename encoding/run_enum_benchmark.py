#!/usr/bin/env python3
"""
連續找 N 個解並比較 baseline vs shape vs learned_shape（無 soft / 無 piece 豁免）。

流程：
  1. 產生 base CNF（僅約束 + piece0 對稱破）
  2. 先封鎖「已知解」的所有有效 D4 對稱變換
  3. 重複 N 次：kissat → 解碼 → 封鎖該解的 D4 對稱 → 記錄時間
  4. 輸出平均時間、各解 JSON、驗證結果

用法：
  python3 run_enum_benchmark.py --puzzle v6 --count 5 \\
    --methods baseline shape learned_shape
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from cnf_io import append_blocking_clauses
from kissat_utils import KISSAT_DEFAULT, resolve_kissat
from generate_cnf import (
    BOARD_H,
    BOARD_W,
    NUM_PIECES,
    apply_puzzle,
    build_cnf,
    write_cnf,
)
from puzzle_defs import PUZZLES
from solution_blocking import (
    all_symmetry_block_clauses,
    build_placement_index,
    parse_kissat_true_vars,
    piece_cells_from_known,
    true_vars_to_piece_cells,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEARNED_MODEL = ROOT / "baselines" / "learned_shape" / "ranker_train_v1_v2_v3.joblib"

METHOD_CHOICES = ["baseline", "shape", "shape_prime", "shape_prime2", "learned_shape"]


@contextmanager
def env_patch(updates: dict[str, str | None]):
    saved: dict[str, str | None] = {}
    for k, v in updates.items():
        saved[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def learned_env(model: Path, top_k: float, cascade_l0: float) -> dict[str, str]:
    return {
        "LEARNED_SHAPE_MODEL": str(model),
        "LEARNED_SHAPE_TOP_K": str(top_k),
        "LEARNED_SHAPE_CASCADE_L0": str(cascade_l0),
        "DEAD_POCKET_FEATURES": "v2",
    }


def verify_no_touch(piece_cells: dict[int, frozenset]) -> bool:
    for pi, ci in piece_cells.items():
        exc: set[tuple[int, int]] = set()
        for r, c in ci:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    exc.add((r + dr, c + dc))
        for pj, cj in piece_cells.items():
            if pj <= pi:
                continue
            if any(cell in exc for cell in cj):
                return False
    return len(piece_cells) == NUM_PIECES and sum(len(c) for c in piece_cells.values()) == 66


def board_to_rows(piece_cells: dict[int, frozenset]) -> list[str]:
    grid = [["." for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    for p, cells in piece_cells.items():
        sym = str(p + 1) if p < 9 else chr(ord("A") + p - 9)
        for r, c in cells:
            grid[r][c] = sym
    return [" ".join(row) for row in grid]


def piece_cells_to_json(piece_cells: dict[int, frozenset]) -> dict:
    return {
        "piece_cells": {str(p): [list(c) for c in sorted(cells)] for p, cells in piece_cells.items()},
        "board": board_to_rows(piece_cells),
        "valid": verify_no_touch(piece_cells),
    }


def run_kissat(kissat: Path, cnf: Path, sol_out: Path) -> float:
    t0 = time.perf_counter()
    with open(sol_out, "w", encoding="latin-1") as f:
        proc = subprocess.run(
            [str(kissat), str(cnf)],
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    text = sol_out.read_text(encoding="latin-1", errors="replace")
    if "UNSATISFIABLE" in text:
        raise RuntimeError(f"UNSAT at {cnf}")
    if proc.returncode not in (0, 10) or "s SATISFIABLE" not in text:
        raise RuntimeError(f"kissat failed rc={proc.returncode} for {cnf}")
    return elapsed


def collect_block_clauses(
    piece_cells: dict[int, frozenset],
    plc_index: dict[tuple[int, frozenset], int],
) -> list[list[int]]:
    variants = all_symmetry_block_clauses(piece_cells, plc_index)
    return [bv for _, bv in variants]


def enumerate_solutions(
    method: str,
    puzzle: str,
    count: int,
    out_dir: Path,
    kissat: Path,
    block_user_known: bool,
    *,
    build_env: dict[str, str] | None = None,
) -> dict:
    apply_puzzle(puzzle)
    print(f"\n{'='*60}\n方法: {method}\n{'='*60}", file=sys.stderr)

    t_build = time.perf_counter()
    with env_patch(build_env or {}):
        total_vars, clauses, placements, by_piece = build_cnf(method)
    cnf_build_sec = time.perf_counter() - t_build
    plc_index = build_placement_index(by_piece)
    var_to_plc = {p["var"]: p for p in placements}

    base_cnf = out_dir / f"{method}_base.cnf"
    work_cnf = out_dir / f"{method}_work.cnf"
    write_cnf(base_cnf, total_vars, clauses)
    shutil.copy2(base_cnf, work_cnf)

    all_block_literals: list[list[int]] = []

    if block_user_known:
        from generate_cnf import KNOWN_SOLUTION
        known_pc = piece_cells_from_known(KNOWN_SOLUTION)
        blks = collect_block_clauses(known_pc, plc_index)
        all_block_literals.extend(blks)
        append_blocking_clauses(work_cnf, blks)
        print(f"  封鎖 puzzle 已知解: {len(blks)} 條對稱 blocking", file=sys.stderr)

    solutions: list[dict] = []
    times: list[float] = []

    for i in range(count):
        sol_path = out_dir / f"{method}_sol_{i+1}.txt"
        elapsed = run_kissat(kissat, work_cnf, sol_path)
        times.append(elapsed)

        true_vars = parse_kissat_true_vars(sol_path.read_text(encoding="latin-1", errors="replace"))
        pc = true_vars_to_piece_cells(true_vars, var_to_plc)
        ok = verify_no_touch(pc)
        rec = piece_cells_to_json(pc)
        rec["index"] = i + 1
        rec["time_sec"] = round(elapsed, 3)
        solutions.append(rec)

        print(f"  解 {i+1}: {elapsed:.2f}s  valid={ok}", file=sys.stderr)
        if not ok:
            print(f"  [WARN] 解 {i+1} 未通過 no-touch 驗證", file=sys.stderr)

        new_blks = collect_block_clauses(pc, plc_index)
        # 只追加尚未存在的 clause（以 tuple 去重）
        existing = {tuple(b) for b in all_block_literals}
        added = []
        for bl in new_blks:
            if tuple(bl) not in existing:
                existing.add(tuple(bl))
                all_block_literals.append(bl)
                added.append(bl)
        append_blocking_clauses(work_cnf, added)
        print(f"    +{len(added)} blocking（累計 {len(all_block_literals)}）", file=sys.stderr)

    mean_t = statistics.mean(times)
    stdev_t = statistics.stdev(times) if len(times) > 1 else 0.0

    return {
        "method": method,
        "puzzle": puzzle,
        "num_solutions": count,
        "times_sec": [round(t, 3) for t in times],
        "mean_sec": round(mean_t, 3),
        "stdev_sec": round(stdev_t, 3),
        "total_blocking_clauses": len(all_block_literals),
        "placements": len(placements),
        "cnf_clauses_base": len(clauses),
        "cnf_build_sec": round(cnf_build_sec, 3),
        "build_env": build_env,
        "solutions": solutions,
    }


def write_results_md(
    path: Path,
    baseline: dict,
    shape: dict,
    shape_prime: dict,
    shape_prime2: dict,
    puzzle: str,
) -> None:
    def spu(res: dict) -> float:
        return baseline["mean_sec"] / res["mean_sec"] if res["mean_sec"] > 0 else 0

    lines = [
        f"# 拼圖 {puzzle} 枚舉實驗（baseline / shape / shape' / shape''）\n",
        "## 設計",
        "- **baseline**：完整 placement + no-touch",
        "- **shape**：邊角 margin=4 二元死區",
        "- **shape'**：內部封閉 singleton + 全盤二元死區",
        "- **shape''**：**強化 singleton（任何<6空腔）** + 全盤二元 + **三元死區**",
        "- 無 soft_prune、無 piece 豁免",
        f"- 各方法連續找 **{baseline['num_solutions']}** 解，D4 對稱封鎖\n",
        "## CNF 規模（base）",
        "| 方法 | Placements | Base 子句數 |",
        "|------|------------|-------------|",
        f"| baseline | {baseline['placements']:,} | {baseline['cnf_clauses_base']:,} |",
        f"| shape | {shape['placements']:,} | {shape['cnf_clauses_base']:,} |",
        f"| shape' | {shape_prime['placements']:,} | {shape_prime['cnf_clauses_base']:,} |",
        f"| shape'' | {shape_prime2['placements']:,} | {shape_prime2['cnf_clauses_base']:,} |",
        "",
        "## 平均求解時間",
        "| 方法 | 各次 (s) | 平均 (s) | 標準差 |",
        "|------|----------|----------|--------|",
        f"| baseline | {baseline['times_sec']} | **{baseline['mean_sec']}** | {baseline['stdev_sec']} |",
        f"| shape | {shape['times_sec']} | **{shape['mean_sec']}** | {shape['stdev_sec']} |",
        f"| shape' | {shape_prime['times_sec']} | **{shape_prime['mean_sec']}** | {shape_prime['stdev_sec']} |",
        f"| shape'' | {shape_prime2['times_sec']} | **{shape_prime2['mean_sec']}** | {shape_prime2['stdev_sec']} |",
        "",
        f"- shape'' / baseline：**≈ {spu(shape_prime2):.2f}×**",
        f"- shape' / baseline：**≈ {spu(shape_prime):.2f}×**",
        f"- shape / baseline：**≈ {spu(shape):.2f}×**",
        "",
        "## 驗證",
    ]
    for tag, res in [
        ("baseline", baseline),
        ("shape", shape),
        ("shape'", shape_prime),
        ("shape''", shape_prime2),
    ]:
        vals = [s["valid"] for s in res["solutions"]]
        lines.append(f"- {tag}: {sum(vals)}/{len(vals)} 解通過 no-touch")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_md_prune(
    path: Path,
    puzzle: str,
    results: dict[str, dict],
    *,
    learned_label: str = "learned_shape",
) -> None:
    """baseline / shape / learned_shape 三方法枚舉摘要。"""
    order = ["baseline", "shape", "learned_shape"]
    present = [m for m in order if m in results]
    if len(present) < 2:
        return
    n = results[present[0]]["num_solutions"]
    lines = [
        f"# 拼圖 {puzzle} 枚舉實驗（baseline / shape / {learned_label}）\n",
        "## 設計",
        "- D4 對稱封鎖：先封鎖已知解，每找到一解再封鎖其 D4",
        f"- 各方法連續找 **{n}** 解",
        f"- **learned_shape**：L0 cascade + @25% oracle（v123 ranker）\n",
        "## CNF 規模（base）",
        "| 方法 | Placements | Base 子句數 | build (s) |",
        "|------|------------|-------------|-----------|",
    ]
    for m in present:
        r = results[m]
        lines.append(
            f"| {m} | {r['placements']:,} | {r['cnf_clauses_base']:,} | {r.get('cnf_build_sec', '?')} |"
        )
    lines.extend(["", "## 平均求解時間", "| 方法 | 各次 (s) | 平均 (s) | 標準差 |", "|------|----------|----------|--------|"])
    base_mean = results.get("baseline", {}).get("mean_sec")
    for m in present:
        r = results[m]
        lines.append(
            f"| {m} | {r['times_sec']} | **{r['mean_sec']}** | {r['stdev_sec']} |"
        )
    if base_mean and "shape" in results and results["shape"]["mean_sec"] > 0:
        lines.append("")
        lines.append(f"- shape / baseline：**≈ {base_mean / results['shape']['mean_sec']:.2f}×**")
    if base_mean and "learned_shape" in results and results["learned_shape"]["mean_sec"] > 0:
        lines.append(
            f"- learned / baseline：**≈ {base_mean / results['learned_shape']['mean_sec']:.2f}×**"
        )
    if "shape" in results and "learned_shape" in results and results["learned_shape"]["mean_sec"] > 0:
        lines.append(
            f"- learned / shape：**≈ {results['shape']['mean_sec'] / results['learned_shape']['mean_sec']:.2f}×**"
        )
    lines.extend(["", "## 驗證"])
    for m in present:
        vals = [s["valid"] for s in results[m]["solutions"]]
        lines.append(f"- {m}: {sum(vals)}/{len(vals)} 解通過 no-touch")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="v2")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-block-known", action="store_true")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "shape", "shape_prime", "shape_prime2"],
        choices=METHOD_CHOICES,
        help="要跑的方法",
    )
    ap.add_argument("--learned-model", type=Path, default=DEFAULT_LEARNED_MODEL)
    ap.add_argument("--learned-top-k", type=float, default=25.0)
    ap.add_argument("--cascade-l0", type=float, default=0.4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (root / "baselines" / args.puzzle / "enum_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        args.kissat = resolve_kissat(args.kissat)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    summary = {
        "puzzle": args.puzzle,
        "count": args.count,
        "block_user_known": not args.no_block_known,
        "methods_run": args.methods,
    }

    results: dict[str, dict] = {}
    learned_env_dict = learned_env(args.learned_model, args.learned_top_k, args.cascade_l0)
    for m in args.methods:
        build_env = learned_env_dict if m == "learned_shape" else None
        results[m] = enumerate_solutions(
            m,
            args.puzzle,
            args.count,
            out_dir,
            args.kissat,
            block_user_known=not args.no_block_known,
            build_env=build_env,
        )
    summary["learned_model"] = str(args.learned_model)
    summary["learned_top_k"] = args.learned_top_k
    summary["cascade_l0_frac"] = args.cascade_l0
    summary.update(results)

    json_path = out_dir / "results.json"
    merged = {}
    if json_path.is_file():
        try:
            merged = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            pass
    merged.update(summary)
    json_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    need = ("baseline", "shape", "shape_prime", "shape_prime2")
    if all(k in merged and isinstance(merged[k], dict) and "mean_sec" in merged[k] for k in need):
        write_results_md(
            out_dir / "RESULTS.md",
            merged["baseline"],
            merged["shape"],
            merged["shape_prime"],
            merged["shape_prime2"],
            args.puzzle,
        )

    prune_need = ("baseline", "shape", "learned_shape")
    if all(k in merged and isinstance(merged[k], dict) and "mean_sec" in merged[k] for k in prune_need):
        write_results_md_prune(out_dir / "RESULTS_prune.md", args.puzzle, merged)

    print(f"\n完成。結果目錄: {out_dir}", file=sys.stderr)
    for m in args.methods:
        print(f"  {m:12s} 平均 {results[m]['mean_sec']}s", file=sys.stderr)


if __name__ == "__main__":
    main()
