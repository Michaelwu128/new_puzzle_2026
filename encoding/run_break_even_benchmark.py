#!/usr/bin/env python3
"""
Break-even：D4 連續 block 求下一解，累積時間（含一次 CNF build）何時反超 baseline。

用法：
  python3 run_break_even_benchmark.py --puzzle v6 --k-max 30 \\
    --methods baseline learned_shape \\
    --learned-model ../baselines/learned_shape/ranker_train_v1_v2_v3_v4_v5.joblib

  # 30 解內未反超，接著跑到 50（不重 build、不重解 1–30）
  python3 run_break_even_benchmark.py --puzzle v6 --k-max 50 --resume
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kissat_utils import KISSAT_DEFAULT, resolve_kissat
from run_enum_benchmark import (
    ROOT,
    collect_block_clauses,
    env_patch,
    learned_env,
    piece_cells_to_json,
    run_kissat,
)
from generate_cnf import KNOWN_SOLUTION, apply_puzzle, build_cnf, write_cnf
from puzzle_defs import PUZZLES
from solution_blocking import (
    build_placement_index,
    parse_kissat_true_vars,
    piece_cells_from_known,
    true_vars_to_piece_cells,
)
from cnf_io import append_blocking_clauses

DEFAULT_LEARNED_MODEL = ROOT / "baselines" / "learned_shape" / "ranker_train_v1_v2_v3_v4_v5.joblib"
METHOD_CHOICES = ["baseline", "learned_shape", "shape"]


def cumulative(build_sec: float, times: list[float]) -> list[float]:
    out: list[float] = []
    total = build_sec
    for t in times:
        total += t
        out.append(round(total, 3))
    return out


def compute_break_even(cum_a: dict[str, list[float]], ref: str, other: str) -> int | None:
    """最小 k（1-based）使 other 累積 < ref；無則 None。"""
    if ref not in cum_a or other not in cum_a:
        return None
    a, b = cum_a[ref], cum_a[other]
    n = min(len(a), len(b))
    for i in range(n):
        if b[i] < a[i]:
            return i + 1
    return None


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _placements_path(out_dir: Path, method: str) -> Path:
    return out_dir / f"{method}_placements.json"


def _save_placements(out_dir: Path, method: str, placements: list[dict]) -> None:
    slim = [{"var": p["var"], "piece": p["piece"], "cells": [list(c) for c in p["cells"]]} for p in placements]
    _placements_path(out_dir, method).write_text(json.dumps(slim), encoding="utf-8")


def _load_placements_index(out_dir: Path, method: str):
    path = _placements_path(out_dir, method)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path} for resume")
    placements = json.loads(path.read_text(encoding="utf-8"))
    for p in placements:
        p["cells"] = [tuple(c) for c in p["cells"]]
    by_piece: dict[int, list] = {}
    for p in placements:
        by_piece.setdefault(p["piece"], []).append(p)
    plc_index = build_placement_index(by_piece)
    var_to_plc = {p["var"]: p for p in placements}
    return placements, by_piece, plc_index, var_to_plc


def run_method(
    method: str,
    puzzle: str,
    k_max: int,
    out_dir: Path,
    kissat: Path,
    *,
    build_env: dict[str, str] | None,
    resume: bool,
    prior: dict | None,
) -> dict:
    apply_puzzle(puzzle)
    work_cnf = out_dir / f"{method}_work.cnf"

    times: list[float] = list(prior.get("times_sec", [])) if prior else []
    solutions: list[dict] = list(prior.get("solutions", [])) if prior else []
    cnf_build_sec = float(prior.get("cnf_build_sec", 0)) if prior else 0.0
    placements_n = prior.get("placements") if prior else None
    clauses_n = prior.get("cnf_clauses_base") if prior else None
    total_blocking = int(prior.get("total_blocking_clauses", 0)) if prior else 0
    stopped = prior.get("stopped", "running") if prior else "running"

    start_k = len(times)
    if resume and start_k >= k_max:
        print(f"  [{method}] 已有 {start_k} 解，>= k_max={k_max}，跳過", file=sys.stderr)
        return _pack_method(
            method, puzzle, times, solutions, cnf_build_sec,
            placements_n or 0, clauses_n or 0, total_blocking, stopped,
        )

    if resume and start_k > 0 and work_cnf.is_file():
        print(f"  [{method}] resume：從第 {start_k + 1} 解繼續（build {cnf_build_sec:.1f}s 已付）", file=sys.stderr)
        placements, by_piece, plc_index, var_to_plc = _load_placements_index(out_dir, method)
        placements_n = len(placements)
        clauses_n = clauses_n or prior.get("cnf_clauses_base", 0)
    elif resume and start_k > 0:
        raise FileNotFoundError(f"resume 需要 {work_cnf} 與 placements 快取")
    else:
        print(f"\n{'='*60}\n方法: {method}（新建）\n{'='*60}", file=sys.stderr)
        t0 = time.perf_counter()
        with env_patch(build_env or {}):
            total_vars, clauses, placements, by_piece = build_cnf(method)
        cnf_build_sec = time.perf_counter() - t0
        plc_index = build_placement_index(by_piece)
        var_to_plc = {p["var"]: p for p in placements}
        write_cnf(work_cnf, total_vars, clauses)
        placements_n = len(placements)
        clauses_n = len(clauses)
        total_blocking = 0
        all_block_literals: list[list[int]] = []
        known_pc = piece_cells_from_known(KNOWN_SOLUTION)
        blks = collect_block_clauses(known_pc, plc_index)
        all_block_literals.extend(blks)
        append_blocking_clauses(work_cnf, blks)
        total_blocking = len(all_block_literals)
        print(f"  build {cnf_build_sec:.1f}s | 封鎖已知解 {len(blks)} 條", file=sys.stderr)
        _save_placements(out_dir, method, placements)

    if resume and start_k > 0:
        all_block_literals = []  # blocking already in work_cnf

    for i in range(start_k, k_max):
        if stopped == "unsat":
            break
        sol_path = out_dir / f"{method}_sol_{i+1}.txt"
        try:
            elapsed = run_kissat(kissat, work_cnf, sol_path)
        except RuntimeError as e:
            if "UNSAT" in str(e):
                print(f"  [{method}] 第 {i+1} 解 UNSAT，停止", file=sys.stderr)
                stopped = "unsat"
                break
            raise
        times.append(round(elapsed, 3))
        true_vars = parse_kissat_true_vars(sol_path.read_text(encoding="latin-1", errors="replace"))
        pc = true_vars_to_piece_cells(true_vars, var_to_plc)
        rec = piece_cells_to_json(pc)
        rec["index"] = i + 1
        rec["time_sec"] = round(elapsed, 3)
        rec["cumulative_sec"] = cumulative(cnf_build_sec, times)[-1]
        solutions.append(rec)
        print(f"  解 {i+1}: {elapsed:.2f}s  cumul={rec['cumulative_sec']:.1f}s", file=sys.stderr)

        new_blks = collect_block_clauses(pc, plc_index)
        append_blocking_clauses(work_cnf, new_blks)
        total_blocking += len(new_blks)

    if stopped != "unsat" and len(times) >= k_max:
        stopped = "k_max"

    return _pack_method(
        method, puzzle, times, solutions, cnf_build_sec,
        placements_n or 0, clauses_n or 0, total_blocking, stopped, build_env,
    )


def _pack_method(
    method, puzzle, times, solutions, cnf_build_sec,
    placements, clauses, total_blocking, stopped, build_env=None,
) -> dict:
    cum = cumulative(cnf_build_sec, times)
    return {
        "method": method,
        "puzzle": puzzle,
        "num_solutions": len(times),
        "k_max_requested": None,
        "times_sec": times,
        "cumulative_sec": cum,
        "cnf_build_sec": round(cnf_build_sec, 3),
        "placements": placements,
        "cnf_clauses_base": clauses,
        "total_blocking_clauses": total_blocking,
        "stopped": stopped,
        "mean_sec": round(statistics.mean(times), 3) if times else 0.0,
        "stdev_sec": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "build_env": build_env,
        "solutions": solutions,
    }


def write_results_md(path: Path, state: dict) -> None:
    puzzle = state["puzzle"]
    k_max = state["k_max"]
    methods = state["methods"]
    be = state.get("break_even", {})
    lines = [
        f"# v{puzzle[-1]} Break-even：D4 連續 block 累積時間\n",
        "## 定義",
        "- `累積(k) = CNF build（一次）+ Σ 前 k 次 Kissat`",
        "- **k\\***：learned 累積首次 **<** baseline 累積的 k（1-based）；無則 `> k_max`",
        "",
        f"**k_max** = {k_max} | **run_at** = {state.get('run_at', '')}",
        "",
    ]
    if "learned_shape" in methods and "baseline" in methods:
        ks = be.get("learned_vs_baseline")
        lines.append(f"**Break-even learned vs baseline**：{ks if ks is not None else f'> {k_max}（未反超）'}")
        lines.append("")

  # table at last k
    headers = ["k", "baseline 累積", "learned 累積"]
    if "shape" in methods:
        headers.append("shape 累積")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    n = max(len(methods[m].get("cumulative_sec", [])) for m in methods)
    for k in range(1, n + 1):
        row = [str(k)]
        for m in ["baseline", "learned_shape", "shape"]:
            if m not in methods:
                continue
            cum = methods[m].get("cumulative_sec", [])
            row.append(f"{cum[k-1]:.1f} s" if k <= len(cum) else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## 各方法摘要", ""])
    for m, d in methods.items():
        lines.append(
            f"- **{m}**：build {d['cnf_build_sec']}s | "
            f"{d['num_solutions']} 解 | stopped={d['stopped']} | "
            f"mean solve {d['mean_sec']}s"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Break-even cumulative time benchmark")
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="v6")
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "learned_shape"],
        choices=METHOD_CHOICES,
    )
    ap.add_argument("--learned-model", type=Path, default=DEFAULT_LEARNED_MODEL)
    ap.add_argument("--learned-top-k", type=float, default=25.0)
    ap.add_argument("--cascade-l0", type=float, default=0.4)
    ap.add_argument("--resume", action="store_true", help="從 results.json / work.cnf 接續")
    args = ap.parse_args()

    out_dir = args.out_dir or (ROOT / "baselines" / args.puzzle / "break_even")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"

    try:
        args.kissat = resolve_kissat(args.kissat)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    prior_state = load_state(json_path) if args.resume else {}
    prior_methods = prior_state.get("methods", {}) if args.resume else {}

    learned_env_dict = learned_env(args.learned_model, args.learned_top_k, args.cascade_l0)
    results: dict[str, dict] = {}

    for m in args.methods:
        build_env = learned_env_dict if m == "learned_shape" else None
        prior_m = prior_methods.get(m) if args.resume else None
        results[m] = run_method(
            m,
            args.puzzle,
            args.k_max,
            out_dir,
            args.kissat,
            build_env=build_env,
            resume=args.resume,
            prior=prior_m,
        )
        results[m]["k_max_requested"] = args.k_max

    cum_map = {m: results[m]["cumulative_sec"] for m in results}
    break_even = {}
    if "baseline" in results and "learned_shape" in results:
        break_even["learned_vs_baseline"] = compute_break_even(
            cum_map, "baseline", "learned_shape"
        )
    if "baseline" in results and "shape" in results:
        break_even["shape_vs_baseline"] = compute_break_even(
            cum_map, "baseline", "shape"
        )

    state = {
        "puzzle": args.puzzle,
        "k_max": args.k_max,
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "methods_run": args.methods,
        "learned_model": str(args.learned_model),
        "learned_top_k": args.learned_top_k,
        "cascade_l0_frac": args.cascade_l0,
        "break_even": break_even,
        "methods": results,
    }
    save_state(json_path, state)
    write_results_md(out_dir / "RESULTS.md", state)

    print(f"\n完成 → {out_dir}", file=sys.stderr)
    if break_even.get("learned_vs_baseline") is not None:
        print(f"  break-even learned vs baseline: k={break_even['learned_vs_baseline']}", file=sys.stderr)
    else:
        print(f"  break-even learned vs baseline: > {args.k_max} 或未反超", file=sys.stderr)


if __name__ == "__main__":
    main()
