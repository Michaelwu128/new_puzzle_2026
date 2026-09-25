#!/usr/bin/env python3
"""
v4：L0 cascade learned @25% vs shape 預處理 + D4 SAT vs baseline。

用法：
  # 預處理耗時對照（shape / learned 無 L0 / learned L0@0.4）
  python3 run_v4_l0_benchmark.py --preprocess-only

  # 產 CNF（baseline + shape + learned_l0_k25）
  python3 run_v4_l0_benchmark.py --cnf-only

  # 只跑 Kissat（需已有 CNF）
  python3 run_v4_l0_benchmark.py --solve-only

  # 全部
  python3 run_v4_l0_benchmark.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cnf_io import append_blocking_clauses, read_cnf_header
from generate_cnf import (
    KNOWN_SOLUTION,
    apply_puzzle,
    build_cnf,
    build_placements,
    compute_extra_shape_clauses,
    write_cnf,
)
from kissat_utils import KISSAT_DEFAULT, resolve_kissat
from learned_shape import compute_learned_shape_clauses
from puzzle_defs import PUZZLES
from run_d4_next_sol_seeds import (
    analyze_d4_uniqueness,
    collect_block_clauses,
    load_var_to_plc,
    piece_cells_to_json,
    run_kissat,
    save_placements_meta,
)
from solution_blocking import (
    build_placement_index,
    parse_kissat_true_vars,
    piece_cells_from_known,
    true_vars_to_piece_cells,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "baselines" / "learned_shape" / "ranker_loo_v2_train.joblib"
DEFAULT_SEEDS = list(range(1, 11))


def default_out_dir(puzzle: str) -> Path:
    return ROOT / "baselines" / puzzle / "l0_benchmark"

LEARNED_ENV = {
    "LEARNED_SHAPE_TOP_K": "25",
    "LEARNED_SHAPE_CASCADE_L0": "0.4",
    "DEAD_POCKET_FEATURES": "v2",
}


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


def cnf_paths(out_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / f"{tag}_base.cnf",
        out_dir / f"{tag}_d4.cnf",
        out_dir / f"{tag}_meta.json",
    )


def time_shape_pairwise(puzzle: str) -> dict:
    apply_puzzle(puzzle)
    placements, by_piece, _ = build_placements("baseline")
    t0 = time.perf_counter()
    extra, _dead = compute_extra_shape_clauses(placements, by_piece)
    sec = time.perf_counter() - t0
    return {
        "method": "shape_pairwise",
        "sec": round(sec, 3),
        "n_clauses": len(extra),
    }


def time_learned(
    puzzle: str,
    model_path: Path,
    *,
    top_k: float,
    cascade_l0: float | None,
) -> dict:
    apply_puzzle(puzzle)
    placements, by_piece, _ = build_placements("baseline")
    t0 = time.perf_counter()
    _extra, stats = compute_learned_shape_clauses(
        placements,
        by_piece,
        model_path=model_path,
        top_k_percent=top_k,
        edge_margin=6,
        features_version="v2",
        cascade_l0_frac=cascade_l0,
    )
    total_sec = time.perf_counter() - t0
    return {
        "method": f"learned_k{int(top_k)}" + (f"_l0{cascade_l0}" if cascade_l0 else ""),
        "total_sec": round(total_sec, 3),
        "l0_sec": stats.get("l0_sec"),
        "ml_sec": stats.get("ml_sec"),
        "oracle_sec": stats.get("oracle_sec"),
        "n_clauses": stats["n_clauses"],
        "clause_recall_vs_full": stats["clause_recall_vs_full"],
        "n_oracle_calls": stats["n_oracle_calls"],
        "n_pairs_total": stats["n_pairs_total"],
        "cascade_l0_frac": cascade_l0,
        "n_l0_kept": stats.get("n_l0_kept"),
    }


def generate_method_cnf(
    tag: str,
    build_method: str,
    out_dir: Path,
    *,
    env: dict[str, str | None] | None = None,
    force: bool = False,
) -> dict:
    base_path, d4_path, meta_path = cnf_paths(out_dir, tag)
    manifest_path = out_dir / "cnf_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    if not force and base_path.is_file() and d4_path.is_file() and meta_path.is_file():
        prev = manifest.get(tag, {})
        print(f"  [快取] {tag}", file=sys.stderr)
        return {**prev, "reused": True}

    print(f"\n=== CNF | {tag} (build_cnf={build_method}) ===", file=sys.stderr)
    t_total = time.perf_counter()
    with env_patch(env or {}):
        t0 = time.perf_counter()
        total_vars, clauses, placements, by_piece = build_cnf(build_method)
        cnf_build_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_cnf(base_path, total_vars, clauses)
    cnf_write_base_sec = time.perf_counter() - t0

    plc_index = build_placement_index(by_piece)
    known_pc = piece_cells_from_known(KNOWN_SOLUTION)
    d4_blocks = collect_block_clauses(known_pc, plc_index)

    shutil.copy2(base_path, d4_path)
    t0 = time.perf_counter()
    append_blocking_clauses(d4_path, d4_blocks)
    cnf_write_d4_sec = time.perf_counter() - t0

    save_placements_meta(meta_path, placements)
    _, clauses_d4 = read_cnf_header(d4_path)

    entry = {
        "tag": tag,
        "build_method": build_method,
        "placements": len(placements),
        "vars": total_vars,
        "clauses_base": len(clauses),
        "clauses_d4": clauses_d4,
        "cnf_build_sec": round(cnf_build_sec, 3),
        "cnf_write_base_sec": round(cnf_write_base_sec, 3),
        "cnf_write_d4_sec": round(cnf_write_d4_sec, 3),
        "cnf_total_sec": round(time.perf_counter() - t_total, 3),
        "env": {k: v for k, v in (env or {}).items() if v is not None},
        "reused": False,
    }
    print(
        f"  clauses={entry['clauses_base']:,}  build={entry['cnf_build_sec']:.1f}s  "
        f"total={entry['cnf_total_sec']:.1f}s",
        file=sys.stderr,
    )
    manifest[tag] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return entry


def solve_tag(
    puzzle: str,
    tag: str,
    out_dir: Path,
    kissat: Path,
    seeds: list[int],
    cnf_entry: dict,
) -> dict:
    _, d4_path, meta_path = cnf_paths(out_dir, tag)
    var_to_plc = load_var_to_plc(meta_path)
    apply_puzzle(puzzle)
    known_pc = piece_cells_from_known(KNOWN_SOLUTION)

    per_seed: list[dict] = []
    times: list[float] = []
    for seed in seeds:
        sol_path = out_dir / f"{tag}_seed{seed}.txt"
        status, elapsed = run_kissat(kissat, d4_path, sol_path, seed)
        rec: dict = {"seed": seed, "status": status, "solve_sec": round(elapsed, 3)}
        if status == "SATISFIABLE":
            times.append(elapsed)
            true_vars = parse_kissat_true_vars(
                sol_path.read_text(encoding="latin-1", errors="replace")
            )
            pc = true_vars_to_piece_cells(true_vars, var_to_plc)
            rec.update(piece_cells_to_json(pc))
            rec["_piece_cells"] = pc
        per_seed.append(rec)
        print(f"  {tag} seed={seed}  {status}  {elapsed:.2f}s", file=sys.stderr)

    d4_stats = analyze_d4_uniqueness(per_seed, known_pc)
    for rec in per_seed:
        rec.pop("_piece_cells", None)

    result = {
        "tag": tag,
        "cnf": cnf_entry,
        "per_seed": per_seed,
        **d4_stats,
    }
    if times:
        result["solve_mean_sec"] = round(statistics.mean(times), 3)
        result["solve_stdev_sec"] = (
            round(statistics.stdev(times), 3) if len(times) > 1 else 0.0
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--cnf-only", action="store_true")
    ap.add_argument("--solve-only", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=float, default=25.0)
    ap.add_argument("--cascade-l0", type=float, default=0.4)
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--puzzle", choices=sorted(PUZZLES.keys()), default="v4")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--learned-tag",
        default="learned_l0_k25",
        help="learned CNF / SAT 方法 tag（多模型對照時改不同名）",
    )
    ap.add_argument(
        "--learned-only",
        action="store_true",
        help="只產 learned CNF + SAT（重用 manifest 內 baseline/shape）",
    )
    ap.add_argument(
        "--report-name",
        default=None,
        help="結果 JSON 檔名（預設 v4_l0_benchmark.json）",
    )
    ap.add_argument("--force-regen-cnf", action="store_true")
    ap.add_argument(
        "--include-no-l0",
        action="store_true",
        help="預處理對照也跑無 L0 的 learned @25%%（較慢）",
    )
    ap.add_argument(
        "--no-l0",
        action="store_true",
        help="CNF/SAT 不設 LEARNED_SHAPE_CASCADE_L0（全量 ML，與 sweep 無 L0 一致）",
    )
    args = ap.parse_args()

    if not (args.preprocess_only or args.cnf_only or args.solve_only or args.all):
        args.all = True

    if args.out_dir is None:
        args.out_dir = default_out_dir(args.puzzle)
    puzzle = args.puzzle
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "puzzle": puzzle,
        "model": str(args.model),
        "learned_tag": args.learned_tag,
        "top_k_percent": args.top_k,
        "cascade_l0_frac": None if args.no_l0 else args.cascade_l0,
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }

    if args.preprocess_only:
        print(f"\n=== 預處理耗時對照 | {puzzle} ===", file=sys.stderr)
        apply_puzzle(puzzle)
        t0 = time.perf_counter()
        _tv, _cl, _pl, _bp = build_cnf("baseline")
        baseline_sec = time.perf_counter() - t0

        shape_row = time_shape_pairwise(puzzle)
        learned_l0 = time_learned(
            puzzle, args.model, top_k=args.top_k, cascade_l0=args.cascade_l0
        )
        preprocess = {
            "baseline_cnf_build_sec": round(baseline_sec, 3),
            "shape": shape_row,
            "learned_l0_k25": learned_l0,
        }
        if args.include_no_l0:
            preprocess["learned_k25_no_l0"] = time_learned(
                puzzle, args.model, top_k=args.top_k, cascade_l0=None
            )
        report["preprocess"] = preprocess

        print("\n--- 預處理摘要 ---", file=sys.stderr)
        print(f"  baseline CNF build: {baseline_sec:.1f}s", file=sys.stderr)
        print(f"  shape pairwise:     {shape_row['sec']:.1f}s  ({shape_row['n_clauses']:,} clauses)", file=sys.stderr)
        if args.include_no_l0 and "learned_k25_no_l0" in preprocess:
            no_l0 = preprocess["learned_k25_no_l0"]
            print(
                f"  learned @25% no L0: {no_l0['total_sec']:.1f}s  "
                f"(ml={no_l0['ml_sec']}s oracle={no_l0['oracle_sec']}s "
                f"recall={no_l0['clause_recall_vs_full']:.3f})",
                file=sys.stderr,
            )
        print(
            f"  learned L0@{args.cascade_l0} @25%: {learned_l0['total_sec']:.1f}s  "
            f"(l0={learned_l0['l0_sec']}s ml={learned_l0['ml_sec']}s oracle={learned_l0['oracle_sec']}s "
            f"recall={learned_l0['clause_recall_vs_full']:.3f})",
            file=sys.stderr,
        )
        faster = learned_l0["total_sec"] < shape_row["sec"]
        print(
            f"  L0 learned vs shape: {'快' if faster else '慢'} "
            f"{shape_row['sec'] / learned_l0['total_sec']:.2f}×",
            file=sys.stderr,
        )

    learned_env: dict[str, str | None] = {
        "DEAD_POCKET_FEATURES": "v2",
        "LEARNED_SHAPE_MODEL": str(args.model),
        "LEARNED_SHAPE_TOP_K": str(args.top_k),
    }
    if args.no_l0:
        learned_env["LEARNED_SHAPE_CASCADE_L0"] = None
    else:
        learned_env["LEARNED_SHAPE_CASCADE_L0"] = str(args.cascade_l0)

    manifest_path = args.out_dir / "cnf_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    if args.cnf_only or args.all:
        apply_puzzle(puzzle)
        cnf_results: dict = {}
        if not args.learned_only:
            cnf_results["baseline"] = generate_method_cnf(
                "baseline", "baseline", args.out_dir, force=args.force_regen_cnf
            )
            cnf_results["shape"] = generate_method_cnf(
                "shape", "shape", args.out_dir, force=args.force_regen_cnf
            )
        cnf_results[args.learned_tag] = generate_method_cnf(
            args.learned_tag,
            "learned_shape",
            args.out_dir,
            env=learned_env,
            force=args.force_regen_cnf,
        )
        report["cnf"] = cnf_results
        manifest.update(cnf_results)

        if args.all and not args.learned_only:
            shape_sec = cnf_results["shape"]["cnf_build_sec"]
            learned_sec = cnf_results[args.learned_tag]["cnf_build_sec"]
            report["preprocess_from_cnf"] = {
                "baseline_cnf_build_sec": cnf_results["baseline"]["cnf_build_sec"],
                "shape_cnf_build_sec": shape_sec,
                "learned_cnf_build_sec": learned_sec,
                "learned_vs_shape_ratio": round(shape_sec / learned_sec, 3)
                if learned_sec > 0
                else None,
            }
            print("\n--- CNF 預處理摘要 ---", file=sys.stderr)
            print(f"  baseline: {cnf_results['baseline']['cnf_build_sec']:.1f}s", file=sys.stderr)
            print(f"  shape:    {shape_sec:.1f}s", file=sys.stderr)
            print(
                f"  {args.learned_tag}: {learned_sec:.1f}s  ({shape_sec/learned_sec:.2f}× vs shape)",
                file=sys.stderr,
            )
        elif args.learned_only:
            le = cnf_results[args.learned_tag]
            print(
                f"\n--- learned CNF | {args.learned_tag} ---",
                file=sys.stderr,
            )
            print(
                f"  build={le['cnf_build_sec']:.1f}s  clauses={le['clauses_base']:,}",
                file=sys.stderr,
            )
            report["preprocess_from_cnf"] = {
                "learned_tag": args.learned_tag,
                "learned_cnf_build_sec": le["cnf_build_sec"],
                "learned_clauses_base": le["clauses_base"],
            }

    if args.solve_only or args.all:
        try:
            args.kissat = resolve_kissat(args.kissat)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        sat_results: dict = {}
        sat_tags = [args.learned_tag] if args.learned_only else ("baseline", "shape", args.learned_tag)
        for tag in sat_tags:
            entry = manifest.get(tag)
            if not entry:
                print(f"[ERROR] 缺少 {tag} CNF，請先 --cnf-only", file=sys.stderr)
                sys.exit(1)
            sat_results[tag] = solve_tag(
                puzzle, tag, args.out_dir, args.kissat, args.seeds, entry
            )

        baseline_mean = sat_results.get("baseline", {}).get("solve_mean_sec")
        if baseline_mean is None and manifest.get("baseline", {}).get("reused"):
            # learned-only：從先前 v2 run 的 report 取 baseline 參考
            prev_report = args.out_dir / f"{puzzle}_l0_benchmark.json"
            if prev_report.is_file():
                try:
                    baseline_mean = json.loads(prev_report.read_text()).get("sat", {}).get(
                        "baseline", {}
                    ).get("solve_mean_sec")
                except (json.JSONDecodeError, AttributeError):
                    pass
        if baseline_mean:
            for tag, row in sat_results.items():
                mean = row.get("solve_mean_sec")
                if mean and mean > 0:
                    row["vs_baseline"] = (
                        round(baseline_mean / mean, 3) if tag != "baseline" else 1.0
                    )
        report["sat"] = sat_results
        if baseline_mean and args.learned_only:
            report["baseline_ref_mean_sec"] = baseline_mean

        print("\n--- SAT 摘要 (D4, 10 seeds) ---", file=sys.stderr)
        for tag, row in sat_results.items():
            mean = row.get("solve_mean_sec", "?")
            vs = row.get("vs_baseline", "")
            vs_s = f"  vs baseline={vs}×" if vs != "" else ""
            print(f"  {tag}: mean={mean}s{vs_s}", file=sys.stderr)

    out_path = args.out_dir / (args.report_name or f"{puzzle}_l0_benchmark.json")
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev.update(report)
            report = prev
        except json.JSONDecodeError:
            pass
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n完成。{out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
