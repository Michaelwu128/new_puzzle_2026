#!/usr/bin/env python3
"""
Learned dead-pocket 實驗編排：Phase 1 top-K sweep、Phase 2 D4+seed SAT。

用法：
  # Phase 1：oracle 效率曲線（不寫完整 CNF）
  python3 run_learned_shape_benchmark.py --phase sweep --puzzle v1 v2

  # Phase 2：對選定 K 產 CNF + D4 求解
  python3 run_learned_shape_benchmark.py --phase sat --puzzle v1 v2 --top-k 25 100

  # 全部
  python3 run_learned_shape_benchmark.py --phase all --puzzle v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_cnf import apply_puzzle, build_cnf, build_placements, write_cnf
from kissat_utils import KISSAT_DEFAULT, resolve_kissat
from learned_shape import oracle_selected_pairs, score_candidate_pairs
from puzzle_defs import puzzle_names

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "baselines" / "learned_shape" / "ranker_loo_v1_train.joblib"
DEFAULT_RF_MODEL = ROOT / "baselines" / "learned_shape" / "model_splitB.joblib"
SWEEP_K = [1, 5, 10, 25, 50, 100]


def ranker_path_for_train(train_puzzles: list[str]) -> Path:
    if len(train_puzzles) == 1:
        p = ROOT / "baselines" / "learned_shape" / f"ranker_loo_{train_puzzles[0]}_train.joblib"
    else:
        tag = "_".join(train_puzzles)
        p = ROOT / "baselines" / "learned_shape" / f"ranker_train_{tag}.joblib"
    return p if p.is_file() else DEFAULT_RF_MODEL


def ranker_path_for_loo(train_puzzle: str) -> Path:
    return ranker_path_for_train([train_puzzle])


def phase_sweep(
    puzzle: str,
    model_path: Path,
    out_dir: Path,
    *,
    k_values: list[float],
    edge_margin: int = 6,
    features_version: str = "v2",
    train_puzzle: str | None = None,
    cascade_l0_frac: float | None = None,
) -> dict:
    from collections import defaultdict

    from generate_cnf import compute_extra_shape_clauses

    apply_puzzle(puzzle)
    placements, by_piece, _ = build_placements("baseline")

    print(f"\n=== Phase 1 sweep | {puzzle} ===", file=sys.stderr)
    cache_path = out_dir / f"full_dead_{puzzle}.json"

    t0 = time.perf_counter()
    if cache_path.is_file():
        full_dead = {
            tuple(p) for p in json.loads(cache_path.read_text(encoding="utf-8"))
        }
        print(f"  [快取] full dead pairs: {len(full_dead):,}", file=sys.stderr)
        full_scan_sec = 0.0
        # band-exclusion 在 v1/v2 皆為 0；dead-pocket 子句數 = |full_dead|
        full_clauses = [[]] * len(full_dead)  # 僅用 len()，不物化
        full_clause_sec = 0.0
    else:
        result = compute_extra_shape_clauses(
            placements, by_piece, edge_margin=edge_margin, collect_dead_pairs=True
        )
        full_clauses, full_dead = result
        full_scan_sec = time.perf_counter() - t0
        full_clause_sec = full_scan_sec
        cache_path.write_text(
            json.dumps([list(p) for p in sorted(full_dead)], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  寫入快取 {cache_path.name}", file=sys.stderr)

    print(
        f"  full dead pairs: {len(full_dead):,}  scan={full_scan_sec:.1f}s  "
        f"full_clauses={len(full_clauses):,}",
        file=sys.stderr,
    )

    rows: list[dict] = []
    t_score = time.perf_counter()
    ranked, score_meta = score_candidate_pairs(
        placements,
        by_piece,
        model_path=model_path,
        edge_margin=edge_margin,
        features_version=features_version,
        cascade_l0_frac=cascade_l0_frac,
    )
    n_total = score_meta["n_pairs_total"]
    score_sec = time.perf_counter() - t_score
    l0_sec = score_meta.get("l0_sec", 0.0) or 0.0
    ml_sec = score_meta.get("ml_sec", 0.0) or 0.0
    print(
        f"  scored {n_total:,} pairs in {score_sec:.1f}s  "
        f"(l0={l0_sec}s ml={ml_sec}s) model={score_meta.get('model_type')}",
        file=sys.stderr,
    )

    for k in k_values:
        t_k = time.perf_counter()
        if n_total == 0:
            stats = {
                "n_pairs_total": 0,
                "n_oracle_calls": 0,
                "oracle_call_fraction": 0.0,
                "n_clauses": 0,
                "clause_recall_vs_full": 1.0,
                "oracle_sec": 0.0,
            }
        else:
            k_n = max(1, int(np.ceil(n_total * k / 100.0)))
            selected = ranked[:k_n]
            _clauses, stats = oracle_selected_pairs(
                selected,
                full_dead_pairs=full_dead,
                top_k_percent=k,
            )
            stats["n_pairs_total"] = n_total
            stats["oracle_call_fraction"] = round(k_n / n_total, 4)
        elapsed = time.perf_counter() - t_k
        rows.append(
            {
                "puzzle": puzzle,
                "top_k_percent": k,
                "n_clauses": stats["n_clauses"],
                "n_oracle_calls": stats["n_oracle_calls"],
                "n_pairs_total": stats["n_pairs_total"],
                "oracle_call_fraction": stats["oracle_call_fraction"],
                "clause_recall_vs_full": stats["clause_recall_vs_full"],
                "oracle_sec": stats["oracle_sec"],
                "score_sec": round(score_sec, 3),
                "l0_sec": l0_sec,
                "ml_sec": ml_sec,
                "total_sec": round(score_sec + elapsed, 3),
                "full_shape_clauses": len(full_dead),
                "clause_fraction_vs_full_shape": round(
                    stats["n_clauses"] / len(full_dead) if full_dead else 1.0, 4
                ),
            }
        )
        print(
            f"  k={k}%  recall={stats['clause_recall_vs_full']:.4f}  "
            f"clauses={stats['n_clauses']:,}  oracle={stats['oracle_sec']:.1f}s",
            file=sys.stderr,
        )

    result = {
        "puzzle": puzzle,
        "train_puzzle": train_puzzle,
        "model": str(model_path),
        "features_version": features_version,
        "n_full_dead_pairs": len(full_dead),
        "full_dead_scan_sec": round(full_scan_sec, 3),
        "full_shape_clauses": len(full_dead),
        "full_shape_pairwise_sec": round(full_clause_sec, 3),
        "cascade_l0_frac": cascade_l0_frac,
        "score_l0_sec": l0_sec,
        "score_ml_sec": ml_sec,
        "score_total_sec": round(score_sec, 3),
        "sweep": rows,
    }
    out_path = out_dir / f"sweep_{puzzle}.json"
    if train_puzzle and train_puzzle != puzzle:
        tag = train_puzzle.replace("+", "_")
        out_path = out_dir / f"sweep_{puzzle}_train_{tag}.json"
    if cascade_l0_frac is not None:
        out_path = out_path.with_name(
            out_path.stem + f"_l0{cascade_l0_frac}".replace(".", "p") + out_path.suffix
        )
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"  寫出 {out_path}", file=sys.stderr)
    return result


def phase_sat(
    puzzle: str,
    top_k_values: list[float],
    model_path: Path,
    kissat: Path,
    seeds: list[int],
    *,
    solve_only: bool = False,
    skip_existing_seeds: bool = False,
    features_version: str = "v2",
    method_prefix: str = "learned_shape",
    cascade_l0_frac: float | None = None,
) -> None:
    """產 learned_shape CNF 並跑 D4 + Kissat。"""
    import shutil
    import statistics

    from cnf_io import append_blocking_clauses, read_cnf_header
    from generate_cnf import KNOWN_SOLUTION
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

    learned_dir = ROOT / "baselines" / puzzle / "learned_shape"
    learned_dir.mkdir(parents=True, exist_ok=True)

    for k in top_k_values:
        method = f"{method_prefix}_k{int(k)}"
        os.environ["LEARNED_SHAPE_MODEL"] = str(model_path)
        os.environ["LEARNED_SHAPE_TOP_K"] = str(k)
        os.environ["DEAD_POCKET_FEATURES"] = features_version
        os.environ.pop("LEARNED_SHAPE_FULL_PAIRS", None)
        if cascade_l0_frac is not None:
            os.environ["LEARNED_SHAPE_CASCADE_L0"] = str(cascade_l0_frac)
        else:
            os.environ.pop("LEARNED_SHAPE_CASCADE_L0", None)

        print(f"\n=== Phase 2 SAT | {puzzle} | top_k={k}% ===", file=sys.stderr)
        d4_cnf = learned_dir / f"{method}_d4.cnf"
        meta_path = learned_dir / f"{method}_meta.json"
        manifest_path = learned_dir / "cnf_manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                pass

        if solve_only and d4_cnf.is_file() and meta_path.is_file():
            print(f"  [solve-only] 重用 {d4_cnf.name}", file=sys.stderr)
            cnf_entry = manifest.get(method, {})
            apply_puzzle(puzzle)
            known_pc = piece_cells_from_known(KNOWN_SOLUTION)
        else:
            apply_puzzle(puzzle)
            t0 = time.perf_counter()
            total_vars, clauses, placements, _ = build_cnf("learned_shape")
            cnf_sec = time.perf_counter() - t0

            base_cnf = learned_dir / f"{method}_base.cnf"
            write_cnf(base_cnf, total_vars, clauses)

            _, by_piece, _ = build_placements("baseline")
            plc_index = build_placement_index(by_piece)
            known_pc = piece_cells_from_known(KNOWN_SOLUTION)
            d4_blocks = collect_block_clauses(known_pc, plc_index)

            shutil.copy2(base_cnf, d4_cnf)
            append_blocking_clauses(d4_cnf, d4_blocks)
            save_placements_meta(meta_path, placements)

            _, clauses_d4 = read_cnf_header(d4_cnf)
            cnf_entry = {
                "placements": len(placements),
                "vars": total_vars,
                "clauses_base": len(clauses),
                "clauses_d4": clauses_d4,
                "cnf_build_sec": round(cnf_sec, 3),
                "top_k_percent": k,
                "method_tag": method,
            }
            manifest[method] = cnf_entry
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )

        var_to_plc = load_var_to_plc(meta_path)
        per_seed: list[dict] = []
        times: list[float] = []
        for seed in seeds:
            sol_path = learned_dir / f"{method}_seed{seed}.txt"
            if skip_existing_seeds and sol_path.is_file():
                text = sol_path.read_text(encoding="latin-1", errors="replace")
                if "SATISFIABLE" in text[:24]:
                    print(f"  seed={seed}  skip (exists)", file=sys.stderr)
                    true_vars = parse_kissat_true_vars(text)
                    pc = true_vars_to_piece_cells(true_vars, var_to_plc)
                    rec = {
                        "seed": seed,
                        "status": "SATISFIABLE",
                        "solve_sec": None,
                    }
                    rec.update(piece_cells_to_json(pc))
                    rec["_piece_cells"] = pc
                    per_seed.append(rec)
                    continue
            status, elapsed = run_kissat(kissat, d4_cnf, sol_path, seed)
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
            print(f"  seed={seed}  {status}  {elapsed:.2f}s", file=sys.stderr)

        d4_stats = analyze_d4_uniqueness(per_seed, known_pc)
        for rec in per_seed:
            rec.pop("_piece_cells", None)

        result = {
            "method": method,
            "top_k_percent": k,
            "cnf": cnf_entry,
            "per_seed": per_seed,
            **d4_stats,
        }
        if times:
            result["solve_mean_sec"] = round(statistics.mean(times), 3)
            result["solve_stdev_sec"] = (
                round(statistics.stdev(times), 3) if len(times) > 1 else 0.0
            )

        results_path = learned_dir / "results.json"
        merged: dict = {"puzzle": puzzle, "seeds": seeds, "methods": {}}
        if results_path.is_file():
            try:
                merged = json.loads(results_path.read_text())
            except json.JSONDecodeError:
                pass
        merged.setdefault("methods", {})[method] = result
        merged["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        results_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")


def pick_k_star(sweep: dict, min_recall: float = 0.99) -> list[float]:
    chosen: list[float] = []
    for row in sweep.get("sweep", []):
        if row["clause_recall_vs_full"] >= min_recall:
            chosen.append(row["top_k_percent"])
    if not chosen:
        return [100.0]
    return [min(chosen), 100.0] if 100.0 not in chosen else [min(chosen)]


def run_loo_fold(
    train_puzzle: list[str],
    eval_puzzle: str,
    args,
    sweep_results: dict,
) -> None:
    default_multi = ranker_path_for_train(train_puzzle)
    model_path = args.model if args.model != DEFAULT_MODEL else default_multi
    if not model_path.is_file():
        print(f"[WARN] ranker missing: {model_path}", file=sys.stderr)
        return

    train_label = "+".join(train_puzzle)
    print(
        f"\n======== eval {eval_puzzle} | train {train_label} ========",
        file=sys.stderr,
    )
    if args.phase in ("sweep", "all"):
        key = f"{eval_puzzle}_train_{train_label.replace('+', '_')}"
        sweep_results[key] = phase_sweep(
            eval_puzzle,
            model_path,
            args.out_dir,
            k_values=args.sweep_k,
            features_version=args.features,
            train_puzzle=train_label,
            cascade_l0_frac=args.cascade_l0,
        )
    if args.phase in ("sat", "all"):
        if args.top_k:
            k_list = args.top_k
        else:
            sk = f"{eval_puzzle}_train_{train_label.replace('+', '_')}"
            if sk in sweep_results:
                k_list = pick_k_star(sweep_results[sk], args.min_recall)
            else:
                tag = train_label.replace("+", "_")
                sweep_path = args.out_dir / f"sweep_{eval_puzzle}_train_{tag}.json"
                if sweep_path.is_file():
                    k_list = pick_k_star(
                        json.loads(sweep_path.read_text()), args.min_recall
                    )
                else:
                    k_list = [25.0]
        method_prefix = f"ranker_{train_label.replace('+', '_')}"
        phase_sat(
            eval_puzzle,
            k_list,
            model_path,
            args.kissat,
            args.seeds,
            solve_only=args.solve_only,
            skip_existing_seeds=args.skip_existing_seeds,
            features_version=args.features,
            method_prefix=method_prefix,
            cascade_l0_frac=args.cascade_l0,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sweep", "sat", "all"], default="all")
    ap.add_argument("--puzzle", nargs="+", choices=puzzle_names(), default=["v1", "v2"])
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--top-k", nargs="+", type=float, default=None)
    ap.add_argument("--sweep-k", nargs="+", type=float, default=SWEEP_K)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument("--min-recall", type=float, default=0.99)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "baselines" / "learned_shape")
    ap.add_argument("--solve-only", action="store_true")
    ap.add_argument("--skip-existing-seeds", action="store_true")
    ap.add_argument("--features", choices=["v1", "v2"], default="v2")
    ap.add_argument(
        "--cascade-l0",
        type=float,
        default=None,
        help="L0 heur 預篩比例（如 0.4 = 保留 top 40%% 再跑 ML）",
    )
    ap.add_argument("--loo", action="store_true", help="Leave-one-out: v1→v2 與 v2→v1")
    ap.add_argument("--train-puzzle", nargs="+", choices=puzzle_names(), default=None)
    ap.add_argument("--eval-puzzle", choices=puzzle_names(), default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_results: dict[str, dict] = {}
    if args.phase in ("sat", "all"):
        try:
            args.kissat = resolve_kissat(args.kissat)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

    if args.loo or (args.train_puzzle and args.eval_puzzle):
        folds = []
        if args.train_puzzle and args.eval_puzzle:
            folds = [(args.train_puzzle, args.eval_puzzle)]
        else:
            folds = [(["v1"], "v2"), (["v2"], "v1")]
        for train_p, eval_p in folds:
            run_loo_fold(train_p, eval_p, args, sweep_results)
    else:
        if args.phase in ("sweep", "all"):
            for puzzle in args.puzzle:
                sweep_results[puzzle] = phase_sweep(
                    puzzle,
                    args.model,
                    args.out_dir,
                    k_values=args.sweep_k,
                    features_version=args.features,
                    cascade_l0_frac=args.cascade_l0,
                )

        if args.phase in ("sat", "all"):
            for puzzle in args.puzzle:
                if args.top_k:
                    k_list = args.top_k
                elif puzzle in sweep_results:
                    k_list = pick_k_star(sweep_results[puzzle], args.min_recall)
                else:
                    sweep_path = args.out_dir / f"sweep_{puzzle}.json"
                    if sweep_path.is_file():
                        k_list = pick_k_star(
                            json.loads(sweep_path.read_text()), args.min_recall
                        )
                    else:
                        k_list = [25.0, 100.0]
                phase_sat(
                    puzzle,
                    k_list,
                    args.model,
                    args.kissat,
                    args.seeds,
                    solve_only=args.solve_only,
                    skip_existing_seeds=args.skip_existing_seeds,
                    features_version=args.features,
                    cascade_l0_frac=args.cascade_l0,
                )

    summary_path = args.out_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "phase": args.phase,
                "puzzles": args.puzzle,
                "loo": args.loo,
                "features": args.features,
                "model": str(args.model),
                "sweep": sweep_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"\n完成。{summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
