#!/usr/bin/env python3
"""
封鎖 puzzle 已知解後，用多個 Kissat seed 各找「下一個 SAT 解」，比較方法耗時。

用法：
  python3 run_next_sol_seeds.py --puzzle v1 --methods baseline shape --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from generate_cnf import apply_puzzle, build_cnf, write_cnf
from kissat_utils import KISSAT_DEFAULT, resolve_kissat


def run_kissat(kissat: Path, cnf: Path, sol_out: Path, seed: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(kissat), f"--seed={seed}", str(cnf), "-q"],
        capture_output=True,
        text=True,
        encoding="latin-1",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - t0
    sol_out.write_text(proc.stdout, encoding="latin-1", errors="replace")
    status = "UNKNOWN"
    for line in proc.stdout.splitlines():
        if line.startswith("s "):
            status = line[2:].strip()
            break
    return status, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=["v1", "v2"], default="v1")
    ap.add_argument("--methods", nargs="+", default=["baseline", "shape"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="預設 baselines/<puzzle>/next_sol_seeds/",
    )
    args = ap.parse_args()

    try:
        args.kissat = resolve_kissat(args.kissat)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "baselines" / args.puzzle / "next_sol_seeds"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_puzzle(args.puzzle)
    summary: dict = {"puzzle": args.puzzle, "seeds": args.seeds, "methods": {}}

    for method in args.methods:
        print(f"\n=== {args.puzzle} | {method} ===", file=sys.stderr)
        total_vars, clauses, placements, _ = build_cnf(
            method, block_known_solution=True
        )
        n_placements = len(placements)
        cnf_path = out_dir / f"{method}.cnf"
        write_cnf(cnf_path, total_vars, clauses)

        times: list[float] = []
        per_seed: list[dict] = []
        for seed in args.seeds:
            sol_path = out_dir / f"{method}_seed{seed}.txt"
            status, elapsed = run_kissat(args.kissat, cnf_path, sol_path, seed)
            times.append(elapsed)
            per_seed.append({"seed": seed, "status": status, "sec": round(elapsed, 3)})
            print(f"  seed={seed}  {status}  {elapsed:.2f}s", file=sys.stderr)
            if status != "SATISFIABLE":
                print(f"  WARNING: expected SATISFIABLE, got {status}", file=sys.stderr)

        mean_t = statistics.mean(times)
        summary["methods"][method] = {
            "placements": n_placements,
            "vars": total_vars,
            "clauses": len(clauses),
            "per_seed": per_seed,
            "mean_sec": round(mean_t, 3),
            "stdev_sec": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        }

    baseline_mean = summary["methods"].get("baseline", {}).get("mean_sec")
    if baseline_mean and "shape" in summary["methods"]:
        summary["methods"]["shape"]["vs_baseline"] = round(
            baseline_mean / summary["methods"]["shape"]["mean_sec"], 3
        )

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
