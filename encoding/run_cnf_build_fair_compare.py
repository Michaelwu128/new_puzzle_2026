#!/usr/bin/env python3
"""
同一時段連續量測各方法 build_cnf 耗時（公平對照）。

用法：
  python3 run_cnf_build_fair_compare.py --puzzle v6
  python3 run_cnf_build_fair_compare.py --puzzle v2 \\
    --methods baseline shape shape_prime shape_prime2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from generate_cnf import apply_puzzle, build_cnf, write_cnf
from puzzle_defs import PUZZLES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "baselines" / "learned_shape" / "ranker_train_v1_v2_v3_v4_v5.joblib"
DEFAULT_METHODS: dict[str, list[str]] = {
    "v6": ["baseline", "shape", "learned_shape"],
    "v2": ["baseline", "shape", "shape_prime", "shape_prime2"],
}
ALL_METHODS = [
    "baseline",
    "shape",
    "shape_prime",
    "shape_prime2",
    "learned_shape",
]


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


def time_build(
    method: str,
    puzzle: str,
    build_env: dict[str, str] | None,
) -> dict:
    print(f"\n{'='*60}\nbuild_cnf: {method}\n{'='*60}", file=sys.stderr)
    t0 = time.perf_counter()
    with env_patch(build_env or {}):
        total_vars, clauses, placements, _by_piece = build_cnf(method)
    build_sec = time.perf_counter() - t0

    out_dir = ROOT / "baselines" / puzzle / "cnf_build_fair"
    out_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = out_dir / f"{method}_fair.cnf"
    t1 = time.perf_counter()
    write_cnf(cnf_path, total_vars, clauses)
    write_sec = time.perf_counter() - t1

    row = {
        "method": method,
        "cnf_build_sec": round(build_sec, 3),
        "cnf_write_sec": round(write_sec, 3),
        "cnf_total_sec": round(build_sec + write_sec, 3),
        "vars": total_vars,
        "clauses": len(clauses),
        "placements": len(placements),
        "cnf_path": str(cnf_path),
    }
    print(
        f"  build={row['cnf_build_sec']:.1f}s  write={row['cnf_write_sec']:.1f}s  "
        f"clauses={row['clauses']:,}",
        file=sys.stderr,
    )
    return row


def write_md(path: Path, state: dict) -> None:
    rows = state["methods"]
    puzzle = state["puzzle"]
    lines = [
        f"# {puzzle} CNF build 公平對照（同一時段連續跑）",
        "",
        f"**run_at** = {state['run_at']}",
        f"**puzzle** = {puzzle}",
    ]
    if state.get("learned_model"):
        lines.append(f"**learned_model** = {state['learned_model']}")
    lines.extend(
        [
            "",
            "| 方法 | build_cnf (s) | write_cnf (s) | 合計 (s) | placements | clauses |",
            "|------|---------------|---------------|----------|------------|---------|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['cnf_build_sec']} | {r['cnf_write_sec']} | "
            f"{r['cnf_total_sec']} | {r['placements']:,} | {r['clauses']:,} |"
        )
    note = state.get("historical_note", "")
    if note:
        lines.extend(["", "## 歷史對照", "", note])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="v6")
    ap.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=None,
        help="預設：v6=baseline shape learned_shape；v2=baseline shape shape_prime shape_prime2",
    )
    ap.add_argument("--learned-model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--learned-top-k", type=float, default=25.0)
    ap.add_argument("--cascade-l0", type=float, default=0.4)
    args = ap.parse_args()

    methods = args.methods or DEFAULT_METHODS.get(
        args.puzzle, ["baseline", "shape", "learned_shape"]
    )

    apply_puzzle(args.puzzle)
    learned_env_dict = learned_env(args.learned_model, args.learned_top_k, args.cascade_l0)

    started = datetime.now(timezone.utc).astimezone()
    results: list[dict] = []
    for m in methods:
        env = learned_env_dict if m == "learned_shape" else None
        results.append(time_build(m, args.puzzle, env))

    historical_notes = {
        "v6": (
            "break-even: baseline 31.5s, learned 938.3s, shape 1232.5s (build only); "
            "l0 manifest: baseline 42.1s, learned 950.6s, shape 2061.4s."
        ),
        "v2": (
            "d4_next_sol_seeds (不同協議): baseline 13.3s, shape 717.8s, "
            "shape' 713.6s, shape'' 2339.2s."
        ),
    }

    state = {
        "puzzle": args.puzzle,
        "run_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "order": methods,
        "learned_model": str(args.learned_model) if "learned_shape" in methods else "",
        "learned_top_k": args.learned_top_k,
        "cascade_l0_frac": args.cascade_l0,
        "methods": results,
        "historical_note": historical_notes.get(args.puzzle, ""),
    }

    out_dir = ROOT / "baselines" / args.puzzle / "cnf_build_fair"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cnf_build_fair_compare.json"
    md_path = out_dir / "RESULTS.md"
    json_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(md_path, state)
    print(f"\n完成 → {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
