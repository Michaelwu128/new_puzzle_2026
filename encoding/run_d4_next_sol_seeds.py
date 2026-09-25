#!/usr/bin/env python3
"""
封鎖已知解的 D4 對稱後，用多個 Kissat seed 各找「下一個 SAT 解」。

特點：
  - 產出 base / d4 CNF 並快取，重跑求解可 --solve-only 跳過 CNF 生成
  - 記錄 CNF 產生各階段耗時（cnf_manifest.json）
  - 解碼後驗證 no-touch，並統計各 seed 解在 D4 意義下的互異性

用法：
  # 第一次：產 CNF + 求解（預設 seed 1..10）
  python3 run_d4_next_sol_seeds.py --puzzle v1

  # 只產 CNF（shape'' 很慢時可先跑）
  python3 run_d4_next_sol_seeds.py --puzzle v1 --cnf-only

  # 重用已存 CNF，只重跑 Kissat
  python3 run_d4_next_sol_seeds.py --puzzle v1 --solve-only

  # 強制重產 CNF
  python3 run_d4_next_sol_seeds.py --puzzle v1 --force-regen-cnf
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cnf_io import append_blocking_clauses, read_cnf_header
from kissat_utils import KISSAT_DEFAULT, resolve_kissat
from generate_cnf import (
    BOARD_H,
    BOARD_W,
    KNOWN_SOLUTION,
    NUM_PIECES,
    apply_puzzle,
    build_cnf,
    write_cnf,
)
from puzzle_defs import PUZZLES
from solution_blocking import (
    D4_TRANSFORMS,
    build_placement_index,
    parse_kissat_true_vars,
    piece_cells_from_known,
    true_vars_to_piece_cells,
)

DEFAULT_METHODS = ["baseline", "shape", "shape_prime", "shape_prime2"]
DEFAULT_SEEDS = list(range(1, 11))


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
        "piece_cells": {
            str(p): [list(c) for c in sorted(cells)] for p, cells in piece_cells.items()
        },
        "board": board_to_rows(piece_cells),
        "valid": verify_no_touch(piece_cells),
    }


def collect_block_clauses(
    piece_cells: dict[int, frozenset],
    plc_index: dict[tuple[int, frozenset], int],
) -> list[list[int]]:
    from solution_blocking import all_symmetry_block_clauses

    return [bv for _, bv in all_symmetry_block_clauses(piece_cells, plc_index)]


def d4_equivalent(
    a: dict[int, frozenset],
    b: dict[int, frozenset],
) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for tfn in D4_TRANSFORMS.values():
        transformed = {
            pid: frozenset(tfn(r, c) for r, c in cells) for pid, cells in a.items()
        }
        if all(transformed[p] == b[p] for p in b):
            return True
    return False


def analyze_d4_uniqueness(
    per_seed: list[dict],
    known_pc: dict[int, frozenset],
) -> dict:
    """依 D4 等價分類各 seed 的解；並檢查是否與已知解 D4 等價。"""
    classes: list[dict[int, frozenset]] = []
    class_by_seed: dict[int, int] = {}
    equivalent_to_known: dict[int, bool] = {}

    for rec in per_seed:
        seed = rec["seed"]
        pc_raw = rec.get("_piece_cells")
        if pc_raw is None or rec.get("status") != "SATISFIABLE":
            continue

        equivalent_to_known[seed] = d4_equivalent(pc_raw, known_pc)

        assigned = None
        for cid, rep in enumerate(classes):
            if d4_equivalent(pc_raw, rep):
                assigned = cid
                break
        if assigned is None:
            assigned = len(classes)
            classes.append(pc_raw)
        class_by_seed[seed] = assigned

    seeds_per_class: list[list[int]] = [[] for _ in classes]
    for seed, cid in class_by_seed.items():
        seeds_per_class[cid].append(seed)

    return {
        "solutions_unique_d4": len(classes),
        "d4_class_by_seed": {str(k): v for k, v in sorted(class_by_seed.items())},
        "seeds_per_d4_class": [sorted(s) for s in seeds_per_class],
        "equivalent_to_known_by_seed": {
            str(k): v for k, v in sorted(equivalent_to_known.items())
        },
        "any_equivalent_to_known": any(equivalent_to_known.values()),
    }


def save_placements_meta(path: Path, placements: list[dict]) -> None:
    meta = {
        "placements": [
            {"var": p["var"], "piece": p["piece"], "cells": [list(c) for c in p["cells"]]}
            for p in placements
        ]
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_var_to_plc(meta_path: Path) -> dict[int, dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    var_to_plc: dict[int, dict] = {}
    for p in meta["placements"]:
        var_to_plc[p["var"]] = {
            "var": p["var"],
            "piece": p["piece"],
            "cells": [tuple(c) for c in p["cells"]],
        }
    return var_to_plc


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


def cnf_paths(out_dir: Path, method: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / f"{method}_base.cnf",
        out_dir / f"{method}_d4.cnf",
        out_dir / f"{method}_meta.json",
    )


def cnf_cache_valid(out_dir: Path, method: str) -> bool:
    base, d4, meta = cnf_paths(out_dir, method)
    return base.is_file() and d4.is_file() and meta.is_file()


def generate_cnf(
    method: str,
    out_dir: Path,
    *,
    force: bool,
) -> dict:
    base_path, d4_path, meta_path = cnf_paths(out_dir, method)
    manifest_path = out_dir / "cnf_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    if not force and cnf_cache_valid(out_dir, method):
        prev = manifest.get(method, {})
        print(f"  [快取] 沿用既有 CNF: {d4_path.name}", file=sys.stderr)
        return {**prev, "reused": True}

    print(f"\n=== 產生 CNF | {method} ===", file=sys.stderr)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    total_vars, clauses, placements, by_piece = build_cnf(method)
    cnf_build_sec = time.perf_counter() - t0

    plc_index = build_placement_index(by_piece)

    t0 = time.perf_counter()
    write_cnf(base_path, total_vars, clauses)
    cnf_write_base_sec = time.perf_counter() - t0

    known_pc = piece_cells_from_known(KNOWN_SOLUTION)
    d4_blocks = collect_block_clauses(known_pc, plc_index)

    shutil.copy2(base_path, d4_path)
    t0 = time.perf_counter()
    append_blocking_clauses(d4_path, d4_blocks)
    cnf_write_d4_sec = time.perf_counter() - t0

    save_placements_meta(meta_path, placements)

    _, clauses_d4 = read_cnf_header(d4_path)
    cnf_total_sec = time.perf_counter() - t_total

    entry = {
        "placements": len(placements),
        "vars": total_vars,
        "clauses_base": len(clauses),
        "clauses_d4": clauses_d4,
        "d4_blocking_count": len(d4_blocks),
        "cnf_build_sec": round(cnf_build_sec, 3),
        "cnf_write_base_sec": round(cnf_write_base_sec, 3),
        "cnf_write_d4_sec": round(cnf_write_d4_sec, 3),
        "cnf_total_sec": round(cnf_total_sec, 3),
        "base_cnf_path": base_path.name,
        "d4_cnf_path": d4_path.name,
        "meta_path": meta_path.name,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "reused": False,
    }
    print(
        f"  placements={entry['placements']}  base_clauses={entry['clauses_base']:,}  "
        f"d4_clauses={entry['clauses_d4']:,}  d4_blocking={entry['d4_blocking_count']}",
        file=sys.stderr,
    )
    print(
        f"  CNF 耗時: build={entry['cnf_build_sec']:.1f}s  "
        f"write_base={entry['cnf_write_base_sec']:.1f}s  "
        f"write_d4={entry['cnf_write_d4_sec']:.3f}s  "
        f"total={entry['cnf_total_sec']:.1f}s",
        file=sys.stderr,
    )

    manifest[method] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return entry


def solve_method(
    method: str,
    seeds: list[int],
    out_dir: Path,
    kissat: Path,
    known_pc: dict[int, frozenset],
    cnf_entry: dict,
) -> dict:
    _, d4_path, meta_path = cnf_paths(out_dir, method)
    if not d4_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"缺少 {d4_path.name} 或 {meta_path.name}，請先執行 --cnf-only 或完整跑法"
        )

    var_to_plc = load_var_to_plc(meta_path)
    print(f"\n=== 求解 | {method} | seeds={seeds} ===", file=sys.stderr)

    per_seed: list[dict] = []
    solve_times: list[float] = []

    for seed in seeds:
        sol_path = out_dir / f"{method}_seed{seed}.txt"
        status, elapsed = run_kissat(kissat, d4_path, sol_path, seed)
        rec: dict = {
            "seed": seed,
            "status": status,
            "solve_sec": round(elapsed, 3),
        }

        if status == "SATISFIABLE":
            solve_times.append(elapsed)
            true_vars = parse_kissat_true_vars(
                sol_path.read_text(encoding="latin-1", errors="replace")
            )
            pc = true_vars_to_piece_cells(true_vars, var_to_plc)
            sol_json = piece_cells_to_json(pc)
            rec.update(sol_json)
            rec["_piece_cells"] = pc
            print(
                f"  seed={seed}  {status}  {elapsed:.2f}s  valid={rec['valid']}",
                file=sys.stderr,
            )
            if not rec["valid"]:
                print(f"  [WARN] seed={seed} 未通過 no-touch", file=sys.stderr)
        else:
            print(f"  seed={seed}  {status}  {elapsed:.2f}s", file=sys.stderr)
            print(f"  [WARN] 預期 SATISFIABLE，得到 {status}", file=sys.stderr)

        per_seed.append(rec)

    d4_stats = analyze_d4_uniqueness(per_seed, known_pc)

    # 輸出 JSON 前移除內部欄位
    per_seed_out: list[dict] = []
    for rec in per_seed:
        out = {k: v for k, v in rec.items() if k != "_piece_cells"}
        per_seed_out.append(out)

    result: dict = {
        "placements": cnf_entry.get("placements"),
        "vars": cnf_entry.get("vars"),
        "clauses_base": cnf_entry.get("clauses_base"),
        "clauses_d4": cnf_entry.get("clauses_d4"),
        "d4_blocking_count": cnf_entry.get("d4_blocking_count"),
        "cnf": cnf_entry,
        "per_seed": per_seed_out,
        **d4_stats,
    }

    if solve_times:
        result["solve_mean_sec"] = round(statistics.mean(solve_times), 3)
        result["solve_median_sec"] = round(statistics.median(solve_times), 3)
        result["solve_stdev_sec"] = (
            round(statistics.stdev(solve_times), 3) if len(solve_times) > 1 else 0.0
        )
        result["solve_times_sec"] = [round(t, 3) for t in solve_times]
    else:
        result["solve_mean_sec"] = None
        result["solve_median_sec"] = None
        result["solve_stdev_sec"] = None
        result["solve_times_sec"] = []

    print(
        f"  D4 互異解: {d4_stats['solutions_unique_d4']}/{len(seeds)}  "
        f"classes={d4_stats['seeds_per_d4_class']}",
        file=sys.stderr,
    )
    if d4_stats["any_equivalent_to_known"]:
        print("  [WARN] 有解與已知解 D4 等價（blocking 可能異常）", file=sys.stderr)

    return result


def _fmt_int(n: object) -> str:
    return f"{int(n):,}" if isinstance(n, int) else "—"


def write_results_md(path: Path, summary: dict) -> None:
    methods = summary.get("methods", {})
    seeds = summary.get("seeds", [])
    lines = [
        f"# {summary['puzzle']}：D4 blocking · 下一解（{len(seeds)} seed）",
        "",
        "**指令**：`encoding/run_d4_next_sol_seeds.py`",
        "",
        "- 封鎖已知解 **D4 對稱**（8 條 blocking）",
        f"- Kissat `--seed={seeds[0]}..{seeds[-1]}`，各找 1 解",
        "- 詳細：`results.json`、`cnf_manifest.json`",
        "",
        "## CNF 產生耗時",
        "",
        "| 方法 | placements | base 子句 | d4 子句 | CNF total (s) | 快取 |",
        "|------|------------|-----------|---------|---------------|------|",
    ]
    manifest = summary.get("cnf_manifest", {})
    for m in summary.get("methods_run", []):
        c = manifest.get(m, methods.get(m, {}).get("cnf", {}))
        reused = "是" if c.get("reused") else "否"
        lines.append(
            f"| {m} | {c.get('placements', '—')} | "
            f"{_fmt_int(c.get('clauses_base'))} | {_fmt_int(c.get('clauses_d4'))} | "
            f"{c.get('cnf_total_sec', '—')} | {reused} |"
        )

    baseline_mean = methods.get("baseline", {}).get("solve_mean_sec")
    lines.extend(["", "## 求解時間（秒）", ""])
    header = "| seed |" + "".join(f" {m} |" for m in summary.get("methods_run", []))
    sep = "|------|" + "------|" * len(summary.get("methods_run", []))
    lines.extend([header, sep])
    for seed in seeds:
        row = f"| {seed} |"
        for m in summary.get("methods_run", []):
            per = {r["seed"]: r for r in methods.get(m, {}).get("per_seed", [])}
            r = per.get(seed)
            if r and r.get("status") == "SATISFIABLE":
                row += f" {r['solve_sec']} |"
            else:
                row += f" {r['status'] if r else '—'} |"
        lines.append(row)

    lines.append("| **平均** |")
    for m in summary.get("methods_run", []):
        mean = methods.get(m, {}).get("solve_mean_sec")
        lines[-1] += f" **{mean}** |" if mean is not None else " — |"

    lines.append("| **vs baseline** |")
    for m in summary.get("methods_run", []):
        if m == "baseline":
            lines[-1] += " 1× |"
        else:
            vs = methods.get(m, {}).get("vs_baseline")
            lines[-1] += f" ≈{vs}× |" if vs else " — |"

    lines.extend(["", "## D4 互異解（各方法）", ""])
    for m in summary.get("methods_run", []):
        md = methods.get(m, {})
        u = md.get("solutions_unique_d4", "—")
        classes = md.get("seeds_per_d4_class", [])
        warn = " ⚠ 含與已知解 D4 等價" if md.get("any_equivalent_to_known") else ""
        lines.append(f"- **{m}**：{u} 個 D4 等價類 {classes}{warn}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="D4 blocking 後多 seed 各找下一解")
    ap.add_argument("--puzzle", choices=sorted(PUZZLES), default="v1")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--kissat", default=KISSAT_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--cnf-only",
        action="store_true",
        help="只產生並快取 CNF，不跑 Kissat",
    )
    ap.add_argument(
        "--solve-only",
        action="store_true",
        help="只跑 Kissat（需已有 *_d4.cnf 與 cnf_manifest.json）",
    )
    ap.add_argument(
        "--force-regen-cnf",
        action="store_true",
        help="忽略快取，強制重產 CNF",
    )
    args = ap.parse_args()

    if args.cnf_only and args.solve_only:
        print("不能同時指定 --cnf-only 與 --solve-only", file=sys.stderr)
        sys.exit(1)

    if not args.cnf_only:
        try:
            args.kissat = resolve_kissat(args.kissat)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (root / "baselines" / args.puzzle / "d4_next_sol_seeds")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_puzzle(args.puzzle)
    known_pc = piece_cells_from_known(KNOWN_SOLUTION)

    manifest_path = out_dir / "cnf_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    do_cnf = not args.solve_only
    do_solve = not args.cnf_only

    if do_cnf:
        for method in args.methods:
            manifest[method] = generate_cnf(
                method, out_dir, force=args.force_regen_cnf
            )

    summary: dict = {
        "puzzle": args.puzzle,
        "seeds": args.seeds,
        "blocking": "d4_known_solution",
        "d4_blocking_count": 8,
        "methods_run": args.methods,
        "cnf_manifest": manifest,
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "methods": {},
    }

    if do_solve:
        for method in args.methods:
            cnf_entry = manifest.get(method)
            if not cnf_entry:
                if cnf_cache_valid(out_dir, method):
                    _, d4_path, _ = cnf_paths(out_dir, method)
                    _, clauses_d4 = read_cnf_header(d4_path)
                    cnf_entry = {
                        "d4_cnf_path": d4_path.name,
                        "clauses_d4": clauses_d4,
                        "reused": True,
                    }
                else:
                    print(f"[ERROR] 方法 {method} 無 CNF，請先 --cnf-only", file=sys.stderr)
                    sys.exit(1)

            summary["methods"][method] = solve_method(
                method,
                args.seeds,
                out_dir,
                args.kissat,
                known_pc,
                cnf_entry,
            )

        baseline_mean = summary["methods"].get("baseline", {}).get("solve_mean_sec")
        if baseline_mean:
            for m, md in summary["methods"].items():
                mean = md.get("solve_mean_sec")
                if mean and mean > 0:
                    md["vs_baseline"] = (
                        round(baseline_mean / mean, 3) if m != "baseline" else 1.0
                    )

    results_path = out_dir / "results.json"
    merged: dict = {}
    if results_path.is_file():
        try:
            merged = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    for key, val in summary.items():
        if key == "methods":
            merged.setdefault("methods", {})
            merged["methods"].update(val)
        else:
            merged[key] = val
    results_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if do_solve and summary["methods"]:
        write_results_md(out_dir / "RESULTS.md", merged)

    print(f"\n完成。結果目錄: {out_dir}", file=sys.stderr)
    if do_solve:
        for m in args.methods:
            md = summary["methods"].get(m, {})
            mean = md.get("solve_mean_sec")
            u = md.get("solutions_unique_d4")
            print(
                f"  {m:12s}  平均 {mean}s  D4互異 {u}/{len(args.seeds)}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
