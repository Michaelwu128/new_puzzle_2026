#!/usr/bin/env python3
"""並排比較 v2-only vs v123 合訓 v4 L0@25% benchmark 結果。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "baselines" / "v4" / "l0_benchmark"
V2_JSON = OUT / "v4_l0_benchmark.json"
V123_JSON = OUT / "v4_l0_benchmark_v123.json"
COMPARE_JSON = OUT / "v4_l0_compare_v2_vs_v123.json"
COMPARE_MD = OUT / "v4_l0_compare_v2_vs_v123.md"


def _load(path: Path) -> dict:
    if not path.is_file():
        print(f"缺少 {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _learned_cnf(r: dict) -> dict:
    tag = r.get("learned_tag", "learned_l0_k25")
    return r.get("cnf", {}).get(tag) or r.get("cnf", {}).get("learned_l0_k25", {})


def _learned_sat(r: dict) -> dict:
    tag = r.get("learned_tag", "learned_l0_k25")
    return r.get("sat", {}).get(tag) or r.get("sat", {}).get("learned_l0_k25", {})


def main() -> None:
    v2 = _load(V2_JSON)
    v123 = _load(V123_JSON)

    shape_build = v2.get("preprocess_from_cnf", {}).get("shape_cnf_build_sec")
    if shape_build is None:
        shape_build = v2.get("cnf", {}).get("shape", {}).get("cnf_build_sec")

    baseline_sat = v2.get("sat", {}).get("baseline", {}).get("solve_mean_sec")
    shape_sat = v2.get("sat", {}).get("shape", {}).get("solve_mean_sec")

    v2_cnf = _learned_cnf(v2)
    v123_cnf = _learned_cnf(v123)
    v2_sat = _learned_sat(v2)
    v123_sat = _learned_sat(v123)

    compare = {
        "shape_cnf_build_sec": shape_build,
        "baseline_sat_mean_sec": baseline_sat,
        "shape_sat_mean_sec": shape_sat,
        "v2_only": {
            "model": v2.get("model"),
            "learned_tag": v2.get("learned_tag", "learned_l0_k25"),
            "cnf_build_sec": v2_cnf.get("cnf_build_sec"),
            "clauses_base": v2_cnf.get("clauses_base"),
            "extra_dead_clauses": (v2_cnf.get("clauses_base", 0) - 54876183)
            if v2_cnf.get("clauses_base")
            else None,
            "sat_mean_sec": v2_sat.get("solve_mean_sec"),
            "vs_baseline": v2_sat.get("vs_baseline"),
            "vs_shape_sat": round(shape_sat / v2_sat["solve_mean_sec"], 3)
            if shape_sat and v2_sat.get("solve_mean_sec")
            else None,
        },
        "v123": {
            "model": v123.get("model"),
            "learned_tag": v123.get("learned_tag"),
            "cnf_build_sec": v123_cnf.get("cnf_build_sec"),
            "clauses_base": v123_cnf.get("clauses_base"),
            "extra_dead_clauses": (v123_cnf.get("clauses_base", 0) - 54876183)
            if v123_cnf.get("clauses_base")
            else None,
            "sat_mean_sec": v123_sat.get("solve_mean_sec"),
            "vs_baseline": v123_sat.get("vs_baseline"),
            "vs_shape_sat": round(shape_sat / v123_sat["solve_mean_sec"], 3)
            if shape_sat and v123_sat.get("solve_mean_sec")
            else None,
        },
        "sweep_v4_at_25pct_recall": {
            "v2_only": 0.47,
            "v123": 0.421,
        },
    }
    if shape_build and v2_cnf.get("cnf_build_sec"):
        compare["v2_only"]["vs_shape_preprocess"] = round(
            shape_build / v2_cnf["cnf_build_sec"], 3
        )
    if shape_build and v123_cnf.get("cnf_build_sec"):
        compare["v123"]["vs_shape_preprocess"] = round(
            shape_build / v123_cnf["cnf_build_sec"], 3
        )

    COMPARE_JSON.write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# v4 L0@25%：v2-only vs v1+v2+v3 合訓",
        "",
        f"設定：L0=0.4，top-K=25%，D4 + 10 seeds",
        "",
        "## 預處理（CNF build）",
        "",
        "| | v2-only | v123 合訓 | shape（參考） |",
        "|--|---------|-----------|---------------|",
        f"| build (s) | {v2_cnf.get('cnf_build_sec')} | {v123_cnf.get('cnf_build_sec')} | {shape_build} |",
        f"| 額外 dead 子句 | {compare['v2_only']['extra_dead_clauses']:,} | {compare['v123']['extra_dead_clauses']:,} | 1,630,496 |",
        f"| vs shape 速度 | {compare['v2_only'].get('vs_shape_preprocess')}× | {compare['v123'].get('vs_shape_preprocess')}× | 1.00× |",
        "",
        "## SAT 平均（秒）",
        "",
        "| | v2-only | v123 合訓 | shape | baseline |",
        "|--|---------|-----------|-------|----------|",
        f"| mean | {v2_sat.get('solve_mean_sec')} | {v123_sat.get('solve_mean_sec')} | {shape_sat} | {baseline_sat} |",
        f"| vs baseline | {v2_sat.get('vs_baseline')}× | {v123_sat.get('vs_baseline')}× | {round(baseline_sat/shape_sat,3) if baseline_sat and shape_sat else '-'}× | 1.00× |",
        f"| vs shape | {compare['v2_only'].get('vs_shape_sat')}× | {compare['v123'].get('vs_shape_sat')}× | 1.00× | — |",
        "",
        "## Sweep @25% recall（先前 zero-shot）",
        "",
        "| 模型 | recall |",
        "|------|--------|",
        "| v2-only | 47.0% |",
        "| v123 | 42.1% |",
        "",
        f"完整 JSON：`{COMPARE_JSON.name}`",
    ]
    COMPARE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(COMPARE_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
