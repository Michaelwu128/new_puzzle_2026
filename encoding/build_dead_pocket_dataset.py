#!/usr/bin/env python3
"""
從 formal oracle 產生 dead-pocket 訓練資料（分層 neg 抽樣 + 全保留 positive）。

用法：
  python3 build_dead_pocket_dataset.py --puzzle v1 v2 --features v2
  python3 build_dead_pocket_dataset.py --puzzle v2 --max-samples 400000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from puzzle_defs import puzzle_names
from generate_cnf import apply_puzzle, build_placements, has_dead_pocket
from dead_pocket_pairs import build_exc_cache, iter_candidate_pairs, stratify_bucket


def _feature_module(version: str):
    if version == "v2":
        from dead_pocket_features_v2 import FEATURE_NAMES, extract_pair_features

        return FEATURE_NAMES, extract_pair_features
    from dead_pocket_features import FEATURE_NAMES, extract_pair_features

    return FEATURE_NAMES, extract_pair_features


def reservoir_add_item(reservoir: list[dict], item: dict, k: int, seen: int) -> None:
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = random.randint(0, seen - 1)
        if j < k:
            reservoir[j] = item


def build_dataset(
    puzzle: str,
    out_path: Path,
    *,
    max_neg_samples: int,
    max_pos_samples: int,
    time_limit_sec: float | None,
    seed: int,
    edge_margin: int,
    features_version: str = "v2",
    stratified: bool = True,
) -> dict:
    random.seed(seed)
    feature_names, extract_pair_features = _feature_module(features_version)
    apply_puzzle(puzzle)
    placements, by_piece, _ = build_placements("baseline")
    exc_cache = build_exc_cache(placements)

    neg_quotas = {
        "random": max_neg_samples // 2,
        "near_edge": max_neg_samples // 4,
        "band_overlap": max_neg_samples // 4,
    }
    neg_seen = {"random": 0, "near_edge": 0, "band_overlap": 0}
    neg_reservoirs: dict[str, list[dict]] = {k: [] for k in neg_quotas}
    neg_reservoir: list[dict] = []
    pos_reservoir: list[dict] = []
    n_scanned = 0
    n_pos_seen = 0
    n_neg_seen = 0

    t0 = time.perf_counter()
    for plc_a, plc_b in iter_candidate_pairs(placements, by_piece, edge_margin=edge_margin):
        n_scanned += 1
        if time_limit_sec and time.perf_counter() - t0 > time_limit_sec:
            print(f"  [時間上限 {time_limit_sec}s] 停止掃描", file=sys.stderr)
            break

        label = has_dead_pocket(plc_a["cells"], plc_b["cells"], 12, 12)
        feat = extract_pair_features(
            plc_a, plc_b, exc_a=exc_cache[plc_a["var"]], exc_b=exc_cache[plc_b["var"]]
        )
        row = {
            **feat,
            "label": int(label),
            "var_a": plc_a["var"],
            "var_b": plc_b["var"],
            "puzzle": puzzle,
        }

        if label:
            n_pos_seen += 1
            reservoir_add_item(pos_reservoir, row, max_pos_samples, n_pos_seen)
        else:
            n_neg_seen += 1
            if stratified:
                primary = stratify_bucket(plc_a, plc_b)
                # 首選桶滿時 fallback，避免 neg 總量卡住無法早停
                bucket = primary
                for b in (primary, "random", "band_overlap", "near_edge"):
                    if len(neg_reservoirs[b]) < neg_quotas[b]:
                        bucket = b
                        break
                neg_seen[bucket] += 1
                cap = neg_quotas[bucket]
                reservoir_add_item(neg_reservoirs[bucket], row, cap, neg_seen[bucket])
            else:
                reservoir_add_item(neg_reservoir, row, max_neg_samples, n_neg_seen)

        if n_scanned % 500_000 == 0:
            elapsed = time.perf_counter() - t0
            neg_kept = sum(len(v) for v in neg_reservoirs.values()) if stratified else len(neg_reservoir)
            print(
                f"  scanned={n_scanned:,}  pos_seen={n_pos_seen:,}  "
                f"neg_seen={n_neg_seen:,}  pos_kept={len(pos_reservoir):,}  "
                f"neg_kept={neg_kept:,}  {elapsed:.1f}s",
                file=sys.stderr,
            )

        if stratified:
            neg_total = sum(len(v) for v in neg_reservoirs.values())
            neg_full = neg_total >= max_neg_samples
        else:
            neg_total = len(neg_reservoir)
            neg_full = neg_total >= max_neg_samples
        # neg 桶滿且已掃描足夠 pair 後早停（ranker 訓練會 subsample，不必掃完所有 positive）
        if neg_full and n_scanned >= max_neg_samples:
            print(
                f"  [早停] neg={neg_total:,}  pos_kept={len(pos_reservoir):,}",
                file=sys.stderr,
            )
            break

    if stratified:
        neg_rows = [r for b in neg_reservoirs for r in neg_reservoirs[b]]
    else:
        neg_rows = neg_reservoir
    rows: list[dict] = list(pos_reservoir) + neg_rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = feature_names + ["label", "var_a", "var_b", "puzzle"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    elapsed = time.perf_counter() - t0
    stats = {
        "puzzle": puzzle,
        "features_version": features_version,
        "stratified": stratified,
        "max_neg_samples": max_neg_samples,
        "max_pos_samples": max_pos_samples,
        "neg_quotas": neg_quotas if stratified else None,
        "edge_margin": edge_margin,
        "n_scanned": n_scanned,
        "n_pos_seen": n_pos_seen,
        "n_neg_seen": n_neg_seen,
        "n_pos_kept": len(pos_reservoir),
        "n_neg_kept": len(neg_rows),
        "n_rows_written": len(rows),
        "positive_rate_scanned": round(n_pos_seen / n_scanned, 6) if n_scanned else 0,
        "positive_rate_written": round(len(pos_reservoir) / len(rows), 6) if rows else 0,
        "build_sec": round(elapsed, 3),
        "output": str(out_path),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", nargs="+", choices=puzzle_names(), default=["v1", "v2"])
    ap.add_argument("--max-samples", type=int, default=400_000, help="負樣本總 reservoir 上限")
    ap.add_argument("--max-pos-samples", type=int, default=2_000_000)
    ap.add_argument("--time-limit-sec", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edge-margin", type=int, default=6)
    ap.add_argument("--features", choices=["v1", "v2"], default="v2")
    ap.add_argument("--no-stratified", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (root / "baselines" / "learned_shape")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.features == "v1" else "_v2"
    all_stats: dict = {"puzzles": {}, "features": args.features}
    for puzzle in args.puzzle:
        print(f"\n=== 資料集 | {puzzle} | features={args.features} ===", file=sys.stderr)
        out_path = out_dir / f"dataset_{puzzle}{suffix}.csv"
        stats = build_dataset(
            puzzle,
            out_path,
            max_neg_samples=args.max_samples,
            max_pos_samples=args.max_pos_samples,
            time_limit_sec=args.time_limit_sec,
            seed=args.seed,
            edge_margin=args.edge_margin,
            features_version=args.features,
            stratified=not args.no_stratified,
        )
        all_stats["puzzles"][puzzle] = stats
        print(
            f"  寫出 {out_path.name}: {stats['n_rows_written']:,} rows  "
            f"(pos={stats['n_pos_kept']:,}, neg={stats['n_neg_kept']:,}, "
            f"pos_rate={stats['positive_rate_written']:.3f})  {stats['build_sec']:.1f}s",
            file=sys.stderr,
        )

    meta_path = out_dir / f"dataset_meta{suffix}.json"
    meta_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n")
    print(f"\n完成。{meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
