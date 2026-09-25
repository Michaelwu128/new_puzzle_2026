#!/usr/bin/env python3
"""
AI 預篩 + formal oracle 閘門的 learned shape 子句生成。
支援 v1/v2 特徵與 LightGBM ranker / sklearn 分類器。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from dead_pocket_pairs import build_exc_cache, iter_candidate_pairs
from generate_cnf import BOARD_H, BOARD_W, has_dead_pocket


def _feature_module(version: str):
    if version == "v2":
        from dead_pocket_features_v2 import FEATURE_NAMES, extract_pair_features

        return FEATURE_NAMES, extract_pair_features
    from dead_pocket_features import FEATURE_NAMES, extract_pair_features

    return FEATURE_NAMES, extract_pair_features


def load_model(model_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    # legacy RandomForest bundle
    feat_v = os.environ.get("DEAD_POCKET_FEATURES", "v1")
    feat_names, _ = _feature_module(feat_v)
    return {
        "model": bundle.get("model", bundle),
        "model_type": "sklearn_classifier",
        "feature_names": bundle.get("feature_names", feat_names),
        "features_version": feat_v,
        "heur_blend": 0.0,
    }


def predict_scores(model_bundle: dict[str, Any], feat_matrix: np.ndarray) -> np.ndarray:
    model = model_bundle["model"]
    model_type = model_bundle.get("model_type", "sklearn_classifier")
    heur_blend = float(model_bundle.get("heur_blend", 0.0))
    feat_names = model_bundle.get("feature_names", [])
    heur_idx = feat_names.index("heur_score") if "heur_score" in feat_names else None

    if model_type == "lightgbm_ranker":
        ml = model.predict(feat_matrix)
    elif hasattr(model, "predict_proba"):
        ml = model.predict_proba(feat_matrix)[:, 1]
    else:
        ml = model.predict(feat_matrix)

    if heur_idx is not None and heur_blend > 0:
        heur = feat_matrix[:, heur_idx]
        hmin, hmax = float(heur.min()), float(heur.max())
        heur_n = (heur - hmin) / (hmax - hmin) if hmax > hmin else heur
        ml_min, ml_max = float(ml.min()), float(ml.max())
        ml_n = (ml - ml_min) / (ml_max - ml_min) if ml_max > ml_min else ml
        return (1 - heur_blend) * ml_n + heur_blend * heur_n
    return np.asarray(ml, dtype=np.float64)


def score_candidate_pairs(
    placements: list[dict],
    by_piece,
    *,
    model_path: Path,
    edge_margin: int = 6,
    features_version: str | None = None,
    cascade_l0_frac: float | None = None,
) -> tuple[list[tuple[dict, dict, float]], dict[str, Any]]:
    """特徵提取 + 打分。cascade_l0_frac 設定時 L0 heur 預篩，僅對存活 pair 跑完整 ML。"""
    bundle = load_model(model_path)
    feat_v = features_version or bundle.get("features_version") or os.environ.get(
        "DEAD_POCKET_FEATURES", "v2"
    )
    feat_names, extract_pair_features = _feature_module(feat_v)
    if bundle.get("feature_names"):
        feat_names = bundle["feature_names"]

    exc_cache = build_exc_cache(placements)
    pair_list: list[tuple[dict, dict]] = list(
        iter_candidate_pairs(placements, by_piece, edge_margin=edge_margin)
    )
    n_total = len(pair_list)
    meta: dict[str, Any] = {
        "n_pairs_total": n_total,
        "model_type": bundle.get("model_type"),
        "features_version": feat_v,
        "cascade_l0_frac": cascade_l0_frac,
    }
    if n_total == 0:
        return [], meta

    ml_indices: list[int]
    t_l0 = time.perf_counter()
    if cascade_l0_frac is not None and 0 < cascade_l0_frac < 1.0:
        from dead_pocket_features_v2 import cheap_heur_score

        l0_scores = [
            cheap_heur_score(
                plc_a,
                plc_b,
                exc_a=exc_cache[plc_a["var"]],
                exc_b=exc_cache[plc_b["var"]],
            )
            for plc_a, plc_b in pair_list
        ]
        k_l0 = max(1, int(np.ceil(n_total * cascade_l0_frac)))
        ml_indices = list(np.argsort(-np.asarray(l0_scores, dtype=np.float64))[:k_l0])
        meta["l0_sec"] = round(time.perf_counter() - t_l0, 3)
        meta["n_l0_kept"] = len(ml_indices)
    else:
        ml_indices = list(range(n_total))
        meta["l0_sec"] = 0.0
        meta["n_l0_kept"] = n_total

    t_ml = time.perf_counter()
    all_scores = np.full(n_total, -np.inf, dtype=np.float64)
    n_ml = len(ml_indices)
    feat_matrix = np.empty((n_ml, len(feat_names)), dtype=np.float64)
    for j, i in enumerate(ml_indices):
        plc_a, plc_b = pair_list[i]
        feat = extract_pair_features(
            plc_a,
            plc_b,
            exc_a=exc_cache[plc_a["var"]],
            exc_b=exc_cache[plc_b["var"]],
        )
        feat_matrix[j] = [feat[n] for n in feat_names]

    ml_scores = predict_scores(bundle, feat_matrix)
    for j, i in enumerate(ml_indices):
        all_scores[i] = ml_scores[j]
    meta["ml_sec"] = round(time.perf_counter() - t_ml, 3)

    ranked = sorted(
        (
            (pair_list[i][0], pair_list[i][1], float(all_scores[i]))
            for i in range(n_total)
        ),
        key=lambda x: -x[2],
    )
    return ranked, meta


def oracle_selected_pairs(
    selected: list[tuple[dict, dict, float]],
    *,
    full_dead_pairs: set[tuple[int, int]] | None = None,
    top_k_percent: float | None = None,
) -> tuple[list[list[int]], dict]:
    extra: list[list[int]] = []
    found_pairs: set[tuple[int, int]] = set()
    t0 = time.perf_counter()
    for plc_a, plc_b, _score in selected:
        if has_dead_pocket(plc_a["cells"], plc_b["cells"], BOARD_H, BOARD_W):
            va, vb = plc_a["var"], plc_b["var"]
            extra.append([-va, -vb])
            found_pairs.add((min(va, vb), max(va, vb)))
    oracle_sec = time.perf_counter() - t0
    clause_recall = 1.0
    if full_dead_pairs is not None and full_dead_pairs:
        clause_recall = len(found_pairs & full_dead_pairs) / len(full_dead_pairs)
    stats = {
        "n_oracle_calls": len(selected),
        "n_clauses": len(extra),
        "oracle_sec": round(oracle_sec, 3),
        "top_k_percent": top_k_percent,
        "clause_recall_vs_full": round(clause_recall, 4),
    }
    return extra, stats


def compute_full_dead_pair_set(
    placements: list[dict],
    by_piece,
    *,
    edge_margin: int = 6,
) -> set[tuple[int, int]]:
    dead: set[tuple[int, int]] = set()
    for plc_a, plc_b in iter_candidate_pairs(
        placements, by_piece, edge_margin=edge_margin
    ):
        if has_dead_pocket(plc_a["cells"], plc_b["cells"], BOARD_H, BOARD_W):
            va, vb = plc_a["var"], plc_b["var"]
            dead.add((min(va, vb), max(va, vb)))
    return dead


def compute_learned_shape_clauses(
    placements: list[dict],
    by_piece,
    *,
    model_path: Path,
    top_k_percent: float | None = 100.0,
    score_threshold: float | None = None,
    edge_margin: int = 6,
    full_dead_pairs: set[tuple[int, int]] | None = None,
    features_version: str | None = None,
    cascade_l0_frac: float | None = None,
) -> tuple[list[list[int]], dict]:
    """
    AI 排序候選對 → 僅對選中者呼叫 has_dead_pocket → 僅 oracle=True 加子句。
    """
    ranked, meta = score_candidate_pairs(
        placements,
        by_piece,
        model_path=model_path,
        edge_margin=edge_margin,
        features_version=features_version,
        cascade_l0_frac=cascade_l0_frac,
    )
    n_total = meta["n_pairs_total"]
    if n_total == 0:
        return [], {
            "n_pairs_total": 0,
            "n_oracle_calls": 0,
            "n_clauses": 0,
            "oracle_sec": 0.0,
            "top_k_percent": top_k_percent,
            "clause_recall_vs_full": 1.0,
        }

    if score_threshold is not None:
        selected = [(a, b, s) for a, b, s in ranked if s >= score_threshold]
    elif top_k_percent is not None:
        k = max(1, int(np.ceil(n_total * top_k_percent / 100.0)))
        selected = ranked[:k]
    else:
        selected = ranked

    extra, ostats = oracle_selected_pairs(
        selected,
        full_dead_pairs=full_dead_pairs,
        top_k_percent=top_k_percent,
    )
    stats = {
        "n_pairs_total": n_total,
        "n_oracle_calls": ostats["n_oracle_calls"],
        "oracle_call_fraction": round(ostats["n_oracle_calls"] / n_total, 4),
        "n_clauses": ostats["n_clauses"],
        "oracle_sec": ostats["oracle_sec"],
        "top_k_percent": top_k_percent,
        "score_threshold": score_threshold,
        "clause_recall_vs_full": ostats["clause_recall_vs_full"],
        "n_full_dead_pairs": len(full_dead_pairs) if full_dead_pairs else None,
        "model_type": meta.get("model_type"),
        "features_version": meta.get("features_version"),
        "cascade_l0_frac": meta.get("cascade_l0_frac"),
        "l0_sec": meta.get("l0_sec"),
        "ml_sec": meta.get("ml_sec"),
        "n_l0_kept": meta.get("n_l0_kept"),
    }
    cascade_note = ""
    if meta.get("cascade_l0_frac"):
        cascade_note = (
            f"  l0={meta.get('l0_sec')}s ml={meta.get('ml_sec')}s "
            f"kept={meta.get('n_l0_kept'):,}/{n_total:,}"
        )
    print(
        f"  [learned_shape] pairs={n_total:,}  oracle_calls={stats['n_oracle_calls']:,} "
        f"({stats['oracle_call_fraction']*100:.1f}%)  clauses={stats['n_clauses']:,}  "
        f"recall={stats['clause_recall_vs_full']:.3f}  oracle={stats['oracle_sec']:.1f}s  "
        f"feat={meta.get('features_version')} model={meta.get('model_type')}"
        f"{cascade_note}",
        file=sys.stderr,
    )
    return extra, stats
