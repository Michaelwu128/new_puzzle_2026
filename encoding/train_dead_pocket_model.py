#!/usr/bin/env python3
"""
訓練 dead-pocket 預測模型（Split-A: v1→v2；Split-B: 80/20 mixed）。

用法：
  python3 train_dead_pocket_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dead_pocket_features import FEATURE_NAMES

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "baselines" / "learned_shape"


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1 = pd.read_csv(data_dir / "dataset_v1.csv")
    v2 = pd.read_csv(data_dir / "dataset_v2.csv")
    return v1, v2


def recall_at_top_k_percent(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> dict:
    n = len(y_true)
    if n == 0:
        return {"k_percent": k_pct, "recall": 0.0, "n_selected": 0}
    k = max(1, int(np.ceil(n * k_pct / 100.0)))
    idx = np.argsort(-scores)[:k]
    pos_total = int(y_true.sum())
    pos_hit = int(y_true[idx].sum())
    recall = pos_hit / pos_total if pos_total else 1.0
    return {
        "k_percent": k_pct,
        "recall": round(recall, 4),
        "n_selected": k,
        "pos_hit": pos_hit,
        "pos_total": pos_total,
    }


def evaluate_model(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics: dict = {
        "split": split_name,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y, proba)), 4)
    else:
        metrics["roc_auc"] = None

    metrics["recall_at_top_k"] = [
        recall_at_top_k_percent(y, proba, k)
        for k in (1, 5, 10, 25, 50, 100)
    ]
    return metrics


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    split_name: str,
) -> tuple[RandomForestClassifier, dict]:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    weight_ratio = neg / max(pos, 1)
    sample_weight = np.where(y_train == 1, weight_ratio, 1.0)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    report = {
        "train": evaluate_model(model, X_train, y_train, f"{split_name}_train"),
        "test": evaluate_model(model, X_test, y_test, f"{split_name}_test"),
    }
    return model, report


def maybe_subsample(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_rows, replace=False)
    return X[idx], y[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mixed-test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-rows", type=int, default=250_000)
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1, v2 = load_datasets(args.data_dir)
    for df in (v1, v2):
        if not set(FEATURE_NAMES).issubset(df.columns):
            missing = set(FEATURE_NAMES) - set(df.columns)
            print(f"[ERROR] dataset 缺少欄位: {missing}", file=sys.stderr)
            sys.exit(1)

    X_v1 = v1[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_v1 = v1["label"].to_numpy(dtype=np.int32)
    X_v2 = v2[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_v2 = v2["label"].to_numpy(dtype=np.int32)

    X_v1, y_v1 = maybe_subsample(X_v1, y_v1, args.max_train_rows, args.seed)
    X_v2_eval = X_v2
    y_v2_eval = y_v2

    print("=== Split-A: train v1 → test v2 ===", file=sys.stderr)
    model_a, report_a = train_one(X_v1, y_v1, X_v2_eval, y_v2_eval, "splitA")
    joblib.dump(
        {"model": model_a, "feature_names": FEATURE_NAMES, "split": "splitA"},
        out_dir / "model_splitA.joblib",
    )
    print(
        f"  test recall={report_a['test']['recall']}  "
        f"roc_auc={report_a['test'].get('roc_auc')}  "
        f"recall@10%={report_a['test']['recall_at_top_k'][2]['recall']}",
        file=sys.stderr,
    )

    print("=== Split-B: mixed 80/20 ===", file=sys.stderr)
    combined = pd.concat([v1, v2], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(combined))
    n_test = int(len(combined) * args.mixed_test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_all = combined[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_all = combined["label"].to_numpy(dtype=np.int32)
    X_tr, y_tr = maybe_subsample(
        X_all[train_idx], y_all[train_idx], args.max_train_rows, args.seed + 1
    )
    model_b, report_b = train_one(
        X_tr, y_tr, X_all[test_idx], y_all[test_idx], "splitB"
    )
    joblib.dump(
        {"model": model_b, "feature_names": FEATURE_NAMES, "split": "splitB"},
        out_dir / "model_splitB.joblib",
    )
    print(
        f"  test recall={report_b['test']['recall']}  "
        f"roc_auc={report_b['test'].get('roc_auc')}  "
        f"recall@10%={report_b['test']['recall_at_top_k'][2]['recall']}",
        file=sys.stderr,
    )

    report = {
        "splitA_v1_train_v2_test": report_a,
        "splitB_mixed_8020": report_b,
        "feature_names": FEATURE_NAMES,
        "dataset_rows": {"v1": len(v1), "v2": len(v2)},
    }
    report_path = out_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n完成。{report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
