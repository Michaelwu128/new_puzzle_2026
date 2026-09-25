#!/usr/bin/env python3
"""
LightGBM / sklearn ranker 訓練（Leave-one-out: v1↔v2）。

用法：
  python3 train_dead_pocket_ranker.py --train-puzzle v1 --eval-puzzle v2
  python3 train_dead_pocket_ranker.py --loo  # 兩個 fold 都訓練
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from puzzle_defs import puzzle_names

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "baselines" / "learned_shape"


def _feature_names(version: str) -> list[str]:
    if version == "v2":
        from dead_pocket_features_v2 import FEATURE_NAMES

        return FEATURE_NAMES
    from dead_pocket_features import FEATURE_NAMES

    return FEATURE_NAMES


def recall_at_top_k(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> dict:
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


def evaluate_scores(y: np.ndarray, scores: np.ndarray, split_name: str) -> dict:
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    pred = (scores >= np.median(scores)).astype(int)
    metrics: dict = {
        "split": split_name,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
    metrics["recall_at_top_k"] = [
        recall_at_top_k(y, scores, k) for k in (1, 5, 10, 25, 50, 100)
    ]
    return metrics


def maybe_subsample(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_rows, replace=False)
    return X[idx], y[idx]


def make_query_groups(n: int, max_size: int = 5000) -> list[int]:
    """LightGBM lambdarank 每 query 上限 10000；拆成多組。"""
    groups: list[int] = []
    left = n
    while left > 0:
        sz = min(left, max_size)
        groups.append(sz)
        left -= sz
    return groups


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    train_puzzle: str,
    eval_puzzle: str,
    feature_names: list[str],
    features_version: str,
    heur_blend: float = 0.3,
    max_train_rows: int = 300_000,
    train_query_groups: list[int] | None = None,
    seed: int = 42,
) -> tuple[dict, dict]:
    if train_query_groups is not None:
        X_tr, y_tr = X_train, y_train
    else:
        X_tr, y_tr = maybe_subsample(X_train, y_train, max_train_rows, seed)

    heur_idx = feature_names.index("heur_score") if "heur_score" in feature_names else None

    model = None
    model_type = "sklearn_hgb"
    best_recall25: dict = {"score": -1.0, "iteration": 0}
    try:
        import lightgbm as lgb

        if train_query_groups is not None:
            group: list[int] = []
            for g in train_query_groups:
                group.extend(make_query_groups(g))
        else:
            group = make_query_groups(len(y_tr))
        eval_n = min(len(y_eval), 50_000)
        rng = np.random.default_rng(seed)
        ev_idx = (
            rng.choice(len(y_eval), size=eval_n, replace=False)
            if len(y_eval) > eval_n
            else np.arange(len(y_eval))
        )
        X_ev_fit = X_eval[ev_idx]
        y_ev_fit = y_eval[ev_idx]
        eval_group = make_query_groups(len(y_ev_fit))

        best_recall25 = {"score": -1.0, "iteration": 0}

        def recall25_cb(env):
            scores = env.model.predict(X_ev_fit, num_iteration=env.iteration + 1)
            r = recall_at_top_k(y_ev_fit, scores, 25.0)["recall"]
            if r > best_recall25["score"]:
                best_recall25["score"] = r
                best_recall25["iteration"] = env.iteration + 1

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        ranker.fit(
            X_tr,
            y_tr,
            group=group,
            eval_set=[(X_ev_fit, y_ev_fit)],
            eval_group=[eval_group],
            eval_at=[25],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, first_metric_only=True),
                lgb.log_evaluation(period=0),
                recall25_cb,
            ],
        )
        model = ranker
        model_type = "lightgbm_ranker"
    except Exception as exc:
        if "lightgbm" not in str(type(exc).__module__):
            print(f"  [WARN] LightGBM ranker failed ({exc}), fallback HGB", file=sys.stderr)
        from sklearn.ensemble import HistGradientBoostingClassifier

        pos = max(int(y_tr.sum()), 1)
        neg = len(y_tr) - pos
        sw = np.where(y_tr == 1, neg / pos, 1.0)
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=12,
            learning_rate=0.08,
            random_state=seed,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

    def _predict(X: np.ndarray) -> np.ndarray:
        if model_type == "lightgbm_ranker":
            ml = model.predict(X)
        else:
            ml = model.predict_proba(X)[:, 1]
        if heur_idx is not None:
            heur = X[:, heur_idx]
            # normalize heur to [0,1] roughly
            hmin, hmax = heur.min(), heur.max()
            if hmax > hmin:
                heur_n = (heur - hmin) / (hmax - hmin)
            else:
                heur_n = heur
            ml_min, ml_max = ml.min(), ml.max()
            if ml_max > ml_min:
                ml_n = (ml - ml_min) / (ml_max - ml_min)
            else:
                ml_n = ml
            return (1 - heur_blend) * ml_n + heur_blend * heur_n
        return ml

    train_scores = _predict(X_tr)
    eval_scores = _predict(X_eval)

    report = {
        "train_puzzle": train_puzzle,
        "train_puzzles": train_puzzle.split("+") if "+" in train_puzzle else [train_puzzle],
        "eval_puzzle": eval_puzzle,
        "model_type": model_type,
        "features_version": features_version,
        "heur_blend": heur_blend,
        "best_recall25_iteration": best_recall25.get("iteration") if model_type == "lightgbm_ranker" else None,
        "best_recall25_score": round(best_recall25["score"], 4) if model_type == "lightgbm_ranker" and best_recall25["score"] >= 0 else None,
        "train": evaluate_scores(y_tr, train_scores, f"train_{train_puzzle}"),
        "eval": evaluate_scores(y_eval, eval_scores, f"eval_{eval_puzzle}"),
    }

    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "features_version": features_version,
        "heur_blend": heur_blend,
        "train_puzzle": train_puzzle,
        "eval_puzzle": eval_puzzle,
    }
    return bundle, report


def load_dataset(data_dir: Path, puzzle: str, features_version: str) -> pd.DataFrame:
    suffix = "" if features_version == "v1" else "_v2"
    path = data_dir / f"dataset_{puzzle}{suffix}.csv"
    if not path.is_file():
        path = data_dir / f"dataset_{puzzle}.csv"
    return pd.read_csv(path)


def load_multi_train(
    data_dir: Path,
    puzzles: list[str],
    feature_names: list[str],
    features_version: str,
    *,
    max_per_puzzle: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    for i, p in enumerate(puzzles):
        df = load_dataset(data_dir, p, features_version)
        for col in feature_names:
            if col not in df.columns:
                raise KeyError(f"missing feature {col} in {p}")
        X = df[feature_names].to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=np.int32)
        X, y = maybe_subsample(X, y, max_per_puzzle, seed + i * 17)
        xs.append(X)
        ys.append(y)
        groups.append(len(y))
    return np.vstack(xs), np.concatenate(ys), groups


def model_out_name(train_puzzles: list[str]) -> str:
    if len(train_puzzles) == 1:
        return f"ranker_loo_{train_puzzles[0]}_train.joblib"
    return f"ranker_train_{'_'.join(train_puzzles)}.joblib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-puzzle",
        nargs="+",
        choices=puzzle_names(),
        default=None,
        help="一個或多個拼圖；多拼圖合訓時 eval 不可在 train 集合內",
    )
    ap.add_argument("--eval-puzzle", choices=puzzle_names(), default=None)
    ap.add_argument("--loo", action="store_true", help="訓練 v1→v2 與 v2→v1（單拼圖）")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--features", choices=["v1", "v2"], default="v2")
    ap.add_argument("--heur-blend", type=float, default=0.3)
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument(
        "--max-per-puzzle",
        type=int,
        default=100_000,
        help="多拼圖合訓時每拼圖 subsample 上限",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_names = _feature_names(args.features)

    folds: list[tuple[list[str], str]] = []
    if args.loo:
        folds = [(["v1"], "v2"), (["v2"], "v1")]
    elif args.train_puzzle and args.eval_puzzle:
        folds = [(args.train_puzzle, args.eval_puzzle)]
    else:
        folds = [(["v1"], "v2"), (["v2"], "v1")]

    full_report: dict = {"folds": {}, "features": args.features}
    for train_ps, eval_p in folds:
        train_label = "+".join(train_ps)
        if eval_p in train_ps:
            print(f"[ERROR] eval {eval_p} 不可在 train {train_ps} 內", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== ranker | train {train_label} → eval {eval_p} ===", file=sys.stderr)

        if len(train_ps) == 1:
            df_tr = load_dataset(args.data_dir, train_ps[0], args.features)
            for col in feat_names:
                if col not in df_tr.columns:
                    print(f"[ERROR] missing feature {col} in train set", file=sys.stderr)
                    sys.exit(1)
            X_tr = df_tr[feat_names].to_numpy(dtype=np.float64)
            y_tr = df_tr["label"].to_numpy(dtype=np.int32)
            query_groups = None
        else:
            X_tr, y_tr, query_groups = load_multi_train(
                args.data_dir,
                train_ps,
                feat_names,
                args.features,
                max_per_puzzle=args.max_per_puzzle,
                seed=args.seed,
            )

        df_ev = load_dataset(args.data_dir, eval_p, args.features)
        X_ev = df_ev[feat_names].to_numpy(dtype=np.float64)
        y_ev = df_ev["label"].to_numpy(dtype=np.int32)

        bundle, report = train_ranker(
            X_tr,
            y_tr,
            X_ev,
            y_ev,
            train_puzzle=train_label,
            eval_puzzle=eval_p,
            feature_names=feat_names,
            features_version=args.features,
            heur_blend=args.heur_blend,
            max_train_rows=args.max_train_rows,
            train_query_groups=query_groups,
            seed=args.seed,
        )
        key = f"train_{train_label.replace('+', '_')}_eval_{eval_p}"
        full_report["folds"][key] = report

        out_model = out_dir / model_out_name(train_ps)
        joblib.dump(bundle, out_model)
        r25 = report["eval"]["recall_at_top_k"][3]
        print(
            f"  eval recall={report['eval']['recall']}  "
            f"recall@25%={r25['recall']}  auc={report['eval'].get('roc_auc')}  "
            f"→ {out_model.name}",
            file=sys.stderr,
        )

    report_path = out_dir / "ranker_training_report.json"
    if report_path.is_file():
        try:
            prev = json.loads(report_path.read_text())
            prev.setdefault("folds", {}).update(full_report["folds"])
            full_report = {**prev, "folds": prev["folds"], "features": args.features}
        except json.JSONDecodeError:
            pass
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n完成。{report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
