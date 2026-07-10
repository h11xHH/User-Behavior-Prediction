"""modeling/baseline.py

Baseline model
1. Rule scorers: rank candidates by a single transparent signal. No learning.
2. Logistic regression.

Evaluated on expanding-window folds from dataset_builder.
Results aggregated per day type (before/on/after the sale event)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config
from src.dataset_builder import build_folds, make_fold
from src.modeling.features import get_xy
from src.modeling.metrics import evaluate

# score array, higher = more likely to buy
RULE_SCORERS = {
    "rule_cart": lambda X: X["user_item_cart_all"].to_numpy(float),
    "rule_browse": lambda X: X["user_item_browse_all"].to_numpy(float),
    "rule_cart_not_buy": lambda X: X["user_item_cart_not_buy"].to_numpy(float),
    "rule_recency": lambda X: -X["user_item_since_last_hour"].to_numpy(float),
}


def fit_logistic(X_train: pd.DataFrame, y_train, seed: int):
    """Fit StandardScaler + class-balanced LogisticRegression; return (scaler, model)."""
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(X_train), y_train)
    return scaler, model


def score_logistic(fitted, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability as the ranking score."""
    scaler, model = fitted
    return model.predict_proba(scaler.transform(X))[:, 1]


def run_fold(trainval: pd.DataFrame, fold, negative_ratio: int, seed: int) -> list[dict]:
    """Fit every baseline on one fold; return a list of metric rows."""
    train, val = make_fold(trainval, fold, negative_ratio, seed + fold.index)
    X_train, y_train, _ = get_xy(train)
    X_val, y_val, _ = get_xy(val)

    rows = []

    def record(model_name, scores):
        metrics = evaluate(y_val, scores)
        rows.append({"fold": fold.index, "val_date": fold.val_date,
                     "day_type": fold.val_day_type, "model": model_name, **metrics})

    for name, scorer in RULE_SCORERS.items():
        record(name, scorer(X_val))

    fitted = fit_logistic(X_train, y_train, seed + fold.index)
    record("logistic", score_logistic(fitted, X_val))
    return rows


def _print_table(results: pd.DataFrame, metric: str) -> None:
    """Print one metric as folds x models, plus per-day-type and overall means."""
    print(f"\n=== {metric} by fold ===")
    pivot = results.pivot_table(index=["fold", "val_date", "day_type"],
                                columns="model", values=metric)
    print(pivot.round(4).to_string())
    print(f"\n--- {metric} mean by day type ---")
    print(results.groupby(["day_type", "model"])[metric].mean().unstack().round(4).to_string())
    print(f"\n--- {metric} overall mean (across folds) ---")
    print(results.groupby("model")[metric].mean().round(4).to_string())


def main() -> None:
    """Run all baselines across the folds and report per day type."""
    config = load_config()
    try:
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        all_dates = list(config.labeling["prediction_dates"])
        split_cfg = config.raw["split"]
        test_dates = list(split_cfg["test_dates"])
        first_val_date = split_cfg["first_val_date"]
        negative_ratio = int(split_cfg["negative_ratio"])
        seed = int(split_cfg["seed"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. Needs paths.interim_dir, "
            f"labeling.prediction_dates, and the split section."
        ) from error

    trainval_path = interim_dir / "trainval.parquet"
    if not trainval_path.exists():
        raise FileNotFoundError(f"'{trainval_path.name}' not found; run dataset_builder first.")
    trainval = pd.read_parquet(trainval_path)

    trainval_dates = [d for d in all_dates if d not in test_dates]
    folds = build_folds(trainval_dates, first_val_date)

    all_rows = []
    for fold in folds:
        print(f"fold {fold.index}: train {fold.train_dates[0][5:]}..{fold.train_dates[-1][5:]}  "
              f"val {fold.val_date[5:]} ({fold.val_day_type})")
        all_rows.extend(run_fold(trainval, fold, negative_ratio, seed))

    results = pd.DataFrame(all_rows)
    out_path = interim_dir / "baseline_results.csv"
    results.to_csv(out_path, index=False)

    for metric in ["pr_auc", "P@500", "roc_auc"]:
        _print_table(results, metric)
    print(f"\nFull results -> {out_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"baseline failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)