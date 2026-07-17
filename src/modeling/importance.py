"""modeling/importance.py

Permutation importance on val: for each fold, fit on train, then shuffle each
feature on val and measure the pr-auc drop. Averaged across folds. 

Ablation test: retrain with the cart pair, then the whole user_item table removed,
check how much pr-auc drop

Both use downsample20 (20:1) and the normal-day folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config_loader import load_config
from src.dataset_builder import build_folds
from src.modeling.features import get_xy
from src.modeling.metrics import evaluate
from src.modeling.xgboost_model import (DEFAULT_PARAMS, TrainConfig, fold_train_val,
                                        scale_pos_weight, train_xgb)

CART_FEATURES = ["user_item_cart_all", "user_item_cart_not_buy"]
PERM_N_REPEATS = 5


def permutation_importance_cv(trainval, folds, config, params, early_stopping_rounds,
                              n_repeats, seed) -> pd.DataFrame:
    """Per-fold permutation importance on val (pr-auc drop), averaged over folds."""
    per_fold = []
    for fold in folds:
        train, val = fold_train_val(trainval, fold, config.negative_ratio, seed + fold.index)
        X_train, y_train, names = get_xy(train)
        X_val, y_val, _ = get_xy(val)
        model = train_xgb(X_train, y_train, X_val, y_val, params,
                          scale_pos_weight(y_train, config.negative_ratio),
                          early_stopping_rounds, seed)
        result = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                        n_repeats=n_repeats, random_state=seed, n_jobs=1)
        per_fold.append(pd.Series(result.importances_mean, index=names))
        print(f"  perm importance done: fold {fold.index} ({fold.val_date})")
    matrix = pd.concat(per_fold, axis=1)
    table = pd.DataFrame({"perm_importance_mean": matrix.mean(axis=1),
                          "perm_importance_std": matrix.std(axis=1)})
    return table.sort_values("perm_importance_mean", ascending=False)


def run_ablation(trainval, folds, config, params, early_stopping_rounds,
                 ablations: dict, seed) -> pd.DataFrame:
    """Retrain per fold with each feature group removed; return per-fold val metrics."""
    rows = []
    for name, drop_cols in ablations.items():
        for fold in folds:
            train, val = fold_train_val(trainval, fold, config.negative_ratio, seed + fold.index)
            X_train, y_train, _ = get_xy(train)
            X_val, y_val, _ = get_xy(val)
            keep_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
            keep_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
            model = train_xgb(keep_train, y_train, keep_val, y_val, params,
                             scale_pos_weight(y_train, config.negative_ratio),
                             early_stopping_rounds, seed)
            scores = model.predict_proba(keep_val)[:, 1]
            metrics = evaluate(y_val, scores)
            rows.append({"ablation": name, "n_features": keep_train.shape[1],
                         "fold": fold.index, "day_type": fold.val_day_type,
                         "pr_auc": metrics["pr_auc"], "P@100": metrics["P@100"],
                         "P@500": metrics["P@500"]})
    return pd.DataFrame(rows)


def main() -> None:
    """Run permutation importance and ablation on downsample20; save results."""
    config = load_config()
    try:
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        all_dates = list(config.labeling["prediction_dates"])
        split_cfg = config.raw["split"]
        test_dates = list(split_cfg["test_dates"])
        first_val_date = split_cfg["first_val_date"]
        seed = int(split_cfg["seed"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. Needs paths.interim_dir, "
            f"labeling.prediction_dates, and the split section."
        ) from error

    xgb_cfg = config.raw.get("xgboost", {})
    params = {**DEFAULT_PARAMS, **xgb_cfg.get("params", {})}
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 30)

    trainval_path = interim_dir / "trainval.parquet"
    if not trainval_path.exists():
        raise FileNotFoundError(f"'{trainval_path.name}' not found; run dataset_builder first.")
    trainval = pd.read_parquet(trainval_path)

    trainval_dates = [d for d in all_dates if d not in test_dates]
    folds = build_folds(trainval_dates, first_val_date)
    normal_folds = [f for f in folds if f.val_day_type != "on_sale"]

    best = TrainConfig("downsample_20", 20)

    # permutation importance
    print(f"Permutation importance on {len(normal_folds)} normal-day folds "
          f"(n_repeats={PERM_N_REPEATS}) ...")
    perm = permutation_importance_cv(trainval, normal_folds, best, params,
                                     early_stopping_rounds, PERM_N_REPEATS, seed)
    perm.to_csv(interim_dir / "perm_importance.csv")
    print("\n=== top 15 features by permutation importance (mean PR-AUC drop) ===")
    print(perm.head(15).round(5).to_string())

    # ablation test
    _, _, names = get_xy(trainval.head(1))
    pair_cols = [c for c in names if c.startswith("user_item_")]
    ablations = {"full": [], "no_cart": CART_FEATURES, "no_user_item_table": pair_cols}
    print(f"\nAblation: full vs no_cart vs no_user_item_table on {len(normal_folds)} folds ...")
    abl = run_ablation(trainval, normal_folds, best, params, early_stopping_rounds, ablations, seed)
    abl.to_csv(interim_dir / "ablation_results.csv", index=False)

    summary = abl.groupby("ablation").agg(
        n_features=("n_features", "first"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        P100=("P@100", "mean"))
    full_mean = summary.loc["full", "pr_auc_mean"]
    summary["drop_vs_full"] = full_mean - summary["pr_auc_mean"]
    summary["pct_drop"] = 100 * summary["drop_vs_full"] / full_mean
    order = ["full", "no_cart", "no_user_item_table"]
    print("\n=== ablation (normal-day mean val PR-AUC) ===")
    print(summary.loc[order].round(4).to_string())
    print("\nArtifacts -> perm_importance.csv, ablation_results.csv")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"importance failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)