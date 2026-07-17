"""modeling/xgboost_model.py

XGBoost on the same expanding-window folds and metrics as the baseline

Implemented a controlled experiment for downsampling ratios {5,10,20,50} 
and a no-downsampling + scale_pos_weight config. Best ratio is chosen by
mean validation PR-AUC across folds. 

Early stopping on val aucpr.

Gain feature importances.

Retrain on all trainval with the final model and evaluate once on the
held-out test day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config_loader import load_config
from src.dataset_builder import build_folds, downsample_negatives, make_fold
from src.modeling.features import get_xy
from src.modeling.metrics import evaluate

DEFAULT_PARAMS = {
    "max_depth": 6, "learning_rate": 0.1, "n_estimators": 600,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
}


@dataclass
class TrainConfig:
    name: str
    negative_ratio: int | None # None = no downsampling, use scale_pos_weight


def _is_weekend(prediction_date: str) -> bool:
    return datetime.strptime(prediction_date, "%Y-%m-%d").date().weekday() >= 5


def scale_pos_weight(y_train: pd.Series, negative_ratio: int | None) -> float:
    """1.0 when downsampling handles imbalance; natural neg/pos when it doesn't."""
    if negative_ratio and negative_ratio > 0:
        return 1.0
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    return negatives / max(positives, 1)


def fold_train_val(trainval: pd.DataFrame, fold, negative_ratio: int | None,
                   seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a fold's train/val. Downsample train unless negative_ratio is None."""
    if negative_ratio and negative_ratio > 0:
        return make_fold(trainval, fold, negative_ratio, seed)
    train = trainval[trainval["cutoff_date"].isin(fold.train_dates)]
    val = trainval[trainval["cutoff_date"] == fold.val_date]
    return train, val


def train_xgb(X_train, y_train, X_val, y_val, params: dict, spw: float,
              early_stopping_rounds: int, seed: int):
    """Fit an XGBoost classifier with early stopping on validation aucpr."""
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=spw, random_state=seed, n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def gain_importances(model, feature_names: list[str]) -> pd.Series:
    """Gain-based feature importances as a sorted Series (0 for unused features)."""
    score = model.get_booster().get_score(importance_type="gain")
    return pd.Series({f: score.get(f, 0.0) for f in feature_names}).sort_values(ascending=False)


def run_sweep(trainval: pd.DataFrame, folds: list, configs: list[TrainConfig],
              seeds: list[int], params: dict, early_stopping_rounds: int) -> pd.DataFrame:
    """Fit XGBoost for every (config, seed, fold); return a table of val metrics."""
    rows = []
    for config in configs:
        cfg_seeds = seeds if config.negative_ratio else seeds[:1]   # full data: seed varies little
        for seed in cfg_seeds:
            for fold in folds:
                train, val = fold_train_val(trainval, fold, config.negative_ratio, seed + fold.index)
                X_train, y_train, names = get_xy(train)
                X_val, y_val, _ = get_xy(val)
                model = train_xgb(X_train, y_train, X_val, y_val, params,
                                  scale_pos_weight(y_train, config.negative_ratio),
                                  early_stopping_rounds, seed)
                scores = model.predict_proba(X_val)[:, 1]
                metrics = evaluate(y_val, scores)
                rows.append({"config": config.name, "seed": seed, "fold": fold.index,
                             "val_date": fold.val_date, "day_type": fold.val_day_type,
                             "is_weekend": _is_weekend(fold.val_date),
                             "best_trees": int(model.best_iteration) + 1, **metrics})
    return pd.DataFrame(rows)


def train_final(trainval: pd.DataFrame, test: pd.DataFrame, config: TrainConfig,
                n_estimators: int, params: dict, seed: int):
    """Retrain on all trainval with fixed n_estimators; evaluate once on test."""
    if config.negative_ratio and config.negative_ratio > 0:
        train = downsample_negatives(trainval, config.negative_ratio, seed)
    else:
        train = trainval
    X_train, y_train, names = get_xy(train)
    X_test, y_test, _ = get_xy(test)
    final_params = {**params, "n_estimators": n_estimators}
    model = xgb.XGBClassifier(
        n_estimators=final_params["n_estimators"], max_depth=final_params["max_depth"],
        learning_rate=final_params["learning_rate"], subsample=final_params["subsample"],
        colsample_bytree=final_params["colsample_bytree"],
        min_child_weight=final_params["min_child_weight"],
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=scale_pos_weight(y_train, config.negative_ratio),
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    scores = model.predict_proba(X_test)[:, 1]
    return model, evaluate(y_test, scores), gain_importances(model, names)


def main() -> None:
    """Sweep negative ratios, pick the best by val PR-AUC, then evaluate on test."""
    config = load_config()
    try:
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        all_dates = list(config.labeling["prediction_dates"])
        split_cfg = config.raw["split"]
        test_dates = list(split_cfg["test_dates"])
        first_val_date = split_cfg["first_val_date"]
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. Needs paths.interim_dir, "
            f"labeling.prediction_dates, and the split section."
        ) from error

    xgb_cfg = config.raw.get("xgboost", {})
    ratios = xgb_cfg.get("negative_ratios", [5, 10, 20, 50])
    include_full = xgb_cfg.get("include_full", True)
    seeds = xgb_cfg.get("seeds", [42, 43])
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 30)
    params = {**DEFAULT_PARAMS, **xgb_cfg.get("params", {})}

    trainval_path = interim_dir / "trainval.parquet"
    test_path = interim_dir / "test.parquet"
    for path in (trainval_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"'{path.name}' not found; run dataset_builder first.")
    trainval = pd.read_parquet(trainval_path)
    test = pd.read_parquet(test_path)

    trainval_dates = [d for d in all_dates if d not in test_dates]
    folds = build_folds(trainval_dates, first_val_date)

    configs = [TrainConfig(f"downsample_{r}", r) for r in ratios]
    if include_full:
        configs.append(TrainConfig("full_scale_pos_weight", None))

    print(f"Sweeping {len(configs)} configs x up to {len(seeds)} seeds x {len(folds)} folds ...")
    sweep = run_sweep(trainval, folds, configs, seeds, params, early_stopping_rounds)
    sweep.to_csv(interim_dir / "xgb_sweep_results.csv", index=False)

    # report
    print("\n=== mean validation metrics by config (across folds & seeds) ===")
    by_config = sweep.groupby("config")[["pr_auc", "roc_auc", "P@500", "best_trees"]].mean()
    print(by_config.round(4).sort_values("pr_auc", ascending=False).to_string())

    best_name = by_config["pr_auc"].idxmax()
    best_config = next(c for c in configs if c.name == best_name)
    best_rows = sweep[sweep["config"] == best_name]
    print(f"\nBest config by mean val PR-AUC: {best_name}")

    print(f"\n--- {best_name}: PR-AUC by day type ---")
    print(best_rows.groupby("day_type")["pr_auc"].mean().round(4).to_string())
    print(f"\n--- {best_name}: PR-AUC by weekend ---")
    print(best_rows.groupby("is_weekend")["pr_auc"].mean().round(4).to_string())

    # final model on test
    n_estimators = int(np.median(best_rows["best_trees"]))
    print(f"\nRetraining final model ({best_name}, {n_estimators} trees) on all trainval; "
          f"evaluating on test {test_dates} ...")
    model, test_metrics, importances = train_final(trainval, test, best_config,
                                                   n_estimators, params, seeds[0])
    model.save_model(interim_dir / "xgb_final.json")
    importances.to_csv(interim_dir / "xgb_importances.csv", header=["gain"])

    print("\n=== TEST metrics (held-out 12-18) ===")
    for key, value in test_metrics.items():
        print(f"  {key:<10} {value:.4f}" if isinstance(value, float) else f"  {key:<10} {value}")
    print("\n=== top 15 features by gain ===")
    print(importances.head(15).round(2).to_string())
    print(f"\nArtifacts -> xgb_sweep_results.csv, xgb_final.json, xgb_importances.csv")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"xgboost_model failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)