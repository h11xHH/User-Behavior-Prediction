"""dataset_builder.py

Do train/val/test split for modeling.
- Test date: 12-18. Train grows + validate on next date.
- Negative downsampling on each fold's train only.
- Saves trainval.parquet once + fold_spec.json.
- Exposes make_fold() to materialise a fold's downsampled train + natural val in memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.feature_assembler import SALE_EVENT_DATE


@dataclass
class Fold:
    index: int
    train_dates: list[str]
    val_date: str
    val_day_type: str


def day_type(prediction_date: str) -> str:
    """Classify a cutoff date relative to the sale event."""
    d = datetime.strptime(prediction_date, "%Y-%m-%d").date()
    if d < SALE_EVENT_DATE:
        return "before_sale"
    if d == SALE_EVENT_DATE:
        return "on_sale"
    return "after_sale"


def build_folds(trainval_dates: list[str], first_val_date: str) -> list[Fold]:
    """Expanding-window folds: for each val day, train on all earlier trainval days.

    Input
    -----
    sorted trainval cutoff dates (test excluded), and the earliest date allowed to
        serve as a validation fold.

    Output
    ------
    list of Fold(index, train_dates, val_date, val_day_type).
    """
    ordered = sorted(trainval_dates)
    folds: list[Fold] = []
    for val_date in ordered:
        if val_date < first_val_date:
            continue
        train_dates = [d for d in ordered if d < val_date]
        if not train_dates:
            continue
        folds.append(Fold(index=len(folds) + 1, train_dates=train_dates,
                          val_date=val_date, val_day_type=day_type(val_date)))
    return folds


def downsample_negatives(train: pd.DataFrame, negative_ratio: int, seed: int) -> pd.DataFrame:
    """Keep all positives; sample negatives down to negative_ratio x positives.

    Returns a shuffled frame. If there are no positives, returns train unchanged.
    """
    positives = train[train["label"] == 1]
    negatives = train[train["label"] == 0]
    if len(positives) == 0:
        return train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_target = min(len(negatives), negative_ratio * len(positives))
    negatives_kept = negatives.sample(n=n_target, random_state=seed)
    combined = pd.concat([positives, negatives_kept], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_fold(trainval: pd.DataFrame, fold: Fold, negative_ratio: int,
              seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise one fold: downsampled train + natural-rate val.

    Input
    -----
    the full trainval pool, a Fold, the train downsample ratio, and a
        seed for this fold's negative sampling.

    Output
    ------
    (train_df, val_df). Val is left at its natural positive rate.
    """
    train = trainval[trainval["cutoff_date"].isin(fold.train_dates)]
    val = trainval[trainval["cutoff_date"] == fold.val_date]
    train = downsample_negatives(train, negative_ratio, seed)
    return train, val


def _fold_record(fold: Fold, trainval: pd.DataFrame, negative_ratio: int) -> dict:
    """Summarise a fold for fold_spec.json."""
    train = trainval[trainval["cutoff_date"].isin(fold.train_dates)]
    val = trainval[trainval["cutoff_date"] == fold.val_date]
    train_pos = int(train["label"].sum())
    record = asdict(fold)
    record.update(
        train_rows_natural=int(len(train)),
        train_positives=train_pos,
        train_rows_downsampled=int(train_pos + min(len(train) - train_pos,
                                                   negative_ratio * train_pos)),
        val_rows=int(len(val)),
        val_positives=int(val["label"].sum()),
    )
    return record


def main() -> None:
    """Split assembled.parquet into test/trainval and emit the fold spec."""
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
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"paths.interim_dir, labeling.prediction_dates, and a split section with "
            f"test_dates, first_val_date, negative_ratio, seed."
        ) from error

    assembled_path = interim_dir / "assembled.parquet"
    if not assembled_path.exists():
        raise FileNotFoundError(
            f"'{assembled_path.name}' not found; run feature_assembler first."
        )
    assembled = pd.read_parquet(assembled_path)

    # partition into held-out test
    is_test = assembled["cutoff_date"].isin(test_dates)
    test_df = assembled[is_test].reset_index(drop=True)
    trainval_df = assembled[~is_test].reset_index(drop=True)

    test_df.to_parquet(interim_dir / "test.parquet", index=False)
    trainval_df.to_parquet(interim_dir / "trainval.parquet", index=False)

    trainval_dates = [d for d in all_dates if d not in test_dates]
    folds = build_folds(trainval_dates, first_val_date)
    if not folds:
        raise RuntimeError("No folds were produced; check first_val_date vs prediction_dates.")

    spec = {
        "sale_event_date": SALE_EVENT_DATE.isoformat(),
        "test_dates": test_dates,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "trainval_dates": trainval_dates,
        "folds": [_fold_record(f, trainval_df, negative_ratio) for f in folds],
    }
    spec_path = interim_dir / "fold_spec.json"
    with spec_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)


    def line(name, frame):
        pos = int(frame["label"].sum())
        print(f"  {name:<12} {len(frame):>9,} rows  {pos:>6,} pos  ({pos/max(len(frame),1):.3%})")

    print("Split artifacts written:")
    line("test", test_df)
    line("trainval", trainval_df)
    print(f"\nExpanding-window folds ({len(folds)}), train downsample {negative_ratio}:1 "
          f"(natural rates shown for val):")
    print(f"  {'fold':<5}{'train days':<14}{'val':<12}{'type':<13}"
          f"{'train pos':>10}{'val rows':>10}{'val pos':>9}")
    for record in spec["folds"]:
        print(f"  {record['index']:<5}"
              f"{record['train_dates'][0][5:]+'..'+record['train_dates'][-1][5:]:<14}"
              f"{record['val_date'][5:]:<12}{record['val_day_type']:<13}"
              f"{record['train_positives']:>10,}{record['val_rows']:>10,}{record['val_positives']:>9,}")
    print(f"\nfold_spec.json -> {spec_path.name}. "
          f"Phase 7 loads trainval.parquet once and calls make_fold() per fold.")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"dataset_builder failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)