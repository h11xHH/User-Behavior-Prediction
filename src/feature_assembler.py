"""feature_assembler.py

Assemble the tables. For each D, left join every feature onto sample,
then add the time features derived from D. Builds one day at a time,
final table is assembled.parquet.

Assert row count unchange for zero join misses, add a filling logic if
any warning triggered

time features:
  time_is_weekend            | 1 if D falls on Sat/Sun
  time_is_before_sale_event  | 1 if D is before the 2014-12-12 sale event
  time_is_on_sale_event      | 1 if D is the sale event day (2014-12-12)
  time_is_after_sale_event   | 1 if D is after the sale event
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.config_loader import load_config

SALE_EVENT_DATE = date(2014, 12, 12)


def context_features(prediction_date: str) -> dict[str, int]:
    """Return the context feature flags for cutoff date D.

    Uses before / on / after the sale event (a 3-way one-hot) rather than a
    signed day count. A day count is monotonic in the date, so a temporal test
    day can take a value never seen in train/val; the before/on/after flags
    generalise instead -- every 'after' day shares the same encoding, so
    training on 12-13..12-17 informs the 12-18 test day.
    """
    d = datetime.strptime(prediction_date, "%Y-%m-%d").date()
    return {
        "time_is_weekend": 1 if d.weekday() >= 5 else 0, # 5=Sat, 6=Sun
        "time_is_before_sale_event": 1 if d < SALE_EVENT_DATE else 0,
        "time_is_on_sale_event": 1 if d == SALE_EVENT_DATE else 0,
        "time_is_after_sale_event": 1 if d > SALE_EVENT_DATE else 0,
    }


def _read_required(interim_dir: Path, name: str) -> pd.DataFrame:
    """Read a per-day interim file, with a clear error if it is missing."""
    path = interim_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"'{name}' not found in {interim_dir}. Build the sample and all feature "
            f"tables for this date before assembly."
        )
    return pd.read_parquet(path)


def _merge_checked(left: pd.DataFrame, right: pd.DataFrame, on: list[str],
                   n_base: int, label: str) -> pd.DataFrame:
    """Left merge and assert the row count is unchanged."""
    merged = left.merge(right, on=on, how="left")
    if len(merged) != n_base:
        raise RuntimeError(
            f"Join with {label} changed row count {n_base} -> {len(merged)}; "
            f"'{label}' likely has duplicate keys on {on}."
        )
    return merged


def _apply_defensive_fill(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill any numeric nulls per feature-type policy. This should not be called."""
    for col in frame.columns:
        if not frame[col].isnull().any() or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.endswith("_peak_hour"):
            fill_value = -1
        elif col == "item_is_cold":
            fill_value = 1
        else:
            fill_value = 0
        frame[col] = frame[col].fillna(fill_value)
    return frame


def assemble_day(interim_dir: Path, prediction_date: str) -> pd.DataFrame:
    """Assemble the full labelled feature matrix for one prediction day D."""
    sample = _read_required(interim_dir, f"sample__{prediction_date}.parquet")
    feat_user = _read_required(interim_dir, f"feat_user__{prediction_date}.parquet")
    feat_item = _read_required(interim_dir, f"feat_item__{prediction_date}.parquet")
    feat_category = _read_required(interim_dir, f"feat_category__{prediction_date}.parquet")
    feat_user_item = _read_required(interim_dir, f"feat_user_item__{prediction_date}.parquet")
    feat_user_category = _read_required(interim_dir, f"feat_user_category__{prediction_date}.parquet")

    n_base = len(sample)
    df = _merge_checked(sample, feat_item, ["item_id", "cutoff_date"], n_base, "feat_item")
    df = _merge_checked(df, feat_user, ["user_id", "cutoff_date"], n_base, "feat_user")
    df = _merge_checked(df, feat_user_item, ["user_id", "item_id", "cutoff_date"], n_base, "feat_user_item")
    df = _merge_checked(df, feat_category, ["item_category", "cutoff_date"], n_base, "feat_category")
    df = _merge_checked(df, feat_user_category, ["user_id", "item_category", "cutoff_date"],
                        n_base, "feat_user_category")

    for name, value in context_features(prediction_date).items():
        df[name] = value

    # check nulls
    null_counts = df.isnull().sum()
    missing = null_counts[null_counts > 0]
    if not missing.empty:
        print(f"  WARNING D={prediction_date}: join misses introduced nulls in "
              f"{len(missing)} columns (filling defensively): {dict(missing)}")
    df = _apply_defensive_fill(df)
    remaining = int(df.isnull().sum().sum())
    if remaining:
        raise RuntimeError(f"{remaining} nulls remain after fill on D={prediction_date} "
                           f"(non-numeric join miss?).")


    user_cols = [c for c in feat_user.columns if c not in ("user_id", "cutoff_date")]
    item_cols = [c for c in feat_item.columns if c not in ("item_id", "cutoff_date")]
    category_cols = [c for c in feat_category.columns if c not in ("item_category", "cutoff_date")]
    ui_cols = [c for c in feat_user_item.columns if c not in ("user_id", "item_id", "cutoff_date")]
    uc_cols = [c for c in feat_user_category.columns
               if c not in ("user_id", "item_category", "cutoff_date")]
    context_cols = ["time_is_weekend", "time_is_before_sale_event",
                    "time_is_on_sale_event", "time_is_after_sale_event"]
    ordered = (["user_id", "item_id", "cutoff_date", "label"] + context_cols
               + user_cols + item_cols + category_cols + ui_cols + uc_cols)
    return df[ordered]


def save_assembled(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day assembled table to interim as parquet, return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"assembled__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_assembled(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day assembled files into one assembled.parquet."""
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"assembled__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; assemble that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "assembled.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Assemble every configured cutoff date, then combine into assembled.parquet."""
    config = load_config()
    try:
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        dates = list(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"paths.interim_dir, labeling.prediction_dates."
        ) from error

    for prediction_date in dates:
        per_day = interim_dir / f"assembled__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already assembled, skipping")
            continue
        try:
            frame = assemble_day(interim_dir, prediction_date)
            save_assembled(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while assembling D={prediction_date} ({error})."
            ) from error
        positives = int(frame["label"].sum())
        print(f"D={prediction_date}: assembled {frame.shape[0]:,} rows x {frame.shape[1]} cols "
              f"({positives:,} positives, {positives / len(frame):.3%})")

    try:
        combined_path = combine_assembled(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining assembled files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined assembled: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates x {combined.shape[1]} cols "
          f"({int(combined['label'].sum()):,} positives)  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"feature_assembler failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)