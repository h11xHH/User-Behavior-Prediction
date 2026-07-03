"""features/category_features.py

Build the feat_category table. Key is (item_category, cutoff_date)
About 8,900 categories. Single grouped query plus one ratio.

Features produced:
  category_browse_all         browses of items in the category before D
  category_buy_all            purchases of items in the category before D
  category_distinct_item      distinct items seen in the category
  category_conversion_browse_buy = category_buy_all / category_browse_all
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.features.common import day_midnight

_COUNT_COLS = ["category_browse_all", "category_buy_all", "category_distinct_item"]


def build_category_counts(engine: Engine, table: str, prediction_date: str) -> pd.DataFrame:
    """Per-category browse/buy counts and distinct-item count over time < D."""
    d_start = day_midnight(prediction_date)
    sql = f"""
        SELECT
            item_category,
            SUM(behavior_type = 1) AS category_browse_all,
            SUM(behavior_type = 4) AS category_buy_all,
            COUNT(DISTINCT item_id) AS category_distinct_item
        FROM {table}
        WHERE time < '{d_start}'
        GROUP BY item_category
    """
    return read_query(engine, sql)


def build_category_features(engine: Engine, table: str, prediction_date: str) -> pd.DataFrame:
    """Assemble the full feat_category table for one prediction day D.

    Output: DataFrame keyed by (item_category, cutoff_date) with every feature.
    """
    frame = build_category_counts(engine, table, prediction_date)
    frame[_COUNT_COLS] = frame[_COUNT_COLS].astype("int64")

    frame["category_conversion_browse_buy"] = np.where(
        frame["category_browse_all"] > 0,
        frame["category_buy_all"] / frame["category_browse_all"].replace(0, np.nan), 0.0)
    frame["category_conversion_browse_buy"] = frame["category_conversion_browse_buy"].fillna(0.0)

    frame["cutoff_date"] = prediction_date
    key_cols = ["item_category", "cutoff_date"]
    other_cols = [c for c in frame.columns if c not in key_cols]
    return frame[key_cols + other_cols]


def save_category_features(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day feat_category table to interim as Parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"feat_category__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_category_features(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day feat_category files into one feat_category.parquet."""
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"feat_category__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; build that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "feat_category.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Build feat_category for every configured cutoff date, then combine them."""
    config = load_config()
    try:
        table = config.database["table_raw"]
        secrets_path = config.resolve_path(config.database["secrets_file"])
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        dates = list(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"database.table_raw, database.secrets_file, paths.interim_dir, "
            f"labeling.prediction_dates."
        ) from error

    try:
        engine = make_engine(load_secrets(secrets_path))
    except Exception as error:
        raise RuntimeError(f"Could not set up the MySQL connection ({error}).") from error

    for prediction_date in dates:
        per_day = interim_dir / f"feat_category__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already built, skipping")
            continue
        try:
            frame = build_category_features(engine, table, prediction_date)
            save_category_features(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while building feat_category for D={prediction_date} ({error})."
            ) from error
        print(f"D={prediction_date}: feat_category {frame.shape[0]:,} categories x {frame.shape[1]} cols")

    try:
        combined_path = combine_category_features(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining per-day feat_category files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined feat_category: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates "
          f"x {combined.shape[1]} cols  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:  # top-level guard: report clearly, never hide the cause
        print(f"category_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)