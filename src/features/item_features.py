"""features/item_features.py

Build the feat_item table, keyed by (item_id, cutoff_date). 
Builds one prediction day at a time (resumable), then combines all per-day 
tables into one stacked feat_item.parquet.

There are ~2.88M items, so features are computed ONLY for candidate items.

Features produced (matching the design):
  item_category | the item's category (single, verified in Phase 1)
  item_browse_all, item_favorite_all, item_cart_all, item_buy_all
  item_distinct_user | distinct users who interacted with the item
  item_conversion_browse_buy = item_buy_all / item_browse_all
  item_is_cold | 1 if the item was never bought before D
  item_peak_hour, item_browse_peak_hour, item_buy_peak_hour   (-1 if undefined)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.features.common import day_midnight, hist_to_wide, peak_hour, window_start

_PEAK_COLS = ["item_peak_hour", "item_browse_peak_hour", "item_buy_peak_hour"]
_COUNT_COLS = ["item_browse_all", "item_favorite_all", "item_cart_all",
               "item_buy_all", "item_distinct_user"]


def build_item_counts(engine: Engine, table: str, prediction_date: str, lookback: int) -> pd.DataFrame:
    """Per-candidate-item all-history counts, category, and distinct-user count.

    The JOIN restricts to candidate items (interacted with in [D - lookback, D));
    the aggregates themselves range over all history before D.
    """
    d_start = day_midnight(prediction_date)
    w_start = window_start(prediction_date, lookback)
    sql = f"""
        SELECT
            r.item_id,
            MIN(r.item_category)        AS item_category,
            SUM(r.behavior_type = 1)    AS item_browse_all,
            SUM(r.behavior_type = 2)    AS item_favorite_all,
            SUM(r.behavior_type = 3)    AS item_cart_all,
            SUM(r.behavior_type = 4)    AS item_buy_all,
            COUNT(DISTINCT r.user_id)   AS item_distinct_user
        FROM {table} r
        JOIN (
            SELECT DISTINCT item_id FROM {table}
            WHERE time >= '{w_start}' AND time < '{d_start}'
        ) cand ON r.item_id = cand.item_id
        WHERE r.time < '{d_start}'
        GROUP BY r.item_id
    """
    return read_query(engine, sql)


def build_item_hour_histogram(engine: Engine, table: str, prediction_date: str,
                              lookback: int) -> pd.DataFrame:
    """Per (candidate-item, hour-of-day): total / browse / buy counts (long)."""
    d_start = day_midnight(prediction_date)
    w_start = window_start(prediction_date, lookback)
    sql = f"""
        SELECT
            r.item_id,
            HOUR(r.time)             AS hod,
            COUNT(*)                 AS action_cnt,
            SUM(r.behavior_type = 1) AS browse_cnt,
            SUM(r.behavior_type = 4) AS buy_cnt
        FROM {table} r
        JOIN (
            SELECT DISTINCT item_id FROM {table}
            WHERE time >= '{w_start}' AND time < '{d_start}'
        ) cand ON r.item_id = cand.item_id
        WHERE r.time < '{d_start}'
        GROUP BY r.item_id, HOUR(r.time)
    """
    return read_query(engine, sql)


def derive_item_hour_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-item peak hours (overall / browse / buy) from the long histogram."""
    out = pd.DataFrame(index=hist_to_wide(hist, "item_id", "action_cnt").index)
    out["item_peak_hour"] = peak_hour(hist_to_wide(hist, "item_id", "action_cnt"))
    out["item_browse_peak_hour"] = peak_hour(hist_to_wide(hist, "item_id", "browse_cnt"))
    out["item_buy_peak_hour"] = peak_hour(hist_to_wide(hist, "item_id", "buy_cnt"))
    return out


def build_item_features(engine: Engine, table: str, prediction_date: str, lookback: int) -> pd.DataFrame:
    """Assemble the full feat_item table for one prediction day D.

    Output: DataFrame keyed by (item_id, cutoff_date) with every item feature.
    """
    counts = build_item_counts(engine, table, prediction_date, lookback)
    hist = build_item_hour_histogram(engine, table, prediction_date, lookback)

    counts[_COUNT_COLS] = counts[_COUNT_COLS].astype("int64")

    hour_feats = derive_item_hour_features(hist).reset_index()
    frame = counts.merge(hour_feats, on="item_id", how="left")
    # defensive: every candidate item has pre-D rows, so peaks shouldn't be NaN,
    # but fill just in case so the column stays integer.
    frame[_PEAK_COLS] = frame[_PEAK_COLS].fillna(-1).astype("int64")

    frame["item_conversion_browse_buy"] = np.where(
        frame["item_browse_all"] > 0,
        frame["item_buy_all"] / frame["item_browse_all"].replace(0, np.nan), 0.0)
    frame["item_conversion_browse_buy"] = frame["item_conversion_browse_buy"].fillna(0.0)

    frame["item_is_cold"] = (frame["item_buy_all"] == 0).astype("int64")

    frame["cutoff_date"] = prediction_date
    key_cols = ["item_id", "cutoff_date"]
    other_cols = [c for c in frame.columns if c not in key_cols]
    return frame[key_cols + other_cols]


def save_item_features(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day feat_item table to interim as parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"feat_item__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_item_features(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day feat_item files for `dates` into one feat_item.parquet."""
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"feat_item__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; build that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "feat_item.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Build feat_item for every configured cutoff date, then combine them."""
    config = load_config()
    try:
        table = config.database["table_raw"]
        secrets_path = config.resolve_path(config.database["secrets_file"])
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        lookback = int(config.labeling["candidate_lookback_days"])
        dates = list(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"database.table_raw, database.secrets_file, paths.interim_dir, "
            f"labeling.candidate_lookback_days, labeling.prediction_dates."
        ) from error

    try:
        engine = make_engine(load_secrets(secrets_path))
    except Exception as error:
        raise RuntimeError(f"Could not set up the MySQL connection ({error}).") from error

    for prediction_date in dates:
        per_day = interim_dir / f"feat_item__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already built, skipping")
            continue
        try:
            frame = build_item_features(engine, table, prediction_date, lookback)
            save_item_features(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while building feat_item for D={prediction_date} ({error})."
            ) from error
        print(f"D={prediction_date}: feat_item {frame.shape[0]:,} items x {frame.shape[1]} cols")

    try:
        combined_path = combine_item_features(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining per-day feat_item files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined feat_item: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates "
          f"x {combined.shape[1]} cols  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:
        print(f"item_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)