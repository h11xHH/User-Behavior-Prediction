"""features/user_category_features.py

Build the feat_user_category table. Key is (user_id, item_category, cutoff_date)

Candidate: the distinct (user, category) pairs implied by the candidate
(user, item) pairs.

Features produced:
  user_category_browse_all   the user's browses in that category before D
  user_category_buy_all      the user's purchases in that category before D
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.features.common import day_midnight, window_start

_COUNT_COLS = ["user_category_browse_all", "user_category_buy_all"]


def build_user_category_counts(engine: Engine, table: str, prediction_date: str,
                               lookback: int) -> pd.DataFrame:
    """
    Per candidate (user, category): all-history browse and buy counts.

    The JOIN restricts to candidate (user, category) pairs.
    The counts range over all history before D.
    """
    d_start = day_midnight(prediction_date)
    cand_start = window_start(prediction_date, lookback)
    sql = f"""
        SELECT
            r.user_id,
            r.item_category,
            SUM(r.behavior_type = 1) AS user_category_browse_all,
            SUM(r.behavior_type = 4) AS user_category_buy_all
        FROM {table} r
        JOIN (
            SELECT DISTINCT user_id, item_category FROM {table}
            WHERE time >= '{cand_start}' AND time < '{d_start}'
        ) cand ON r.user_id = cand.user_id AND r.item_category = cand.item_category
        WHERE r.time < '{d_start}'
        GROUP BY r.user_id, r.item_category
    """
    return read_query(engine, sql)


def build_user_category_features(engine: Engine, table: str, prediction_date: str,
                                 lookback: int) -> pd.DataFrame:
    """
    Assemble the full feat_user_category table for one prediction day D.

    Output
    ------
    DataFrame with key: (user_id, item_category, cutoff_date).
    """
    frame = build_user_category_counts(engine, table, prediction_date, lookback)
    frame[_COUNT_COLS] = frame[_COUNT_COLS].astype("int64")

    frame["cutoff_date"] = prediction_date
    key_cols = ["user_id", "item_category", "cutoff_date"]
    other_cols = [c for c in frame.columns if c not in key_cols]
    return frame[key_cols + other_cols]


def save_user_category_features(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day feat_user_category table to interim as Parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"feat_user_category__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_user_category_features(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day feat_user_category files into one feat_user_category.parquet."""
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"feat_user_category__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; build that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "feat_user_category.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Build feat_user_category for every configured cutoff date, then combine them."""
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
        per_day = interim_dir / f"feat_user_category__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already built, skipping")
            continue
        try:
            frame = build_user_category_features(engine, table, prediction_date, lookback)
            save_user_category_features(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while building feat_user_category for D={prediction_date} ({error})."
            ) from error
        print(f"D={prediction_date}: feat_user_category {frame.shape[0]:,} pairs x {frame.shape[1]} cols")

    try:
        combined_path = combine_user_category_features(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining per-day feat_user_category files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined feat_user_category: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates "
          f"x {combined.shape[1]} cols  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:  # top-level guard: report clearly, never hide the cause
        print(f"user_category_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)