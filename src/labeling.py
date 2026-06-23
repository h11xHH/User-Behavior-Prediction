"""labeling.py

Build the sample_D table for a prediction day D.

sample_D: candidate pairs (user_id, item_id) with binary label indicating whether
a purchase happens for this pair

For a pair to be a candidate, the user must interacted the item within the period
[D - candiate_lookback_days, D).

The candidate has label 1 if a purhcase happens on day D, else 0.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from sqlalchemy.engine import Engine


def day_bounds(prediction_date: str, lookback_days: int) -> tuple[str, str, str]:
    """Compute the three datetime boundaries used to build a sample for day D.

    Input
    -----
    prediction_date : str
        The day D in 'YYYY-MM-DD' form.
    lookback_days : int
        How many days before D the candidate window covers.

    Output
    ------
    tuple[str, str, str]
        (window_start, d_start, d_end) as 'YYYY-MM-DD HH:MM:SS' strings, 
        where:
          window_start = midnight of (D - lookback_days) -> candidate window opens
          d_start = midnight of D -> candidate window closes / label day opens
          d_end = midnight of (D + 1 day) -> label day closes
    """
    day = datetime.strptime(prediction_date, "%Y-%m-%d")
    window_start = day - timedelta(days=lookback_days)
    d_end = day + timedelta(days=1)
    fmt = "%Y-%m-%d %H:%M:%S"
    return window_start.strftime(fmt), day.strftime(fmt), d_end.strftime(fmt)


def build_sample(engine: Engine, table: str, prediction_date: str, lookback_days: int) -> pd.DataFrame:
    """Build the candidate (user, item) pairs and their label for day D.

    Input
    -----
    engine : Engine
        SQLAlchemy engine.
    table : str
        Source behaviour table (raw_behavior).
    prediction_date : str
        Day D in 'YYYY-MM-DD'.
    lookback_days : int
        Candidate window length in days before D.

    Output
    ------
    pandas.DataFrame
        Columns: user_id (str), item_id (str), cutoff_date (str), label (int 0/1).

    Rules
    -----
    Candidates = DISTINCT (user, item) that interacted in [window_start, d_start).
    Positives = DISTINCT (user, item) that bought (behavior_type=4) in [d_start, d_end).
    label = 1 if the candidate is also a positive, else 0 (LEFT JOIN + NULL check).
    All the heavy work happens in MySQL; only the small result comes back.
    """
    window_start, d_start, d_end = day_bounds(prediction_date, lookback_days)
    sql = f"""
        SELECT
            c.user_id,
            c.item_id,
            CASE WHEN b.user_id IS NULL THEN 0 ELSE 1 END AS label
        FROM (
            SELECT DISTINCT user_id, item_id
            FROM {table}
            WHERE time >= '{window_start}' AND time < '{d_start}'
        ) AS c
        LEFT JOIN (
            SELECT DISTINCT user_id, item_id
            FROM {table}
            WHERE time >= '{d_start}' AND time < '{d_end}' AND behavior_type = 4
        ) AS b
          ON c.user_id = b.user_id AND c.item_id = b.item_id
    """
    frame = read_query(engine, sql)
    frame["cutoff_date"] = prediction_date
    frame["label"] = frame["label"].astype("int64")
    return frame[["user_id", "item_id", "cutoff_date", "label"]]


def save_sample(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save a sample table to interim as Parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"sample__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def main() -> None:
    """Build and save a sample table for each configured prediction date."""
    config = load_config()
    creds = load_secrets(config.resolve_path(config.database["secrets_file"]))
    engine = make_engine(creds)
    table = config.database["table_raw"]
    interim_dir = config.resolve_path(config.paths["interim_dir"])
    lookback = int(config.labeling["candidate_lookback_days"])
    dates = config.labeling["prediction_dates"]

    if not dates:
        print("No prediction_dates set in the labeling section of config.yaml.")
        return

    for prediction_date in dates:
        frame = build_sample(engine, table, prediction_date, lookback)
        path = save_sample(frame, interim_dir, prediction_date)
        n_total = len(frame)
        n_pos = int(frame["label"].sum())
        rate = 100.0 * n_pos / n_total if n_total else 0.0
        print(f"D={prediction_date}: {n_total:,} candidates, "
              f"{n_pos:,} positives ({rate:.3f}%)  ->  {path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Labeling failed: {error}")