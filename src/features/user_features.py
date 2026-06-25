"""features/user_features.py

Phase 5: build the feat_user table, keyed by (user_id, cutoff_date), using only
behaviour with time < D. Builds one prediction day at a time, then combines all
the per-day tables into a single stacked feat_user.parquet spanning every
configured cutoff date.

Pattern (used by every feature builder): MySQL does the heavy GROUP BY, pandas
does the light derivation. Per day, two queries run:
  1. counts query -> per user: windowed browse counts, all-history fav/cart/buy,
     and the user's last action time (for recency).
  2. hour query   -> per (user, hour-of-day): total / browse / buy counts, the
     raw material for the 24-hour histogram, peak hours, fractions, entropy.

Resumability: a day whose per-day file already exists is skipped, so an
interrupted run resumes where it left off. To force a rebuild (e.g. after
changing windows), delete the relevant feat_user__*.parquet files.

Run standalone:
    python -m src.features.user_features
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query

_FMT = "%Y-%m-%d %H:%M:%S"
MORNING_HOURS = [6, 7, 8, 9, 10, 11]
AFTERNOON_HOURS = [12, 13, 14, 15, 16, 17]
NIGHT_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]


def _window_start(prediction_date: str, window_days: int) -> str:
    """Return the 'YYYY-MM-DD HH:MM:SS' start of a window_days window before D."""
    day = datetime.strptime(prediction_date, "%Y-%m-%d")
    return (day - timedelta(days=window_days)).strftime(_FMT)


def build_user_counts(engine: Engine, table: str, prediction_date: str,
                      windows: list[int]) -> pd.DataFrame:
    """Per-user windowed browse counts + all-history fav/cart/buy + last action.

    Output columns: user_id, user_browse_{w}..., user_favorite_all,
    user_cart_all, user_buy_all, last_action_time.
    """
    d_start = datetime.strptime(prediction_date, "%Y-%m-%d").strftime(_FMT)
    browse_selects = []
    for window in windows:
        if window == 0:
            browse_selects.append("SUM(behavior_type = 1) AS user_browse_all")
        else:
            w_start = _window_start(prediction_date, window)
            browse_selects.append(
                f"SUM(behavior_type = 1 AND time >= '{w_start}') AS user_browse_{window}d"
            )
    sql = f"""
        SELECT
            user_id,
            {", ".join(browse_selects)},
            SUM(behavior_type = 2) AS user_favorite_all,
            SUM(behavior_type = 3) AS user_cart_all,
            SUM(behavior_type = 4) AS user_buy_all,
            MAX(time) AS last_action_time
        FROM {table}
        WHERE time < '{d_start}'
        GROUP BY user_id
    """
    return read_query(engine, sql)


def build_user_hour_histogram(engine: Engine, table: str, prediction_date: str) -> pd.DataFrame:
    """Per (user, hour-of-day): total / browse / buy counts (long format)."""
    d_start = datetime.strptime(prediction_date, "%Y-%m-%d").strftime(_FMT)
    sql = f"""
        SELECT
            user_id,
            HOUR(time)             AS hod,
            COUNT(*)               AS action_cnt,
            SUM(behavior_type = 1) AS browse_cnt,
            SUM(behavior_type = 4) AS buy_cnt
        FROM {table}
        WHERE time < '{d_start}'
        GROUP BY user_id, HOUR(time)
    """
    return read_query(engine, sql)


def _peak_hour(wide: pd.DataFrame) -> pd.Series:
    """Argmax hour per user (ties -> earliest hour); -1 where the row is all zeros."""
    peak = wide.idxmax(axis=1)          # idxmax returns the first (smallest) hour on ties
    peak[wide.sum(axis=1) == 0] = -1    # undefined when there are no such actions
    return peak.astype("int64")


def derive_hour_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Turn the long hour histogram into per-user timing features.

    Input : long DataFrame (user_id, hod, action_cnt, browse_cnt, buy_cnt).
    Output: DataFrame indexed by user_id with action_0..23, the three peak hours,
            morning/afternoon/night fractions, and hour entropy.
    Pure pandas, so it is unit-testable without a database.
    """
    all_hours = list(range(24))

    def to_wide(value_col: str) -> pd.DataFrame:
        wide = hist.pivot_table(index="user_id", columns="hod", values=value_col,
                                aggfunc="sum", fill_value=0)
        return wide.reindex(columns=all_hours, fill_value=0)

    action_wide = to_wide("action_cnt")
    browse_wide = to_wide("browse_cnt")
    buy_wide = to_wide("buy_cnt")

    total = action_wide.sum(axis=1)
    out = pd.DataFrame(index=action_wide.index)

    out["user_peak_hour"] = _peak_hour(action_wide)
    out["user_browse_peak_hour"] = _peak_hour(browse_wide)
    out["user_buy_peak_hour"] = _peak_hour(buy_wide)

    out["user_morning_fraction"] = action_wide[MORNING_HOURS].sum(axis=1) / total
    out["user_afternoon_fraction"] = action_wide[AFTERNOON_HOURS].sum(axis=1) / total
    out["user_night_fraction"] = action_wide[NIGHT_HOURS].sum(axis=1) / total

    # Shannon entropy (base 2) of the 24-hour distribution: high = spread out
    proportion = action_wide.div(total, axis=0).to_numpy()
    log_term = np.zeros_like(proportion)
    mask = proportion > 0
    log_term[mask] = np.log2(proportion[mask])
    out["user_hour_entropy"] = -(proportion * log_term).sum(axis=1)

    action_cols = action_wide.rename(columns={h: f"user_action_{h}" for h in all_hours})
    return out.join(action_cols)


def build_user_features(engine: Engine, table: str, prediction_date: str,
                        windows: list[int]) -> pd.DataFrame:
    """Assemble the full feat_user table for one prediction day D.

    Output: DataFrame keyed by (user_id, cutoff_date) with every user feature.
    """
    counts = build_user_counts(engine, table, prediction_date, windows)
    hist = build_user_hour_histogram(engine, table, prediction_date)

    count_cols = [c for c in counts.columns if c not in ("user_id", "last_action_time")]
    counts[count_cols] = counts[count_cols].astype("int64")

    hour_feats = derive_hour_features(hist).reset_index()
    frame = counts.merge(hour_feats, on="user_id", how="left")

    frame["user_conversion_browse_buy"] = np.where(
        frame["user_browse_all"] > 0,
        frame["user_buy_all"] / frame["user_browse_all"].replace(0, np.nan), 0.0)
    frame["user_conversion_cart_buy"] = np.where(
        frame["user_cart_all"] > 0,
        frame["user_buy_all"] / frame["user_cart_all"].replace(0, np.nan), 0.0)
    frame[["user_conversion_browse_buy", "user_conversion_cart_buy"]] = (
        frame[["user_conversion_browse_buy", "user_conversion_cart_buy"]].fillna(0.0))

    d_start = pd.Timestamp(datetime.strptime(prediction_date, "%Y-%m-%d"))
    delta_hours = (d_start - pd.to_datetime(frame["last_action_time"])).dt.total_seconds() / 3600.0
    frame["user_since_last_hour"] = delta_hours.round().astype("int64")

    frame["cutoff_date"] = prediction_date
    frame = frame.drop(columns=["last_action_time"])

    key_cols = ["user_id", "cutoff_date"]
    other_cols = [c for c in frame.columns if c not in key_cols]
    return frame[key_cols + other_cols]


def save_user_features(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day feat_user table to interim as Parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"feat_user__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_user_features(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day feat_user files for `dates` into one feat_user.parquet.

    Input : interim_dir and the list of cutoff dates that have been built.
    Output: path to the combined table, keyed by (user_id, cutoff_date).
    Raises: FileNotFoundError if any per-day file is missing.
    """
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"feat_user__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; build that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "feat_user.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Build feat_user for every configured cutoff date, then combine them.

    Resumable: a day whose per-day file exists is skipped. Each external step
    (config, DB, per-day build, combine) is wrapped so a failure reports what
    failed and why, with the original cause preserved in the traceback.
    """
    config = load_config()

    try:
        table = config.database["table_raw"]
        secrets_path = config.resolve_path(config.database["secrets_file"])
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        windows = [int(w) for w in config.features["count_windows_days"]]
        dates = list(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"database.table_raw, database.secrets_file, paths.interim_dir, "
            f"features.count_windows_days, labeling.prediction_dates."
        ) from error

    try:
        engine = make_engine(load_secrets(secrets_path))
    except Exception as error:
        raise RuntimeError(f"Could not set up the MySQL connection ({error}).") from error

    for prediction_date in dates:
        per_day = interim_dir / f"feat_user__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already built, skipping")
            continue
        try:
            frame = build_user_features(engine, table, prediction_date, windows)
            save_user_features(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while building feat_user for D={prediction_date} ({error})."
            ) from error
        print(f"D={prediction_date}: feat_user {frame.shape[0]:,} users x {frame.shape[1]} cols")

    try:
        combined_path = combine_user_features(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining per-day feat_user files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined feat_user: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates "
          f"x {combined.shape[1]} cols  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:  # top-level guard: report clearly, never hide the cause
        print(f"user_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)