"""features/user_features.py

Build the feat_user table, using only behaviour with time < D. Builds one 
prediction day at a time, then combines all per-day tables into one stacked 
feat_user.parquet.

Features produced:
  user_browse_{w} | windowed browse counts (w from config; 0 -> 'all')
  user_favorite_all, user_cart_all, user_buy_all
  user_conversion_browse_buy = buy_all / browse_all
  user_conversion_cart_buy = buy_all / cart_all
  user_since_last_hour | hours from last action to D
  user_peak_hour, user_browse_peak_hour, user_buy_peak_hour   (-1 if undefined)
  user_morning_fraction (6-11), user_afternoon_fraction (12-17),
  user_night_fraction (18-23 + 0-5)
  user_hour_entropy | Shannon entropy (base 2) of the 24-hour histogram
  user_action_0 .. user_action_23   actions per hour-of-day
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.features.common import ALL_HOURS, day_midnight, hist_to_wide, peak_hour, window_start

MORNING_HOURS = [6, 7, 8, 9, 10, 11]
AFTERNOON_HOURS = [12, 13, 14, 15, 16, 17]
NIGHT_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]


def build_user_counts(engine: Engine, table: str, prediction_date: str,
                      windows: list[int]) -> pd.DataFrame:
    """Per-user windowed browse counts + all-history fav/cart/buy + last action."""
    d_start = day_midnight(prediction_date)
    browse_selects = []
    for window in windows:
        if window == 0:
            browse_selects.append("SUM(behavior_type = 1) AS user_browse_all")
        else:
            w_start = window_start(prediction_date, window)
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
    d_start = day_midnight(prediction_date)
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


def derive_hour_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Turn the long hour histogram into per-user timing features.

    Output: DataFrame indexed by user_id with action_0/1/.../23, the three peak hours,
            morning/afternoon/night fractions, and hour entropy.
    Pure pandas, so it is unit-testable without a database.
    """
    action_wide = hist_to_wide(hist, "user_id", "action_cnt")
    browse_wide = hist_to_wide(hist, "user_id", "browse_cnt")
    buy_wide = hist_to_wide(hist, "user_id", "buy_cnt")

    total = action_wide.sum(axis=1)
    out = pd.DataFrame(index=action_wide.index)

    out["user_peak_hour"] = peak_hour(action_wide)
    out["user_browse_peak_hour"] = peak_hour(browse_wide)
    out["user_buy_peak_hour"] = peak_hour(buy_wide)

    out["user_morning_fraction"] = action_wide[MORNING_HOURS].sum(axis=1) / total
    out["user_afternoon_fraction"] = action_wide[AFTERNOON_HOURS].sum(axis=1) / total
    out["user_night_fraction"] = action_wide[NIGHT_HOURS].sum(axis=1) / total

    # Shannon entropy (base 2) of the 24-hour distribution: high = spread out
    proportion = action_wide.div(total, axis=0).to_numpy()
    log_term = np.zeros_like(proportion)
    mask = proportion > 0
    log_term[mask] = np.log2(proportion[mask])
    out["user_hour_entropy"] = -(proportion * log_term).sum(axis=1)

    action_cols = action_wide.rename(columns={h: f"user_action_{h}" for h in ALL_HOURS})
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

    d_start = pd.Timestamp(day_midnight(prediction_date))
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
    """Stack the per-day feat_user files for `dates` into one feat_user.parquet."""
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
    """Build feat_user for every configured cutoff date, then combine them."""
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
    except Exception as error:
        print(f"user_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)