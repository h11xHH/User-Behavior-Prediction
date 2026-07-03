"""features/user_item_features.py

Build the feat_user_item table, Key is (user_id, item_id, cutoff_date)

Candidate: the pairs that interacted in [D - lookback, D)
=
Candidates in the labeled sample.

Depends on feat_user__D and feat_item__D
-> run after user_features and item_features

Features produced:
  user_item_browse_{w}          windowed browse counts (w from config)
  user_item_browse_distinct_hour distinct hours the pair was browsed
  user_item_cart_all, user_item_buy_all
  user_item_cart_not_buy        1 if carted before D but never bought
  user_item_is_favorited        1 if the pair was ever favourited
  user_item_last_behavior       behavior_type (1-4) of the pair's latest action
  user_item_since_last_hour     hours from the pair's last action to D
  user_item_is_user_peak_hour   1 if the pair acted at the USER's peak hour
  user_item_is_item_peak_hour   1 if the pair acted at the ITEM's peak hour
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.features.common import day_midnight, window_start


def build_user_item_aggregates(engine: Engine, table: str, prediction_date: str,
                               lookback: int, windows: list[int]) -> pd.DataFrame:
    """
    Per candidate pair: windowed browse, cart/buy/fav counts, distinct browse
    hours, last action time, and the last behaviour type.
    """
    d_start = day_midnight(prediction_date)
    cand_start = window_start(prediction_date, lookback)

    browse_selects = []
    for window in windows:
        if window == 0:
            browse_selects.append("SUM(r.behavior_type = 1) AS user_item_browse_all")
        else:
            w_start = window_start(prediction_date, window)
            browse_selects.append(
                f"SUM(r.behavior_type = 1 AND r.time >= '{w_start}') AS user_item_browse_{window}d"
            )
    sql = f"""
        SELECT
            r.user_id,
            r.item_id,
            {", ".join(browse_selects)},
            SUM(r.behavior_type = 2) AS ui_fav_all,
            SUM(r.behavior_type = 3) AS user_item_cart_all,
            SUM(r.behavior_type = 4) AS user_item_buy_all,
            COUNT(DISTINCT IF(r.behavior_type = 1, r.time, NULL)) AS user_item_browse_distinct_hour,
            MAX(r.time) AS last_action_time,
            CAST(SUBSTRING_INDEX(
                GROUP_CONCAT(r.behavior_type ORDER BY r.time DESC, r.behavior_type DESC SEPARATOR ','),
                ',', 1) AS UNSIGNED) AS user_item_last_behavior
        FROM {table} r
        JOIN (
            SELECT DISTINCT user_id, item_id FROM {table}
            WHERE time >= '{cand_start}' AND time < '{d_start}'
        ) cand ON r.user_id = cand.user_id AND r.item_id = cand.item_id
        WHERE r.time < '{d_start}'
        GROUP BY r.user_id, r.item_id
    """
    return read_query(engine, sql)


def build_pair_hour_presence(engine: Engine, table: str, prediction_date: str,
                             lookback: int) -> pd.DataFrame:
    """Distinct (candidate pair, hour-of-day) it ever acted in before D."""
    d_start = day_midnight(prediction_date)
    cand_start = window_start(prediction_date, lookback)
    sql = f"""
        SELECT DISTINCT r.user_id, r.item_id, HOUR(r.time) AS hod
        FROM {table} r
        JOIN (
            SELECT DISTINCT user_id, item_id FROM {table}
            WHERE time >= '{cand_start}' AND time < '{d_start}'
        ) cand ON r.user_id = cand.user_id AND r.item_id = cand.item_id
        WHERE r.time < '{d_start}'
    """
    return read_query(engine, sql)


def derive_peak_flags(pair_hours: pd.DataFrame, user_peak: pd.DataFrame,
                      item_peak: pd.DataFrame) -> pd.DataFrame:
    """Flag whether each pair acted at its user's and its item's peak hour.

    Input : pair_hours (user_id, item_id, hod); user_peak (user_id,
            user_peak_hour); item_peak (item_id, item_peak_hour).
    Output: per-pair DataFrame with user_item_is_user_peak_hour and
            user_item_is_item_peak_hour (0/1).
    Pure pandas -> unit-testable.
    """
    merged = (pair_hours
              .merge(user_peak, on="user_id", how="left")
              .merge(item_peak, on="item_id", how="left"))
    merged["at_user_peak"] = (merged["hod"] == merged["user_peak_hour"]).astype("int64")
    merged["at_item_peak"] = (merged["hod"] == merged["item_peak_hour"]).astype("int64")
    return (merged.groupby(["user_id", "item_id"], as_index=False)
            .agg(user_item_is_user_peak_hour=("at_user_peak", "max"),
                 user_item_is_item_peak_hour=("at_item_peak", "max")))


def _load_peak(interim_dir: Path, kind: str, prediction_date: str,
               key_col: str, peak_col: str) -> pd.DataFrame:
    """Load [key_col, peak_col] from a feat_{kind}__{D} file; error if absent."""
    path = interim_dir / f"feat_{kind}__{prediction_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"'{path.name}' not found. Build {kind} features for {prediction_date} "
            f"before user_item features (it supplies {peak_col})."
        )
    return pd.read_parquet(path, columns=[key_col, peak_col])


def build_user_item_features(engine: Engine, table: str, prediction_date: str,
                             lookback: int, windows: list[int], interim_dir: Path) -> pd.DataFrame:
    """Assemble the full feat_user_item table for one prediction day D."""
    frame = build_user_item_aggregates(engine, table, prediction_date, lookback, windows)

    int_cols = [c for c in frame.columns
                if c not in ("user_id", "item_id", "last_action_time")]
    frame[int_cols] = frame[int_cols].astype("int64")

    # derived flags
    frame["user_item_is_favorited"] = (frame["ui_fav_all"] > 0).astype("int64")
    frame["user_item_cart_not_buy"] = (
        (frame["user_item_cart_all"] > 0) & (frame["user_item_buy_all"] == 0)).astype("int64")
    frame = frame.drop(columns=["ui_fav_all"])

    # recency: whole hours from the pair's last action to midnight of D
    d_midnight = pd.Timestamp(day_midnight(prediction_date))
    delta = (d_midnight - pd.to_datetime(frame["last_action_time"])).dt.total_seconds() / 3600.0
    frame["user_item_since_last_hour"] = delta.round().astype("int64")
    frame = frame.drop(columns=["last_action_time"])

    # peak-hour cross-flags (need the user's and item's peak hours for this day)
    user_peak = _load_peak(interim_dir, "user", prediction_date, "user_id", "user_peak_hour")
    item_peak = _load_peak(interim_dir, "item", prediction_date, "item_id", "item_peak_hour")
    pair_hours = build_pair_hour_presence(engine, table, prediction_date, lookback)
    flags = derive_peak_flags(pair_hours, user_peak, item_peak)
    frame = frame.merge(flags, on=["user_id", "item_id"], how="left")
    flag_cols = ["user_item_is_user_peak_hour", "user_item_is_item_peak_hour"]
    frame[flag_cols] = frame[flag_cols].fillna(0).astype("int64")

    frame["cutoff_date"] = prediction_date

    ordered = (["user_id", "item_id", "cutoff_date"]
               + [f"user_item_browse_{w}d" if w != 0 else "user_item_browse_all" for w in windows]
               + ["user_item_browse_distinct_hour", "user_item_cart_all",
                  "user_item_cart_not_buy", "user_item_buy_all", "user_item_is_favorited",
                  "user_item_last_behavior", "user_item_since_last_hour",
                  "user_item_is_user_peak_hour", "user_item_is_item_peak_hour"])
    return frame[ordered]


def save_user_item_features(frame: pd.DataFrame, interim_dir: Path, prediction_date: str) -> Path:
    """Save the per-day feat_user_item table to interim as Parquet; return the path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / f"feat_user_item__{prediction_date}.parquet"
    frame.to_parquet(path, index=False)
    return path


def combine_user_item_features(interim_dir: Path, dates: list[str]) -> Path:
    """Stack the per-day feat_user_item files into one feat_user_item.parquet."""
    frames = []
    for prediction_date in dates:
        per_day = interim_dir / f"feat_user_item__{prediction_date}.parquet"
        if not per_day.exists():
            raise FileNotFoundError(
                f"Missing per-day file '{per_day.name}'; build that day before combining."
            )
        frames.append(pd.read_parquet(per_day))
    combined = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "feat_user_item.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    """Build feat_user_item for every configured cutoff date, then combine them."""
    config = load_config()
    try:
        table = config.database["table_raw"]
        secrets_path = config.resolve_path(config.database["secrets_file"])
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        lookback = int(config.labeling["candidate_lookback_days"])
        windows = [int(w) for w in config.features["count_windows_days"]]
        dates = list(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}. This module needs: "
            f"database.table_raw, database.secrets_file, paths.interim_dir, "
            f"labeling.candidate_lookback_days, features.count_windows_days, "
            f"labeling.prediction_dates."
        ) from error

    try:
        engine = make_engine(load_secrets(secrets_path))
    except Exception as error:
        raise RuntimeError(f"Could not set up the MySQL connection ({error}).") from error

    for prediction_date in dates:
        per_day = interim_dir / f"feat_user_item__{prediction_date}.parquet"
        if per_day.exists():
            print(f"D={prediction_date}: already built, skipping")
            continue
        try:
            frame = build_user_item_features(engine, table, prediction_date, lookback,
                                             windows, interim_dir)
            save_user_item_features(frame, interim_dir, prediction_date)
        except Exception as error:
            raise RuntimeError(
                f"Failed while building feat_user_item for D={prediction_date} ({error})."
            ) from error
        print(f"D={prediction_date}: feat_user_item {frame.shape[0]:,} pairs x {frame.shape[1]} cols")

    try:
        combined_path = combine_user_item_features(interim_dir, dates)
    except Exception as error:
        raise RuntimeError(f"Failed while combining per-day feat_user_item files ({error}).") from error

    combined = pd.read_parquet(combined_path)
    print(f"\nCombined feat_user_item: {combined.shape[0]:,} rows across "
          f"{combined['cutoff_date'].nunique()} cutoff dates "
          f"x {combined.shape[1]} cols  ->  {combined_path.name}")


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception as error:  # top-level guard: report clearly, never hide the cause
        print(f"user_item_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)