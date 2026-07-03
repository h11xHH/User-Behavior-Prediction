"""validation/recipes.py

Independent recompute primitives for reconciling feature tables against
raw_behavior.

recipe(engine, table, key: dict, prediction_date: str) -> (value, demo_str)

The ~8 primitives below are shared across all six tables; only the key dict
differs between, say, user_browse_all and user_item_browse_all. That is how the
overlap between tables is handled once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_io import read_query
from src.features.common import day_midnight, window_start

BEHAVIOR_NAME = {1: "browse", 2: "favorite", 3: "cart", 4: "buy"}

def _read(engine, sql: str) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    return read_query(engine, sql)


def _scalar(engine, sql: str):
    """Run a query expected to return exactly one value; return it."""
    frame = _read(engine, sql)
    return frame.iloc[0, 0]


def _where_key(key: dict) -> str:
    """Build a 'col = value AND ...' clause from an entity key dict (ids are strings)."""
    return " AND ".join(f"{col} = '{value}'" for col, value in key.items())


def count(behavior: int | None = None, window: int = 0, hour: int | None = None):
    """COUNT of rows for a key, optionally filtered by behaviour / window / hour-of-day."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        conds = [_where_key(key), f"time < '{d_start}'"]
        bits = []
        if behavior is not None:
            conds.append(f"behavior_type = {behavior}")
            bits.append(BEHAVIOR_NAME[behavior])
        else:
            bits.append("all-actions")
        if window and window != 0:
            conds.append(f"time >= '{window_start(prediction_date, window)}'")
            bits.append(f"last {window}d")
        else:
            bits.append("all history")
        if hour is not None:
            conds.append(f"HOUR(time) = {hour}")
            bits.append(f"hour={hour}")
        value = int(_scalar(engine, f"SELECT COUNT(*) FROM {table} WHERE " + " AND ".join(conds)))
        return value, f"COUNT[{', '.join(bits)}, t<{prediction_date}]"
    return recipe


def distinct(col: str, behavior: int | None = None):
    """COUNT(DISTINCT col) for a key, optionally filtered by behaviour."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        conds = [_where_key(key), f"time < '{d_start}'"]
        bits = []
        if behavior is not None:
            conds.append(f"behavior_type = {behavior}")
            bits.append(BEHAVIOR_NAME[behavior])
        sql = f"SELECT COUNT(DISTINCT {col}) FROM {table} WHERE " + " AND ".join(conds)
        value = int(_scalar(engine, sql))
        return value, f"COUNT DISTINCT {col} [{', '.join(bits) or 'all'}, t<{prediction_date}]"
    return recipe


def ratio(num_behavior: int, den_behavior: int):
    """Guarded division of two all-history behaviour counts (0 when denominator is 0)."""
    def recipe(engine, table, key, prediction_date):
        num, _ = count(behavior=num_behavior)(engine, table, key, prediction_date)
        den, _ = count(behavior=den_behavior)(engine, table, key, prediction_date)
        value = num / den if den > 0 else 0.0
        return value, (f"{num} {BEHAVIOR_NAME[num_behavior]} / {den} "
                       f"{BEHAVIOR_NAME[den_behavior]} = {value:.4f}")
    return recipe


def peak_hour(behavior: int | None = None):
    """Argmax hour-of-day (ties -> earliest); -1 if there are no such actions."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        conds = [_where_key(key), f"time < '{d_start}'"]
        label = "all-actions"
        if behavior is not None:
            conds.append(f"behavior_type = {behavior}")
            label = BEHAVIOR_NAME[behavior]
        sql = (f"SELECT HOUR(time) AS hod, COUNT(*) AS c FROM {table} "
               f"WHERE " + " AND ".join(conds) + " GROUP BY hod")
        frame = _read(engine, sql)
        if frame.empty:
            return -1, f"no {label} actions -> -1"
        frame = frame.sort_values(["c", "hod"], ascending=[False, True])
        peak, n = int(frame.iloc[0]["hod"]), int(frame.iloc[0]["c"])
        return peak, f"argmax {label} hourly counts -> hour {peak} (n={n}), ties->earliest"
    return recipe


def fraction(hours):
    """Share of all-action rows that fall in a set of hours-of-day."""
    hours = list(hours)
    hour_list = ",".join(str(h) for h in hours)

    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        base = f"{_where_key(key)} AND time < '{d_start}'"
        total = int(_scalar(engine, f"SELECT COUNT(*) FROM {table} WHERE {base}"))
        num = int(_scalar(engine, f"SELECT COUNT(*) FROM {table} WHERE {base} "
                                  f"AND HOUR(time) IN ({hour_list})"))
        value = num / total if total > 0 else 0.0
        return value, f"{num}/{total} actions in {len(hours)} hours = {value:.4f}"
    return recipe


def entropy():
    """Shannon entropy (base 2) of the 24-hour all-action distribution."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        sql = (f"SELECT HOUR(time) AS hod, COUNT(*) AS c FROM {table} "
               f"WHERE {_where_key(key)} AND time < '{d_start}' GROUP BY hod")
        frame = _read(engine, sql)
        if frame.empty:
            return 0.0, "no actions -> 0"
        counts = frame["c"].to_numpy(dtype=float)
        proportion = counts / counts.sum()
        value = float(-(proportion * np.log2(proportion)).sum())
        return value, f"-sum(p*log2 p) over {len(counts)} active hours = {value:.4f}"
    return recipe


def recency_hours():
    """Whole hours from the key's last action to midnight of D."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        last = _scalar(engine, f"SELECT MAX(time) FROM {table} "
                               f"WHERE {_where_key(key)} AND time < '{d_start}'")
        value = int(round((pd.Timestamp(d_start) - pd.to_datetime(last)).total_seconds() / 3600))
        return value, f"{d_start} - {last} = {value}h"
    return recipe


def last_behavior():
    """behavior_type of the key's most recent action (ties -> highest type)."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        sql = (f"SELECT behavior_type FROM {table} WHERE {_where_key(key)} "
               f"AND time < '{d_start}' ORDER BY time DESC, behavior_type DESC LIMIT 1")
        value = int(_scalar(engine, sql))
        return value, f"latest action's behavior (ties->highest) = {value} ({BEHAVIOR_NAME[value]})"
    return recipe


def attr(col: str, agg: str = "MIN"):
    """A static item/row attribute recomputed with a simple aggregate (e.g. item_category)."""
    def recipe(engine, table, key, prediction_date):
        d_start = day_midnight(prediction_date)
        value = _scalar(engine, f"SELECT {agg}({col}) FROM {table} "
                                f"WHERE {_where_key(key)} AND time < '{d_start}'")
        return value, f"{agg}({col}) = {value}"
    return recipe


# --- composite flags (built from the primitives above) ----------------------

def is_cold():
    """1 if the key was never bought before D."""
    def recipe(engine, table, key, prediction_date):
        buys, _ = count(behavior=4)(engine, table, key, prediction_date)
        value = 1 if buys == 0 else 0
        return value, f"{buys} buys -> cold={value}"
    return recipe


def is_favorited():
    """1 if the key was ever favourited before D."""
    def recipe(engine, table, key, prediction_date):
        favs, _ = count(behavior=2)(engine, table, key, prediction_date)
        value = 1 if favs > 0 else 0
        return value, f"{favs} favourites -> {value}"
    return recipe


def cart_not_buy():
    """1 if carted before D but never bought."""
    def recipe(engine, table, key, prediction_date):
        carts, _ = count(behavior=3)(engine, table, key, prediction_date)
        buys, _ = count(behavior=4)(engine, table, key, prediction_date)
        value = 1 if (carts > 0 and buys == 0) else 0
        return value, f"cart={carts}, buy={buys} -> {value}"
    return recipe


def cross_peak(parent: str):
    """1 if the pair ever acted at its parent (user or item) overall peak hour."""
    parent_col = {"user": "user_id", "item": "item_id"}[parent]

    def recipe(engine, table, key, prediction_date):
        parent_peak, _ = peak_hour()(engine, table, {parent_col: key[parent_col]}, prediction_date)
        if parent_peak == -1:
            return 0, f"{parent} peak undefined -> 0"
        at, _ = count(hour=parent_peak)(engine, table, key, prediction_date)
        value = 1 if at > 0 else 0
        return value, f"{parent} peak hour={parent_peak}; pair acted there {at}x -> {value}"
    return recipe