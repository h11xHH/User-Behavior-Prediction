"""features/common.py

Shared helpers for the feature builders: time-window boundaries, the
hour-histogram pivot, and the peak-hour argmax with a -1 sentinel.

This is subject to change for later user_item, user_category feature constructions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

FMT = "%Y-%m-%d %H:%M:%S"
ALL_HOURS = list(range(24))


def day_midnight(prediction_date: str) -> str:
    """Return 'YYYY-MM-DD 00:00:00' for D."""
    return datetime.strptime(prediction_date, "%Y-%m-%d").strftime(FMT)


def window_start(prediction_date: str, window_days: int) -> str:
    """Return the 'YYYY-MM-DD HH:MM:SS' start of a window_days window before D."""
    day = datetime.strptime(prediction_date, "%Y-%m-%d")
    return (day - timedelta(days=window_days)).strftime(FMT)


def hist_to_wide(hist: pd.DataFrame, index_col: str, value_col: str) -> pd.DataFrame:
    """Pivot a long (entity, hod, value) histogram to wide: entity x hours 0..23.

    Input : long DataFrame with columns [index_col, 'hod', value_col].
    Output: DataFrame indexed by index_col, one column per hour 0/1/.../23 (0-filled).
    """
    wide = hist.pivot_table(index=index_col, columns="hod", values=value_col,
                            aggfunc="sum", fill_value=0)
    return wide.reindex(columns=ALL_HOURS, fill_value=0)


def peak_hour(wide: pd.DataFrame) -> pd.Series:
    """Argmax hour per entity (ties -> earliest hour); -1 where the row is all zeros.

    Input : wide DataFrame indexed by entity, columns 0/1/.../23 of counts.
    Output: Series of the peak hour per entity (int; -1 when undefined).
    """
    peak = wide.idxmax(axis=1)          # idxmax returns the first (smallest) hour on ties
    peak[wide.sum(axis=1) == 0] = -1    # undefined when there are no such actions
    return peak.astype("int64")