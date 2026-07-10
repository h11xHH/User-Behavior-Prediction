"""modeling/features.py

Turn a dataFrame into a numeric feature matrix X and label vector y. 
Shared by the baseline and the later models.

*the keys, label and item_category are dropped.
"""

from __future__ import annotations

import pandas as pd

NON_FEATURE_COLUMNS = ["user_id", "item_id", "cutoff_date", "label", "item_category"]


def get_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, "pd.Series", list[str]]:
    """Return (X, y, feature_names) from a modelling DataFrame."""
    y = frame["label"].astype(int)
    drop = [c for c in NON_FEATURE_COLUMNS if c in frame.columns]
    X = frame.drop(columns=drop)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns present: {non_numeric}. "
                         f"Add them to NON_FEATURE_COLUMNS or encode them.")
    return X, y, list(X.columns)