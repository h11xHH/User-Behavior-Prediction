"""validation/specs.py

One TableSpec per intermediate table.

Each spec names uniqueness key, maps every recomputable column to 
a shared recipe (Layer B, see reconcile), lists non-recomputed attribute 
columns, and gives Layer A ranges and cross-column invariants. 

Different tables use different specs, but same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.validation import recipes as R

MORNING = range(6, 12)
AFTERNOON = range(12, 18)
NIGHT = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]


@dataclass
class TableSpec:
    name: str
    parquet: str
    keys: list[str]
    recipes: dict[str, Callable] # layer B
    attributes: list[str] = field(default_factory=list)
    ranges: dict[str, tuple] = field(default_factory=dict)
    invariants: list[tuple] = field(default_factory=list)

    @property
    def entity_keys(self) -> list[str]:
        """Key columns used to address a row in raw_behavior (cutoff_date -> D param)."""
        return [k for k in self.keys if k != "cutoff_date"]

    @property
    def expected_columns(self) -> list[str]:
        return self.keys + self.attributes + list(self.recipes)


def _user_action_recipes() -> dict:
    """user_action_0 .. user_action_23 -> COUNT of all actions in that hour-of-day."""
    return {f"user_action_{h}": R.count(hour=h) for h in range(24)}


FEAT_USER = TableSpec(
    name="feat_user",
    parquet="feat_user.parquet",
    keys=["user_id", "cutoff_date"],
    recipes={
        "user_browse_1d": R.count(behavior=1, window=1),
        "user_browse_7d": R.count(behavior=1, window=7),
        "user_browse_all": R.count(behavior=1, window=0),
        "user_favorite_all": R.count(behavior=2),
        "user_cart_all": R.count(behavior=3),
        "user_buy_all": R.count(behavior=4),
        "user_conversion_browse_buy": R.ratio(4, 1),
        "user_conversion_cart_buy": R.ratio(4, 3),
        "user_since_last_hour": R.recency_hours(),
        "user_peak_hour": R.peak_hour(),
        "user_browse_peak_hour": R.peak_hour(behavior=1),
        "user_buy_peak_hour": R.peak_hour(behavior=4),
        "user_morning_fraction": R.fraction(MORNING),
        "user_afternoon_fraction": R.fraction(AFTERNOON),
        "user_night_fraction": R.fraction(NIGHT),
        "user_hour_entropy": R.entropy(),
        **_user_action_recipes(),
    },
    ranges={
        "user_peak_hour": (0, 23), "user_browse_peak_hour": (-1, 23),
        "user_buy_peak_hour": (-1, 23), "user_morning_fraction": (0, 1),
        "user_afternoon_fraction": (0, 1), "user_night_fraction": (0, 1),
        "user_hour_entropy": (0, np.log2(24)),
    },
    invariants=[
        ("browse windows nested (1d<=7d<=all)",
         lambda d: bool((d.user_browse_1d <= d.user_browse_7d).all()
                        and (d.user_browse_7d <= d.user_browse_all).all())),
        ("day-part fractions sum to 1",
         lambda d: bool(np.isclose(
             d[["user_morning_fraction", "user_afternoon_fraction", "user_night_fraction"]].sum(1),
             1.0).all())),
        ("buy_peak_hour = -1 iff no buys",
         lambda d: bool(((d.user_buy_all == 0) == (d.user_buy_peak_hour == -1)).all())),
    ],
)


FEAT_ITEM = TableSpec(
    name="feat_item",
    parquet="feat_item.parquet",
    keys=["item_id", "cutoff_date"],
    recipes={
        "item_category": R.attr("item_category", "MIN"),
        "item_browse_all": R.count(behavior=1),
        "item_favorite_all": R.count(behavior=2),
        "item_cart_all": R.count(behavior=3),
        "item_buy_all": R.count(behavior=4),
        "item_distinct_user": R.distinct("user_id"),
        "item_conversion_browse_buy": R.ratio(4, 1),
        "item_is_cold": R.is_cold(),
        "item_peak_hour": R.peak_hour(),
        "item_browse_peak_hour": R.peak_hour(behavior=1),
        "item_buy_peak_hour": R.peak_hour(behavior=4),
    },
    ranges={
        "item_peak_hour": (0, 23), "item_browse_peak_hour": (-1, 23),
        "item_buy_peak_hour": (-1, 23), "item_is_cold": (0, 1),
    },
    invariants=[
        ("item_is_cold = 1 iff no buys",
         lambda d: bool(((d.item_buy_all == 0) == (d.item_is_cold == 1)).all())),
        ("buy_peak_hour = -1 iff cold",
         lambda d: bool(((d.item_is_cold == 1) == (d.item_buy_peak_hour == -1)).all())),
    ],
)


FEAT_CATEGORY = TableSpec(
    name="feat_category",
    parquet="feat_category.parquet",
    keys=["item_category", "cutoff_date"],
    recipes={
        "category_browse_all": R.count(behavior=1),
        "category_buy_all": R.count(behavior=4),
        "category_distinct_item": R.distinct("item_id"),
        "category_conversion_browse_buy": R.ratio(4, 1),
    },
    ranges={"category_conversion_browse_buy": (0, 1)},
    invariants=[
        ("conversion in [0,1]",
         lambda d: bool(d.category_conversion_browse_buy.between(0, 1).all())),
    ],
)


FEAT_USER_ITEM = TableSpec(
    name="feat_user_item",
    parquet="feat_user_item.parquet",
    keys=["user_id", "item_id", "cutoff_date"],
    recipes={
        "user_item_browse_1d": R.count(behavior=1, window=1),
        "user_item_browse_7d": R.count(behavior=1, window=7),
        "user_item_browse_all": R.count(behavior=1, window=0),
        "user_item_browse_distinct_hour": R.distinct("time", behavior=1),
        "user_item_cart_all": R.count(behavior=3),
        "user_item_cart_not_buy": R.cart_not_buy(),
        "user_item_buy_all": R.count(behavior=4),
        "user_item_is_favorited": R.is_favorited(),
        "user_item_last_behavior": R.last_behavior(),
        "user_item_since_last_hour": R.recency_hours(),
        "user_item_is_user_peak_hour": R.cross_peak("user"),
        "user_item_is_item_peak_hour": R.cross_peak("item"),
    },
    ranges={
        "user_item_cart_not_buy": (0, 1), "user_item_is_favorited": (0, 1),
        "user_item_last_behavior": (1, 4), "user_item_is_user_peak_hour": (0, 1),
        "user_item_is_item_peak_hour": (0, 1),
    },
    invariants=[
        ("browse windows nested",
         lambda d: bool((d.user_item_browse_1d <= d.user_item_browse_7d).all()
                        and (d.user_item_browse_7d <= d.user_item_browse_all).all())),
        ("cart_not_buy = 1 iff cart>0 and buy==0",
         lambda d: bool((d.user_item_cart_not_buy ==
                         ((d.user_item_cart_all > 0) & (d.user_item_buy_all == 0)).astype(int)).all())),
        ("distinct browse hours <= browse_all",
         lambda d: bool((d.user_item_browse_distinct_hour <= d.user_item_browse_all).all())),
    ],
)


FEAT_USER_CATEGORY = TableSpec(
    name="feat_user_category",
    parquet="feat_user_category.parquet",
    keys=["user_id", "item_category", "cutoff_date"],
    recipes={
        "user_category_browse_all": R.count(behavior=1),
        "user_category_buy_all": R.count(behavior=4),
    },
)

ALL_SPECS = [FEAT_USER, FEAT_ITEM, FEAT_CATEGORY, FEAT_USER_ITEM, FEAT_USER_CATEGORY]