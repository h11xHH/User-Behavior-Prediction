"""validation/reconcile.py

The reconciliation engine. For one TableSpec it runs:
- Layer A (whole table): expected columns present, key uniqueness,
    no nulls, value ranges, cutoff-date coverage, and cross-column invariants.
- Layer B (sampled reconciliation): draw N keys stratified across cutoff dates;
    for each, recompute every mapped feature from raw_behavior via recipes.py and
    compare to the stored value, printing a worked demonstration per key.

Returns a TableResult with pass/fail tallies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.specs import TableSpec

_ACTION_PREFIX = "user_action_"


@dataclass
class TableResult:
    name: str
    structural_pass: int = 0
    structural_fail: int = 0
    values_checked: int = 0
    values_matched: int = 0
    keys_checked: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.structural_fail == 0 and self.values_checked == self.values_matched


def _matches(stored, recomputed, tol: float = 1e-9) -> bool:
    """Compare stored vs recomputed with float tolerance; exact for non-numerics."""
    if recomputed is None:
        return pd.isna(stored)
    try:
        return math.isclose(float(stored), float(recomputed), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(stored) == str(recomputed)


# Layer A

def run_structural(spec: TableSpec, frame: pd.DataFrame, expected_dates: int,
                   result: TableResult) -> None:
    """Whole-table structural checks; records pass/fail into result."""
    def check(label: str, passed: bool) -> None:
        if passed:
            result.structural_pass += 1
        else:
            result.structural_fail += 1
            result.messages.append(f"  [FAIL] {label}")

    missing = [c for c in spec.expected_columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in spec.expected_columns]
    check(f"all expected columns present (missing={missing})", not missing)
    check(f"no unexpected columns (extra={extra})", not extra)
    check("key is unique", not frame.duplicated(spec.keys).any())
    check("no null values", int(frame.isnull().sum().sum()) == 0)

    n_dates = frame["cutoff_date"].nunique()
    check(f"cutoff_date coverage = {expected_dates} (found {n_dates})",
          n_dates == expected_dates)

    for col, (lo, hi) in spec.ranges.items():
        if col in frame.columns:
            within = bool(frame[col].between(lo, hi).all())
            check(f"{col} within [{lo}, {hi}]", within)

    for label, fn in spec.invariants:
        try:
            check(f"invariant: {label}", bool(fn(frame)))
        except Exception as error:  # noqa: BLE001
            check(f"invariant: {label} (raised {type(error).__name__})", False)


# Layer B

def _stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n rows spread across distinct cutoff dates (one per date, then extra)."""
    rng = np.random.default_rng(seed)
    dates = sorted(frame["cutoff_date"].unique())
    picks = []
    chosen = list(rng.choice(dates, size=min(n, len(dates)), replace=False))
    for d in chosen:
        sub = frame[frame["cutoff_date"] == d]
        picks.append(sub.iloc[int(rng.integers(len(sub)))])
    if n > len(dates):  # more keys than dates: fill the remainder at random
        extra = frame.sample(n - len(dates), random_state=seed)
        picks.extend(row for _, row in extra.iterrows())
    return pd.DataFrame(picks).reset_index(drop=True)


def run_reconciliation(spec: TableSpec, frame: pd.DataFrame, engine, table: str,
                       n: int, seed: int, result: TableResult, printer=print) -> None:
    """Sampled per-key recompute-and-compare with printed demonstrations."""
    sample = _stratified_sample(frame, n, seed)
    action_cols = [c for c in spec.recipes if c.startswith(_ACTION_PREFIX)]
    rng = np.random.default_rng(seed + 1)

    for _, row in sample.iterrows():
        key = {k: row[k] for k in spec.entity_keys}
        prediction_date = row["cutoff_date"]
        key_str = ", ".join(f"{k}={v}" for k, v in key.items())
        printer(f"\n{spec.name}  {key_str}  cutoff_date={prediction_date}")
        result.keys_checked += 1

        # only demonstrate a few of the 24 hour columns to keep output readable
        shown_actions = set(rng.choice(action_cols, size=min(3, len(action_cols)),
                                       replace=False)) if action_cols else set()

        matched_here = checked_here = 0
        for col, recipe in spec.recipes.items():
            recomputed, demo = recipe(engine, table, key, prediction_date)
            stored = row[col]
            ok = _matches(stored, recomputed)
            checked_here += 1
            matched_here += int(ok)
            result.values_checked += 1
            result.values_matched += int(ok)
            demonstrate = col in shown_actions or not col.startswith(_ACTION_PREFIX)
            if demonstrate or not ok:
                mark = "OK " if ok else "XX "
                printer(f"  {mark}{col:<32} stored={_fmt(stored):<12} "
                        f"recomputed={_fmt(recomputed):<12} {demo}")
        printer(f"  -> {matched_here}/{checked_here} values matched for this key")


def _fmt(value) -> str:
    """Compact formatting for stored/recomputed scalars."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def reconcile_table(spec: TableSpec, interim_dir: Path, engine, table: str,
                    expected_dates: int, n: int, seed: int, printer=print) -> TableResult:
    """Run Layer A + Layer B for one table; print a header and summary."""
    result = TableResult(name=spec.name)
    path = interim_dir / spec.parquet
    if not path.exists():
        result.structural_fail += 1
        result.messages.append(f"  [FAIL] combined file '{spec.parquet}' not found")
        printer(f"\n==== {spec.name} ====\n  [FAIL] {spec.parquet} not found; build/combine it first.")
        return result

    frame = pd.read_parquet(path)
    printer(f"\n{'='*72}\n{spec.name}  ({len(frame):,} rows, {frame.shape[1]} cols)\n{'='*72}")

    printer("Layer A - structural checks:")
    run_structural(spec, frame, expected_dates, result)
    if result.structural_fail == 0:
        printer(f"  all {result.structural_pass} structural checks passed")
    else:
        for msg in result.messages:
            printer(msg)
        printer(f"  {result.structural_pass} passed, {result.structural_fail} failed")

    printer(f"\nLayer B - sampled reconciliation ({n} keys stratified across dates):")
    run_reconciliation(spec, frame, engine, table, n, seed, result, printer)

    status = "PASS" if result.ok else "FAIL"
    printer(f"\n[{status}] {spec.name}: "
            f"{result.structural_pass}/{result.structural_pass + result.structural_fail} structural, "
            f"{result.values_matched}/{result.values_checked} values across "
            f"{result.keys_checked} keys")
    return result