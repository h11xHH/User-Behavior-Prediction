"""validate_sample.py

Checks a saved sample__{D}.parquet:
  1. integrity check + varify the (user, item) pair is unique
  2. Print the history of a few pairs, manually verify the labels
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine, read_query
from src.labeling import day_bounds


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line for one check and return the boolean."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"   ({detail})" if detail else ""))
    return bool(condition)


def _is_string_col(series: pd.Series) -> bool:
    """True if a column is stored as text (object/str/string), not numeric."""
    return str(series.dtype) in ("object", "str", "string")


def _spot_check(engine: Engine, table: str, rows: pd.DataFrame,
                window_start: str, d_start: str, d_end: str, kind: str) -> None:
    """Pull each pair's full history and show whether it qualifies + bought on D."""
    for _, row in rows.iterrows():
        user_id, item_id = row["user_id"], row["item_id"]
        sql = (f"SELECT behavior_type, time FROM {table} "
               f"WHERE user_id='{user_id}' AND item_id='{item_id}' ORDER BY time")
        history = read_query(engine, sql)
        in_window = ((history["time"] >= pd.Timestamp(window_start))
                     & (history["time"] < pd.Timestamp(d_start))).any()
        bought_on_d = ((history["time"] >= pd.Timestamp(d_start))
                       & (history["time"] < pd.Timestamp(d_end))
                       & (history["behavior_type"] == 4)).any()
        expected = "1" if kind == "POSITIVE" else "0"
        print(f"\n  {kind} pair (user={user_id}, item={item_id}) -- label should be {expected}")
        print(f"    interacted in candidate window : {bool(in_window)}  (must be True)")
        print(f"    bought on day D                : {bool(bought_on_d)}  "
              f"(must be {'True' if kind == 'POSITIVE' else 'False'})")
        print(history.head(20).to_string(index=False))


def validate(engine: Engine, table: str, interim_dir: Path,
             prediction_date: str, lookback: int) -> bool:
    """Run all checks on the sample for one prediction date. Returns overall pass."""
    path = interim_dir / f"sample__{prediction_date}.parquet"
    frame = pd.read_parquet(path)
    window_start, d_start, d_end = day_bounds(prediction_date, lookback)

    print(f"\n=== Validating {path.name}  (D={prediction_date}, lookback={lookback}) ===")
    print(f"shape: {frame.shape[0]:,} rows x {frame.shape[1]} cols\n")

    ok = True
    # ---------- internal invariants ----------
    ok &= _check("columns are exactly [user_id, item_id, cutoff_date, label]",
                 list(frame.columns) == ["user_id", "item_id", "cutoff_date", "label"],
                 str(list(frame.columns)))
    n_null = int(frame.isnull().sum().sum())
    ok &= _check("no missing values", n_null == 0, f"{n_null} nulls")
    label_vals = set(int(v) for v in frame["label"].unique())
    ok &= _check("label values are only {0,1}", label_vals.issubset({0, 1}), str(sorted(label_vals)))
    ok &= _check("cutoff_date is constant and equals D",
                 bool((frame["cutoff_date"] == prediction_date).all()))
    n_dup = int(frame.duplicated(["user_id", "item_id"]).sum())
    ok &= _check("(user_id, item_id) pairs are UNIQUE (candidate set is distinct)",
                 n_dup == 0, f"{n_dup} duplicate pairs")
    ok &= _check("ids stored as strings, not numbers",
                 _is_string_col(frame["user_id"]) and _is_string_col(frame["item_id"]),
                 f"user_id={frame['user_id'].dtype}, item_id={frame['item_id'].dtype}")

    n_total = len(frame)
    n_pos = int(frame["label"].sum())
    print(f"\ncandidates={n_total:,}  positives={n_pos:,}  "
          f"pos_rate={100.0 * n_pos / n_total:.3f}%")

    # ---------- agreement with MySQL ----------
    cand_sql = (f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT user_id, item_id "
                f"FROM {table} WHERE time>='{window_start}' AND time<'{d_start}') t")
    cand_n = int(read_query(engine, cand_sql).iloc[0]["n"])
    ok &= _check("candidate count matches an independent MySQL recomputation",
                 cand_n == n_total, f"MySQL={cand_n:,} vs parquet={n_total:,}")

    pos_sql = (f"SELECT COUNT(*) AS n FROM "
               f"(SELECT DISTINCT user_id, item_id FROM {table} "
               f"WHERE time>='{window_start}' AND time<'{d_start}') c "
               f"JOIN (SELECT DISTINCT user_id, item_id FROM {table} "
               f"WHERE time>='{d_start}' AND time<'{d_end}' AND behavior_type=4) b "
               f"ON c.user_id=b.user_id AND c.item_id=b.item_id")
    pos_n = int(read_query(engine, pos_sql).iloc[0]["n"])
    ok &= _check("positive count matches an independent MySQL recomputation",
                 pos_n == n_pos, f"MySQL={pos_n:,} vs parquet={n_pos:,}")

    # informational: recall against all buyers on D
    buy_sql = (f"SELECT COUNT(DISTINCT user_id, item_id) AS n FROM {table} "
               f"WHERE time>='{d_start}' AND time<'{d_end}' AND behavior_type=4")
    buy_n = int(read_query(engine, buy_sql).iloc[0]["n"])
    print(f"\n(info) distinct buy-pairs on D = {buy_n:,}  ->  recall = "
          f"{100.0 * n_pos / buy_n:.1f}%")

    # ---------- spot checks ----------
    print("\n--- spot checks: a couple of positives and negatives, with history ---")
    _spot_check(engine, table, frame[frame.label == 1].head(2),
                window_start, d_start, d_end, "POSITIVE")
    _spot_check(engine, table, frame[frame.label == 0].head(2),
                window_start, d_start, d_end, "NEGATIVE")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- review above"))
    return ok


def main() -> None:
    """Validate the sample for each configured prediction date."""
    config = load_config()
    creds = load_secrets(config.resolve_path(config.database["secrets_file"]))
    engine = make_engine(creds)
    table = config.database["table_raw"]
    interim_dir = config.resolve_path(config.paths["interim_dir"])
    lookback = int(config.labeling["candidate_lookback_days"])
    for prediction_date in config.labeling["prediction_dates"]:
        validate(engine, table, interim_dir, prediction_date, lookback)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Validation failed: {error}")