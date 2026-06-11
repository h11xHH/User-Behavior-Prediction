"""exploration.py

Data exploration & quality assurance. Run several read-only queries,
the output are: summary in terminal, four charts and a report framework
stored in reports/.

Run: python -m src.exploration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # render straight to PNG
import matplotlib.pyplot as plt
import pandas as pd

from src.config_loader import Config, load_config
from src.data_io import count_rows, load_secrets, make_engine, read_query
from sqlalchemy.engine import Engine


BEHAVIOR_LABELS: dict[int, str] = {1: "browse", 2: "favorite", 3: "cart", 4: "buy"}


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def count_csv_data_lines(csv_path: str | Path) -> int:
    """Count data rows in the raw CSV, later compare to that in MySQL table.

    Input
    -----
    csv_path : str | Path
        Path to the raw CSV file.

    Output
    ------
    int
        Number of data rows in the file (excluding header).
    """
    with open(csv_path, "r", encoding="utf-8") as handle:
        total_lines = sum(1 for _ in handle)
    return max(total_lines - 1, 0)


def check_missing_values(engine: Engine, table: str) -> pd.DataFrame:
    """Count NULL or empty values in each column.

    Output
    ------
    pandas.DataFrame
        One row with a count per column.
    """
    sql = f"""
        SELECT
            SUM(user_id IS NULL OR user_id = '')             AS user_id_missing,
            SUM(item_id IS NULL OR item_id = '')             AS item_id_missing,
            SUM(behavior_type IS NULL)                       AS behavior_type_missing,
            SUM(item_category IS NULL OR item_category = '') AS item_category_missing,
            SUM(time IS NULL)                                AS time_missing
        FROM {table}
    """
    return read_query(engine, sql)


def check_behavior_domain(engine: Engine, table: str) -> pd.DataFrame:
    """Return the count of rows per behavior_type value.

    Output
    ------
    pandas.DataFrame
        Columns: behavior_type, cnt. Confirms only {1,2,3,4} occur and gives the
        raw distribution.
    """
    sql = f"""
        SELECT behavior_type, COUNT(*) AS cnt
        FROM {table}
        GROUP BY behavior_type
        ORDER BY behavior_type
    """
    return read_query(engine, sql)


def check_time_range(engine: Engine, table: str) -> dict[str, Any]:
    """Return min/max timestamp, hour bounds, and any NULL-time count.

    Output
    ------
    dict
        Keys: min_time, max_time, min_hour, max_hour, null_time.
    """
    sql = f"""
        SELECT
            MIN(time)       AS min_time,
            MAX(time)       AS max_time,
            MIN(HOUR(time)) AS min_hour,
            MAX(HOUR(time)) AS max_hour,
            SUM(time IS NULL) AS null_time
        FROM {table}
    """
    row = read_query(engine, sql).iloc[0]
    return {
        "min_time": str(row["min_time"]),
        "max_time": str(row["max_time"]),
        "min_hour": int(row["min_hour"]),
        "max_hour": int(row["max_hour"]),
        "null_time": int(row["null_time"]),
    }


def check_duplicate_rows(engine: Engine, table: str) -> dict[str, int]:
    """Count exact-duplicate rows (all five columns identical).

    Output
    ------
    dict
        Keys: total_rows, distinct_rows, extra_rows (= total - distinct).

    Logic
    ----------------
    KEEP duplicates and only quantify them. extra_rows tells us how many 
    such repeats exist.
    """
    sql = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT user_id, item_id, behavior_type, item_category, time)
                AS distinct_rows
        FROM {table}
    """
    row = read_query(engine, sql).iloc[0]
    total = int(row["total_rows"])
    distinct = int(row["distinct_rows"])
    return {"total_rows": total, "distinct_rows": distinct, "extra_rows": total - distinct}


def check_item_category_consistency(engine: Engine, table: str) -> int:
    """Count items that map to more than one item_category.

    Output
    ------
    int
        Number of item_ids associated with multiple categories.
    """
    sql = f"""
        SELECT COUNT(*) AS inconsistent_items FROM (
            SELECT item_id
            FROM {table}
            GROUP BY item_id
            HAVING COUNT(DISTINCT item_category) > 1
        ) AS multi
    """
    return int(read_query(engine, sql).iloc[0]["inconsistent_items"])


# ---------------------------------------------------------------------------
# Key statistics
# ---------------------------------------------------------------------------

def get_scale(engine: Engine, table: str) -> dict[str, int]:
    """Return distinct counts of users, items, and categories.

    Output
    ------
    dict
        Keys: n_users, n_items, n_categories.
    """
    sql = f"""
        SELECT
            COUNT(DISTINCT user_id)       AS n_users,
            COUNT(DISTINCT item_id)       AS n_items,
            COUNT(DISTINCT item_category) AS n_categories
        FROM {table}
    """
    row = read_query(engine, sql).iloc[0]
    return {
        "n_users": int(row["n_users"]),
        "n_items": int(row["n_items"]),
        "n_categories": int(row["n_categories"]),
    }


def get_daily_breakdown(engine: Engine, table: str) -> pd.DataFrame:
    """Return per-day counts, split by behavior type, in one pass.

    Output
    ------
    pandas.DataFrame
        Columns: d (date), total, browse, favorite, cart, buy.
        Feeds the daily-volume and buys-per-day charts and locates the sale peak.
    """
    sql = f"""
        SELECT
            DATE(time)              AS d,
            COUNT(*)                AS total,
            SUM(behavior_type = 1)  AS browse,
            SUM(behavior_type = 2)  AS favorite,
            SUM(behavior_type = 3)  AS cart,
            SUM(behavior_type = 4)  AS buy
        FROM {table}
        GROUP BY DATE(time)
        ORDER BY d
    """
    frame = read_query(engine, sql)
    frame["d"] = pd.to_datetime(frame["d"])
    for column in ("total", "browse", "favorite", "cart", "buy"):
        frame[column] = frame[column].astype("int64")
    return frame


def get_target_prevalence(engine: Engine, table: str) -> dict[str, int]:
    """Return purchase counts that preview the class-imbalance problem.

    Output
    ------
    dict
        Keys: n_buy_events (rows with behavior_type=4),
              n_distinct_buy_pairs (distinct (user,item) that ever bought).
    """
    sql = f"""
        SELECT
            COUNT(*)                          AS n_buy_events,
            COUNT(DISTINCT user_id, item_id)  AS n_distinct_buy_pairs
        FROM {table}
        WHERE behavior_type = 4
    """
    row = read_query(engine, sql).iloc[0]
    return {
        "n_buy_events": int(row["n_buy_events"]),
        "n_distinct_buy_pairs": int(row["n_distinct_buy_pairs"]),
    }


def get_user_activity(engine: Engine, table: str) -> pd.DataFrame:
    """Return number of actions per user.

    Output
    ------
    pandas.DataFrame
        Columns: user_id, action_count. One row per user, used for 
        the activity histogram and percentile summary.
    """
    sql = f"""
        SELECT user_id, COUNT(*) AS action_count
        FROM {table}
        GROUP BY user_id
    """
    return read_query(engine, sql)


def get_buyer_and_cold_item_counts(engine: Engine, table: str, n_items: int) -> dict[str, int]:
    """Return how many users ever buy and how many items are never bought.

    Input
    -----
    n_items : int
        Total distinct items (from get_scale), used to derive cold-item count.

    Output
    ------
    dict
        Keys: n_buyers, n_items_bought, n_items_never_bought.
    """
    sql = f"""
        SELECT
            COUNT(DISTINCT user_id) AS n_buyers,
            COUNT(DISTINCT item_id) AS n_items_bought
        FROM {table}
        WHERE behavior_type = 4
    """
    row = read_query(engine, sql).iloc[0]
    n_buyers = int(row["n_buyers"])
    n_items_bought = int(row["n_items_bought"])
    return {
        "n_buyers": n_buyers,
        "n_items_bought": n_items_bought,
        "n_items_never_bought": n_items - n_items_bought,
    }


def get_sparsity(engine: Engine, table: str, n_users: int) -> float:
    """Return the average number of distinct items a user interacts with.

    Input
    -----
    n_users : int
        Total distinct users (from get_scale).

    Output
    ------
    float
        distinct (user,item) pairs divided by number of users. A small value
        relative to the catalogue confirms why a candidate set is needed later.
    """
    sql = f"SELECT COUNT(DISTINCT user_id, item_id) AS n_pairs FROM {table}"
    n_pairs = int(read_query(engine, sql).iloc[0]["n_pairs"])
    return n_pairs / n_users if n_users else 0.0


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    """Save a figure to PNG and close it (shared by all chart functions)."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def chart_daily_volume(daily: pd.DataFrame, out_path: Path) -> Path:
    """Line chart of total behaviours per day."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily["d"], daily["total"], marker="o", linewidth=1.5)
    ax.set_title("Total behaviours per day")
    ax.set_xlabel("date")
    ax.set_ylabel("number of behaviours")
    fig.autofmt_xdate()
    return _save_fig(fig, out_path)


def chart_behavior_split(behavior: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of counts per behaviour type."""
    labels = [BEHAVIOR_LABELS.get(int(b), str(b)) for b in behavior["behavior_type"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, behavior["cnt"])
    ax.set_title("Behaviour distribution")
    ax.set_xlabel("behaviour")
    ax.set_ylabel("count")
    return _save_fig(fig, out_path)


def chart_buys_per_day(daily: pd.DataFrame, out_path: Path) -> Path:
    """Line chart of purchases per day."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily["d"], daily["buy"], marker="o", color="tab:red", linewidth=1.5)
    ax.set_title("Purchases per day")
    ax.set_xlabel("date")
    ax.set_ylabel("number of purchases")
    fig.autofmt_xdate()
    return _save_fig(fig, out_path)


def chart_user_activity_hist(user_activity: pd.DataFrame, out_path: Path) -> Path:
    """Histogram of actions per user (log y-axis)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(user_activity["action_count"], bins=50)
    ax.set_yscale("log")
    ax.set_title("Actions per user (log scale)")
    ax.set_xlabel("actions per user")
    ax.set_ylabel("number of users (log)")
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(results: dict[str, Any], chart_files: dict[str, str], out_path: Path) -> Path:
    """Write the markdown report from collected results and chart files, with placeholders.

    Input
    -----
    results : dict
        All computed numbers (scale, integrity, prevalence, etc.).
    chart_files : dict
        Mapping of chart key -> filename.
    out_path : Path
        Where to write the markdown report.

    Output
    ------
    Path
        The path written. The report contains every number and chart, and placeholders.
    """
    scale = results["scale"]
    prevalence = results["prevalence"]
    time_range = results["time_range"]
    duplicates = results["duplicates"]
    buyers = results["buyers"]
    activity = results["activity_summary"]

    behavior_lines = "\n".join(
        f"| {BEHAVIOR_LABELS.get(int(bt), bt)} | {cnt:,} | {pct:.2f}% |"
        for bt, cnt, pct in results["behavior_rows"]
    )

    pair_pos_rate = (
        100.0 * prevalence["n_distinct_buy_pairs"] / results["distinct_pairs"]
        if results["distinct_pairs"]
        else 0.0
    )

    text = f"""# 01 — Exploration Report

## A. Integrity checks

| Check | Result |
|---|---|
| Rows in MySQL table | {results['row_count']:,} |
| Data rows in CSV (excl. header) | {results['csv_lines']:,} |
| Row count matches file | {"yes" if results['row_count'] == results['csv_lines'] else "NO — investigate"} |
| Missing/empty values (any column) | {results['total_missing']:,} |
| behavior_type values present | {results['behavior_values']} |
| Time range | {time_range['min_time']} → {time_range['max_time']} |
| Hour range | {time_range['min_hour']:02d}–{time_range['max_hour']:02d} |
| NULL timestamps | {time_range['null_time']:,} |
| Exact-duplicate rows (kept, not removed) | {duplicates['extra_rows']:,} |
| Items with >1 category | {results['inconsistent_items']:,} |

> tmp

## B. Key statistics

| Metric | Value |
|---|---|
| Distinct users | {scale['n_users']:,} |
| Distinct items | {scale['n_items']:,} |
| Distinct categories | {scale['n_categories']:,} |
| Total purchase events | {prevalence['n_buy_events']:,} |
| Distinct (user,item) buy pairs | {prevalence['n_distinct_buy_pairs']:,} |
| Distinct (user,item) pairs overall | {results['distinct_pairs']:,} |
| Of all pairs, fraction that ever buy | {pair_pos_rate:.3f}% |
| Users with ≥1 purchase | {buyers['n_buyers']:,} of {scale['n_users']:,} |
| Items never purchased (cold) | {buyers['n_items_never_bought']:,} of {scale['n_items']:,} |
| Avg distinct items per user | {results['sparsity']:.1f} |

### Behaviour distribution (the funnel)

| behaviour | count | share |
|---|---|---|
{behavior_lines}

### Actions per user

| percentile | actions |
|---|---|
| 50th (median) | {activity['p50']:.0f} |
| 90th | {activity['p90']:.0f} |
| 99th | {activity['p99']:.0f} |
| max | {activity['max']:.0f} |

> tmp

## Charts

![Daily volume]({chart_files['daily_volume']})

![Behaviour split]({chart_files['behavior_split']})

![Purchases per day]({chart_files['buys_per_day']})

![Actions per user]({chart_files['user_activity_hist']})

---
Time Range: *(`data.date_start: {time_range['min_time'][:10]}`, `data.date_end: {time_range['max_time'][:10]}`).*
"""
    out_path.write_text(text, encoding="utf-8")
    return out_path

# ---------------------------------------------------------------------------

def run_exploration(config: Config) -> None:
    """Run every check, print a summary, save charts, and write the report.

    Input
    -----
    config : Config
        Loaded project configuration.

    Output
    ------
    None (side effects: console output, chart PNGs, and the markdown report).
    """
    creds = load_secrets(config.resolve_path(config.database["secrets_file"]))
    engine = make_engine(creds)
    table = config.database["table_raw"]
    reports_dir = config.resolve_path(config.paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # run all queries
    row_count = count_rows(engine, table)
    csv_lines = count_csv_data_lines(config.resolve_path(config.paths["raw_csv"]))
    missing = check_missing_values(engine, table)
    behavior = check_behavior_domain(engine, table)
    time_range = check_time_range(engine, table)
    duplicates = check_duplicate_rows(engine, table)
    inconsistent_items = check_item_category_consistency(engine, table)

    scale = get_scale(engine, table)
    daily = get_daily_breakdown(engine, table)
    prevalence = get_target_prevalence(engine, table)
    user_activity = get_user_activity(engine, table)
    buyers = get_buyer_and_cold_item_counts(engine, table, scale["n_items"])
    sparsity = get_sparsity(engine, table, scale["n_users"])

    distinct_pairs = round(sparsity * scale["n_users"])  # reuse the pair count

    # derive a few summaries for the report
    total_behaviours = int(behavior["cnt"].sum())
    behavior_rows = [
        (int(bt), int(cnt), 100.0 * int(cnt) / total_behaviours)
        for bt, cnt in zip(behavior["behavior_type"], behavior["cnt"])
    ]
    activity_summary = {
        "p50": float(user_activity["action_count"].quantile(0.50)),
        "p90": float(user_activity["action_count"].quantile(0.90)),
        "p99": float(user_activity["action_count"].quantile(0.99)),
        "max": float(user_activity["action_count"].max()),
    }

    # charts
    chart_files = {
        "daily_volume": "chart_daily_volume.png",
        "behavior_split": "chart_behavior_split.png",
        "buys_per_day": "chart_buys_per_day.png",
        "user_activity_hist": "chart_user_activity_hist.png",
    }
    chart_daily_volume(daily, reports_dir / chart_files["daily_volume"])
    chart_behavior_split(behavior, reports_dir / chart_files["behavior_split"])
    chart_buys_per_day(daily, reports_dir / chart_files["buys_per_day"])
    chart_user_activity_hist(user_activity, reports_dir / chart_files["user_activity_hist"])

    # print a concise terminal summary
    peak = daily.loc[daily["total"].idxmax()]
    print("\n===== Data exploration summary =====")
    print(f"rows (MySQL) : {row_count:,}   |   csv data rows: {csv_lines:,}"
          f"   |   match: {row_count == csv_lines}")
    print(f"time range   : {time_range['min_time']} -> {time_range['max_time']}"
          f"   (hours {time_range['min_hour']:02d}-{time_range['max_hour']:02d})")
    print(f"missing      : {int(missing.iloc[0].sum()):,} total"
          f"   |   dup rows: {duplicates['extra_rows']:,}"
          f"   |   multi-category items: {inconsistent_items:,}")
    print(f"scale        : {scale['n_users']:,} users, {scale['n_items']:,} items,"
          f" {scale['n_categories']:,} categories")
    print(f"behaviours   : " + ", ".join(
        f"{BEHAVIOR_LABELS.get(bt, bt)}={cnt:,}" for bt, cnt, _ in behavior_rows))
    print(f"buys         : {prevalence['n_buy_events']:,} events,"
          f" {prevalence['n_distinct_buy_pairs']:,} distinct (user,item) pairs")
    print(f"peak day     : {peak['d'].date()} with {int(peak['total']):,} behaviours")
    print(f"buyers       : {buyers['n_buyers']:,}/{scale['n_users']:,} users"
          f"   |   cold items: {buyers['n_items_never_bought']:,}/{scale['n_items']:,}")
    print(f"sparsity     : {sparsity:.1f} distinct items per user (avg)")

    # create the report frame
    results = {
        "row_count": row_count,
        "csv_lines": csv_lines,
        "total_missing": int(missing.iloc[0].sum()),
        "behavior_values": sorted(int(b) for b in behavior["behavior_type"]),
        "time_range": time_range,
        "duplicates": duplicates,
        "inconsistent_items": inconsistent_items,
        "scale": scale,
        "prevalence": prevalence,
        "distinct_pairs": distinct_pairs,
        "buyers": buyers,
        "sparsity": sparsity,
        "behavior_rows": behavior_rows,
        "activity_summary": activity_summary,
    }
    report_path = build_report(results, chart_files, reports_dir / "01_exploration_report.md")
    print(f"\nDraft report : {report_path}")
    print(f"Charts saved : {reports_dir}\n")


if __name__ == "__main__":
    try:
        run_exploration(load_config())
    except Exception as error:
        print(f"Exploration failed: {error}")