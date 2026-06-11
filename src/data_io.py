"""data_io.py

Database access layer for the project. Responsibilities:
  - read MySQL credentials from the (git-ignored) secrets file;
  - build a SQLAlchemy engine configured for LOCAL INFILE bulk loading;
  - create the raw table and bulk-load the raw CSV into it;
  - run read-only SQL queries and return the result as a pandas DataFrame.

Note: LOAD DATA LOCAL INFILE needs `local_infile` enabled on BOTH sides — the
client (we pass it in make_engine) and the server (you run `SET GLOBAL
local_infile = 1;` once as the MySQL admin).

Run standalone to (re)load the CSV into MySQL and print the row count:
    python -m src.data_io
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config_loader import load_config


def load_secrets(secrets_path: str | Path) -> dict[str, Any]:
    """Read the MySQL credentials from the git-ignored secrets file.

    Input
    -----
    secrets_path : str | Path
        Path to config/secrets.yaml (relative to the project root or absolute).

    Output
    ------
    dict[str, Any]
        The 'mysql' sub-mapping with keys host, port, user, password, database.

    Raises
    ------
    FileNotFoundError
        If the secrets file is missing (with a hint to copy the example).
    ValueError
        If the file lacks a 'mysql' section or a required key.
    """
    path = Path(secrets_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Secrets file not found at '{path}'. Copy "
            f"config/secrets.example.yaml to config/secrets.yaml and fill it in."
        )

    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)

    if not isinstance(parsed, dict) or "mysql" not in parsed:
        raise ValueError("secrets.yaml must contain a top-level 'mysql:' section.")

    creds = parsed["mysql"]
    required_keys = ("host", "port", "user", "password", "database")
    missing = [key for key in required_keys if key not in creds]
    if missing:
        raise ValueError(f"secrets.yaml 'mysql' section is missing key(s): {missing}")
    return creds


def make_engine(creds: dict[str, Any]) -> Engine:
    """Build a SQLAlchemy engine for MySQL with LOCAL INFILE enabled.

    Input
    -----
    creds : dict[str, Any]
        MySQL connection settings (host, port, user, password, database).

    Output
    ------
    sqlalchemy.engine.Engine
        Engine usable by pandas (pd.read_sql) and for raw SQL execution.

    Logic
    -----
    `connect_args={"local_infile": 1}` permits LOAD DATA LOCAL INFILE on the
    client side. The server must independently have local_infile turned on.
    """
    url = (
        f"mysql+pymysql://{creds['user']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['database']}"
    )
    return create_engine(url, connect_args={"local_infile": 1})


# DDL for the raw table. IDs are VARCHAR to honour the project rule "IDs are
# strings, never do arithmetic on them"; behavior_type is a small int (1-4);
# time is a real DATETIME so SQL date functions work in the exploration phase.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    user_id       VARCHAR(20) NOT NULL,
    item_id       VARCHAR(20) NOT NULL,
    behavior_type TINYINT     NOT NULL,
    item_category VARCHAR(20) NOT NULL,
    time          DATETIME    NOT NULL
)
"""

# Bulk load. The raw 'time' values are hour-only ("YYYY-MM-DD HH"), so we read
# the raw field into a variable and convert it with STR_TO_DATE (minutes and
# seconds default to 00). TRIM TRAILING '\r' makes the load robust to
# Windows-style CRLF line endings, where the last field would otherwise keep a
# trailing carriage return and fail to parse.
_LOAD_SQL = r"""
LOAD DATA LOCAL INFILE '{csv_path}'
INTO TABLE {table}
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(user_id, item_id, behavior_type, item_category, @time_raw)
SET time = STR_TO_DATE(TRIM(TRAILING '\r' FROM @time_raw), '%Y-%m-%d %H')
"""


def _ensure_indexes(engine: Engine, table_name: str) -> None:
    """Create indexes used by the exploration queries; ignore if they exist.

    Input
    -----
    engine : Engine
        The SQLAlchemy engine.
    table_name : str
        Name of the raw table.

    Output
    ------
    None

    Logic / optimization
    --------------------
    The exploration phase groups by user, item, and time a lot. Indexes on those
    columns make those group-bys fast. We add them AFTER the bulk load (indexing
    during load would slow the load). On a re-load the table still exists, so a
    duplicate-index error is expected and safely ignored.
    """
    statements = [
        f"CREATE INDEX idx_{table_name}_user ON {table_name} (user_id)",
        f"CREATE INDEX idx_{table_name}_item ON {table_name} (item_id)",
        f"CREATE INDEX idx_{table_name}_time ON {table_name} (time)",
    ]
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        for statement in statements:
            try:
                cursor.execute(statement)
            except Exception:
                # Index already exists from a previous load — safe to skip.
                pass
        raw_conn.commit()
        cursor.close()
    finally:
        raw_conn.close()


def create_and_load_raw_table(engine: Engine, csv_path: str | Path, table_name: str) -> int:
    """Create the raw table (if needed), bulk-load the CSV, and return row count.

    Input
    -----
    engine : Engine
        SQLAlchemy engine built by make_engine (local_infile enabled).
    csv_path : str | Path
        Absolute path to the raw CSV.
    table_name : str
        Target table name (from config.database['table_raw']).

    Output
    ------
    int
        Number of rows in the table after loading.

    Logic
    -----
    1. CREATE TABLE IF NOT EXISTS with the schema above.
    2. TRUNCATE first so re-running this gives a clean, idempotent reload rather
       than appending duplicates.
    3. LOAD DATA LOCAL INFILE the whole file in one fast bulk operation.
    4. Add helper indexes, then count the rows.

    We execute via the raw DBAPI cursor (not SQLAlchemy text()) because the load
    statement contains '%' characters in the date format string, which the
    driver's parameter handling would otherwise misinterpret.
    """
    csv_literal = Path(csv_path).resolve().as_posix().replace("'", "''")
    create_sql = _CREATE_TABLE_SQL.format(table=table_name)
    truncate_sql = f"TRUNCATE TABLE {table_name}"
    load_sql = _LOAD_SQL.format(table=table_name, csv_path=csv_literal)

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute(create_sql)
        cursor.execute(truncate_sql)
        cursor.execute(load_sql)
        raw_conn.commit()
        cursor.close()
    finally:
        raw_conn.close()

    _ensure_indexes(engine, table_name)
    return count_rows(engine, table_name)


def count_rows(engine: Engine, table_name: str) -> int:
    """Return the number of rows in a table.

    Input
    -----
    engine : Engine
        SQLAlchemy engine.
    table_name : str
        Table to count.

    Output
    ------
    int
        Row count.
    """
    result = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name}", engine)
    return int(result.iloc[0]["n"])


def read_query(engine: Engine, sql: str) -> pd.DataFrame:
    """Run a read-only SQL query and return the result as a DataFrame.

    Input
    -----
    engine : Engine
        SQLAlchemy engine.
    sql : str
        A SELECT statement. (Avoid '%' in queries, or escape it as '%%'.)

    Output
    ------
    pandas.DataFrame
        The query result.
    """
    return pd.read_sql(sql, engine)


if __name__ == "__main__":
    # One-shot loader: read config + secrets, connect, bulk-load the CSV, report.
    try:
        config = load_config()
        creds = load_secrets(config.resolve_path(config.database["secrets_file"]))
        engine = make_engine(creds)
        csv_path = config.resolve_path(config.paths["raw_csv"])
        table_name = config.database["table_raw"]

        print(f"Loading '{csv_path}' into MySQL table '{table_name}' ...")
        row_count = create_and_load_raw_table(engine, csv_path, table_name)
        print(f"Done. Rows in '{table_name}': {row_count:,}")
    except Exception as error:  # noqa: BLE001  (CLI entry: report any failure clearly)
        print(f"Load failed: {error}")