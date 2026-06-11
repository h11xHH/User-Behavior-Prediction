"""data_io.py

Database access layer for the project. It do the following:
1. reads credentials from the secrets file;
2. builds a SQLAlchemy engine with LOCAL INFILE enabled;
3. creates the raw table and bulk-loads the CSV into it;
4. runs read-only SQL queries and returns the result as a pandas DataFrame.

Must enable local_infile.

Run: python -m src.data_io
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
    """Read the MySQL credentials from the secrets file.

    Input
    -----
    secrets_path : str | Path
        Path to config/secrets.yaml.

    Output
    ------
    dict[str, Any]
        The 'mysql' sub-mapping with keys host, port, user, password, database.
    """
    path = Path(secrets_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found at '{path}'. ")

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
    """
    url = (
        f"mysql+pymysql://{creds['user']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['database']}"
    )
    return create_engine(url, connect_args={"local_infile": 1})


# Using real DATETIME for time so SQL date functions work.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    user_id       VARCHAR(20) NOT NULL,
    item_id       VARCHAR(20) NOT NULL,
    behavior_type TINYINT     NOT NULL,
    item_category VARCHAR(20) NOT NULL,
    time          DATETIME    NOT NULL
)
"""


# Raw time is YYYY-MM-DD HH, use STR_TO_DATE to convert to DATETIME.
# Remove trailing \r to avoid parsing errors.
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
    columns make those group-bys fast. They are added after the bulk load. On a 
    re-load the table still exists, so a duplicate-index error is expected and 
    safely ignored.
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
        SQLAlchemy engine built by make_engine.
    csv_path : str | Path
        Absolute path to the raw CSV.
    table_name : str
        From config.database['table_raw'].

    Output
    ------
    int
        Number of rows in the table after loading.
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
    try:
        config = load_config()
        creds = load_secrets(config.resolve_path(config.database["secrets_file"]))
        engine = make_engine(creds)
        csv_path = config.resolve_path(config.paths["raw_csv"])
        table_name = config.database["table_raw"]

        print(f"Loading '{csv_path}' into MySQL table '{table_name}' ...")
        row_count = create_and_load_raw_table(engine, csv_path, table_name)
        print(f"Done. Rows in '{table_name}': {row_count:,}")
    except Exception as error:
        print(f"Load failed: {error}")