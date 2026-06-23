"""config_loader.py

Loads and validates the project configuration from `config/config.yaml` and
returns as objects for future use.

Run: python -m src.config_loader
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Define sections that must be present in config.yaml
REQUIRED_SECTIONS: tuple[str, ...] = ("paths", "data", "database", "labeling")


@dataclass(frozen=True)
class Config:
    """Immutable dataclass for the project configuration.

    Attributes
    ----------
    raw : dict[str, Any]
        The full YAML.
    project_root : Path
        Absolute path to the project root.
    """

    raw: dict[str, Any]
    project_root: Path

    @property
    def paths(self) -> dict[str, Any]:
        """Return the `paths` section of the config (dict)."""
        return self.raw["paths"]

    @property
    def data(self) -> dict[str, Any]:
        """Return the `data` section of the config (dict)."""
        return self.raw["data"]
 
    @property
    def database(self) -> dict[str, Any]:
        """Return the `database` section of the config (dict)."""
        return self.raw["database"]

    @property
    def labeling(self) -> dict[str, Any]:
        """Return the `labeling` section of the config (dict)."""
        return self.raw["labeling"]

    def resolve_path(self, relative_path: str) -> Path:
        """Turn a relative path into an absolute path.

        Input
        -----
        relative_path : str
            A path as written in config.yaml.

        Output
        ------
        Path
            The absolute path.

        Logic
        -----
        This is to ensure the project can run on different machines.
        """
        return self.project_root / relative_path


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Read and validate config.yaml, returning a Config object.

    Input
    -----
    config_path : str | Path
        Path to the YAML config file, relative to the current working directory.

    Output
    ------
    Config
        A frozen Config holding the parsed YAML and the resolved project root.
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at '{path}'. ")

    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed: Any = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"Could not parse YAML in '{path}': {error}") from error

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Config root must be a mapping (key: value), got {type(parsed).__name__}."
        )

    missing = [section for section in REQUIRED_SECTIONS if section not in parsed]
    if missing:
        raise ValueError(
            f"Config is missing required section(s): {missing}. "
            f"Expected all of: {list(REQUIRED_SECTIONS)}."
        )

    project_root = path.parent.parent

    return Config(raw=parsed, project_root=project_root)


if __name__ == "__main__":
    try:
        config = load_config()
        print("Config loaded successfully.")
        print(f"  project_root : {config.project_root}")
        print(f"  raw_csv      : {config.resolve_path(config.paths['raw_csv'])}")
        print(f"  table_raw  : {config.database['table_raw']}")
        print(f" prediction_dates : {config.labeling['prediction_dates']}")
        print(f" candidate_lookback_days : {config.labeling['candidate_lookback_days']}")
    except (FileNotFoundError, ValueError) as error:
        print(f"Failed to load config: {error}")