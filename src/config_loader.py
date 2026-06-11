"""config_loader.py

Loads and validates the project configuration from `config/config.yaml` and
returns it as a typed object the rest of the pipeline can rely on.

Run standalone to print the loaded config:
    python -m src.config_loader
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Sections that MUST be present in config.yaml for the current stage of the
# project. We deliberately start with only `paths` and append new required
# sections as later phases introduce them (e.g. "data" in Phase 1). This keeps
# the config and its validation in lock-step with the code that actually exists.
REQUIRED_SECTIONS: tuple[str, ...] = ("paths", "data", "database")


@dataclass(frozen=True)
class Config:
    """Immutable container for the project configuration.

    Attributes
    ----------
    raw : dict[str, Any]
        The full parsed YAML, so any key is still reachable as the config grows.
    project_root : Path
        Absolute path to the project root (the parent of the `config/` folder).
        All relative paths in the YAML are resolved against this.
    """

    raw: dict[str, Any]
    project_root: Path

    @property
    def paths(self) -> dict[str, Any]:
        """Return the `paths` section of the config (dict)."""
        return self.raw["paths"]

    @property
    def data(self) -> dict[str, Any]:
        """Return the `data` section of the config (dict). Added in Phase 1."""
        return self.raw["data"]
 
    @property
    def database(self) -> dict[str, Any]:
        """Return the `database` section of the config (dict). Added in Phase 1."""
        return self.raw["database"]
    
    # NOTE: accessors for future sections (labeling, features, model) will be
    # added here when the phase that needs them arrives — not before.

    def resolve_path(self, relative_path: str) -> Path:
        """Turn a config-relative path string into an absolute Path.

        Input
        -----
        relative_path : str
            A path as written in config.yaml, relative to the project root
            (e.g. "data/raw/user_behavior.csv").

        Output
        ------
        Path
            The absolute path, built by joining `project_root` and the input.

        Logic
        -----
        Keeping every path relative in the YAML and resolving it here means the
        repo can be cloned to any machine/folder and still run unchanged.
        """
        return self.project_root / relative_path


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Read and validate config.yaml, returning a Config object.

    Input
    -----
    config_path : str | Path
        Path to the YAML config file, relative to the current working directory
        or absolute. Defaults to "config/config.yaml".

    Output
    ------
    Config
        A frozen Config holding the parsed YAML and the resolved project root.

    Logic / validation
    -------------------
    1. Resolve the config path and confirm the file exists.
    2. Parse the YAML safely (yaml.safe_load avoids executing arbitrary tags).
    3. Confirm the result is a dict and that every REQUIRED_SECTIONS key is
       present, so a malformed config fails here with a clear message instead of
       somewhere deep in a later phase.
    4. Derive project_root as the parent of the `config/` directory.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If the file is not valid YAML, is not a mapping, or is missing a
        required section.
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Run from the project root, or pass the correct path."
        )

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

    # config/ lives directly under the project root, so the root is two levels
    # up from the file itself (…/project_root/config/config.yaml).
    project_root = path.parent.parent

    return Config(raw=parsed, project_root=project_root)


if __name__ == "__main__":
    # Standalone check: load the default config and print a short summary so you
    # can confirm the environment is wired up correctly before any real work.
    try:
        config = load_config()
        print("Config loaded successfully.")
        print(f"  project_root : {config.project_root}")
        print(f"  raw_csv      : {config.resolve_path(config.paths['raw_csv'])}")
        print(f"  table_raw  : {config.resolve_path(config.paths['table_raw'])}")
    except (FileNotFoundError, ValueError) as error:
        print(f"Failed to load config: {error}")