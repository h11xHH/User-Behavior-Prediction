"""validation/validate_features.py

Reconcile every intermediate feature table. Runs Layer A and B.
"""

from __future__ import annotations

import sys
import traceback

from src.config_loader import load_config
from src.data_io import load_secrets, make_engine
from src.validation.reconcile import reconcile_table
from src.validation.specs import ALL_SPECS

SAMPLE_KEYS = 8 # number of stratified across Ds
SEED = 20141218


def main() -> None:
    """Validate all feature tables; summarise and exit non-zero on any failure."""
    config = load_config()
    try:
        table = config.database["table_raw"]
        secrets_path = config.resolve_path(config.database["secrets_file"])
        interim_dir = config.resolve_path(config.paths["interim_dir"])
        expected_dates = len(config.labeling["prediction_dates"])
    except KeyError as error:
        raise RuntimeError(
            f"config/config.yaml is missing required key {error}."
        ) from error

    try:
        engine = make_engine(load_secrets(secrets_path))
    except Exception as error:
        raise RuntimeError(f"Could not set up the MySQL connection ({error}).") from error

    results = []
    for spec in ALL_SPECS:
        results.append(reconcile_table(spec, interim_dir, engine, table,
                                       expected_dates, SAMPLE_KEYS, SEED))

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    all_ok = True
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        all_ok = all_ok and r.ok
        print(f"  [{status}] {r.name:<20} "
              f"structural {r.structural_pass}/{r.structural_pass + r.structural_fail}, "
              f"values {r.values_matched}/{r.values_checked} ({r.keys_checked} keys)")

    if not all_ok:
        print("\nValidation FAILED - see [FAIL]/XX lines above.")
        sys.exit(1)
    print("\nAll intermediate tables reconciled successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # top-level guard: report clearly, never hide the cause
        print(f"validate_features failed: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)