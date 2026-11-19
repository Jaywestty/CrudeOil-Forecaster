"""Weekly refresh: fetch -> build -> validate -> diff -> gate -> promote.

Usage:
    python -m pipeline.weekly            # what the scheduled GitHub Action runs

Steps
  1. fetch fresh raw snapshots and build the staged dataset (production recipe);
     stop with an error if validation fails
  2. write the diff report against the live data; stop if nothing changed
  3. make sure a current ("champion") model exists for monitoring: in CI the
     repository has no model file, so it is trained from the live data
  4. run the promotion gate with --apply (no overrides: a human must review
     anything that needs one)

Exit code 0 = promoted or nothing new; non-zero = a step or gate failed.
When running inside GitHub Actions, outputs are written to $GITHUB_OUTPUT:
  promoted=true|false, data_last=<last week>, reason=<short status>
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path


import modeling
from pipeline import compare, promote, refresh

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAGING = DATA / "staging"
MODEL = ROOT / "models" / "sarimax_model.pkl"
REPORTS = ROOT / "reports" / "data"


def set_output(**values) -> None:
    """Expose results to later workflow steps (no-op outside GitHub Actions)."""
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")


def ensure_champion(model_path: Path = MODEL, data_dir: Path = DATA) -> None:
    if model_path.exists():
        return
    print("No current model file: training the champion from the live data for monitoring...", flush=True)
    weekly, transformed = promote.load_dataset_dir(data_dir)
    fitted = modeling.fit(weekly, transformed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(fitted, f)


def main(argv=None) -> int:
    rc = refresh.main(["--fetch", "--out", str(STAGING)])
    if rc != 0:
        set_output(promoted="false", reason="validation-failed")
        return rc

    compare.main([str(DATA), str(STAGING), "--out", str(REPORTS / "staged_vs_current.md")])
    last = json.loads((STAGING / promote.MANIFEST).read_text(encoding="utf-8"))["weekly"]["last"]
    if not promote.has_new_data(DATA, STAGING):
        print(f"No new data (live data already runs through {last}).")
        set_output(promoted="false", data_last=last, reason="no-new-data")
        return 0

    ensure_champion()
    rc = promote.main(["--apply"])
    promoted = rc == 0 and not promote.has_new_data(DATA, STAGING)  # live data now equals staging
    set_output(promoted=str(promoted).lower(), data_last=last,
               reason="promoted" if promoted else "gate-failed")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
