"""Fetch, build, validate and (optionally) write the modelling dataset.

Usage:
    python -m pipeline.refresh --fetch                  # fetch fresh raw snapshots, build to data/staging
    python -m pipeline.refresh                          # rebuild offline from existing snapshots
    python -m pipeline.refresh --end 2025-12-31         # build only from observations up to a date
    python -m pipeline.refresh --recipe v2-release-aware

Recipes (pipeline/build.py): v1-notebook reproduces the original data exactly;
v1.1-complete-weeks (default) also drops a partial final week; v2-release-aware
also shifts monthly series to their publication dates.

Output goes to data/staging/ by default, never straight over the files the
model is trained on. Promoting a staged dataset is a separate, deliberate
step (roadmap step 3b), because new data changes the model and every report.
A manifest (manifest.json) records sources, recipe, row counts, date range,
file hashes and validation results for every build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import build, sources
from pipeline.validate import validate

ROOT = Path(__file__).resolve().parent.parent
RECIPES = build.RECIPES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(end: str | None, recipe_name: str, out: Path, raw_dir: Path, fetch: bool, as_of: pd.Timestamp) -> int:
    recipe = RECIPES[recipe_name]
    if fetch:
        print("Fetching raw series...", flush=True)
        sources.save_snapshots(sources.fetch_all(), raw_dir)
    raw = sources.load_snapshots(raw_dir)
    end = end or str(as_of.date())

    weekly = build.build_weekly(raw, end=end, recipe=recipe)
    transformed = build.build_transformed(weekly)
    report = validate(weekly, transformed, as_of=as_of)
    for w in report.warnings:
        print(f"  WARNING: {w}")
    if not report.ok:
        for e in report.errors:
            print(f"  ERROR: {e}")
        print("Validation failed; nothing written.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    weekly_path, transformed_path = out / "oil_macro_weekly.csv", out / "oil_macro_transformed.csv"
    weekly.to_csv(weekly_path, lineterminator="\n")
    transformed.to_csv(transformed_path, lineterminator="\n")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recipe": recipe.name,
        "release_lag_days": recipe.release_lag_days,
        "end": end,
        "sources": {name: {"source": src, "series_id": sid,
                           "raw_last_date": str(raw[name].index[-1].date()),
                           "raw_sha256": _sha(raw_dir / f"{name}.csv")}
                    for name, (src, sid) in sources.SERIES.items()},
        "weekly": {"rows": len(weekly), "first": str(weekly.index[0].date()), "last": str(weekly.index[-1].date()),
                   "last_brent": float(weekly["brent"].iloc[-1]), "sha256": _sha(weekly_path)},
        "transformed": {"rows": len(transformed), "sha256": _sha(transformed_path)},
        "validation": report.as_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(weekly)} weeks ({weekly.index[0].date()} -> {weekly.index[-1].date()}, "
          f"last Brent ${weekly['brent'].iloc[-1]:.2f}) to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download fresh raw snapshots first")
    ap.add_argument("--end", help="use observations up to this date (default: today)")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default=build.RECIPE_V1_1.name)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "staging")
    ap.add_argument("--raw-dir", type=Path, default=sources.RAW_DIR)
    args = ap.parse_args(argv)
    return run(args.end, args.recipe, args.out, args.raw_dir, args.fetch, pd.Timestamp.now().normalize())


if __name__ == "__main__":
    raise SystemExit(main())
