"""Compare two datasets (e.g. current vs staged) and write a markdown diff report.

Usage:
    python -m pipeline.compare data data/staging --out reports/data/diff.md

Reports, per column: rows that differ on dates both datasets share, the
largest difference and where the differences start. Differences on shared
dates are revisions or vintage changes at the source; new dates are new data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ("oil_macro_weekly.csv", "oil_macro_transformed.csv")


def diff_frames(old: pd.DataFrame, new: pd.DataFrame, rtol: float = 1e-9) -> pd.DataFrame:
    common = old.index.intersection(new.index)
    rows = []
    for col in old.columns.intersection(new.columns):
        a, b = old.loc[common, col], new.loc[common, col]
        changed = (a - b).abs() > rtol * np.maximum(1.0, a.abs())
        rows.append({
            "column": col,
            "shared rows": len(common),
            "rows changed": int(changed.sum()),
            "max abs change": float((a - b).abs().max()) if len(common) else 0.0,
            "first changed": str(changed[changed].index[0].date()) if changed.any() else "",
            "last changed": str(changed[changed].index[-1].date()) if changed.any() else "",
        })
    return pd.DataFrame(rows)


def render(old_dir: Path, new_dir: Path) -> str:
    from evaluation.report import md_table

    parts = [f"# Data diff: `{old_dir}` → `{new_dir}`", ""]
    for name in FILES:
        old = pd.read_csv(old_dir / name, index_col=0, parse_dates=True)
        new = pd.read_csv(new_dir / name, index_col=0, parse_dates=True)
        added, removed = new.index.difference(old.index), old.index.difference(new.index)
        parts += [
            f"## {name}",
            "",
            f"- Old: {len(old)} rows, {old.index[0].date()} → {old.index[-1].date()}",
            f"- New: {len(new)} rows, {new.index[0].date()} → {new.index[-1].date()}",
            f"- Dates added: {len(added)}" + (f" ({added[0].date()} → {added[-1].date()})" if len(added) else ""),
            f"- Dates removed: {len(removed)}",
            "",
            md_table(diff_frames(old, new), {"max abs change": 4}),
            "",
        ]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    text = render(args.old, args.new)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
