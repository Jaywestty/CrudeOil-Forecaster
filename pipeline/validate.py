"""Checks a dataset must pass before the pipeline writes it.

Errors block the write: the data is structurally wrong or physically
implausible. Warnings are written into the manifest for a human to review:
the data is plausible but unusual (e.g. a very large weekly move), or stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.build import TRANSFORMED_COLUMNS, WEEKLY_COLUMNS, build_transformed

# Plausible bounds for weekly levels. Wide on purpose: they catch unit or
# parsing errors (e.g. inventories in barrels instead of thousands), not
# unusual markets.
BOUNDS = {
    "brent":           (5.0, 300.0),
    "inventories":     (100_000.0, 2_000_000.0),
    "industrial_prod": (40.0, 200.0),
    "dollar_index":    (50.0, 200.0),
    "fed_funds":       (-1.0, 25.0),
    "vix":             (5.0, 100.0),
}
MAX_ABS_WEEKLY_BRENT_LOG_RETURN = 0.30  # larger moves are flagged for review (e.g. March-April 2020)
MAX_STALENESS_DAYS = 21


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def validate(weekly: pd.DataFrame, transformed: pd.DataFrame, as_of: pd.Timestamp | None = None) -> ValidationReport:
    rep = ValidationReport()

    # Structure
    if list(weekly.columns) != WEEKLY_COLUMNS:
        rep.errors.append(f"weekly columns {list(weekly.columns)} != {WEEKLY_COLUMNS}")
        return rep
    if list(transformed.columns) != TRANSFORMED_COLUMNS:
        rep.errors.append(f"transformed columns {list(transformed.columns)} != {TRANSFORMED_COLUMNS}")
        return rep
    if len(weekly) < 260:
        rep.errors.append(f"only {len(weekly)} weekly rows; need at least 5 years")
    if not weekly.index.is_monotonic_increasing or weekly.index.has_duplicates:
        rep.errors.append("weekly index is not strictly increasing")
    gaps = weekly.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(weeks=1)).all():
        rep.errors.append(f"weekly index has {(gaps != pd.Timedelta(weeks=1)).sum()} gaps or irregular steps")
    if weekly.isna().any().any() or transformed.isna().any().any():
        rep.errors.append("missing values present")
    if not np.isfinite(transformed.to_numpy()).all():
        rep.errors.append("non-finite values in transformed data")

    # Plausibility
    for col, (lo, hi) in BOUNDS.items():
        bad = weekly[(weekly[col] < lo) | (weekly[col] > hi)]
        if len(bad):
            rep.errors.append(f"{col}: {len(bad)} values outside [{lo}, {hi}], e.g. "
                              f"{bad.index[0].date()}={bad[col].iloc[0]}")

    # Internal consistency: transformed must be derivable from weekly
    expected = build_transformed(weekly)
    if not expected.index.equals(transformed.index) or not np.allclose(
        expected.to_numpy(), transformed.to_numpy(), rtol=1e-9, atol=1e-12
    ):
        rep.errors.append("transformed data does not match build_transformed(weekly)")

    # Review flags
    big = transformed[transformed["brent_return"].abs() > MAX_ABS_WEEKLY_BRENT_LOG_RETURN]
    for d, v in big["brent_return"].items():
        rep.warnings.append(f"large weekly Brent move {v:+.2f} log on {d.date()}")
    if as_of is not None:
        age = (pd.Timestamp(as_of) - weekly.index[-1]).days
        if age > MAX_STALENESS_DAYS:
            rep.warnings.append(f"data is {age} days old (last week {weekly.index[-1].date()})")
    return rep
