"""Score prediction intervals and apply the step-1c selection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import hac_mean_se, interval_score

Z = 1.959963984540054


def summarize_intervals(intervals: pd.DataFrame, group_extra: list[str] | None = None) -> pd.DataFrame:
    """Coverage (with HAC 95% CI), mean width and mean interval score per group."""
    keys = ["model", "method", "horizon", "level"] + (group_extra or [])
    rows = []
    for key, g in intervals.groupby(keys, sort=False):
        g = g.sort_values("origin")
        hit = ((g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])).to_numpy(float)
        h, level = int(g["horizon"].iloc[0]), float(g["level"].iloc[0])
        cov = hit.mean()
        se = hac_mean_se(hit, lags=max(h - 1, 0) + 1)  # +1 also absorbs clustering of misses in time
        rows.append(
            {
                **dict(zip(keys, key)),
                "n": len(g),
                "coverage": cov,
                "cov_ci_lo": max(0.0, cov - Z * se),
                "cov_ci_hi": min(1.0, cov + Z * se),
                "calibrated": bool(cov - Z * se <= level <= cov + Z * se),
                "mean_width": float((g["hi"] - g["lo"]).mean()),
                "interval_score": float(interval_score(g["y_true"], g["lo"], g["hi"], level).mean()),
            }
        )
    return pd.DataFrame(rows)


def select_methods(summary: pd.DataFrame, levels=(0.8, 0.95), horizons=(1, 4, 12)) -> pd.DataFrame:
    """Pre-registered rule: among calibrated methods, pick the lowest interval score.

    If no method is calibrated at a horizon/level, pick the one whose coverage
    is closest to nominal and say so.
    """
    rows = []
    s = summary[summary["level"].isin(levels) & summary["horizon"].isin(horizons)]
    for (h, level), g in s.groupby(["horizon", "level"]):
        cal = g[g["calibrated"]]
        if not cal.empty:
            best = cal.sort_values("interval_score").iloc[0]
            basis = "calibrated, lowest interval score"
        else:
            best = g.assign(gap=(g["coverage"] - level).abs()).sort_values("gap").iloc[0]
            basis = "none calibrated; closest coverage"
        rows.append(
            {
                "horizon": int(h),
                "level": level,
                "model": best["model"],
                "method": best["method"],
                "coverage": best["coverage"],
                "mean_width": best["mean_width"],
                "interval_score": best["interval_score"],
                "n_calibrated": int(g["calibrated"].sum()),
                "basis": basis,
            }
        )
    return pd.DataFrame(rows)


def regime_scorecard(
    by_regime: pd.DataFrame, levels=(0.8, 0.95), horizons=(1, 4, 12)
) -> tuple[pd.DataFrame, str]:
    """Count horizon×level cells where a method is calibrated in *every* regime.

    Pre-registered success criterion for volatility scaling: some vol-scaled
    method must score more such cells than every unscaled method.
    Returns (scorecard sorted best first, verdict sentence).
    """
    s = by_regime[by_regime["level"].isin(levels) & by_regime["horizon"].isin(horizons)]
    cells = s.groupby(["model", "method", "horizon", "level"])["calibrated"].all().reset_index()
    card = (
        cells.groupby(["model", "method"])["calibrated"]
        .agg(cells_calibrated_in_all_regimes="sum", cells="size")
        .reset_index()
        .sort_values("cells_calibrated_in_all_regimes", ascending=False)
        .reset_index(drop=True)
    )
    is_vol = card["method"].str.endswith(" + vol-scaled")
    best_vol = card.loc[is_vol, "cells_calibrated_in_all_regimes"].max()
    best_plain = card.loc[~is_vol, "cells_calibrated_in_all_regimes"].max()
    if pd.isna(best_vol):
        verdict = "No vol-scaled methods were evaluated."
    elif best_vol > best_plain:
        verdict = (
            f"**Success:** best vol-scaled method is calibrated in all regimes in {int(best_vol)} cells, "
            f"vs {int(best_plain)} for the best unscaled method."
        )
    else:
        verdict = (
            f"**Not met:** best vol-scaled method reaches {int(best_vol)} cells, "
            f"vs {int(best_plain)} for the best unscaled method."
        )
    return card, verdict


def regime_label(target_dates: pd.Series) -> pd.Series:
    """2022 (war spike and collapse) vs 2023 onwards (calmer)."""
    year = pd.to_datetime(target_dates).dt.year
    return np.where(year <= 2022, "2022 crisis", "2023+ calmer")
