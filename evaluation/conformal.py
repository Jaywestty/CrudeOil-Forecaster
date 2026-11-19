"""Conformal prediction intervals built on top of any point forecaster.

Conformal prediction sets the interval width from the forecaster's own past
errors, not from model assumptions. If recent errors are representative of the
next one, an interval at level L covers the truth with probability ≥ L.

Two methods, both run per horizon:

split conformal (rolling window)
    Take the absolute log errors of the last ``window`` *resolved* forecasts
    and use their L-quantile as the half-width (in log space).

adaptive conformal inference, ACI (Gibbs & Candès, 2021)
    Time series break the "recent errors are representative" assumption when
    the regime changes. ACI tracks its own misses and adjusts the quantile it
    uses: after a miss it asks for a wider interval, after a hit slightly
    narrower. With step size gamma the long-run miss rate converges to 1 - L
    even when the error distribution shifts.

Leakage rule: an h-step forecast made at origin s is scored only once its
target date s+h has passed. At origin t, only forecasts with target_date <= t
enter the calibration set or the ACI update. ``tests/test_evaluation.py``
checks that changing unresolved outcomes never changes an interval.

Errors are measured as |log(y_true / y_pred)| so that a $5 miss at $40 oil
and a $12 miss at $100 oil count roughly the same.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NATIVE = "native (model's own)"
SPLIT = "split conformal"
ACI = "adaptive conformal (ACI)"
VOL_SUFFIX = " + vol-scaled"
SPLIT_VOL = SPLIT + VOL_SUFFIX
ACI_VOL = ACI + VOL_SUFFIX


def trailing_volatility(y: pd.Series, window: int = 12) -> pd.Series:
    """Std of the last ``window`` weekly log returns, ending at each date.

    Known at each date (it only uses returns up to and including that week),
    so it can scale both past errors and today's interval without leakage.

    Volatility-scaled conformal: errors are divided by the volatility at their
    forecast origin, so a $10 miss in a turbulent month and a $3 miss in a calm
    one can count as equally "surprising". Today's interval is then the scaled
    quantile times *today's* volatility, so it widens and narrows with the market.
    """
    r = np.log(y).diff()
    return r.rolling(window, min_periods=window).std(ddof=1)


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: the ceil((m+1)·L)-th smallest score.

    The +1 is what gives the coverage guarantee. If it exceeds the sample
    size the exact interval is infinite; we cap at the largest observed score
    and note it in the report (only possible with very few scores).
    """
    m = len(scores)
    if m == 0:
        return float("nan")
    k = math.ceil((m + 1) * level)
    if k <= 0:
        return 0.0
    if k > m:
        return float(np.max(scores))
    return float(np.partition(scores, k - 1)[k - 1])


def _native_rows(g: pd.DataFrame, levels) -> list[dict]:
    rows = []
    for level in levels:
        tag = round(level * 100)
        lo_col, hi_col = f"lo_{tag}", f"hi_{tag}"
        if lo_col not in g or g[lo_col].isna().all():
            continue
        for r in g.itertuples(index=False):
            rows.append(_row(r, level, NATIVE, getattr(r, lo_col), getattr(r, hi_col)))
    return rows


def _row(r, level, method, lo, hi) -> dict:
    return {
        "model": r.model,
        "origin": r.origin,
        "target_date": r.target_date,
        "horizon": r.horizon,
        "level": level,
        "method": method,
        "y_true": r.y_true,
        "y_pred": r.y_pred,
        "lo": lo,
        "hi": hi,
    }


def conformal_intervals(
    forecasts: pd.DataFrame,
    levels=(0.5, 0.8, 0.95),
    window: int = 156,
    min_calib: int = 52,
    gamma: float = 0.005,
    scale_col: str | None = None,
    include_native: bool = True,
) -> pd.DataFrame:
    """Native, split-conformal and ACI intervals for one model's backtest output.

    ``forecasts`` must contain a single model with weekly, consecutive origins.
    With ``scale_col`` (e.g. trailing volatility at each origin), errors are
    divided by that scale and intervals multiplied by it; method names get a
    " + vol-scaled" suffix. Rows with a missing scale get no conformal interval.
    Returns one row per (origin, horizon, level, method).
    """
    if forecasts["model"].nunique() != 1:
        raise ValueError("Pass one model at a time")
    split_name, aci_name = (SPLIT_VOL, ACI_VOL) if scale_col else (SPLIT, ACI)
    out: list[dict] = []
    for h, g in forecasts.groupby("horizon"):
        g = g.sort_values("origin").reset_index(drop=True)
        if include_native:
            out += _native_rows(g, levels)

        origins = g["origin"].to_numpy()
        targets = g["target_date"].to_numpy()
        if np.any(np.diff(targets) <= np.timedelta64(0)):
            raise ValueError("Target dates must be strictly increasing within a horizon")
        y_true, y_pred = g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float)
        scale = g[scale_col].to_numpy(float) if scale_col else np.ones(len(g))
        if np.any(scale[~np.isnan(scale)] <= 0):
            raise ValueError("Scale must be positive")
        scores = np.abs(np.log(y_true / y_pred)) / scale  # NaN where scale is missing

        alpha = {L: 1 - L for L in levels}  # ACI state: current miscoverage target
        aci_lo = {L: np.full(len(g), np.nan) for L in levels}
        aci_hi = {L: np.full(len(g), np.nan) for L in levels}
        resolved_before = 0

        rows = list(g.itertuples(index=False))
        for i, r in enumerate(rows):
            # Forecasts whose target date has passed by this origin are now scoreable.
            resolved = int(np.searchsorted(targets, origins[i], side="right"))

            # ACI update: learn from every interval that just resolved.
            for j in range(resolved_before, resolved):
                for L in levels:
                    if np.isnan(aci_lo[L][j]):
                        continue
                    miss = float(not (aci_lo[L][j] <= y_true[j] <= aci_hi[L][j]))
                    alpha[L] += gamma * ((1 - L) - miss)
            resolved_before = resolved

            calib = scores[max(0, resolved - window) : resolved]
            calib = calib[~np.isnan(calib)]
            if len(calib) < min_calib or np.isnan(scale[i]):
                continue
            s = scale[i]
            for L in levels:
                q = conformal_quantile(calib, L) * s
                out.append(_row(r, L, split_name, y_pred[i] * math.exp(-q), y_pred[i] * math.exp(q)))

                eff = 1 - alpha[L]  # level ACI asks for right now
                q_aci = conformal_quantile(calib, min(max(eff, 0.0), 1.0)) * s
                lo, hi = y_pred[i] * math.exp(-q_aci), y_pred[i] * math.exp(q_aci)
                aci_lo[L][i], aci_hi[L][i] = lo, hi
                out.append(_row(r, L, aci_name, lo, hi))
    return pd.DataFrame(out)
