"""Production model specification and fitting.

One place defines the model the API serves, so train.py (Render build) and the
promotion gate (pipeline/promote.py) always fit exactly the same thing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG_COLS = ["dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff"]
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 52)
MAXITER = 200  # the default 50 stopped before convergence; backtests converged within ~75
SPEC = f"SARIMAX{ORDER}{SEASONAL_ORDER} with {len(EXOG_COLS)} macro drivers"


def align(weekly: pd.DataFrame, transformed: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Brent levels and exogenous drivers on their shared weekly dates."""
    common = weekly.index.intersection(transformed.index)
    return weekly.loc[common, "brent"], transformed.loc[common, EXOG_COLS]


def fit(weekly: pd.DataFrame, transformed: pd.DataFrame):
    y, X = align(weekly, transformed)
    model = SARIMAX(y, exog=X, order=ORDER, seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=MAXITER)


def fit_summary(results) -> dict:
    params = np.asarray(results.params, dtype=float)
    return {
        "spec": SPEC,
        "converged": bool(results.mle_retvals.get("converged", False)),
        "iterations": int(results.mle_retvals.get("iterations", -1)),
        "aic": round(float(results.aic), 2),
        "n_obs": int(results.nobs),
        "params_finite": bool(np.isfinite(params).all()),
    }
