"""Production model specification and fitting.

One place defines the model the API serves, so train.py (Render build) and the
promotion gate (pipeline/promote.py) always fit exactly the same thing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace import kalman_filter as kf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Forecasting only needs the Kalman filter's final state. A fitted results
# object also keeps a 55x55 covariance matrix per week for every smoothed,
# filtered and gain quantity (~490 MB pickled at 1,088 weeks), which crashed
# Render's 512 MB instance. Serving models keep only the final state (~3 MB).
SERVING_MEMORY_FLAGS = (kf.MEMORY_NO_FILTERED | kf.MEMORY_NO_GAIN
                        | kf.MEMORY_NO_SMOOTHING | kf.MEMORY_NO_PREDICTED_COV)

EXOG_COLS = ["dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff"]
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 52)
MAXITER = 200  # the default 50 stopped before convergence; backtests converged within ~75
SPEC = f"SARIMAX{ORDER}{SEASONAL_ORDER} with {len(EXOG_COLS)} macro drivers"


def align(weekly: pd.DataFrame, transformed: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Brent levels and exogenous drivers on their shared weekly dates."""
    common = weekly.index.intersection(transformed.index)
    return weekly.loc[common, "brent"], transformed.loc[common, EXOG_COLS]


def _model(weekly: pd.DataFrame, transformed: pd.DataFrame) -> SARIMAX:
    y, X = align(weekly, transformed)
    return SARIMAX(y, exog=X, order=ORDER, seasonal_order=SEASONAL_ORDER,
                   enforce_stationarity=False, enforce_invertibility=False)


def fit(weekly: pd.DataFrame, transformed: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _model(weekly, transformed).fit(disp=False, maxiter=MAXITER)


def serving_model(results, weekly: pd.DataFrame, transformed: pd.DataFrame):
    """A small copy of a fitted model for serving: same parameters, final state only.

    Built on a FRESH model object: the fitted one keeps references to every
    array from estimation, so filtering through it would pickle them all.
    Forecasts and their intervals are identical to the full results.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _model(weekly, transformed).filter(
            results.params, conserve_memory=SERVING_MEMORY_FLAGS, cov_type="none"
        )


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
