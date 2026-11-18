"""Features and direct multi-horizon targets for the ML challengers.

Two rules keep this leakage-free:

1. **Features at row t use only data up to t.** Every feature is a function of
   the past (lags, trailing windows), so appending future data never changes
   an existing row. ``tests/test_evaluation.py`` checks this property directly.

2. **A training label must be observable at fit time.** The h-week target for
   row s is log(y[s+h] / y[s]). When fitting at origin t, row s can be used
   only if s + h <= t; otherwise the label would contain prices from after t.
   This is the classic look-ahead trap in direct multi-horizon forecasting.

The features use only the target history and the same five exogenous
variables SARIMAX gets, so any difference in results comes from the model,
not from extra data.
"""

from __future__ import annotations

import numpy as np

WARMUP = 52  # rows needed before every trailing window is defined


def feature_names(n_exog: int) -> list[str]:
    names = ["ret_1w", "ret_4w", "ret_12w", "ret_26w", "vol_12w", "dev_52w_mean"]
    names += [f"exog{i}_now" for i in range(n_exog)]
    names += [f"exog{i}_sum_4w" for i in range(n_exog)]
    return names


def build_features(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Feature matrix with one row per time step; rows before WARMUP are NaN."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = len(y), X.shape[1]
    logy = np.log(y)
    r = np.r_[np.nan, np.diff(logy)]  # weekly log return ending at t

    F = np.full((n, 6 + 2 * k), np.nan)
    for t in range(WARMUP, n):
        F[t, 0] = r[t]
        F[t, 1] = logy[t] - logy[t - 4]
        F[t, 2] = logy[t] - logy[t - 12]
        F[t, 3] = logy[t] - logy[t - 26]
        F[t, 4] = np.std(r[t - 11 : t + 1], ddof=1)
        F[t, 5] = logy[t] - np.log(np.mean(y[t - 51 : t + 1]))
        F[t, 6 : 6 + k] = X[t]
        F[t, 6 + k :] = X[t - 3 : t + 1].sum(axis=0)
    return F


def direct_target(y: np.ndarray, h: int) -> np.ndarray:
    """log(y[s+h] / y[s]) for each row s; NaN where s+h is beyond the data."""
    logy = np.log(np.asarray(y, dtype=float))
    out = np.full(len(logy), np.nan)
    out[: len(logy) - h] = logy[h:] - logy[:-h]
    return out


def training_rows(n_obs: int, h: int) -> slice:
    """Rows whose features exist and whose h-step label is known at the last row.

    With history y[0..n-1], the label for row s needs y[s+h], so s <= n-1-h.
    """
    return slice(WARMUP, n_obs - h)
