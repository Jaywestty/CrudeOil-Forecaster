"""Walk-forward (expanding-window) backtest engine.

At each forecast origin t the engine:
  1. hands every forecaster the data up to and including t, and nothing later
     (oracle forecasters additionally get realised future exogenous values);
  2. optionally re-estimates parameters (every ``refit_every`` weeks);
  3. records the 1..H-week-ahead forecasts against what actually happened.

This simulates using the model in production: each forecast uses only
information that existed on the day it was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from evaluation.forecasters import ForecastContext, Forecaster


@dataclass(frozen=True)
class BacktestConfig:
    first_origin: int  # positional index of the first origin (last observation used)
    horizon: int = 12
    step: int = 1  # weeks between consecutive origins
    refit_every: int | None = 52  # weeks between parameter re-estimation; None = fit once

    def origins(self, n_obs: int) -> range:
        # The last origin must leave `horizon` observations to score against.
        return range(self.first_origin, n_obs - self.horizon, self.step)


def _read_only(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a.flags.writeable = False
    return a


def run_backtest(
    y: pd.Series,
    X: pd.DataFrame,
    forecasters: Sequence[Forecaster],
    config: BacktestConfig,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run the walk-forward backtest and return one row per (origin, model, horizon)."""
    if not y.index.equals(X.index):
        raise ValueError("y and X must share an index")
    names = [f.name for f in forecasters]
    if len(set(names)) != len(names):
        raise ValueError(f"Forecaster names must be unique: {names}")

    yv, Xv, dates = y.to_numpy(float), X.to_numpy(float), y.index
    h = config.horizon
    origins = config.origins(len(yv))
    if len(origins) == 0:
        raise ValueError("No forecast origins: check first_origin and horizon")

    records: list[dict] = []
    last_fit: int | None = None
    for i, o in enumerate(origins):
        y_hist, X_hist = _read_only(yv[: o + 1]), _read_only(Xv[: o + 1])

        if last_fit is None:
            refit_due = True
        else:
            refit_due = config.refit_every is not None and o - last_fit >= config.refit_every
        if refit_due:
            for f in forecasters:
                f.fit(y_hist, X_hist)
            last_fit = o

        X_future = _read_only(Xv[o + 1 : o + 1 + h])
        for f in forecasters:
            ctx = ForecastContext(y_hist, X_hist, h, X_future if f.uses_future_exog else None)
            fc = f.forecast(ctx)
            mean = np.asarray(fc.mean, dtype=float)
            if mean.shape != (h,) or not np.all(np.isfinite(mean)):
                raise ValueError(f"{f.name} returned an invalid forecast at origin {dates[o].date()}")
            for k in range(1, h + 1):
                records.append(
                    {
                        "origin": dates[o],
                        "target_date": dates[o + k],
                        "horizon": k,
                        "model": f.name,
                        "oracle": f.uses_future_exog,
                        "y_origin": yv[o],
                        "y_true": yv[o + k],
                        "y_pred": mean[k - 1],
                        "lower": np.nan if fc.lower is None else float(fc.lower[k - 1]),
                        "upper": np.nan if fc.upper is None else float(fc.upper[k - 1]),
                        "params_fitted_through": dates[last_fit],
                    }
                )
        if progress:
            progress(i + 1, len(origins))

    return pd.DataFrame.from_records(records)
