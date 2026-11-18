"""Forecasting models evaluated by the backtest.

Every forecaster sees only a ``ForecastContext``, built by the backtest engine
from data available at the forecast origin. Future exogenous values are passed
only to forecasters that declare ``uses_future_exog = True``; those are oracle
benchmarks (an upper bound) and cannot be deployed, because nobody knows next
month's dollar index or VIX in advance.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ForecastContext:
    """Everything a forecaster is allowed to know at the forecast origin."""

    y_hist: np.ndarray  # target up to and including the origin
    X_hist: np.ndarray  # exogenous values up to and including the origin
    horizon: int
    X_future: np.ndarray | None = None  # oracle forecasters only


@dataclass(frozen=True)
class Forecast:
    mean: np.ndarray
    lower: np.ndarray | None = None  # 95% interval
    upper: np.ndarray | None = None


class Forecaster(ABC):
    name: str
    uses_future_exog: bool = False

    def fit(self, y: np.ndarray, X: np.ndarray) -> None:
        """Estimate parameters on data up to the origin. Baselines need none."""

    @abstractmethod
    def forecast(self, ctx: ForecastContext) -> Forecast: ...


# ── Baselines ───────────────────────────────────────────────────────────────


class Naive(Forecaster):
    """Random walk: every future week equals the last observed price.

    Intervals assume i.i.d. weekly changes, so their width grows with sqrt(h).
    """

    name = "Naive"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        last = ctx.y_hist[-1]
        mean = np.full(ctx.horizon, last)
        sigma = np.std(np.diff(ctx.y_hist), ddof=1)
        half = Z_95 * sigma * np.sqrt(np.arange(1, ctx.horizon + 1))
        return Forecast(mean, mean - half, mean + half)


class SeasonalNaive(Forecaster):
    """Each future week equals the same week one season (52 weeks) earlier."""

    def __init__(self, period: int = 52):
        self.period = period
        self.name = f"SeasonalNaive({period})"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        if ctx.horizon > self.period:
            raise ValueError("Seasonal naive needs horizon <= period")
        n = len(ctx.y_hist)
        idx = n - self.period + np.arange(ctx.horizon)
        return Forecast(ctx.y_hist[idx].astype(float))


class Drift(Forecaster):
    """Random walk with drift: extend the line from the first to the last point."""

    name = "Drift"

    def forecast(self, ctx: ForecastContext) -> Forecast:
        y = ctx.y_hist
        slope = (y[-1] - y[0]) / (len(y) - 1)
        return Forecast(y[-1] + slope * np.arange(1, ctx.horizon + 1))


# ── SARIMAX ─────────────────────────────────────────────────────────────────


class SarimaxEstimator:
    """Fits SARIMAX parameters and re-filters them on newer data.

    Several forecasters share one estimator, so the expensive maximum-likelihood
    fit (~40s) runs once per refit point. Between refits, ``filtered`` re-runs
    the Kalman filter with frozen parameters, which updates the model state
    with the latest observations at a cost of well under a second.
    """

    def __init__(
        self,
        use_exog: bool,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 52),
        maxiter: int = 200,
    ):
        self.use_exog = use_exog
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self.fit_log: list[dict] = []
        self._result = None
        self._fit_len: int | None = None
        self._cache_len: int | None = None
        self._cache_result = None

    def fit(self, y: np.ndarray, X: np.ndarray) -> None:
        if self._fit_len == len(y):
            return  # already fitted on this exact history (shared by variants)
        model = SARIMAX(
            y,
            exog=X if self.use_exog else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        start = None if self._result is None else self._result.params  # warm start
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False, maxiter=self.maxiter, start_params=start)
        self.fit_log.append(
            {
                "use_exog": self.use_exog,
                "n_obs": len(y),
                "converged": bool(result.mle_retvals.get("converged", False)),
                "iterations": int(result.mle_retvals.get("iterations", -1)),
                "seconds": round(time.perf_counter() - t0, 1),
                "aic": round(float(result.aic), 2),
            }
        )
        self._result = result
        self._fit_len = len(y)
        self._cache_len = None

    def filtered(self, y: np.ndarray, X: np.ndarray):
        if self._result is None:
            raise RuntimeError("fit() must be called before forecasting")
        if self._cache_len != len(y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cache_result = self._result.apply(y, exog=X if self.use_exog else None)
            self._cache_len = len(y)
        return self._cache_result


ExogStrategy = Literal["none", "persist", "mean", "oracle"]


class SarimaxForecaster(Forecaster):
    """SARIMAX forecast under one assumption about future exogenous values.

    none     no exogenous variables (plain SARIMA)
    persist  hold the last observed values constant (what scenario_engine does)
    mean     use the historical average of each variable up to the origin
    oracle   use the realised future values (leakage, upper bound only)
    """

    def __init__(self, name: str, estimator: SarimaxEstimator, exog_strategy: ExogStrategy):
        if (exog_strategy == "none") == estimator.use_exog:
            raise ValueError("exog_strategy 'none' must pair with a no-exog estimator and vice versa")
        self.name = name
        self.estimator = estimator
        self.exog_strategy = exog_strategy
        self.uses_future_exog = exog_strategy == "oracle"

    def fit(self, y: np.ndarray, X: np.ndarray) -> None:
        self.estimator.fit(y, X)

    def _future_exog(self, ctx: ForecastContext) -> np.ndarray | None:
        h = ctx.horizon
        if self.exog_strategy == "none":
            return None
        if self.exog_strategy == "persist":
            return np.tile(ctx.X_hist[-1], (h, 1))
        if self.exog_strategy == "mean":
            return np.tile(ctx.X_hist.mean(axis=0), (h, 1))
        if ctx.X_future is None:
            raise ValueError("Oracle forecaster received no future exogenous values")
        return ctx.X_future

    def forecast(self, ctx: ForecastContext) -> Forecast:
        result = self.estimator.filtered(ctx.y_hist, ctx.X_hist)
        fc = result.get_forecast(ctx.horizon, exog=self._future_exog(ctx))
        ci = np.asarray(fc.conf_int(alpha=0.05))
        return Forecast(np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1])


def default_forecasters() -> list[Forecaster]:
    """The model set compared in the step-1 backtest."""
    with_exog = SarimaxEstimator(use_exog=True)
    no_exog = SarimaxEstimator(use_exog=False)
    return [
        Naive(),
        SeasonalNaive(52),
        Drift(),
        SarimaxForecaster("SARIMA (no exog)", no_exog, "none"),
        SarimaxForecaster("SARIMAX exog=persist", with_exog, "persist"),
        SarimaxForecaster("SARIMAX exog=mean", with_exog, "mean"),
        SarimaxForecaster("SARIMAX exog=oracle", with_exog, "oracle"),
    ]
