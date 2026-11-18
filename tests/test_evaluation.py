"""Tests for the walk-forward evaluation toolkit.

The most important tests here are the leakage guards: they prove that a
forecaster cannot see data after its forecast origin unless it is explicitly
declared an oracle.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation import metrics
from evaluation.backtest import BacktestConfig, run_backtest
from evaluation.data import EXOG_COLS, load_dataset, train_size
from evaluation.forecasters import (
    Drift,
    Forecast,
    ForecastContext,
    Forecaster,
    Naive,
    SarimaxEstimator,
    SarimaxForecaster,
    SeasonalNaive,
)


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-03", periods=n, freq="W-SUN")
    y = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    X = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=idx, columns=["a", "b"])
    return y, X


def _ctx(y, h, X=None):
    y = np.asarray(y, dtype=float)
    X = np.zeros((len(y), 1)) if X is None else X
    return ForecastContext(y, X, h)


# ── Baselines ───────────────────────────────────────────────────────────────


def test_naive_repeats_last_value_with_widening_interval():
    fc = Naive().forecast(_ctx([1.0, 3.0, 2.0, 4.0], 3))
    np.testing.assert_allclose(fc.mean, [4.0, 4.0, 4.0])
    widths = fc.upper - fc.lower
    np.testing.assert_allclose(widths / widths[0], np.sqrt([1, 2, 3]))


def test_seasonal_naive_uses_same_week_last_season():
    y = np.arange(10, dtype=float)  # period 4: last season is [6, 7, 8, 9]
    fc = SeasonalNaive(period=4).forecast(_ctx(y, 3))
    np.testing.assert_allclose(fc.mean, [6.0, 7.0, 8.0])


def test_seasonal_naive_rejects_horizon_longer_than_period():
    with pytest.raises(ValueError):
        SeasonalNaive(period=4).forecast(_ctx(np.arange(10.0), 5))


def test_drift_extends_first_to_last_line():
    fc = Drift().forecast(_ctx([10.0, 12.0, 14.0], 2))
    np.testing.assert_allclose(fc.mean, [16.0, 18.0])


# ── Metrics ─────────────────────────────────────────────────────────────────


def test_point_metrics_known_values():
    y, p = [100.0, 200.0], [110.0, 180.0]
    assert metrics.mae(y, p) == pytest.approx(15.0)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics.mape(y, p) == pytest.approx(10.0)
    assert metrics.mase_scale([1.0, 3.0, 2.0]) == pytest.approx(1.5)


def test_directional_accuracy_ignores_flat_calls():
    acc = metrics.directional_accuracy(y_origin=[10, 10, 10], y_true=[11, 9, 12], y_pred=[12, 11, 10])
    assert acc == pytest.approx(1 / 2)  # third call is flat -> excluded
    assert np.isnan(metrics.directional_accuracy([10], [11], [10]))


def test_interval_coverage():
    assert metrics.interval_coverage([1, 5, 9], [0, 0, 0], [2, 6, 8]) == pytest.approx(2 / 3)
    assert np.isnan(metrics.interval_coverage([1], [np.nan], [np.nan]))


def test_diebold_mariano_detects_clearly_better_forecast():
    rng = np.random.default_rng(1)
    good, bad = rng.normal(0, 1, 300), rng.normal(0, 3, 300)
    stat, p = metrics.diebold_mariano(good, bad, horizon=1)
    assert stat < 0 and p < 0.001


def test_diebold_mariano_no_difference_for_equivalent_forecasts():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    _, p = metrics.diebold_mariano(a, b, horizon=4)
    assert p > 0.05


# ── Backtest engine & leakage guards ────────────────────────────────────────


class SpyForecaster(Forecaster):
    """Records what it was shown and tries to cheat by mutating the history."""

    def __init__(self, name="Spy", oracle=False):
        self.name = name
        self.uses_future_exog = oracle
        self.seen = []
        self.fits = []

    def fit(self, y, X):
        self.fits.append(len(y))

    def forecast(self, ctx):
        self.seen.append((len(ctx.y_hist), len(ctx.X_hist), ctx.X_future))
        with pytest.raises(ValueError):
            ctx.y_hist[-1] = 1e9  # history is read-only
        return Forecast(np.full(ctx.horizon, ctx.y_hist[-1]))


def test_forecasters_only_see_data_up_to_origin():
    y, X = _series()
    spy = SpyForecaster()
    cfg = BacktestConfig(first_origin=59, horizon=4, step=5, refit_every=None)
    run_backtest(y, X, [spy], cfg)
    for origin, (ny, nx, fut) in zip(cfg.origins(len(y)), spy.seen):
        assert ny == nx == origin + 1
        assert fut is None


def test_only_oracle_receives_future_exog_and_it_is_the_realised_values():
    y, X = _series()
    oracle = SpyForecaster("Oracle", oracle=True)
    cfg = BacktestConfig(first_origin=59, horizon=4, step=30, refit_every=None)
    run_backtest(y, X, [SpyForecaster(), oracle], cfg)
    first_origin = cfg.origins(len(y))[0]
    np.testing.assert_allclose(oracle.seen[0][2], X.to_numpy()[first_origin + 1 : first_origin + 5])


def test_backtest_rows_align_truth_with_target_date():
    y, X = _series()
    cfg = BacktestConfig(first_origin=59, horizon=3, step=10, refit_every=None)
    out = run_backtest(y, X, [Naive()], cfg)
    assert len(out) == len(cfg.origins(len(y))) * 3
    for _, r in out.sample(10, random_state=0).iterrows():
        assert r["y_true"] == y.loc[r["target_date"]]
        assert r["y_origin"] == y.loc[r["origin"]]
        assert (r["target_date"] - r["origin"]).days == 7 * r["horizon"]


def test_last_origin_leaves_a_full_horizon():
    y, X = _series(n=100)
    cfg = BacktestConfig(first_origin=50, horizon=12)
    assert cfg.origins(len(y))[-1] + 12 == len(y) - 1


def test_refit_schedule():
    y, X = _series()
    spy = SpyForecaster()
    run_backtest(y, X, [spy], BacktestConfig(first_origin=59, horizon=1, step=1, refit_every=20))
    assert spy.fits == [60, 80, 100]


def test_duplicate_model_names_rejected():
    y, X = _series()
    with pytest.raises(ValueError):
        run_backtest(y, X, [Naive(), Naive()], BacktestConfig(first_origin=59))


# ── SARIMAX wrapper ─────────────────────────────────────────────────────────


def test_sarimax_strategies_and_shared_fit():
    y, X = _series(n=80)
    est = SarimaxEstimator(use_exog=True, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), maxiter=50)
    persist = SarimaxForecaster("p", est, "persist")
    oracle = SarimaxForecaster("o", est, "oracle")
    yv, Xv = y.to_numpy(), X.to_numpy()
    persist.fit(yv[:60], Xv[:60])
    oracle.fit(yv[:60], Xv[:60])
    assert len(est.fit_log) == 1  # second fit on the same history is skipped

    fc = persist.forecast(ForecastContext(yv[:70], Xv[:70], 5))
    assert fc.mean.shape == (5,) and np.all(fc.lower < fc.upper)
    with pytest.raises(ValueError):
        oracle.forecast(ForecastContext(yv[:70], Xv[:70], 5))  # no future exog supplied
    assert oracle.uses_future_exog and not persist.uses_future_exog


def test_sarimax_strategy_must_match_estimator():
    with pytest.raises(ValueError):
        SarimaxForecaster("bad", SarimaxEstimator(use_exog=False), "persist")


# ── Real dataset ────────────────────────────────────────────────────────────


def test_dataset_is_regular_weekly_and_matches_train_split():
    y, X = load_dataset()
    assert list(X.columns) == EXOG_COLS
    assert y.index.freqstr.startswith("W")
    assert train_size(len(y)) == int(len(y) * 0.8)
