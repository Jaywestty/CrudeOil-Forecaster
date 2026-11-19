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


# ── Step 1b: features, direct ML challengers, multiple testing ──────────────

from evaluation.features import WARMUP, build_features, direct_target, training_rows  # noqa: E402
from evaluation.forecasters import DirectQuantileGBM, DirectRidge  # noqa: E402


def test_features_do_not_change_when_future_data_is_appended():
    y, X = _series(n=150)
    short = build_features(y.to_numpy()[:100], X.to_numpy()[:100])
    full = build_features(y.to_numpy(), X.to_numpy())
    np.testing.assert_array_equal(short, full[:100])  # NaN-aware equality
    assert np.isnan(full[:WARMUP]).all() and not np.isnan(full[WARMUP:]).any()


def test_direct_target_and_training_rows_never_use_unobserved_prices():
    y = np.exp(np.arange(10, dtype=float))  # log y = 0, 1, 2, ...
    np.testing.assert_allclose(direct_target(y, 3)[:7], 3.0)
    assert np.isnan(direct_target(y, 3)[7:]).all()
    n, h = 120, 12
    rows = range(n)[training_rows(n, h)]
    assert rows[0] == WARMUP and rows[-1] + h == n - 1  # last label uses the origin's own price


def _signal_series(n=600, seed=3):
    """Prices whose next-week return is driven by today's exog0, a known, learnable signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    r = np.r_[0.0, 0.02 * X[:-1, 0] + rng.normal(0, 0.005, n - 1)]
    idx = pd.date_range("2000-01-02", periods=n, freq="W-SUN")
    return pd.Series(60 * np.exp(np.cumsum(r)), index=idx), pd.DataFrame(X, index=idx, columns=["a", "b"])


@pytest.mark.parametrize("model_cls", [DirectRidge, DirectQuantileGBM])
def test_direct_models_learn_a_real_signal(model_cls):
    """Positive control: a null result on oil must not be a pipeline bug."""
    y, X = _signal_series()
    cfg = BacktestConfig(first_origin=499, horizon=1, step=1, refit_every=None)
    out = run_backtest(y, X, [Naive(), model_cls(max_horizon=1)], cfg)
    s = metrics.summarize(out, scale=1.0)
    rel = s.loc[s["model"] == model_cls.name, "rel_MAE"].item()
    assert rel < 0.6


def test_direct_forecasts_are_valid_and_gbm_intervals_are_ordered():
    y, X = _signal_series(n=200)
    yv, Xv = y.to_numpy(), X.to_numpy()
    ridge, gbm = DirectRidge(max_horizon=4), DirectQuantileGBM(max_horizon=4)
    for m in (ridge, gbm):
        m.fit(yv[:180], Xv[:180])
    ctx = ForecastContext(yv[:190], Xv[:190], 4)
    fr, fg = ridge.forecast(ctx), gbm.forecast(ctx)
    assert fr.mean.shape == (4,) and fr.lower is None
    assert np.all(fg.lower <= fg.mean) and np.all(fg.mean <= fg.upper)
    with pytest.raises(ValueError):
        ridge.forecast(ForecastContext(yv[:190], Xv[:190], 5))  # beyond trained horizons
    assert ridge.fit_log[0]["details"].startswith("alpha")


def test_holm_adjustment_known_values():
    np.testing.assert_allclose(metrics.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    adj = metrics.holm([0.2, np.nan])
    assert adj[0] == pytest.approx(0.2) and np.isnan(adj[1])


def test_promotion_requires_lower_error_and_significance_after_holm():
    summary = pd.DataFrame(
        {
            "horizon": [1, 1, 1],
            "model": ["A", "B", "C"],
            "MAE": [1.0, 1.0, 1.0],
            "rel_MAE": [0.9, 0.95, 1.1],
            "DM_p": [0.001, 0.03, 0.03],
        }
    )
    d = metrics.promotion_decisions(summary, ["A", "B", "C"], [1]).set_index("model")["verdict"]
    assert d["A"] == "PROMOTE"
    assert d["B"] == "better, not significant"  # 0.03 alone would pass; Holm makes it 0.06
    assert d["C"] == "worse than naive"


# ── Step 1c: multi-level intervals, conformal prediction, calibration ───────

from evaluation import calibration, conformal  # noqa: E402


def _fake_forecasts(sd, h=1, seed=0, model="M"):
    """Backtest-shaped frame: flat $100 forecasts with log errors drawn at scale sd[i]."""
    rng = np.random.default_rng(seed)
    sd = np.asarray(sd, dtype=float)
    origins = pd.date_range("2000-01-02", periods=len(sd), freq="W-SUN")
    return pd.DataFrame(
        {
            "model": model,
            "origin": origins,
            "target_date": origins + pd.Timedelta(weeks=h),
            "horizon": h,
            "y_pred": 100.0,
            "y_true": 100.0 * np.exp(rng.normal(0, 1, len(sd)) * sd),
        }
    )


def _coverage(iv, method, level, start=None):
    g = iv[(iv["method"] == method) & (iv["level"] == level)]
    if start is not None:
        g = g[g["origin"] >= start]
    return ((g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])).mean()


def test_conformal_quantile_uses_finite_sample_rank():
    s = np.arange(1.0, 10.0)  # 9 scores
    assert conformal.conformal_quantile(s, 0.5) == 5.0  # ceil(10 * 0.5) = 5th smallest
    assert conformal.conformal_quantile(s, 0.95) == 9.0  # rank 10 > 9 -> capped at the max


def test_interval_score_known_values():
    np.testing.assert_allclose(metrics.interval_score([10.0], [8.0], [12.0], 0.8), [4.0])
    np.testing.assert_allclose(metrics.interval_score([14.0], [8.0], [12.0], 0.8), [4.0 + 10 * 2.0])


def test_hac_se_matches_iid_and_grows_with_autocorrelation():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 5000)
    assert metrics.hac_mean_se(x, lags=4) == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.15)
    smooth = np.convolve(x, np.ones(8), mode="same")  # strongly autocorrelated
    iid_se = smooth.std() / np.sqrt(len(smooth))
    assert metrics.hac_mean_se(smooth, lags=8) > 2 * iid_se


def test_conformal_intervals_never_use_unresolved_outcomes():
    fc = _fake_forecasts(np.full(400, 0.05), h=4)
    cutoff = fc["origin"].iloc[300]
    base = conformal.conformal_intervals(fc)

    tampered = fc.copy()
    future = tampered["target_date"] > cutoff  # outcomes unknown at the cutoff origin
    tampered.loc[future, "y_true"] *= 3.0
    after = conformal.conformal_intervals(tampered)

    key = ["origin", "level", "method"]
    a = base[base["origin"] <= cutoff].set_index(key).sort_index()
    b = after[after["origin"] <= cutoff].set_index(key).sort_index()
    np.testing.assert_allclose(a[["lo", "hi"]].to_numpy(), b[["lo", "hi"]].to_numpy())


def test_split_conformal_hits_nominal_coverage_on_stable_errors():
    """Positive control: with exchangeable errors the guarantee must hold."""
    fc = _fake_forecasts(np.full(1500, 0.05), h=1, seed=5)
    iv = conformal.conformal_intervals(fc)
    for level in (0.8, 0.95):
        assert _coverage(iv, conformal.SPLIT, level) == pytest.approx(level, abs=0.03)


def test_aci_recovers_faster_than_split_after_volatility_jump():
    sd = np.r_[np.full(800, 0.02), np.full(800, 0.06)]  # error scale triples at week 800
    fc = _fake_forecasts(sd, h=1, seed=6)
    iv = conformal.conformal_intervals(fc, window=156, gamma=0.005)
    shift = fc["origin"].iloc[800]
    just_after = iv[(iv["origin"] >= shift) & (iv["origin"] < shift + pd.Timedelta(weeks=156))]
    assert _coverage(just_after, conformal.ACI, 0.95) > _coverage(just_after, conformal.SPLIT, 0.95) + 0.03


def test_conformal_rejects_mixed_models():
    fc = pd.concat([_fake_forecasts([0.05] * 60, model="A"), _fake_forecasts([0.05] * 60, model="B")])
    with pytest.raises(ValueError):
        conformal.conformal_intervals(fc)


def test_backtest_records_nested_multi_level_intervals():
    y, X = _series()
    out = run_backtest(y, X, [Naive()], BacktestConfig(first_origin=59, horizon=3, step=20, refit_every=None))
    assert {"lo_50", "hi_50", "lo_80", "hi_80", "lo_95", "hi_95"} <= set(out.columns)
    assert (out["lo_95"] <= out["lo_80"]).all() and (out["lo_80"] <= out["lo_50"]).all()
    assert (out["hi_50"] <= out["hi_80"]).all() and (out["hi_80"] <= out["hi_95"]).all()
    np.testing.assert_allclose(out["lo_95"], out["lower"])


def test_calibration_summary_and_selection_rule():
    rows = []
    origins = pd.date_range("2020-01-05", periods=400, freq="W-SUN")
    rng = np.random.default_rng(7)
    y = 100 + rng.normal(0, 1, 400)
    z95 = 1.959963984540054
    for method, half in [("right", z95), ("too narrow", 0.5), ("too wide", 6.0)]:
        rows.append(
            pd.DataFrame(
                {"model": "M", "method": method, "horizon": 1, "level": 0.95, "origin": origins,
                 "y_true": y, "lo": 100 - half, "hi": 100 + half}
            )
        )
    s = calibration.summarize_intervals(pd.concat(rows)).set_index("method")
    assert s.loc["right", "calibrated"] and not s.loc["too narrow", "calibrated"]
    assert not s.loc["too wide", "calibrated"]  # 100% coverage has zero variance, so 95% is outside its CI
    choice = calibration.select_methods(s.reset_index(), levels=(0.95,), horizons=(1,))
    assert choice.loc[0, "method"] == "right"


# ── Step 1c experiment: volatility-scaled conformal ─────────────────────────


def test_trailing_volatility_uses_only_past_returns():
    y, _ = _series(n=100)
    full = conformal.trailing_volatility(y, window=12)
    short = conformal.trailing_volatility(y.iloc[:60], window=12)
    pd.testing.assert_series_equal(full.iloc[:60], short)
    expected = np.log(y).diff().iloc[48:60].std(ddof=1)
    assert full.iloc[59] == pytest.approx(expected)
    assert full.iloc[:12].isna().all()


def test_vol_scaling_restores_coverage_in_both_regimes():
    """Positive control: error scale switches between calm and turbulent blocks.

    Given a scale that tracks the true volatility, scaled conformal must cover
    ~95% in both kinds of block, while unscaled conformal under-covers in the
    turbulent blocks.
    """
    sd = np.tile(np.r_[np.full(100, 0.02), np.full(100, 0.06)], 8)
    fc = _fake_forecasts(sd, h=1, seed=8)
    fc["vol"] = sd
    plain = conformal.conformal_intervals(fc)
    scaled = conformal.conformal_intervals(fc, scale_col="vol", include_native=False)
    turbulent = fc.loc[fc.index % 200 >= 100, "origin"]
    calm = fc.loc[fc.index % 200 < 100, "origin"]

    def cov(iv, method, origins):
        g = iv[(iv["method"] == method) & (iv["level"] == 0.95) & iv["origin"].isin(origins)]
        return ((g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])).mean()

    assert cov(scaled, conformal.SPLIT_VOL, turbulent) == pytest.approx(0.95, abs=0.03)
    assert cov(scaled, conformal.SPLIT_VOL, calm) == pytest.approx(0.95, abs=0.03)
    assert cov(plain, conformal.SPLIT, turbulent) < 0.90
    assert set(scaled["method"]) == {conformal.SPLIT_VOL, conformal.ACI_VOL}  # no duplicate native rows


def test_vol_scaling_rejects_non_positive_scale():
    fc = _fake_forecasts(np.full(80, 0.05))
    fc["vol"] = 0.0
    with pytest.raises(ValueError):
        conformal.conformal_intervals(fc, scale_col="vol")


def test_regime_scorecard_verdict():
    rows = []
    for method, ok in [("split conformal", [True, False]), ("split conformal + vol-scaled", [True, True])]:
        for regime, flag in zip(["2022 crisis", "2023+ calmer"], ok):
            rows.append({"model": "M", "method": method, "horizon": 1, "level": 0.95,
                         "regime": regime, "calibrated": flag})
    card, verdict = calibration.regime_scorecard(pd.DataFrame(rows), levels=(0.95,), horizons=(1,))
    assert card.iloc[0]["method"].endswith("vol-scaled")
    assert verdict.startswith("**Success")
