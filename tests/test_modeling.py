"""The serving model must be small but forecast exactly like the fitted model."""

import pickle

import numpy as np
import pandas as pd
import pytest

import modeling


@pytest.fixture(scope="module")
def fitted_and_serving():
    # A small, fast specification on real data: the memory behaviour is the
    # same, only the state dimension is smaller than production's.
    w = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True).iloc[-300:]
    t = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True).iloc[-300:]
    orig = (modeling.ORDER, modeling.SEASONAL_ORDER)
    modeling.ORDER, modeling.SEASONAL_ORDER = (1, 1, 1), (1, 0, 0, 4)
    try:
        fitted = modeling.fit(w, t)
        serving = modeling.serving_model(fitted, w, t)
    finally:
        modeling.ORDER, modeling.SEASONAL_ORDER = orig
    _, X = modeling.align(w, t)
    return fitted, serving, np.tile(X.iloc[-1].to_numpy(float), (12, 1))


def test_serving_model_forecasts_and_bands_match_the_fit(fitted_and_serving):
    fitted, serving, exog = fitted_and_serving
    a, b = fitted.get_forecast(12, exog=exog), serving.get_forecast(12, exog=exog)
    np.testing.assert_allclose(np.asarray(b.predicted_mean), np.asarray(a.predicted_mean), rtol=1e-10)
    for alpha in (0.05, 0.2):
        ci = np.asarray(b.conf_int(alpha=alpha))
        assert np.isfinite(ci).all()
        np.testing.assert_allclose(ci, np.asarray(a.conf_int(alpha=alpha)), rtol=1e-10)


def test_serving_model_is_much_smaller_and_survives_pickling(fitted_and_serving):
    fitted, serving, exog = fitted_and_serving
    small, big = pickle.dumps(serving), pickle.dumps(fitted)
    assert len(small) < len(big) / 5
    restored = pickle.loads(small)
    np.testing.assert_allclose(np.asarray(restored.get_forecast(12, exog=exog).predicted_mean),
                               np.asarray(serving.get_forecast(12, exog=exog).predicted_mean))
