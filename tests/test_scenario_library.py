"""Tests for historical-analogue scenario calibration (step 2a)."""

import numpy as np
import pandas as pd
import pytest

import llm_explainer
from evaluation.data import load_dataset
from scenario_library import (
    analogue_path,
    event_paths,
    load_events,
    origin_position,
    week_cols,
)


def _prices(n=40, start="2020-01-05"):
    idx = pd.date_range(start, periods=n, freq="W-SUN")
    return pd.Series(np.exp(np.arange(n) * 0.01) * 50, index=idx)  # +1% log per week


def _events(rows):
    return pd.DataFrame(
        [{"scenario": s, "date": pd.Timestamp(d), "event": e, "confounded": c} for s, d, e, c in rows]
    )


# ── Timing: no look-ahead ──────────────────────────────────────────


def test_origin_is_strictly_before_event_even_on_an_observation_date():
    idx = pd.date_range("2020-01-05", periods=5, freq="W-SUN")
    assert origin_position(idx, pd.Timestamp("2020-01-15")) == 1  # Wed -> previous Sunday (01-12)
    assert origin_position(idx, pd.Timestamp("2020-01-12")) == 0  # event on a Sunday -> week before
    with pytest.raises(ValueError):
        origin_position(idx, pd.Timestamp("2020-01-01"))


def test_event_paths_measure_cumulative_log_moves_from_origin():
    y = _prices()
    paths = event_paths(y, _events([("opec_cut", "2020-02-05", "e", False)]), horizon=3)
    row = paths.iloc[0]
    assert row["origin"] == pd.Timestamp("2020-02-02")
    np.testing.assert_allclose([row["w1"], row["w2"], row["w3"]], [0.01, 0.02, 0.03])


def test_events_without_full_horizon_are_dropped():
    y = _prices(n=20)
    late = y.index[-3] + pd.Timedelta(days=2)
    paths = event_paths(y, _events([("opec_cut", late, "late", False)]), horizon=12)
    assert paths.empty


def test_later_prices_do_not_change_an_events_path():
    y = _prices(n=40)
    ev = _events([("opec_cut", "2020-02-05", "e", False)])
    short = event_paths(y.iloc[:20], ev, horizon=5)
    longer = event_paths(y, ev, horizon=5)
    pd.testing.assert_series_equal(short.iloc[0][week_cols(5)], longer.iloc[0][week_cols(5)])


# ── Analogue paths ─────────────────────────────────────────────────


def _toy_paths():
    rows = []
    for d, move, conf in [("2010-01-01", 0.10, False), ("2011-01-01", 0.20, False), ("2012-01-01", -0.50, True)]:
        rows.append({"scenario": "opec_cut", "date": pd.Timestamp(d), "confounded": conf,
                     **{f"w{k}": move * k / 2 for k in (1, 2)}})
    return pd.DataFrame(rows)


def test_analogue_path_is_median_with_min_max():
    a = analogue_path(_toy_paths(), "opec_cut", horizon=2)
    np.testing.assert_allclose(a["median"], [0.05, 0.10])
    np.testing.assert_allclose(a["low"], [-0.25, -0.50])
    np.testing.assert_allclose(a["high"], [0.10, 0.20])
    assert a["n_events"] == 3


def test_leave_one_out_excludes_the_held_out_event():
    a = analogue_path(_toy_paths(), "opec_cut", horizon=2, exclude_date="2011-01-01")
    assert a["n_events"] == 2
    np.testing.assert_allclose(a["median"], [(0.05 - 0.25) / 2, (0.10 - 0.50) / 2])


def test_sensitivity_drops_confounded_events():
    a = analogue_path(_toy_paths(), "opec_cut", horizon=2, include_confounded=False)
    assert a["n_events"] == 2
    np.testing.assert_allclose(a["high"], [0.10, 0.20])


def test_unknown_or_empty_scenario_raises():
    with pytest.raises(ValueError):
        analogue_path(_toy_paths(), "demand_boom", horizon=2)


# ── The real catalogue ─────────────────────────────────────────────


def test_catalogue_is_valid_and_covers_every_scenario():
    events = load_events()
    assert set(events["scenario"]) == set(llm_explainer.AVAILABLE_SCENARIOS)
    assert events.groupby("scenario").size().min() >= 4
    assert (events["why_it_qualifies"].str.len() > 0).all()


def test_every_catalogued_event_has_a_full_12_week_path():
    y, _ = load_dataset()
    events = load_events()
    paths = event_paths(y, events, horizon=12)
    assert len(paths) == len(events)
    assert paths[week_cols(12)].notna().all().all()


def test_events_before_the_data_starts_are_skipped_not_fatal():
    y = _prices(n=40)
    events = _events([("opec_cut", "2019-06-01", "too early", False), ("opec_cut", "2020-02-05", "ok", False)])
    paths = event_paths(y, events, horizon=3)
    assert list(paths["event"]) == ["ok"]


def test_non_positive_prices_are_rejected():
    y = _prices(n=20)
    y.iloc[5] = 0.0
    with pytest.raises(ValueError):
        event_paths(y, _events([("opec_cut", "2020-02-05", "e", False)]), horizon=3)
