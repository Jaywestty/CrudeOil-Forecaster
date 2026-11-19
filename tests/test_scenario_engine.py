import pandas as pd
import pytest

from conftest import FakeForecastResult


def _index(n=12):
    return pd.date_range("2024-01-01", periods=n, freq="W")


class TestDirectionCorrection:

    def test_bullish_scenario_flips_wrong_direction(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        # Model wrongly outputs a price DROP for a bullish scenario
        shocked = FakeForecastResult([78.0] * 12, idx)

        engine, _ = make_engine([baseline, shocked])
        result = engine.run_scenario("opec_cut", forecast_weeks=12)

        corrected_week12 = float(result["shocked_forecast"].iloc[-1])
        baseline_week12 = float(result["baseline_forecast"].iloc[-1])

        assert corrected_week12 > baseline_week12
        assert corrected_week12 == pytest.approx(82.0, abs=0.01)

    def test_bullish_scenario_no_correction_when_already_correct(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        shocked = FakeForecastResult([84.0] * 12, idx)

        engine, _ = make_engine([baseline, shocked])
        result = engine.run_scenario("geopolitical_tension", forecast_weeks=12)

        assert float(result["shocked_forecast"].iloc[-1]) == pytest.approx(84.0, abs=0.01)

    def test_bearish_scenario_flips_wrong_direction(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        # Model wrongly outputs a price RISE for a bearish scenario
        shocked = FakeForecastResult([83.0] * 12, idx)

        engine, _ = make_engine([baseline, shocked])
        result = engine.run_scenario("global_recession", forecast_weeks=12)

        corrected_week12 = float(result["shocked_forecast"].iloc[-1])
        baseline_week12 = float(result["baseline_forecast"].iloc[-1])

        assert corrected_week12 < baseline_week12
        assert corrected_week12 == pytest.approx(77.0, abs=0.01)

    def test_bearish_scenario_no_correction_when_already_correct(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        shocked = FakeForecastResult([75.0] * 12, idx)

        engine, _ = make_engine([baseline, shocked])
        result = engine.run_scenario("rate_hike", forecast_weeks=12)

        assert float(result["shocked_forecast"].iloc[-1]) == pytest.approx(75.0, abs=0.01)

    def test_unknown_scenario_key_raises(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)

        engine, _ = make_engine([baseline])

        with pytest.raises(ValueError):
            engine.run_scenario("not_a_real_scenario", forecast_weeks=12)


class TestExogMatrix:

    def test_shocks_are_added_to_latest_values(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)

        engine, _ = make_engine(
            [baseline],
            latest_overrides={"dollar_return": 0.01, "vix_diff": 5.0},
        )

        shocks = {"dollar_return": -0.02, "vix_diff": 2.0}
        exog = engine._build_exog_matrix(shocks, forecast_weeks=6)

        assert len(exog) == 6
        assert exog["dollar_return"].iloc[0] == pytest.approx(-0.01)
        assert exog["vix_diff"].iloc[0] == pytest.approx(7.0)
        assert exog["indpro_return"].iloc[0] == pytest.approx(0.0)

    def test_missing_shock_keys_default_to_zero_offset(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)

        engine, _ = make_engine([baseline], latest_overrides={"fed_funds_diff": 0.25})
        exog = engine._build_exog_matrix({}, forecast_weeks=3)

        assert exog["fed_funds_diff"].iloc[0] == pytest.approx(0.25)


class TestMacroProbabilityAdjustment:

    def test_high_vix_boosts_fear_scenarios(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={"vix": 30.0})

        probs = {"geopolitical_tension": 0.5, "global_recession": 0.3, "demand_boom": 0.2}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert adjusted["geopolitical_tension"] > probs["geopolitical_tension"] / sum(probs.values())
        assert "vix_high_geo" in log
        assert "vix_high_rec" in log
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.001)

    def test_low_vix_boosts_demand_boom(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={"vix": 12.0})

        probs = {"demand_boom": 0.5, "opec_cut": 0.5}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert "vix_low" in log
        assert adjusted["demand_boom"] > 0.5
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.001)

    def test_strong_dollar_boosts_rate_hike_and_recession(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={"dollar_return": 0.02})

        probs = {"rate_hike": 0.4, "global_recession": 0.3, "opec_cut": 0.3}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert "dollar_strong_rate" in log
        assert "dollar_strong_rec" in log
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.001)

    def test_inventory_draw_boosts_supply_scenarios(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={"inventory_pct": -0.05})

        probs = {"opec_cut": 0.5, "geopolitical_tension": 0.5}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert "inventory_draw_opec" in log
        assert "inventory_draw_geo" in log
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.001)

    def test_no_macro_signals_returns_unchanged_ratios(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={
            "vix_diff": 20.0, "dollar_return": 0.0, "inventory_pct": 0.0, "fed_funds_diff": 0.0
        })

        probs = {"opec_cut": 0.6, "demand_boom": 0.4}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert log == {}
        assert adjusted["opec_cut"] == pytest.approx(0.6, abs=0.001)
        assert adjusted["demand_boom"] == pytest.approx(0.4, abs=0.001)


class TestProbabilisticScenario:

    def test_low_weight_scenarios_are_skipped_and_weights_renormalised(self, make_engine):
        idx = _index()
        # The baseline is computed once and reused, so the sequence is:
        # 1 baseline + 1 shocked forecast per surviving scenario.
        baseline = FakeForecastResult([80.0] * 12, idx)
        shocked = FakeForecastResult([85.0] * 12, idx)
        engine, fake_model = make_engine([baseline, shocked])

        probs = {"geopolitical_tension": 0.97, "demand_boom": 0.03}
        result = engine.run_probabilistic_scenario(probs, forecast_weeks=12)

        assert "geopolitical_tension" in result["scenario_results"]
        assert "demand_boom" not in result["scenario_results"]
        assert "demand_boom" in result["dropped_scenarios"]
        assert result["scenario_probabilities"] == {"geopolitical_tension": 1.0}
        # Renormalised weights: the expected path equals the only scenario's path,
        # not 97% of it (which would wrongly pull the price towards $0).
        assert result["weighted_forecast"][-1] == pytest.approx(85.0)
        assert fake_model.get_forecast.call_count == 2

    def test_baseline_is_computed_once_for_all_scenarios(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        geo_shocked = FakeForecastResult([90.0] * 12, idx)
        rec_shocked = FakeForecastResult([70.0] * 12, idx)
        engine, fake_model = make_engine([baseline, geo_shocked, rec_shocked])

        probs = {"geopolitical_tension": 0.5, "global_recession": 0.5}
        result = engine.run_probabilistic_scenario(probs, forecast_weeks=12)

        assert fake_model.get_forecast.call_count == 3
        assert result["scenario_week12"] == {"geopolitical_tension": 90.0, "global_recession": 70.0}
        assert result["price_low"] <= result["price_expected"] <= result["price_high"]
        assert result["price_expected"] == pytest.approx(80.0)

    def test_result_includes_baseline_bands_and_dates(self, make_engine):
        idx = _index()
        engine, _ = make_engine(
            [FakeForecastResult([80.0] * 12, idx), FakeForecastResult([85.0] * 12, idx)]
        )
        result = engine.run_probabilistic_scenario({"opec_cut": 1.0}, forecast_weeks=12)
        assert len(result["baseline_band_80"]["lower"]) == 12
        assert len(result["baseline_band_95"]["upper"]) == 12
        assert result["forecast_dates"] is None  # mocked data has no dates


class TestVixUsesLevel:

    def test_large_weekly_vix_change_alone_does_not_trigger_fear_rule(self, make_engine):
        engine, _ = make_engine([], latest_overrides={"vix_diff": 30.0, "vix": 18.0})
        probs = {"geopolitical_tension": 0.5, "demand_boom": 0.5}
        _, log = engine._adjust_probabilities_with_macro(probs)
        assert "vix_high_geo" not in log and "vix_low" not in log

    def test_missing_vix_level_skips_signal(self, make_engine):
        engine, _ = make_engine([], latest_overrides={"vix_diff": 30.0})
        _, log = engine._adjust_probabilities_with_macro({"geopolitical_tension": 1.0})
        assert not any(k.startswith("vix") for k in log)


class TestForecastOrigin:

    def test_model_is_extended_to_latest_week(self, monkeypatch, latest_values):
        """A model fitted on fewer weeks must be re-filtered over the full dataset."""
        from unittest.mock import MagicMock
        import scenario_engine

        idx = pd.date_range("2020-01-05", periods=10, freq="W-SUN")
        weekly = pd.DataFrame({"brent": range(70, 80)}, index=idx)
        cols = ["dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff"]
        transformed = pd.DataFrame(0.0, index=idx, columns=cols)

        stale = MagicMock()
        stale.nobs = 8  # fitted on 8 of the 10 weeks
        extended = MagicMock()
        stale.apply.return_value = extended

        monkeypatch.setattr(scenario_engine, "load_data", lambda: weekly)
        monkeypatch.setattr(scenario_engine, "load_transformed_data", lambda: transformed)
        monkeypatch.setattr(scenario_engine, "load_model", lambda: stale)
        monkeypatch.setattr(scenario_engine, "get_latest_values", lambda w, t: latest_values)

        engine = scenario_engine.ScenarioEngine()

        assert engine.model is extended
        y_arg = stale.apply.call_args.args[0]
        assert len(y_arg) == 10
        assert engine.forecast_origin == idx[-1]
        assert engine.forecast_dates(2) == ["2020-03-15", "2020-03-22"]

    def test_up_to_date_model_is_not_refiltered(self, monkeypatch, latest_values):
        from unittest.mock import MagicMock
        import scenario_engine

        idx = pd.date_range("2020-01-05", periods=10, freq="W-SUN")
        cols = ["dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff"]
        current = MagicMock()
        current.nobs = 10
        monkeypatch.setattr(scenario_engine, "load_data", lambda: pd.DataFrame({"brent": range(60, 70)}, index=idx))
        monkeypatch.setattr(scenario_engine, "load_transformed_data", lambda: pd.DataFrame(0.0, index=idx, columns=cols))
        monkeypatch.setattr(scenario_engine, "load_model", lambda: current)
        monkeypatch.setattr(scenario_engine, "get_latest_values", lambda w, t: latest_values)

        engine = scenario_engine.ScenarioEngine()
        assert engine.model is current
        current.apply.assert_not_called()


class TestHistoricalContext:

    def _engine(self, monkeypatch, latest_values):
        from unittest.mock import MagicMock
        import scenario_engine

        idx = pd.date_range("2020-01-05", periods=60, freq="W-SUN")
        weekly = pd.DataFrame({"brent": [50.0] * 30 + [60.0] * 30}, index=idx)
        events = pd.DataFrame({
            "scenario": ["opec_cut", "opec_cut", "opec_cut"],
            "date": pd.to_datetime(["2020-03-04", "2020-06-17", "2020-07-08"]),
            "event": ["a", "b", "c"],
            "why_it_qualifies": ["x"] * 3,
            "confounded_by": ["", "", "covid"],
        })
        events["confounded"] = events["confounded_by"] != ""
        model = MagicMock()
        model.get_forecast.side_effect = lambda steps, exog: FakeForecastResult([80.0] * steps, _index(steps))
        monkeypatch.setattr(scenario_engine, "load_data", lambda: weekly)
        monkeypatch.setattr(scenario_engine, "load_transformed_data", lambda: pd.DataFrame())
        monkeypatch.setattr(scenario_engine, "load_model", lambda: model)
        monkeypatch.setattr(scenario_engine, "load_events", lambda: events)
        monkeypatch.setattr(scenario_engine, "get_latest_values", lambda w, t: dict(latest_values, brent_price=100.0))
        return scenario_engine.ScenarioEngine()

    def test_context_summarises_past_events(self, monkeypatch, latest_values):
        engine = self._engine(monkeypatch, latest_values)
        ctx = engine.historical_context("opec_cut", 12)
        # Event a (origin wk 8, price 50) ends at week 20: still 50 -> 0%.
        # Events b and c start at 50 and cross into the 60 regime -> +20%.
        assert ctx["n_events"] == 3 and ctx["week"] == 12
        assert ctx["low_pct"] == 0.0 and ctx["high_pct"] == pytest.approx(20.0)
        assert ctx["median_pct"] == pytest.approx(20.0)
        assert ctx["price_low"] == 100.0 and ctx["price_high"] == pytest.approx(120.0)
        assert [e["confounded"] for e in ctx["events"]] == [False, False, True]

    def test_horizon_beyond_event_data_is_capped(self, monkeypatch, latest_values):
        engine = self._engine(monkeypatch, latest_values)
        assert engine.historical_context("opec_cut", 24)["week"] == 12

    def test_unknown_scenario_has_no_context(self, monkeypatch, latest_values):
        engine = self._engine(monkeypatch, latest_values)
        assert engine.historical_context("demand_boom", 12) is None

    def test_probabilistic_result_includes_context_for_used_scenarios(self, monkeypatch, latest_values):
        engine = self._engine(monkeypatch, latest_values)
        result = engine.run_probabilistic_scenario({"opec_cut": 0.5, "demand_boom": 0.5}, 12)
        assert set(result["historical_context"]) == {"opec_cut"}

    def test_missing_price_data_means_no_context(self, make_engine):
        idx = _index()
        engine, _ = make_engine([FakeForecastResult([80.0] * 12, idx), FakeForecastResult([85.0] * 12, idx)])
        assert engine.event_paths is None
        result = engine.run_probabilistic_scenario({"opec_cut": 1.0}, 12)
        assert result["historical_context"] == {}
