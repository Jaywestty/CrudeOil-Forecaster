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
        engine, _ = make_engine([baseline], latest_overrides={"vix_diff": 30.0})

        probs = {"geopolitical_tension": 0.5, "global_recession": 0.3, "demand_boom": 0.2}
        adjusted, log = engine._adjust_probabilities_with_macro(probs)

        assert adjusted["geopolitical_tension"] > probs["geopolitical_tension"] / sum(probs.values())
        assert "vix_high_geo" in log
        assert "vix_high_rec" in log
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.001)

    def test_low_vix_boosts_demand_boom(self, make_engine):
        idx = _index()
        baseline = FakeForecastResult([80.0] * 12, idx)
        engine, _ = make_engine([baseline], latest_overrides={"vix_diff": 5.0})

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

    def test_low_weight_scenarios_are_skipped(self, make_engine):
        idx = _index()
        # Sequence: 1 outer baseline call, then per surviving scenario
        # (only geopolitical_tension clears 5%): 1 baseline + 1 shocked
        outer_baseline = FakeForecastResult([80.0] * 12, idx)
        scenario_baseline = FakeForecastResult([80.0] * 12, idx)
        shocked = FakeForecastResult([85.0] * 12, idx)

        engine, fake_model = make_engine(
            [outer_baseline, scenario_baseline, shocked]
        )

        probs = {"geopolitical_tension": 0.97, "demand_boom": 0.03}
        result = engine.run_probabilistic_scenario(probs, forecast_weeks=12)

        assert "geopolitical_tension" in result["scenario_results"]
        assert "demand_boom" not in result["scenario_results"]

    def test_expected_price_is_clamped_within_range(self, make_engine):
        idx = _index()
        # Sequence: 1 outer baseline, then 2 scenarios each contributing
        # 1 baseline + 1 shocked call = 5 calls total.
        outer_baseline = FakeForecastResult([80.0] * 12, idx)
        geo_baseline = FakeForecastResult([80.0] * 12, idx)
        geo_shocked = FakeForecastResult([90.0] * 12, idx)
        rec_baseline = FakeForecastResult([80.0] * 12, idx)
        rec_shocked = FakeForecastResult([70.0] * 12, idx)

        engine, _ = make_engine(
            [outer_baseline, geo_baseline, geo_shocked, rec_baseline, rec_shocked]
        )

        probs = {"geopolitical_tension": 0.5, "global_recession": 0.5}
        result = engine.run_probabilistic_scenario(probs, forecast_weeks=12)

        assert result["price_low"] <= result["price_expected"] <= result["price_high"]