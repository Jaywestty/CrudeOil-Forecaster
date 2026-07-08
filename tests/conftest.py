import pandas as pd
import pytest
from unittest.mock import MagicMock

import scenario_engine
import llm_explainer

import importlib
import sys
from fastapi.testclient import TestClient
import json


class FakeForecastResult:
    """Stand-in for the object returned by SARIMAXResults.get_forecast()."""

    def __init__(self, mean_values, index):
        self.predicted_mean = pd.Series(mean_values, index=index)

    def conf_int(self, alpha=0.05):
        lower = self.predicted_mean - 2.0
        upper = self.predicted_mean + 2.0
        return pd.DataFrame({"lower": lower, "upper": upper})


def _format_price_change(baseline, shocked):
    difference = shocked - baseline
    pct_change = (difference / baseline) * 100 if baseline else 0.0
    sign = "+" if difference >= 0 else ""
    return {
        "baseline": round(float(baseline), 2),
        "shocked": round(float(shocked), 2),
        "difference": round(float(difference), 2),
        "pct_change": round(float(pct_change), 2),
        "formatted": f"{sign}${difference:.2f} ({sign}{pct_change:.1f}%)",
    }


@pytest.fixture
def latest_values():
    return {
        "brent_price": 82.50,
        "dollar_return": 0.0,
        "indpro_return": 0.0,
        "inventory_pct": 0.0,
        "fed_funds_diff": 0.0,
        "vix_diff": 0.0,
    }


@pytest.fixture
def make_engine(monkeypatch, latest_values):
    """
    Factory fixture for ScenarioEngine with all data/model loading mocked.

    forecast_sequence: list of FakeForecastResult objects returned in
    order on successive calls to model.get_forecast(). run_scenario()
    calls get_forecast twice per run (baseline, then shocked), in that
    order, so pass [baseline_result, shocked_result].
    """
    def _make(forecast_sequence, latest_overrides=None):
        values = dict(latest_values)
        if latest_overrides:
            values.update(latest_overrides)

        fake_model = MagicMock()
        fake_model.get_forecast.side_effect = list(forecast_sequence)

        monkeypatch.setattr(scenario_engine, "load_data", lambda: pd.DataFrame())
        monkeypatch.setattr(scenario_engine, "load_transformed_data", lambda: pd.DataFrame())
        monkeypatch.setattr(scenario_engine, "load_model", lambda: fake_model)
        monkeypatch.setattr(scenario_engine, "get_latest_values", lambda w, t: values)
        monkeypatch.setattr(scenario_engine, "format_price_change", _format_price_change)

        engine = scenario_engine.ScenarioEngine()
        return engine, fake_model

    return _make


@pytest.fixture
def make_explainer(monkeypatch):
    """
    Factory fixture for LLMExplainer with the Groq client mocked.
    Pass the raw string the fake LLM should return as message content.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def _make(chat_response_content):
        fake_message = MagicMock()
        fake_message.content = chat_response_content
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        monkeypatch.setattr(llm_explainer, "Groq", lambda api_key: fake_client)
        explainer = llm_explainer.LLMExplainer()
        return explainer, fake_client

    return _make



@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    weekly_index = pd.date_range("2024-01-01", periods=5, freq="W")
    weekly_data = pd.DataFrame({"brent": [80, 81, 82, 83, 84]}, index=weekly_index)

    latest_values = {
        "brent_price": 84.0,
        "dollar_return": 0.0,
        "indpro_return": 0.0,
        "inventory_pct": 0.0,
        "fed_funds_diff": 0.0,
        "vix_diff": 0.0,
    }

    def fake_forecast(steps, exog):
        idx = pd.date_range("2024-02-01", periods=steps, freq="W")
        result = MagicMock()
        result.predicted_mean = pd.Series([85.0] * steps, index=idx)
        result.conf_int.return_value = pd.DataFrame(
            {"lower": [83.0] * steps, "upper": [87.0] * steps}, index=idx
        )
        return result

    fake_model = MagicMock()
    fake_model.get_forecast.side_effect = fake_forecast

    monkeypatch.setattr(scenario_engine, "load_data", lambda: weekly_data)
    monkeypatch.setattr(scenario_engine, "load_transformed_data", lambda: pd.DataFrame())
    monkeypatch.setattr(scenario_engine, "load_model", lambda: fake_model)
    monkeypatch.setattr(scenario_engine, "get_latest_values", lambda w, t: latest_values)

    fake_message = MagicMock()
    fake_message.content = json.dumps({
        "scenario_key": "opec_cut",
        "magnitude_modifier": 1.0,
        "confidence": "high",
        "reasoning": "test",
        "scenario_context": "test query",
        "specific_entity": None,
        "forecast_weeks": 12
    })
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create.return_value = fake_completion

    monkeypatch.setattr(llm_explainer, "Groq", lambda api_key: fake_groq_client)

    sys.modules.pop("app", None)
    import app as app_module
    importlib.reload(app_module)

    client = TestClient(app_module.app)
    yield client, app_module, fake_groq_client

    sys.modules.pop("app", None)