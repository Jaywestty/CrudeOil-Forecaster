import pytest
import json
from unittest.mock import MagicMock


class TestHealthAndMeta:

    def test_root_returns_online_status(self, api_client):
        client, _, _ = api_client
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "online"

    def test_health_endpoint(self, api_client):
        client, _, _ = api_client
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "online"}

    def test_scenarios_lists_all_five(self, api_client):
        client, _, _ = api_client
        response = client.get("/scenarios")

        assert response.status_code == 200
        keys = [s["key"] for s in response.json()["scenarios"]]
        assert set(keys) == {
            "opec_cut", "global_recession", "rate_hike",
            "geopolitical_tension", "demand_boom"
        }


class TestCurrentPriceEndpoint:

    def test_uses_live_price_when_fred_succeeds(self, api_client):
        client, app_module, _ = api_client
        client.app_ref = app_module
        app_module.fetch_live_brent_price = lambda: (88.5, "2026-07-01", "FRED live")

        response = client.get("/current-price")

        assert response.status_code == 200
        body = response.json()
        assert body["price"] == 88.5
        assert body["source"] == "FRED live"

    def test_falls_back_to_dataset_when_fred_fails(self, api_client):
        client, app_module, _ = api_client
        app_module.fetch_live_brent_price = lambda: (None, None, "dataset")

        response = client.get("/current-price")

        assert response.status_code == 200
        body = response.json()
        assert body["price"] == 84.0
        assert "dataset" in body["source"]


class TestSimulateDirectEndpoint:

    def test_valid_scenario_key_returns_200(self, api_client):
        client, _, _ = api_client
        response = client.post("/simulate-direct?scenario_key=opec_cut&forecast_weeks=12")

        assert response.status_code == 200
        assert response.json()["scenario_name"] == "OPEC Production Cut (10%)"

    def test_unknown_scenario_key_returns_400(self, api_client):
        client, _, _ = api_client
        response = client.post("/simulate-direct?scenario_key=not_real&forecast_weeks=12")

        assert response.status_code == 400


class TestSimulateEndpoint:

    def test_valid_query_returns_full_response(self, api_client):
        client, _, _ = api_client
        response = client.post(
            "/simulate", json={"query": "What if OPEC cuts?", "forecast_weeks": 12}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["scenario_key"] == "opec_cut"
        assert len(body["weekly_forecasts"]) == 12
        assert "explanation" in body
        assert "uncertainty_note" in body


class TestSimulateProbabilisticEndpoint:

    def test_valid_query_returns_weighted_range(self, api_client):
        client, _, fake_groq_client = api_client

        prob_payload = json.dumps({
            "probabilities": {"opec_cut": 0.6, "geopolitical_tension": 0.4},
            "primary_driver": "opec_cut",
            "confidence": "high",
            "reasoning": "test",
            "scenario_context": "test query",
            "specific_entity": None
        })
        fake_message = MagicMock()
        fake_message.content = prob_payload
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_completion = MagicMock()
        fake_completion.choices = [fake_choice]
        fake_groq_client.chat.completions.create.return_value = fake_completion

        response = client.post(
            "/simulate-probabilistic",
            json={"query": "OPEC cuts and tensions rise", "forecast_weeks": 12}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["price_low"] <= body["price_expected"] <= body["price_high"]
        assert "opec_cut" in body["scenario_breakdown"]
        assert "geopolitical_tension" in body["scenario_breakdown"]
    def test_response_includes_v21_fields(self, api_client):
        client, _, fake_groq_client = api_client
        payload = json.dumps({
            "probabilities": {"opec_cut": 0.97, "rate_hike": 0.03},
            "primary_driver": "opec_cut",
            "confidence": "high",
            "reasoning": "test",
            "scenario_context": "test query",
            "specific_entity": None
        })
        msg = MagicMock()
        msg.content = payload
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        fake_groq_client.chat.completions.create.return_value = completion

        body = client.post("/simulate-probabilistic", json={"query": "q", "forecast_weeks": 8}).json()

        assert len(body["scenario_breakdown"]["opec_cut"]["forecast"]) == 8
        assert len(body["statistical_band"]["level_95"]["lower"]) == 8
        assert "backtest" in body["statistical_band"]["note"]
        assert sum(body["final_probabilities"].values()) == pytest.approx(1.0, abs=0.001)
        assert "rate_hike" in body["dropped_scenarios"]
        assert body["model_info"]["llm"].startswith("openai/gpt-oss-120b")


class TestRootModelInfo:

    def test_root_reports_models(self, api_client):
        client, _, _ = api_client
        info = client.get("/").json()["model_info"]
        assert "SARIMAX" in info["forecaster"]
        assert "gpt-oss-120b" in info["llm"]
