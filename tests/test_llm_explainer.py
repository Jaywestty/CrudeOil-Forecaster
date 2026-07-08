import json
import pytest


class TestParseUserQuery:

    def test_valid_response_is_parsed(self, make_explainer):
        payload = json.dumps({
            "scenario_key": "opec_cut",
            "magnitude_modifier": 1.5,
            "confidence": "high",
            "reasoning": "Clear supply cut scenario",
            "scenario_context": "OPEC announces a 10% cut",
            "specific_entity": None,
            "forecast_weeks": 12
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_user_query("What if OPEC cuts production?")

        assert result["scenario_key"] == "opec_cut"
        assert result["magnitude_modifier"] == 1.5

    def test_markdown_fenced_response_is_cleaned(self, make_explainer):
        payload = "```json\n" + json.dumps({
            "scenario_key": "demand_boom",
            "magnitude_modifier": 1.0,
            "confidence": "medium",
            "reasoning": "China reopening",
            "scenario_context": "China reopens strongly",
            "specific_entity": "China",
            "forecast_weeks": 12
        }) + "\n```"
        explainer, _ = make_explainer(payload)

        result = explainer.parse_user_query("China reopens")

        assert result["scenario_key"] == "demand_boom"

    def test_list_response_triggers_fallback(self, make_explainer):
        explainer, _ = make_explainer(json.dumps(["geopolitical_tension"]))

        result = explainer.parse_user_query("something ambiguous")

        assert result["scenario_key"] == "geopolitical_tension"
        assert result["confidence"] == "low"
        assert result["scenario_context"] == "something ambiguous"

    def test_malformed_json_triggers_fallback(self, make_explainer):
        explainer, _ = make_explainer("not valid json at all")

        result = explainer.parse_user_query("Russia halts exports")

        assert result["scenario_key"] == "geopolitical_tension"
        assert result["confidence"] == "low"
        assert result["scenario_context"] == "Russia halts exports"

    def test_invalid_scenario_key_falls_back_to_default(self, make_explainer):
        payload = json.dumps({
            "scenario_key": "not_a_real_scenario",
            "magnitude_modifier": 1.0,
            "confidence": "high",
            "reasoning": "irrelevant",
            "scenario_context": "",
            "specific_entity": None,
            "forecast_weeks": 12
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_user_query("Some vague query")

        assert result["scenario_key"] == "geopolitical_tension"
        assert result["confidence"] == "low"
        assert result["scenario_context"] == "Some vague query"

    def test_missing_scenario_context_is_filled_with_query(self, make_explainer):
        payload = json.dumps({
            "scenario_key": "rate_hike",
            "magnitude_modifier": 1.0,
            "confidence": "high",
            "reasoning": "Fed hike",
            "scenario_context": "",
            "specific_entity": None,
            "forecast_weeks": 12
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_user_query("Fed raises rates")

        assert result["scenario_context"] == "Fed raises rates"


class TestParseQueryProbabilistic:

    def test_probabilities_are_normalized_to_one(self, make_explainer):
        payload = json.dumps({
            "probabilities": {"opec_cut": 0.6, "geopolitical_tension": 0.6},
            "primary_driver": "opec_cut",
            "confidence": "high",
            "reasoning": "Both supply factors present",
            "scenario_context": "OPEC cuts and tensions rise",
            "specific_entity": None
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_query_probabilistic("OPEC cuts and tensions rise")

        total = sum(result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_invalid_scenario_keys_are_filtered_out(self, make_explainer):
        payload = json.dumps({
            "probabilities": {"opec_cut": 0.5, "made_up_scenario": 0.5},
            "primary_driver": "opec_cut",
            "confidence": "medium",
            "reasoning": "test",
            "scenario_context": "test query",
            "specific_entity": None
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_query_probabilistic("test query")

        assert "made_up_scenario" not in result["probabilities"]
        assert result["probabilities"]["opec_cut"] == pytest.approx(1.0, abs=0.001)

    def test_empty_probabilities_falls_back_to_default(self, make_explainer):
        payload = json.dumps({
            "probabilities": {"made_up_scenario": 1.0},
            "primary_driver": "made_up_scenario",
            "confidence": "low",
            "reasoning": "test",
            "scenario_context": "test query",
            "specific_entity": None
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_query_probabilistic("test query")

        assert result["probabilities"] == {"geopolitical_tension": 1.0}

    def test_malformed_json_triggers_fallback(self, make_explainer):
        explainer, _ = make_explainer("this is not json")

        result = explainer.parse_query_probabilistic("some query")

        assert result["probabilities"] == {"geopolitical_tension": 1.0}
        assert result["primary_driver"] == "geopolitical_tension"
        assert result["scenario_context"] == "some query"

    def test_missing_scenario_context_is_filled_with_query(self, make_explainer):
        payload = json.dumps({
            "probabilities": {"demand_boom": 1.0},
            "primary_driver": "demand_boom",
            "confidence": "high",
            "reasoning": "test",
            "scenario_context": "",
            "specific_entity": None
        })
        explainer, _ = make_explainer(payload)

        result = explainer.parse_query_probabilistic("China demand surges")

        assert result["scenario_context"] == "China demand surges"