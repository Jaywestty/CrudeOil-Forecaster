"""Step 4a: input limits, caching, observability and error handling."""

import json
import logging
from unittest.mock import MagicMock

import groq
import pytest

from serving.cache import TTLCache, make_key, normalise_query
from serving.observability import JsonFormatter, Metrics, request_id_from

PROB_PAYLOAD = json.dumps({
    "probabilities": {"opec_cut": 0.6, "geopolitical_tension": 0.4},
    "primary_driver": "opec_cut", "confidence": "high", "reasoning": "test",
    "scenario_context": "test query", "specific_entity": None,
})


def _llm_returns(fake_client, content):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    fake_client.chat.completions.create.side_effect = None
    fake_client.chat.completions.create.return_value = completion


def _simulate(client, query="What if OPEC cuts production?", weeks=12, **kw):
    return client.post("/simulate-probabilistic", json={"query": query, "forecast_weeks": weeks}, **kw)


# ── Cache unit tests ───────────────────────────────────────────────


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_cache_expires_entries_after_ttl():
    clock = FakeClock()
    c = TTLCache(maxsize=10, ttl_seconds=60, clock=clock)
    c.set("k", 1)
    clock.t = 59
    assert c.get("k") == 1
    clock.t = 60
    assert c.get("k") is None


def test_cache_evicts_least_recently_used():
    c = TTLCache(maxsize=2, ttl_seconds=60, clock=FakeClock())
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")      # "a" is now most recently used
    c.set("c", 3)   # evicts "b"
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    assert c.stats()["evictions"] == 1


def test_cache_key_ignores_case_and_spacing_but_not_what_changes_answers():
    base = make_key("What if OPEC cuts?", 12, "2026-11-15", "m")
    assert make_key("  what IF   opec cuts? ", 12, "2026-11-15", "m") == base
    assert make_key("What if OPEC cuts?", 8, "2026-11-15", "m") != base       # horizon
    assert make_key("What if OPEC cuts?", 12, "2026-11-22", "m") != base      # new data
    assert make_key("What if OPEC cuts?", 12, "2026-11-15", "other") != base  # new model
    assert normalise_query(" A \n b ") == "a b"


# ── Observability unit tests ───────────────────────────────────────


def test_request_ids_are_reused_only_when_safe_to_log():
    assert request_id_from("abc-123") == "abc-123"
    minted = request_id_from("bad id\ninjected")
    assert minted != "bad id\ninjected" and len(minted) == 16
    assert len(request_id_from(None)) == 16


def test_json_log_lines_are_valid_json_with_fields():
    record = logging.LogRecord("oil_api", logging.INFO, __file__, 1, "request", None, None)
    record.fields = {"request_id": "r1", "latency_ms": 12.5}
    line = json.loads(JsonFormatter().format(record))
    assert line["msg"] == "request" and line["request_id"] == "r1" and line["latency_ms"] == 12.5


def test_metrics_percentiles_and_fallbacks():
    m = Metrics()
    for ms in range(1, 101):
        m.observe("/x", 200, float(ms))
    m.observe("/x", 500, 5.0)
    m.record_fallbacks(["parse_probabilistic"])
    snap = m.snapshot()
    route = snap["routes"]["/x"]
    assert route["requests"] == 101 and route["errors_5xx"] == 1
    assert 45 <= route["latency_ms"]["p50"] <= 55 and route["latency_ms"]["p95"] >= 90
    assert snap["llm_fallbacks"] == {"parse_probabilistic": 1}


# ── T1: input limits ───────────────────────────────────────────────


@pytest.mark.parametrize("body", [
    {"query": "What if OPEC cuts?", "forecast_weeks": 0},
    {"query": "What if OPEC cuts?", "forecast_weeks": 13},
    {"query": "What if OPEC cuts?", "forecast_weeks": None},
    {"query": "", "forecast_weeks": 12},
    {"query": "   a   ", "forecast_weeks": 12},
    {"query": "x" * 501, "forecast_weeks": 12},
])
def test_invalid_input_is_rejected_before_any_llm_call(api_client, body):
    client, _, fake = api_client
    before = fake.chat.completions.create.call_count
    assert client.post("/simulate-probabilistic", json=body).status_code == 422
    assert fake.chat.completions.create.call_count == before


def test_direct_simulation_horizon_is_bounded(api_client):
    client, _, _ = api_client
    assert client.post("/simulate-direct?scenario_key=opec_cut&forecast_weeks=13").status_code == 422


# ── T2: caching ────────────────────────────────────────────────────


def test_identical_request_is_served_from_cache_without_llm_calls(api_client):
    client, _, fake = api_client
    _llm_returns(fake, PROB_PAYLOAD)
    first = _simulate(client)
    calls_after_first = fake.chat.completions.create.call_count
    second = _simulate(client, query="  what if opec CUTS production? ")  # same question, different spacing
    assert first.status_code == second.status_code == 200
    assert fake.chat.completions.create.call_count == calls_after_first
    assert second.json() == first.json()
    assert client.get("/metrics").json()["cache"]["hits"] == 1


def test_different_horizon_is_not_served_from_cache(api_client):
    client, _, fake = api_client
    _llm_returns(fake, PROB_PAYLOAD)
    _simulate(client, weeks=12)
    calls = fake.chat.completions.create.call_count
    _simulate(client, weeks=8)
    assert fake.chat.completions.create.call_count > calls


def test_degraded_answers_are_flagged_and_not_cached(api_client):
    client, _, fake = api_client
    fake.chat.completions.create.side_effect = groq.GroqError("rate limited")
    first = _simulate(client)
    assert first.status_code == 200 and first.json()["llm_fallback"] is True

    _llm_returns(fake, PROB_PAYLOAD)  # the LLM recovers
    second = _simulate(client)
    assert second.json()["llm_fallback"] is False  # recomputed, not the cached fallback
    fallbacks = client.get("/metrics").json()["llm_fallbacks"]
    assert fallbacks.get("parse_probabilistic", 0) >= 1


# ── T3 / T4: request ids and metrics ───────────────────────────────


def test_every_response_has_a_request_id(api_client):
    client, _, _ = api_client
    r = client.get("/health")
    assert len(r.headers["X-Request-ID"]) == 16
    r = client.get("/health", headers={"X-Request-ID": "trace-42"})
    assert r.headers["X-Request-ID"] == "trace-42"


def test_metrics_report_latency_status_and_cache(api_client):
    client, _, fake = api_client
    _llm_returns(fake, PROB_PAYLOAD)
    _simulate(client)
    client.post("/simulate-probabilistic", json={"query": "", "forecast_weeks": 12})
    m = client.get("/metrics").json()
    route = m["routes"]["/simulate-probabilistic"]
    assert route["by_status"] == {"200": 1, "422": 1}
    assert route["latency_ms"]["p50"] is not None and route["latency_ms"]["p95"] is not None
    assert {"hits", "misses", "hit_rate", "size"} <= set(m["cache"])
    assert "uptime_seconds" in m and "model_info" in m


# ── Errors and shared state ────────────────────────────────────────


def test_server_errors_hide_internal_details(api_client, monkeypatch):
    client, app_module, fake = api_client
    _llm_returns(fake, PROB_PAYLOAD)

    def boom(*a, **k):
        raise RuntimeError("secret connection string")

    monkeypatch.setattr(app_module.engine, "run_probabilistic_scenario", boom)
    r = _simulate(client)
    assert r.status_code == 500
    assert "secret" not in r.text and r.headers["X-Request-ID"] in r.json()["detail"]


def test_simulate_never_mutates_the_shared_scenario_table(api_client):
    client, app_module, fake = api_client
    _llm_returns(fake, json.dumps({
        "scenario_key": "opec_cut", "magnitude_modifier": 2.0, "confidence": "high",
        "reasoning": "t", "scenario_context": "t", "specific_entity": None, "forecast_weeks": 40,
    }))
    before = dict(app_module.SCENARIOS["opec_cut"]["shocks"])
    r = client.post("/simulate", json={"query": "What if OPEC cuts hard?", "forecast_weeks": 12})
    assert r.status_code == 200
    assert app_module.SCENARIOS["opec_cut"]["shocks"] == before
    assert r.json()["shocks_applied"]["inventory_pct"] == pytest.approx(2 * before["inventory_pct"])
    assert len(r.json()["weekly_forecasts"]) == 12  # LLM's 40 weeks clamped to the validated maximum
