"""Tests for the LLM evaluation harness (step 5b). No network calls."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import v2_payload
from evaluation import run_llm_eval as ev
from llm_schema import SCENARIO_KEYS


def test_case_file_is_well_formed():
    cases = ev.load_cases()
    assert len(cases) == 40 and len({c["id"] for c in cases}) == 40
    for c in cases:
        exp = c["expect"]
        assert set(exp.get("primary_any", [])) <= set(SCENARIO_KEYS)
        assert set(exp.get("min_weight", {})) <= set(SCENARIO_KEYS)
        assert set(exp.get("max_weight", {})) <= set(SCENARIO_KEYS)
        assert "in_scope" in exp
    # forbidden words must not appear inside ordinary oil vocabulary
    for c in cases:
        for word in c["expect"].get("forbidden_text", []):
            for normal in ("barrel", "brent", "crude", "arrangement", "embargo"):
                assert word.lower() not in normal


def test_score_checks_each_expectation():
    case = {"expect": {"primary_any": ["opec_cut"], "min_weight": {"opec_cut": 0.5},
                       "max_weight": {"demand_boom": 0.2}, "in_scope": True, "forbidden_text": ["HACKED"]}}
    good = {"probabilities": {"opec_cut": 0.8, "demand_boom": 0.2}, "primary_driver": "opec_cut",
            "in_scope": True, "reasoning": "Supply cut", "scenario_context": "q"}
    assert ev.score(case, good)["passed"]
    assert not ev.score(case, {**good, "primary_driver": "demand_boom"})["passed"]
    assert not ev.score(case, {**good, "reasoning": "hacked!"})["passed"]  # case-insensitive
    assert ev.score(case, {**good, "in_scope": None})["checks"].get("in_scope") is None  # v1 has no flag


def test_bearish_check_sums_recession_and_rates():
    case = {"expect": {"bearish_min": 0.5, "in_scope": True}}
    r = {"probabilities": {"global_recession": 0.3, "rate_hike": 0.25, "opec_cut": 0.45}, "in_scope": True}
    assert ev.score(case, r)["checks"]["bearish"] is True


def test_mean_pairwise_l1():
    a, b, c = np.array([1, 0, 0, 0, 0.]), np.array([0, 1, 0, 0, 0.]), np.array([1, 0, 0, 0, 0.])
    assert ev.mean_pairwise_l1([a, b, c]) == pytest.approx((2 + 0 + 2) / 3)
    assert ev.mean_pairwise_l1([a]) is None


def _rows(**over):
    base = dict(version="v2", case_id="c", category="single", repeat=0, valid=True, infra_error=False,
                error=None, primary="opec_cut", primary_ok=True, in_scope=True, passed=True,
                checks="{}", probabilities={"opec_cut": 1.0}, latency_s=1.0)
    base.update(over)
    return base


def test_summary_excludes_infrastructure_errors_from_validity():
    rows = pd.DataFrame([_rows(), _rows(repeat=1, valid=False, infra_error=True, error="RateLimitError",
                                        primary_ok=None, passed=False, probabilities={})])
    m = ev.summarise(rows)
    assert m["infra_errors"] == 1 and m["T1_valid"] == 1.0


def test_summary_consistency_and_offtopic_rates():
    rows = pd.DataFrame([
        _rows(case_id="a", repeat=0, probabilities={"opec_cut": 1.0}),
        _rows(case_id="a", repeat=1, probabilities={"opec_cut": 0.5, "geopolitical_tension": 0.5}),
        _rows(case_id="o", category="off_topic", in_scope=False, primary_ok=None),
        _rows(case_id="m", category="mixed", in_scope=False, primary_ok=None),
    ])
    m = ev.summarise(rows)
    assert m["T4_consistency_l1"] == pytest.approx(1.0)
    # real questions: two runs of "a" (kept) and one of "m" (wrongly rejected) -> 1/3
    assert m["T6_offtopic_recall"] == 1.0 and m["T6_false_reject"] == pytest.approx(1 / 3)


def test_t6_decides_enforce_only_when_both_thresholds_met():
    enforce = {"T6_offtopic_recall": 0.95, "T6_false_reject": 0.03}
    flag = {"T6_offtopic_recall": 0.95, "T6_false_reject": 0.10}
    assert ev.verdicts(enforce)[-1]["met"] == "enforce rejection"
    assert ev.verdicts(flag)[-1]["met"] == "flag only"


def test_run_end_to_end_with_a_simulated_model(make_explainer):
    explainer, client = make_explainer(v2_payload(opec_cut=0.9, geopolitical_tension=0.1))
    cases = [c for c in ev.load_cases() if c["id"] in ("opec-1", "geo-1")]
    waits = []
    rows = ev.run(cases, {"v2": (ev.parse_v2, 2), "v1": (ev.parse_v1, 1)}, explainer,
                  min_interval=5.0, sleep=waits.append)
    assert len(rows) == 6
    v2 = rows[rows["version"] == "v2"]
    assert v2["valid"].all()
    assert v2.set_index(["case_id", "repeat"])["primary_ok"].to_dict() == {
        ("opec-1", 0): True, ("opec-1", 1): True, ("geo-1", 0): False, ("geo-1", 1): False}
    assert not rows[rows["version"] == "v1"]["valid"].any()  # v1 cannot read the v2 reply format
    assert waits and all(w <= 5.0 for w in waits)  # calls are spaced out
    report = ev.render({"model": "m", "cases": 2, "v2_repeats": 2, "v1_repeats": 1, "run_at": "now"},
                       {v: ev.summarise(rows[rows["version"] == v]) for v in ("v2", "v1")}, rows)
    assert "against the targets" in report and "geo-1" in report


def test_main_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert ev.main([]) == 1
