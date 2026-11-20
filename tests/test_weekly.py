"""Tests for the weekly refresh orchestration (step 3c)."""

import json

import pandas as pd
import pytest

from pipeline import promote, weekly


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, staging = tmp_path / "data", tmp_path / "data" / "staging"
    staging.mkdir(parents=True)
    w = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True)
    w.to_csv(data / promote.WEEKLY)
    t.to_csv(data / promote.TRANSFORMED)

    def fake_refresh(argv):
        w.to_csv(staging / promote.WEEKLY)
        t.to_csv(staging / promote.TRANSFORMED)
        (staging / promote.MANIFEST).write_text(json.dumps({"weekly": {"last": "2026-11-15"}}), encoding="utf-8")
        return 0

    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    # CI itself runs with GITHUB_ACTIONS=true, so pin the secret check's inputs.
    monkeypatch.setenv("FRED_API_KEY", "test")
    monkeypatch.setenv("EIA_API_KEY", "test")
    monkeypatch.setattr(weekly, "DATA", data)
    monkeypatch.setattr(weekly, "STAGING", staging)
    monkeypatch.setattr(weekly, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(weekly.refresh, "main", fake_refresh)
    monkeypatch.setattr(weekly.compare, "main", lambda argv: 0)
    return {"data": data, "staging": staging, "out": out, "w": w, "t": t}


def _outputs(path):
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def test_no_new_data_stops_before_the_gate(env, monkeypatch):
    monkeypatch.setattr(weekly.promote, "main", lambda argv: pytest.fail("gate should not run"))
    assert weekly.main() == 0
    assert _outputs(env["out"]) == {"promoted": "false", "data_last": "2026-11-15", "reason": "no-new-data"}


def test_validation_failure_is_reported(env, monkeypatch):
    monkeypatch.setattr(weekly.refresh, "main", lambda argv: 1)
    assert weekly.main() == 1
    assert _outputs(env["out"])["reason"] == "validation-failed"


def test_new_data_runs_gate_and_reports_promotion(env, monkeypatch):
    env["w"].iloc[:-1].to_csv(env["data"] / promote.WEEKLY)  # live data is one week behind
    monkeypatch.setattr(weekly, "ensure_champion", lambda: None)

    def fake_gate(argv):
        assert argv == ["--apply"]  # never overrides automatically
        for name in (promote.WEEKLY, promote.TRANSFORMED):
            (env["data"] / name).write_bytes((env["staging"] / name).read_bytes())
        return 0

    monkeypatch.setattr(weekly.promote, "main", fake_gate)
    assert weekly.main() == 0
    assert _outputs(env["out"])["promoted"] == "true"


def test_gate_failure_is_not_reported_as_promoted(env, monkeypatch):
    env["w"].iloc[:-1].to_csv(env["data"] / promote.WEEKLY)
    monkeypatch.setattr(weekly, "ensure_champion", lambda: None)
    monkeypatch.setattr(weekly.promote, "main", lambda argv: 1)
    assert weekly.main() == 1
    assert _outputs(env["out"]) == {"promoted": "false", "data_last": "2026-11-15", "reason": "gate-failed"}


def test_set_output_is_a_noop_outside_github(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    weekly.set_output(promoted="true")  # must not raise


def test_missing_secrets_fail_fast_on_github_actions(env, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("EIA_API_KEY")
    monkeypatch.setattr(weekly.refresh, "main", lambda argv: pytest.fail("must not fetch without secrets"))
    assert weekly.main() == 1
    assert _outputs(env["out"])["reason"] == "missing-secrets"


def test_missing_secrets_only_warn_locally(env, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("EIA_API_KEY")
    monkeypatch.setattr(weekly.promote, "main", lambda argv: pytest.fail("gate should not run"))
    assert weekly.main() == 0  # proceeds with the keyless fallback; no new data in this fixture
    assert "EIA_API_KEY not set" in capsys.readouterr().out
