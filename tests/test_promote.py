"""Tests for the promotion gate and model registry (step 3b)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

import modeling
from pipeline import promote
from pipeline.build import PRODUCTION_RECIPE


class FakeForecast:
    def __init__(self, mean, half=5.0):
        self.predicted_mean = np.asarray(mean, dtype=float)
        self._half = half

    def conf_int(self, alpha=0.05):
        return np.column_stack([self.predicted_mean - self._half, self.predicted_mean + self._half])


class FakeResults:
    """Stands in for a fitted SARIMAX: forecasts a flat path at `level`."""

    def __init__(self, level=90.0, converged=True, nobs=100):
        self.level, self.nobs = level, nobs
        self.mle_retvals = {"converged": converged, "iterations": 30}
        self.params = np.array([0.1, -0.2, 1.0])
        self.aic = 1234.5

    def get_forecast(self, steps, exog=None):
        return FakeForecast([self.level] * steps)

    def apply(self, y, exog=None):
        return FakeResults(level=float(y.iloc[-1]))  # behaves like a no-change forecast


# ── Fixtures: current and staged dataset folders ───────────────────


def _write_dir(path, weekly, transformed, recipe=PRODUCTION_RECIPE.name, ok=True):
    path.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(path / promote.WEEKLY)
    transformed.to_csv(path / promote.TRANSFORMED)
    manifest = {"recipe": recipe, "validation": {"ok": ok, "errors": [], "warnings": []},
                "weekly": {"first": str(weekly.index[0].date()), "last": str(weekly.index[-1].date()),
                           "last_brent": float(weekly["brent"].iloc[-1])}}
    (path / promote.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def dirs(tmp_path):
    w = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True)
    current, staging = tmp_path / "current", tmp_path / "staging"
    _write_dir(current, w.iloc[:-10], t.iloc[:-10])
    _write_dir(staging, w, t)  # same history + 10 new weeks
    return {"current": current, "staging": staging, "models": tmp_path / "models",
            "report": tmp_path / "report.md", "w": w, "t": t}


def _run(d, apply=False, overrides=(), reason=None, fit=lambda w, t: FakeResults()):
    return promote.run(d["current"], d["staging"], d["models"] / "model.pkl", d["models"] / "registry.json",
                       d["report"], apply, list(overrides), reason, fit_fn=fit,
                       serving_fn=lambda results, w, t: results)


# ── Individual gates ───────────────────────────────────────────────


def test_g1_requires_valid_data_with_the_production_recipe():
    assert promote.gate_data({"validation": {"ok": True}, "recipe": PRODUCTION_RECIPE.name}, PRODUCTION_RECIPE.name).passed
    assert not promote.gate_data({"validation": {"ok": False}, "recipe": PRODUCTION_RECIPE.name}, PRODUCTION_RECIPE.name).passed
    assert not promote.gate_data({"validation": {"ok": True}, "recipe": "v1-notebook"}, PRODUCTION_RECIPE.name).passed


def test_g2_allows_new_dates_but_not_removals_or_big_revisions(dirs):
    w = dirs["w"]
    assert promote.gate_continuity(w.iloc[:-10], w).passed
    assert not promote.gate_continuity(w, w.iloc[:-1]).passed  # a date disappeared
    revised = w.copy()
    revised.iloc[:100, 2] *= 1.01  # ~9% of industrial production rows revised
    gate = promote.gate_continuity(w, revised)
    assert not gate.passed and "industrial_prod" in gate.detail


def test_g3_requires_convergence():
    assert promote.gate_fit(modeling.fit_summary(FakeResults())).passed
    assert not promote.gate_fit(modeling.fit_summary(FakeResults(converged=False))).passed


def test_g4_rejects_implausible_or_non_finite_forecasts(dirs):
    t = dirs["t"]
    assert promote.gate_sanity(FakeResults(level=90.0), t, last_price=95.0).passed
    assert not promote.gate_sanity(FakeResults(level=300.0), t, last_price=95.0).passed
    assert not promote.gate_sanity(FakeResults(level=np.nan), t, last_price=95.0).passed


def test_monitoring_compares_current_model_with_no_change(dirs):
    w, t = dirs["w"], dirs["t"]
    m = promote.monitor_champion(FakeResults(), w, t, w.index[-10:])
    assert m["weeks"] == 10 and m["mae_vs_naive"] == pytest.approx(1.0)  # the fake *is* a no-change forecast
    assert promote.monitor_champion(None, w, t, w.index[-10:]) is None


# ── The whole gate ─────────────────────────────────────────────────


def test_dry_run_changes_nothing(dirs):
    before = (dirs["current"] / promote.WEEKLY).read_bytes()
    assert _run(dirs) == 0
    assert (dirs["current"] / promote.WEEKLY).read_bytes() == before
    assert not (dirs["models"] / "registry.json").exists()
    assert "Approved (dry run)" in dirs["report"].read_text(encoding="utf-8")


def test_apply_promotes_data_model_and_registry_chain(dirs):
    assert _run(dirs, apply=True) == 0
    assert (dirs["current"] / promote.WEEKLY).read_bytes() == (dirs["staging"] / promote.WEEKLY).read_bytes()
    assert (dirs["models"] / "model.pkl").exists()
    reg = json.loads((dirs["models"] / "registry.json").read_text(encoding="utf-8"))
    assert len(reg) == 1 and reg[0]["previous_version"] is None
    assert reg[0]["data"]["last"] == str(dirs["w"].index[-1].date())
    assert all(g["passed"] for g in reg[0]["gates"])

    # A second promotion (new data: a revised last price) links to the first,
    # so the history is a chain you can roll back along.
    from pipeline.build import build_transformed
    w2 = dirs["w"].copy()
    w2.iloc[-1, 0] += 1.0
    _write_dir(dirs["staging"], w2, build_transformed(w2))
    assert _run(dirs, apply=True) == 0
    reg = json.loads((dirs["models"] / "registry.json").read_text(encoding="utf-8"))
    assert reg[1]["previous_version"] == reg[0]["version"]


def test_failed_gate_blocks_promotion(dirs):
    before = (dirs["current"] / promote.WEEKLY).read_bytes()
    assert _run(dirs, apply=True, fit=lambda w, t: FakeResults(converged=False)) == 1
    assert (dirs["current"] / promote.WEEKLY).read_bytes() == before
    assert "Not promoted" in dirs["report"].read_text(encoding="utf-8")


def test_g2_can_be_overridden_only_with_a_reason(dirs):
    revised = dirs["w"].copy()
    revised.iloc[:200, 4] += 0.1  # a recipe change shifts many fed funds rows
    _write_dir(dirs["staging"], revised, dirs["t"])
    assert _run(dirs, apply=True) == 1
    with pytest.raises(SystemExit):
        _run(dirs, apply=True, overrides=["G2"])  # no reason given
    assert _run(dirs, apply=True, overrides=["G2"], reason="reviewed recipe change") == 0
    reg = json.loads((dirs["models"] / "registry.json").read_text(encoding="utf-8"))
    assert reg[0]["override_reason"] == "reviewed recipe change"
    assert [g["overridden"] for g in reg[0]["gates"]] == [False, True, False, False]


def test_broken_fit_cannot_be_overridden(dirs):
    with pytest.raises(SystemExit):
        _run(dirs, apply=True, overrides=["G3"], reason="please",
             fit=lambda w, t: FakeResults(converged=False))


def test_failure_while_preparing_leaves_live_files_untouched(dirs, monkeypatch):
    before = (dirs["current"] / promote.WEEKLY).read_bytes()

    class Unpicklable(FakeResults):
        def __reduce__(self):
            raise RuntimeError("cannot serialise model")

    with pytest.raises(RuntimeError):
        _run(dirs, apply=True, fit=lambda w, t: Unpicklable())
    assert (dirs["current"] / promote.WEEKLY).read_bytes() == before
    assert not (dirs["models"] / "registry.json").exists()


def test_identical_staged_data_is_not_promoted_again(dirs):
    for name in (promote.WEEKLY, promote.TRANSFORMED):
        shutil.copyfile(dirs["staging"] / name, dirs["current"] / name)
    assert not promote.has_new_data(dirs["current"], dirs["staging"])
    assert _run(dirs, apply=True) == 0
    assert not (dirs["models"] / "registry.json").exists()  # no duplicate entry
