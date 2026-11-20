"""Tests for the data pipeline (step 3a)."""

import json

import numpy as np
import pandas as pd
import pytest
import requests

from pipeline import build, compare, refresh, sources
from pipeline.validate import validate


# ── Parsing ────────────────────────────────────────────────────────


def test_parse_fred_csv_drops_missing_markers():
    text = "observation_date,DCOILBRENTEU\n2024-01-02,77.0\n2024-01-03,.\n2024-01-04,78.5\n"
    s = sources.parse_fred_csv(text, "DCOILBRENTEU")
    assert list(s.values) == [77.0, 78.5]
    assert s.index[-1] == pd.Timestamp("2024-01-04")


def test_parse_fred_csv_rejects_unexpected_responses():
    with pytest.raises(ValueError):
        sources.parse_fred_csv("<!DOCTYPE html><html></html>", "WCRSTUS1")


def test_parse_eia_json_sorts_and_dedupes():
    payload = {"response": {"data": [
        {"period": "2024-01-12", "value": 420000}, {"period": "2024-01-05", "value": 421000},
        {"period": "2024-01-12", "value": 420000},
    ]}}
    s = sources.parse_eia_json(payload, "WCRSTUS1")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-12"]


# ── Retries ────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status=200, text="ok"):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def get(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)


def test_get_retries_transient_failures(no_sleep):
    session = FakeSession([requests.ConnectionError("dropped"), FakeResponse(503), FakeResponse(200)])
    assert sources._get(session, "u", {}).status_code == 200
    assert session.calls == 3


def test_get_does_not_retry_client_errors(no_sleep):
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        sources._get(session, "u", {})
    assert session.calls == 1


def test_get_gives_up_after_max_attempts(no_sleep):
    session = FakeSession([requests.Timeout("slow")] * sources.RETRIES)
    with pytest.raises(RuntimeError):
        sources._get(session, "u", {})
    assert session.calls == sources.RETRIES


# ── Build ──────────────────────────────────────────────────────────


def _raw(days=120, start="2024-01-01"):
    daily = pd.date_range(start, periods=days, freq="D")
    business = daily[daily.dayofweek < 5]
    monthly = pd.date_range(start, periods=days // 28 + 1, freq="MS")
    weekly_fri = pd.date_range(start, periods=days // 7, freq="W-FRI")
    return {
        "brent": pd.Series(np.linspace(70, 80, len(business)), index=business),
        "inventories": pd.Series(np.linspace(800_000, 810_000, len(weekly_fri)), index=weekly_fri),
        "industrial_prod": pd.Series(100.0 + np.arange(len(monthly)), index=monthly),
        "dollar_index": pd.Series(np.linspace(110, 112, len(business)), index=business),
        "fed_funds": pd.Series(5.0 + 0.1 * np.arange(len(monthly)), index=monthly),
        "vix": pd.Series(np.linspace(12, 20, len(business)), index=business),
    }


def test_weekly_uses_last_for_prices_and_mean_for_indices():
    raw = _raw()
    weekly = build.build_weekly(raw, end="2024-04-30")
    week = weekly.index[3]
    in_week = raw["brent"][(raw["brent"].index > week - pd.Timedelta(days=7)) & (raw["brent"].index <= week)]
    assert weekly.loc[week, "brent"] == pytest.approx(in_week.iloc[-1])
    vix_week = raw["vix"][(raw["vix"].index > week - pd.Timedelta(days=7)) & (raw["vix"].index <= week)]
    assert weekly.loc[week, "vix"] == pytest.approx(vix_week.mean())
    assert (weekly.index.dayofweek == 6).all()  # weeks end on Sunday


def test_monthly_values_are_carried_forward_through_the_month():
    weekly = build.build_weekly(_raw(), end="2024-04-30")
    feb = weekly.loc["2024-02-04":"2024-02-25", "industrial_prod"]
    assert (feb == 101.0).all()


def test_end_date_excludes_later_observations():
    raw = _raw()
    early = build.build_weekly(raw, end="2024-03-10")
    full = build.build_weekly(raw, end="2024-04-30")
    assert early.index[-1] <= pd.Timestamp("2024-03-10")
    pd.testing.assert_frame_equal(early.iloc[:-1], full.loc[early.index[:-1]])


def test_release_lag_delays_monthly_values():
    raw = _raw()
    v1 = build.build_weekly(raw, end="2024-04-30", recipe=build.RECIPE_V1)
    v2 = build.build_weekly(raw, end="2024-04-30", recipe=build.RECIPE_V2)
    # February's industrial production (101) is only public ~45 days after Feb 1.
    # With the lag, January's value (100) only arrives mid-February, so the
    # lagged dataset also starts later (rows without it are dropped).
    assert v2.index[0] > v1.index[0]
    assert v1.loc["2024-02-25", "industrial_prod"] == 101.0
    assert v2.loc["2024-02-25", "industrial_prod"] == 100.0
    assert v2.loc["2024-03-24", "industrial_prod"] == 101.0


def test_transformed_values():
    weekly = build.build_weekly(_raw(), end="2024-04-30")
    t = build.build_transformed(weekly)
    w0, w1 = weekly.iloc[0], weekly.iloc[1]
    assert t.iloc[0]["brent_return"] == pytest.approx(np.log(w1["brent"] / w0["brent"]))
    assert t.iloc[0]["inventory_pct"] == pytest.approx(w1["inventories"] / w0["inventories"] - 1)
    assert t.iloc[0]["vix_diff"] == pytest.approx(w1["vix"] - w0["vix"])
    assert list(t.columns) == build.TRANSFORMED_COLUMNS


# ── Validation ─────────────────────────────────────────────────────


@pytest.fixture
def committed():
    """The live dataset (whatever was last promoted)."""
    w = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True)
    return w, t


@pytest.fixture
def notebook_dataset():
    """The original Colab-built dataset, frozen as a reproducibility reference."""
    base = "tests/fixtures/notebook_dataset/"
    w = pd.read_csv(base + "oil_macro_weekly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv(base + "oil_macro_transformed.csv", index_col=0, parse_dates=True)
    return w, t


def test_committed_data_passes_validation(committed):
    rep = validate(*committed)
    assert rep.ok, rep.errors
    assert any("2020-03" in w for w in rep.warnings)  # the COVID crash is flagged for review


def test_validation_catches_gaps_nans_bounds_and_inconsistency(committed):
    w, t = committed
    assert not validate(w.drop(w.index[100]), build.build_transformed(w.drop(w.index[100]))).ok
    bad = w.copy()
    bad.iloc[50, 1] = 836.0  # inventories in million barrels by mistake
    assert any("inventories" in e for e in validate(bad, build.build_transformed(bad)).errors)
    tampered = t.copy()
    tampered.iloc[10, 0] += 0.01
    assert any("does not match" in e for e in validate(w, tampered).errors)
    nan = w.copy()
    nan.iloc[5, 0] = np.nan
    assert not validate(nan, t).ok


def test_staleness_is_a_warning_not_an_error(committed):
    rep = validate(*committed, as_of=committed[0].index[-1] + pd.Timedelta(days=60))
    assert rep.ok and any("days old" in w for w in rep.warnings)


# ── Reproducibility against the committed dataset ──────────────────


def test_raw_snapshots_reproduce_the_notebook_dataset(notebook_dataset):
    """Columns without source revisions must match the notebook-built data exactly."""
    w, _ = notebook_dataset
    rebuilt = build.build_weekly(sources.load_snapshots(), end="2025-12-31", recipe=build.RECIPE_V1)
    assert rebuilt.index.equals(w.index)
    for col in ("inventories", "dollar_index", "fed_funds", "vix"):
        np.testing.assert_allclose(rebuilt[col], w[col], rtol=1e-12, err_msg=col)
    # Known source revisions: a few 2006 Brent days and recent industrial production.
    diff = compare.diff_frames(w, rebuilt).set_index("column")
    assert diff.loc["brent", "rows changed"] <= 10
    assert diff.loc["industrial_prod", "first changed"] >= "2025-01-01"


# ── End to end (offline) ───────────────────────────────────────────


def test_offline_refresh_writes_dataset_and_manifest(tmp_path):
    code = refresh.run(end="2025-12-31", recipe_name="v1-notebook", out=tmp_path,
                       raw_dir=sources.RAW_DIR, fetch=False, as_of=pd.Timestamp("2026-01-10"))
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["ok"] and manifest["weekly"]["last"] == "2026-01-04"
    assert set(manifest["sources"]) == set(sources.SERIES)
    assert (tmp_path / "oil_macro_weekly.csv").exists() and (tmp_path / "oil_macro_transformed.csv").exists()


def test_failed_validation_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "validate", lambda *a, **k: type("R", (), {
        "ok": False, "errors": ["boom"], "warnings": [], "as_dict": lambda self: {}})())
    code = refresh.run(end="2025-12-31", recipe_name="v1-notebook", out=tmp_path,
                       raw_dir=sources.RAW_DIR, fetch=False, as_of=pd.Timestamp("2026-01-10"))
    assert code == 1 and not any(tmp_path.iterdir())


def test_partial_final_week_is_dropped_except_in_notebook_recipe():
    raw = _raw()
    wednesday = "2024-03-13"  # week ending Sunday 2024-03-17 has not finished
    v1 = build.build_weekly(raw, end=wednesday, recipe=build.RECIPE_V1)
    v11 = build.build_weekly(raw, end=wednesday, recipe=build.RECIPE_V1_1)
    assert v1.index[-1] == pd.Timestamp("2024-03-17")   # notebook kept the half week
    assert v11.index[-1] == pd.Timestamp("2024-03-10")  # production keeps only ended weeks
    pd.testing.assert_frame_equal(v11, v1.iloc[:-1])


# ── FRED official API (refresh run #1 timed out on the website CSV) ──


class CapturingSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, params, kwargs))
        return self.response


class JsonResponse(FakeResponse):
    def __init__(self, payload, status=200):
        super().__init__(status)
        self._payload = payload

    def json(self):
        return self._payload


def test_parse_fred_api_json_drops_missing_and_reports_errors():
    payload = {"observations": [{"date": "2024-01-02", "value": "77.0"},
                                {"date": "2024-01-03", "value": "."},
                                {"date": "2024-01-04", "value": "78.5"}]}
    s = sources.parse_fred_api_json(payload, "DCOILBRENTEU")
    assert list(s.values) == [77.0, 78.5]
    with pytest.raises(ValueError, match="Bad Request"):
        sources.parse_fred_api_json({"error_message": "Bad Request. Variable api_key is not set."}, "X")


def test_fetch_fred_uses_official_api_when_key_is_set(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k123")
    session = CapturingSession(JsonResponse({"observations": [{"date": "2024-01-02", "value": "77.0"}]}))
    s = sources.fetch_fred("DCOILBRENTEU", session)
    url, params, kwargs = session.calls[0]
    assert url == sources.FRED_API
    assert params == {"series_id": "DCOILBRENTEU", "api_key": "k123", "file_type": "json"}
    assert kwargs["timeout"] == (10, 60)
    assert s.iloc[0] == 77.0 and sources.fred_source() == "FRED API"


def test_fetch_fred_falls_back_to_csv_without_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    session = CapturingSession(FakeResponse(200, text="observation_date,INDPRO\n2024-01-01,102.5\n"))
    s = sources.fetch_fred("INDPRO", session)
    assert session.calls[0][0] == sources.FRED_CSV and s.iloc[0] == 102.5
    assert "no FRED_API_KEY" in sources.fred_source()


def test_errors_never_contain_the_api_key(no_sleep, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "super-secret-key")
    with pytest.raises(requests.HTTPError) as client_err:
        sources.fetch_fred("DCOILBRENTEU", CapturingSession(JsonResponse({}, status=400)))
    assert "super-secret-key" not in str(client_err.value)

    session = FakeSession([requests.ReadTimeout("Read timed out for url ...?api_key=super-secret-key")] * sources.RETRIES)
    with pytest.raises(RuntimeError) as timeout_err:
        sources.fetch_fred("DCOILBRENTEU", session)
    assert "super-secret-key" not in str(timeout_err.value) and "ReadTimeout" in str(timeout_err.value)


def test_eia_demo_key_refusal_explains_how_to_fix(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="eia.gov/opendata/register"):
        sources.fetch_eia("WCRSTUS1", CapturingSession(JsonResponse({}, status=403)))


def test_eia_errors_with_a_real_key_are_not_misreported(monkeypatch):
    monkeypatch.setenv("EIA_API_KEY", "real-key")
    with pytest.raises(requests.HTTPError) as err:
        sources.fetch_eia("WCRSTUS1", CapturingSession(JsonResponse({}, status=403)))
    assert "real-key" not in str(err.value)
