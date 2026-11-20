import logging
import os
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from llm_explainer import LLMExplainer, track_fallbacks
from scenario_engine import ScenarioEngine, SCENARIOS
from serving.cache import TTLCache, make_key
from serving.observability import (
    REQUEST_ID_HEADER, Metrics, configure_logging, log_event, request_id_from,
)

# ── Limits ─────────────────────────────────────────────────────────
# The backtests validated horizons up to 12 weeks, and every query costs
# several LLM calls, so both are bounded at the edge of the API.
MAX_WEEKS = 12
MAX_QUERY_CHARS = 500
STALE_DATA_DAYS = 21  # the weekly refresh should keep data younger than this

log = configure_logging(os.getenv("LOG_LEVEL", "INFO"))
metrics = Metrics()
response_cache = TTLCache(
    maxsize=int(os.getenv("CACHE_MAX_ENTRIES", "256")),
    ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", str(6 * 3600))),
)


# ── Live Brent Price from FRED ─────────────────────────────────────
# The SARIMAX model uses the static dataset for its baseline forecast.
# But the price displayed in the UI sidebar should be the REAL current
# market price — not the last row of a CSV that was pulled months ago.
#
# We fetch this from FRED (Federal Reserve Economic Data) using the
# DCOILBRENTEU series — the same source used to build the training data.
# This call runs once per /current-price request (lightweight).
#
# If the FRED call fails for any reason (no API key, network issue,
# FRED outage), we fall back to the dataset price with a clear label
# so the user knows what they're looking at.

def fetch_live_brent_price():
    """
    Fetch the latest Brent crude spot price from FRED.

    FRED series: DCOILBRENTEU
      - Daily Brent crude oil price in USD per barrel
      - Published by the U.S. Energy Information Administration
      - Updated on business days with 1-day lag
      - Same primary source used to build the training dataset

    Returns:
        tuple: (price: float, date: str, source: str)
               source is "FRED live" on success, "dataset" on fallback
    """
    try:
        from fredapi import Fred
        fred   = Fred(api_key=os.getenv("FRED_API_KEY"))
        series = fred.get_series("DCOILBRENTEU")

        # FRED sometimes has trailing NaN values for the most recent days
        # (data not yet published). Drop them to get the last real value.
        series = series.dropna()

        price = round(float(series.iloc[-1]), 2)
        date  = series.index[-1].strftime("%Y-%m-%d")

        log_event(log, "live price fetched", price=price, date=date)
        return price, date, "FRED live"

    except Exception as e:
        # Any failure — missing API key, network issue, FRED outage —
        # is handled silently. The caller decides the fallback.
        log_event(log, "live price unavailable", logging.WARNING, error=type(e).__name__)
        return None, None, "dataset"

# ── App Initialization ─────────────────────────────────────────────
app = FastAPI(
    title="Oil Price Scenario Forecasting API",
    description="""
    A macroeconomic scenario simulation system for crude oil price forecasting.
    Powered by SARIMAX econometric modeling and gpt-oss-120b via Groq.

    ## Endpoints
    - **/simulate** — Original single-scenario simulation (deterministic)
    - **/simulate-probabilistic** — New multi-scenario probability-weighted simulation
    - **/simulate-direct** — Run a scenario by key, no LLM parsing

    ## How /simulate-probabilistic works
    1. LLM assigns probability weights across all relevant scenarios
    2. Live macro signals (VIX, dollar, inventories) adjust those weights
    3. SARIMAX runs separately for each scenario (no retraining)
    4. Results combined into a weighted expected price + scenario spread,
       plus the model's 80% / 95% statistical band around the no-shock baseline
    5. LLM explains in institutional research language

    ## Evaluation
    Walk-forward backtests (reports/backtest, reports/calibration) show no model,
    including this one, beats a no-change forecast for weekly Brent prices in
    2022–2025. Treat outputs as scenario analysis, not price prediction.
    """,
    version="2.3.0"
)

# Coverage of the statistical band, measured in the walk-forward backtests
# (reports/backtest/report.md and reports/calibration/report.md).
STATISTICAL_BAND_NOTE = (
    "Statistical range around the no-shock baseline, from the model's own error model. "
    "In walk-forward backtests over 2022–2025, the 95% band contained the actual price "
    "about 85% of the time at 1 week, 92% at 4 weeks and 97% at 12 weeks. During the "
    "2022 crisis, bands like this covered only about 71–85%. Read it as a range for "
    "typical conditions, not a worst case."
)

# Step 2a (reports/scenarios/report.md): historical event medians did not
# forecast better than the engine or a no-change forecast, so history is shown
# as context only.
HISTORICAL_NOTE = (
    "What Brent did in the weeks after past events of the same type, measured from "
    "the last price before each event. This is history, not a forecast: events with "
    "the same label had very different effects, and a leave-one-out test found the "
    "historical median did not predict better than assuming no change."
)

# ── CORS ───────────────────────────────────────────────────────────
# CORS only governs browsers calling the API directly. The Streamlit app calls
# it server-to-server, so it is unaffected. Set ALLOWED_ORIGINS (comma
# separated) to restrict browser access; the default keeps the API open.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)


# ── Request logging, request ids and metrics ──────────────────────
@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request_id_from(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    request.state.cache = None
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        metrics.observe(route_path, status, latency_ms)
        log_event(log, "request", request_id=request_id, method=request.method, path=route_path,
                  status=status, latency_ms=round(latency_ms, 1), cache=request.state.cache)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def _server_error(request: Request, exc: Exception) -> HTTPException:
    """Log the full error; give the client a generic message and the request id."""
    rid = getattr(request.state, "request_id", None)
    log.error("simulation failed", exc_info=exc, extra={"fields": {"request_id": rid}})
    return HTTPException(status_code=500,
                         detail=f"Internal error while running the simulation (request id {rid}).")

# ── Load Components Once at Startup ───────────────────────────────
# Both are expensive to initialize. Loading once here means every
# request reuses the same in-memory objects — no reload per request.
log_event(log, "starting API")
engine    = ScenarioEngine()
explainer = LLMExplainer()
log_event(log, "API ready", llm=explainer.model,
          data_as_of=None if engine.forecast_origin is None else str(engine.forecast_origin.date()))


# ── Pydantic Models ────────────────────────────────────────────────
# Pydantic validates the shape of data coming in and going out.
# If a request body doesn't match, FastAPI returns 422 automatically.

class ScenarioRequest(BaseModel):
    """Shared request body for both /simulate endpoints."""
    query:          str = Field(min_length=3, max_length=MAX_QUERY_CHARS)
    forecast_weeks: int = Field(12, ge=1, le=MAX_WEEKS)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query must contain at least 3 non-space characters")
        return v

class WeeklyForecast(BaseModel):
    week:     int
    baseline: float
    scenario: float
    change:   float

class ImpactSummary(BaseModel):
    baseline:   float
    shocked:    float
    difference: float
    pct_change: float
    formatted:  str

class ScenarioResponse(BaseModel):
    """Response model for /simulate (single scenario)."""
    scenario_key:      str
    scenario_name:     str
    scenario_desc:     str
    current_price:     float
    parsed_confidence: str
    parsed_reasoning:  str
    weekly_forecasts:  list[WeeklyForecast]
    impact_week1:      ImpactSummary
    impact_week12:     ImpactSummary
    explanation:       str
    uncertainty_note:  str
    shocks_applied:    dict

class ScenarioBreakdownItem(BaseModel):
    """One scenario's contribution inside a probabilistic result."""
    name:         str
    probability:  float
    week12_price: float          # price at the final forecast week (name kept for compatibility)
    forecast:     list[float] = []  # full weekly path

class Band(BaseModel):
    lower: list[float]
    upper: list[float]

class StatisticalBand(BaseModel):
    level_80: Band
    level_95: Band
    note:     str

class HistoricalEvent(BaseModel):
    date:       str
    event:      str
    move_pct:   float
    confounded: bool

class HistoricalContext(BaseModel):
    """What Brent did after past events of this type. Descriptive, not a forecast."""
    week:       int
    n_events:   int
    median_pct: float
    low_pct:    float
    high_pct:   float
    price_low:  float
    price_high: float
    events:     list[HistoricalEvent]

class ModelInfo(BaseModel):
    forecaster: str
    llm:        str
    data_as_of: Optional[str] = None

class ProbabilisticResponse(BaseModel):
    """Response model for /simulate-probabilistic."""
    price_expected:         float
    price_low:              float
    price_high:             float
    baseline_week12:        float
    current_price:          float
    primary_driver:         str
    primary_driver_name:    str
    reasoning:              str
    scenario_context:       str
    scenario_breakdown:     dict[str, ScenarioBreakdownItem]
    original_probabilities: dict
    adjusted_probabilities: dict
    macro_adjustments:      dict
    weighted_forecast:      list[float]
    baseline_forecast:      list[float]
    explanation:            str
    uncertainty_note:       str
    forecast_weeks:         int
    # Added in v2.1 (all optional so older clients keep working)
    final_probabilities:    dict = {}
    dropped_scenarios:      dict = {}
    forecast_dates:         Optional[list[str]] = None
    statistical_band:       Optional[StatisticalBand] = None
    model_info:             Optional[ModelInfo] = None
    # Added in v2.2: descriptive history per scenario (see reports/scenarios)
    historical_context:     dict[str, HistoricalContext] = {}
    historical_note:        str = ""
    # Added in v2.3: true when an LLM step failed and a safe fallback was used
    llm_fallback:           bool = False


def _model_info() -> ModelInfo:
    origin = engine.forecast_origin
    return ModelInfo(
        forecaster="SARIMAX(1,1,1)(1,0,1,52) with 5 macro drivers",
        llm=f"{explainer.model} (Groq)",
        data_as_of=None if origin is None else str(origin.date()),
    )


def _round_list(values) -> list[float]:
    return [round(float(v), 2) for v in values]


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms API is running."""
    return {
        "status":     "online",
        "message":    "Oil Price Scenario Forecasting API v2.3",
        "docs":       "/docs",
        "model_info": _model_info().model_dump(),
    }


@app.get("/health")
def health():
    """
    Readiness check, cheap enough for keep-alive pings.

    Always returns HTTP 200 while the process can serve (so a keep-alive ping
    never fails), with status "degraded" and warnings when something needs
    attention, e.g. data older than the weekly refresh should allow.
    """
    warnings = []
    origin = engine.forecast_origin
    age_days = None
    if origin is None:
        warnings.append("data date unknown")
    else:
        age_days = int((pd.Timestamp.now().normalize() - origin).days)
        if age_days > STALE_DATA_DAYS:
            warnings.append(f"data is {age_days} days old; the weekly refresh may have stopped")
    return {
        "status": "degraded" if warnings else "ok",
        "model_loaded": engine.model is not None,
        "data_as_of": None if origin is None else str(origin.date()),
        "data_age_days": age_days,
        "warnings": warnings,
    }


@app.get("/metrics")
def get_metrics():
    """Request counts, latency percentiles, cache and LLM-fallback statistics since start."""
    return {**metrics.snapshot(), "cache": response_cache.stats(), "model_info": _model_info().model_dump()}


@app.get("/scenarios")
def list_scenarios():
    """Return all available predefined scenarios for the UI dropdown."""
    return {
        "scenarios": [
            {
                "key":         key,
                "name":        s["name"],
                "description": s["description"]
            }
            for key, s in SCENARIOS.items()
        ]
    }


@app.get("/current-price")
def get_current_price():
    """
    Return the current Brent crude oil price.

    Priority:
      1. Live price from FRED API (DCOILBRENTEU series) — real market price
      2. Fallback: last row of training dataset — if FRED call fails

    The live price is DISPLAY ONLY — it does not affect the SARIMAX
    model's baseline forecast, which always uses the training dataset.
    This distinction is intentional: the model was calibrated on dataset
    values, and changing its baseline input would alter forecast results.
    The sidebar price is purely informational.
    """
    live_price, live_date, source = fetch_live_brent_price()

    if live_price:
        return {
            "price":  live_price,
            "date":   live_date,
            "unit":   "USD/barrel",
            "source": source
        }

    # Fallback to dataset price with honest label
    return {
        "price":  round(engine.latest['brent_price'], 2),
        "date":   str(engine.weekly_data.index[-1].date()),
        "unit":   "USD/barrel",
        "source": "dataset (FRED unavailable)"
    }


# ─────────────────────────────────────────────────────────────────────
# ORIGINAL ENDPOINT — /simulate
# Unchanged pipeline from your working version.
# Kept intact so the existing Streamlit frontend works without changes.
# ─────────────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=ScenarioResponse)
def run_simulation(request: ScenarioRequest, http: Request):
    """
    Original single-scenario simulation endpoint.

    Pipeline:
    1. LLM parses query → picks ONE best-fit scenario + magnitude modifier
    2. Shocks scaled by modifier (e.g. 2.0 = double severity)
    3. SARIMAX runs baseline + shocked forecast
    4. LLM explains the result referencing user's actual question
    5. Returns deterministic single-scenario output
    """
    try:
        # Step 1 — Parse query to single scenario
        parsed = explainer.parse_user_query(request.query)

        # Guard: parse_user_query() returns a fallback dict on failure,
        # but if something upstream goes wrong and a list slips through,
        # this catches it before .get() crashes with the list error.
        if not isinstance(parsed, dict):
            parsed = {
                "scenario_key":       "geopolitical_tension",
                "magnitude_modifier":  1.0,
                "confidence":         "low",
                "reasoning":          "Unexpected parse format -- using default",
                "scenario_context":   request.query,
                "specific_entity":    None,
                "forecast_weeks":     request.forecast_weeks
            }

        scenario_key = parsed["scenario_key"]
        modifier     = parsed.get("magnitude_modifier", 1.0)
        # The LLM may suggest a horizon; keep it inside the validated range.
        try:
            weeks = int(parsed.get("forecast_weeks", request.forecast_weeks))
        except (TypeError, ValueError):
            weeks = request.forecast_weeks
        weeks = min(max(weeks, 1), MAX_WEEKS)

        # Step 2 — Scale shocks by magnitude modifier
        # e.g. modifier 2.0 doubles all shock values for "catastrophic" events
        modified_shocks = {
            k: v * modifier
            for k, v in SCENARIOS[scenario_key]["shocks"].items()
        }

        # Pass the scaled shocks in; never mutate the shared SCENARIOS table
        # (unsafe with concurrent requests, and left corrupted on any error).
        result = engine.run_scenario(scenario_key, weeks, shocks=modified_shocks)

        # Step 3 — Generate explanation and uncertainty note
        explanation = explainer.explain_results(result, parsed)
        uncertainty = explainer.generate_uncertainty_note(result)

        # Step 4 — Build weekly forecast list for response
        weekly = []
        for i, (b, s) in enumerate(zip(
            result["baseline_forecast"].values,
            result["shocked_forecast"].values
        ), 1):
            weekly.append(WeeklyForecast(
                week     = i,
                baseline = round(float(b), 2),
                scenario = round(float(s), 2),
                change   = round(float(s - b), 2)
            ))

        return ScenarioResponse(
            scenario_key      = scenario_key,
            scenario_name     = result["scenario_name"],
            scenario_desc     = result["scenario_desc"],
            current_price     = round(float(result["current_price"]), 2),
            parsed_confidence = parsed.get("confidence", "medium"),
            parsed_reasoning  = parsed.get("reasoning", ""),
            weekly_forecasts  = weekly,
            impact_week1      = ImpactSummary(**result["impact_week1"]),
            impact_week12     = ImpactSummary(**result["impact_week12"]),
            explanation       = explanation,
            uncertainty_note  = uncertainty,
            shocks_applied    = modified_shocks
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _server_error(http, e)


# ─────────────────────────────────────────────────────────────────────
# NEW ENDPOINT — /simulate-probabilistic
# Institutional-grade probabilistic simulation.
# ─────────────────────────────────────────────────────────────────────

@app.post("/simulate-probabilistic", response_model=ProbabilisticResponse)
def run_probabilistic_simulation(request: ScenarioRequest, http: Request):
    """
    New probabilistic multi-scenario simulation endpoint.

    Pipeline:
    1. LLM assigns probability weights across multiple scenarios
       based on the user's query (not just one best-fit scenario)
    2. Live macro signals (VIX, dollar, inventories) adjust those weights
       using real current data from our dataset
    3. SARIMAX runs separately for each scenario — same model, no retraining
    4. Results combined: weighted expected price + min/max range
    5. LLM explains the distribution in institutional research language

    Output example for "What if tensions rise AND demand grows?":
      Expected price: $87.00
      Range:          $79 – $90
      Primary driver: Geopolitical Tension (45%)
    """
    data_version = "none" if engine.forecast_origin is None else str(engine.forecast_origin.date())
    key = make_key(request.query, request.forecast_weeks, data_version, explainer.model)
    cached = response_cache.get(key)
    if cached is not None:
        http.state.cache = "hit"
        return cached
    http.state.cache = "miss"
    fallbacks = track_fallbacks()

    try:
        # Step 1 — Parse query to probability distribution
        parsed = explainer.parse_query_probabilistic(request.query)

        # Guard: same list-vs-dict safety net as /simulate
        if not isinstance(parsed, dict):
            parsed = {
                "probabilities":    {"geopolitical_tension": 1.0},
                "primary_driver":   "geopolitical_tension",
                "confidence":       "low",
                "reasoning":        "Unexpected parse format -- using default",
                "scenario_context": request.query,
                "specific_entity":  None
            }

        probabilities = parsed["probabilities"]
        weeks         = request.forecast_weeks

        # Step 2 — Run probabilistic simulation
        # (macro signal adjustment happens inside run_probabilistic_scenario)
        result = engine.run_probabilistic_scenario(probabilities, weeks)

        # Step 3 — Generate explanation and uncertainty note
        explanation = explainer.explain_probabilistic_results(result, parsed)
        uncertainty = explainer.generate_probabilistic_uncertainty_note(result)

        # Step 4 — Build scenario breakdown for response
        breakdown = {
            key: ScenarioBreakdownItem(
                name         = data["scenario_name"],
                probability  = data["probability"],
                week12_price = data["week12_price"],
                forecast     = _round_list(data["forecast"]),
            )
            for key, data in result["scenario_results"].items()
        }
        band80, band95 = result["baseline_band_80"], result["baseline_band_95"]
        statistical_band = StatisticalBand(
            level_80 = Band(lower=_round_list(band80["lower"]), upper=_round_list(band80["upper"])),
            level_95 = Band(lower=_round_list(band95["lower"]), upper=_round_list(band95["upper"])),
            note     = STATISTICAL_BAND_NOTE,
        )

        # Primary driver display name
        primary_key  = result["primary_driver"]
        primary_name = (
            SCENARIOS[primary_key]["name"]
            if primary_key in SCENARIOS
            else primary_key
        )

        response = ProbabilisticResponse(
            price_expected         = result["price_expected"],
            price_low              = result["price_low"],
            price_high             = result["price_high"],
            baseline_week12        = round(result["baseline_week12"], 2),
            current_price          = round(float(result["current_price"]), 2),
            primary_driver         = primary_key,
            primary_driver_name    = primary_name,
            reasoning              = parsed.get("reasoning", ""),
            scenario_context       = parsed.get("scenario_context", request.query),
            scenario_breakdown     = breakdown,
            original_probabilities = result["original_probabilities"],
            adjusted_probabilities = result["adjusted_probabilities"],
            final_probabilities    = result["scenario_probabilities"],
            dropped_scenarios      = result["dropped_scenarios"],
            macro_adjustments      = result["macro_adjustments"],
            weighted_forecast      = _round_list(result["weighted_forecast"]),
            baseline_forecast      = _round_list(result["baseline_forecast"]),
            explanation            = explanation,
            uncertainty_note       = uncertainty,
            forecast_weeks         = weeks,
            forecast_dates         = result["forecast_dates"],
            statistical_band       = statistical_band,
            model_info             = _model_info(),
            historical_context     = result.get("historical_context", {}),
            historical_note        = HISTORICAL_NOTE,
            llm_fallback           = bool(fallbacks),
        )
        if fallbacks:
            # Degraded answer: count it, and do not cache it, so a short LLM
            # outage is not served from cache for hours afterwards.
            metrics.record_fallbacks(fallbacks)
            http.state.cache = "skip-degraded"
        else:
            response_cache.set(key, response)
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _server_error(http, e)


# ─────────────────────────────────────────────────────────────────────
# ORIGINAL ENDPOINT — /simulate-direct
# Unchanged from your working version.
# ─────────────────────────────────────────────────────────────────────

@app.post("/simulate-direct")
def run_direct_simulation(scenario_key: str, forecast_weeks: int = Query(12, ge=1, le=MAX_WEEKS)):
    """
    Run a scenario directly by key — bypasses LLM parsing entirely.
    Useful for testing specific scenarios from the /docs page.
    Example: POST /simulate-direct?scenario_key=opec_cut
    """
    if scenario_key not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario. Available: {list(SCENARIOS.keys())}"
        )

    result = engine.run_scenario(scenario_key, forecast_weeks)

    return {
        "scenario_name": result["scenario_name"],
        "current_price": round(float(result["current_price"]), 2),
        "impact_week1":  result["impact_week1"],
        "impact_week12": result["impact_week12"],
        "baseline_mean": round(float(result["baseline_forecast"].mean()), 2),
        "scenario_mean": round(float(result["shocked_forecast"].mean()), 2),
    }