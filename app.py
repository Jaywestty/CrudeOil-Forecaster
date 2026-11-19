from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
import os
from scenario_engine import ScenarioEngine, SCENARIOS
from llm_explainer import LLMExplainer


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

        print(f"✅ Live Brent price fetched: ${price} as of {date}")
        return price, date, "FRED live"

    except Exception as e:
        # Any failure — missing API key, network issue, FRED outage —
        # is handled silently. The caller decides the fallback.
        print(f"  Live price fetch failed: {e}")
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
    version="2.1.0"
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

# ── CORS ───────────────────────────────────────────────────────────
# Allow any domain to call this API.
# Required because Streamlit Cloud (different domain) calls Render.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Components Once at Startup ───────────────────────────────
# Both are expensive to initialize. Loading once here means every
# request reuses the same in-memory objects — no reload per request.
print("🚀 Starting Oil Forecasting API...")
engine    = ScenarioEngine()
explainer = LLMExplainer()
print("✅ API ready\n")


# ── Pydantic Models ────────────────────────────────────────────────
# Pydantic validates the shape of data coming in and going out.
# If a request body doesn't match, FastAPI returns 422 automatically.

class ScenarioRequest(BaseModel):
    """Shared request body for both /simulate endpoints."""
    query:          str
    forecast_weeks: Optional[int] = 12

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
        "message":    "Oil Price Scenario Forecasting API v2.1",
        "docs":       "/docs",
        "model_info": _model_info().model_dump(),
    }


@app.get("/health")
def health():
    """
    Dedicated health check for the keepalive worker.
    Lightweight — just confirms the server is awake.
    The keepalive background worker pings this every 280 seconds
    to prevent Render's free tier from spinning the service down.
    """
    return {"status": "online"}


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
def run_simulation(request: ScenarioRequest):
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
        weeks        = parsed.get("forecast_weeks", request.forecast_weeks)

        # Step 2 — Scale shocks by magnitude modifier
        # e.g. modifier 2.0 doubles all shock values for "catastrophic" events
        modified_shocks = {
            k: v * modifier
            for k, v in SCENARIOS[scenario_key]["shocks"].items()
        }

        # Temporarily override the scenario shocks, restore after run
        original_shocks = SCENARIOS[scenario_key]["shocks"].copy()
        SCENARIOS[scenario_key]["shocks"] = modified_shocks
        result = engine.run_scenario(scenario_key, weeks)
        SCENARIOS[scenario_key]["shocks"] = original_shocks

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
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────
# NEW ENDPOINT — /simulate-probabilistic
# Institutional-grade probabilistic simulation.
# ─────────────────────────────────────────────────────────────────────

@app.post("/simulate-probabilistic", response_model=ProbabilisticResponse)
def run_probabilistic_simulation(request: ScenarioRequest):
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

        return ProbabilisticResponse(
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
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────
# ORIGINAL ENDPOINT — /simulate-direct
# Unchanged from your working version.
# ─────────────────────────────────────────────────────────────────────

@app.post("/simulate-direct")
def run_direct_simulation(scenario_key: str, forecast_weeks: int = 12):
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