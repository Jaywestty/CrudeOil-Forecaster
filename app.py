from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
from scenario_engine import ScenarioEngine, SCENARIOS
from llm_explainer import LLMExplainer

# ── App Initialization ─────────────────────────────────────────────
app = FastAPI(
    title="Oil Price Scenario Forecasting API",
    description="""
    A macroeconomic scenario simulation system for crude oil price forecasting.
    Powered by SARIMAX econometric modeling and LLaMA 3.3 via Groq.

    ## Endpoints
    - **/simulate** — Original single-scenario simulation (deterministic)
    - **/simulate-probabilistic** — New multi-scenario probability-weighted simulation
    - **/simulate-direct** — Run a scenario by key, no LLM parsing

    ## How /simulate-probabilistic works
    1. LLM assigns probability weights across all relevant scenarios
    2. Live macro signals (VIX, dollar, inventories) adjust those weights
    3. SARIMAX runs separately for each scenario (no retraining)
    4. Results combined into a weighted expected price + confidence range
    5. LLM explains in institutional research language
    """,
    version="2.0.0"
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
    week12_price: float

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


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms API is running."""
    return {
        "status":  "online",
        "message": "Oil Price Scenario Forecasting API v2.0",
        "docs":    "/docs"
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
    """Return the most recent Brent crude oil price in the dataset."""
    return {
        "price": round(engine.latest['brent_price'], 2),
        "date":  str(engine.weekly_data.index[-1].date()),
        "unit":  "USD/barrel"
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
                week12_price = data["week12_price"]
            )
            for key, data in result["scenario_results"].items()
        }

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
            adjusted_probabilities = result["scenario_probabilities"],
            macro_adjustments      = result["macro_adjustments"],
            weighted_forecast      = result["weighted_forecast"].tolist(),
            baseline_forecast      = result["baseline_forecast"].tolist(),
            explanation            = explanation,
            uncertainty_note       = uncertainty,
            forecast_weeks         = weeks
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