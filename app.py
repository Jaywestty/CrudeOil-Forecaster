from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
from scenario_engine import ScenarioEngine, SCENARIOS
from llm_explainer import LLMExplainer

# App Initialization
app = FastAPI(
    title="Oil Price Scenario Forecasting API",
    description="""
    A macroeconomic scenario simulation system for crude oil price forecasting.
    Powered by SARIMAX econometric modeling and LLaMA 3.3 via Groq.
    
    ## How it works
    1. Submit a natural language scenario query
    2. LLM parses your intent into structured shock parameters  
    3. SARIMAX model generates baseline + counterfactual forecasts
    4. LLM explains the results in plain economic language
    """,
    version="1.0.0"
)

# CORS = Cross Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Components Once at Startup
print("🚀 Starting Oil Forecasting API...")
engine   = ScenarioEngine()
explainer = LLMExplainer()
print("✅ API ready\n")


# Request/Response Models 
class ScenarioRequest(BaseModel):
    """What the frontend sends to us."""
    query:          str            
    forecast_weeks: Optional[int] = 12  

class WeeklyForecast(BaseModel):
    """One week of forecast data."""
    week:     int
    baseline: float
    scenario: float
    change:   float

class ImpactSummary(BaseModel):
    """Price impact at a specific horizon."""
    baseline:   float
    shocked:    float
    difference: float
    pct_change: float
    formatted:  str

class ScenarioResponse(BaseModel):
    """What we send back to the frontend."""
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


# Endpoints

@app.get("/")
def root():
    """Health check endpoint — confirms API is running."""
    return {
        "status":  "online",
        "message": "Oil Price Scenario Forecasting API",
        "docs":    "Visit /docs for interactive API documentation"
    }


@app.get("/scenarios")
def list_scenarios():
    """
    Return all available predefined scenarios.
    Streamlit uses this to populate the scenario dropdown.
    """
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


@app.post("/simulate", response_model=ScenarioResponse)
def run_simulation(request: ScenarioRequest):
    """
    Main endpoint — runs the full simulation pipeline:
    1. Parse natural language query → scenario + shocks
    2. Run baseline forecast
    3. Run shocked forecast
    4. Generate LLM explanation
    5. Return structured response

    This is the endpoint Streamlit calls when user clicks 'Run Scenario'
    """
    try:
        # Step 1: Parse user query 
        parsed = explainer.parse_user_query(request.query)
        scenario_key = parsed["scenario_key"]
        modifier     = parsed.get("magnitude_modifier", 1.0)
        weeks        = parsed.get("forecast_weeks", request.forecast_weeks)

        # Step 2: Apply magnitude modifier to shocks
        import copy
        modified_shocks = {
            k: v * modifier
            for k, v in SCENARIOS[scenario_key]["shocks"].items()
        }

        # Temporarily override shocks with scaled version
        original_shocks = SCENARIOS[scenario_key]["shocks"].copy()
        SCENARIOS[scenario_key]["shocks"] = modified_shocks

        # Step 3: Run scenario simulation
        result = engine.run_scenario(scenario_key, weeks)

        # Restore original shocks immediately after
        SCENARIOS[scenario_key]["shocks"] = original_shocks

        # Step 4: Generate LLM explanation 
        explanation    = explainer.explain_results(result, parsed)
        uncertainty    = explainer.generate_uncertainty_note(result)

        # Step 5: Build weekly forecast list
        weekly = []
        for i, (b, s) in enumerate(zip(
            result["baseline_forecast"].values,
            result["shocked_forecast"].values
        ), 1):
            weekly.append(WeeklyForecast(
                week=i,
                baseline=round(float(b), 2),
                scenario=round(float(s), 2),
                change=round(float(s - b), 2)
            ))

        # Step 6: Build response 
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
        # Known errors — bad scenario key, missing data, etc.
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unknown errors — log full traceback for debugging
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-direct")
def run_direct_simulation(
    scenario_key:   str,
    forecast_weeks: int = 12
):
    """
    Run a scenario directly by key — bypasses LLM parsing.
    Useful for testing specific scenarios without natural language.
    Example: POST /simulate-direct?scenario_key=opec_cut
    """
    if scenario_key not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario. Available: {list(SCENARIOS.keys())}"
        )

    result = engine.run_scenario(scenario_key, forecast_weeks)

    return {
        "scenario_name":    result["scenario_name"],
        "current_price":    round(float(result["current_price"]), 2),
        "impact_week1":     result["impact_week1"],
        "impact_week12":    result["impact_week12"],
        "baseline_mean":    round(float(result["baseline_forecast"].mean()), 2),
        "scenario_mean":    round(float(result["shocked_forecast"].mean()), 2),
    }