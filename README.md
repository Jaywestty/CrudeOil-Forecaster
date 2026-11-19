# 🛢️ Crude Oil Price Scenario Forecaster

> **Probabilistic macroeconomic simulation engine for Brent crude oil pricing — powered by SARIMAX econometrics, gpt-oss-120b, and live macro signal adjustment.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://crude-forecaster.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://crudeoil-forecaster.onrender.com/docs)

---

## 🌐 Live Deployments

| Service | URL | Description |
|---|---|---|
| 🖥️ Frontend (Streamlit) | [crude-forecaster.streamlit.app](https://crude-forecaster.streamlit.app/) | Interactive probabilistic simulation UI |
| ⚙️ Backend (FastAPI) | [crudeoil-forecaster.onrender.com](https://crudeoil-forecaster.onrender.com/) | REST API + auto-generated docs at `/docs` |

> ⚠️ **Cold start notice:** The API runs on Render's free tier and spins down after 15 minutes of inactivity. The first request after a period of sleep takes **30–60 seconds** to wake up — this is expected.

---

## 📌 What This Project Does

This system answers oil market scenario questions in plain English and returns a **probability-weighted price distribution** — not a single deterministic answer.

Instead of saying:
> *"Oil will rise by $4 under this scenario"*

It says:
> *"Expected price: $87. Range: $80–$93. Primary driver: Geopolitical Tension (55% weight), adjusted by current VIX and inventory signals."*

That difference is how professional energy trading desks actually think.

**Example queries the system handles:**

- *"What happens if OPEC cuts production by 10%?"*
- *"If Russia-Ukraine tensions escalate, what is the price range?"*
- *"What if global economic slowdown coincides with OPEC cuts?"*
- *"How does a strong US dollar affect crude oil prices?"*
- *"What is the expected price range for next quarter?"*

---

## 🏗️ System Architecture

```
User types scenario query (plain English)
              │
              ▼
     ┌─────────────────┐
     │  Streamlit UI   │  ← crude-forecaster.streamlit.app
     └────────┬────────┘
              │  POST /simulate-probabilistic
              ▼
     ┌─────────────────┐
     │   FastAPI API   │  ← crudeoil-forecaster.onrender.com
     └────────┬────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
┌──────────────┐   ┌──────────────┐
│ gpt-oss-120b │   │   SARIMAX    │
│ (Groq)       │   │   Model      │
│              │   │   (.pkl)     │
│ Assigns      │   │              │
│ probability  │   │ Runs once    │
│ weights to   │   │ per scenario │
│ scenarios    │   │ (no retrain) │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────────────────────────┐
│   Macro Signal Adjustment Layer │
│                                 │
│  Live VIX, dollar, inventory    │
│  signals shift LLM probabilities│
│  based on current market state  │
└─────────────────────┬───────────┘
                      │
                      ▼
        Weighted forecast distribution
        Expected price + range
        Scenario breakdown + explanation
              │
              ▼
     ┌─────────────────┐
     │  Streamlit UI   │
     │  renders result │
     └─────────────────┘
```

---

## 🧠 The Probabilistic Engine — How It Works

This is the core upgrade that separates the system from a standard single-scenario simulator.

### Step 1 — LLM Probability Assignment

Instead of picking one best-fit scenario, gpt-oss-120b reads the query and distributes probability weights across all relevant scenarios:

```
Query: "What if Middle East tensions rise while demand also grows?"

LLM returns:
  {
    "geopolitical_tension": 0.55,
    "demand_boom":          0.30,
    "opec_cut":             0.15
  }
```

### Step 2 — Macro Signal Adjustment

The LLM estimates are then adjusted using **live macroeconomic signals** pulled from the dataset — real data, not guesses:

```
Current conditions read from dataset:
  VIX level, dollar direction, inventory trend, Fed funds

Adjustment rules:
  VIX > 25      → boost geopolitical + recession weight
  VIX < 15      → boost demand boom weight
  Dollar rising → boost rate hike + recession weight
  Dollar falling → boost demand boom + OPEC cut weight
  Inventories ↓  → boost OPEC cut + geopolitical weight
  Inventories ↑  → boost recession weight

After adjustment, weights are renormalised to sum to 1.0
```

### Step 3 — SARIMAX Runs Per Scenario

The SARIMAX model runs **once per scenario** — same trained model, different exogenous inputs each time. No retraining required:

```
Scenario A (geopolitical, 55%) → SARIMAX → Week-12 price: $91
Scenario B (demand boom, 30%)  → SARIMAX → Week-12 price: $88
Scenario C (opec cut, 15%)     → SARIMAX → Week-12 price: $89
```

### Step 4 — Weighted Expected Price

```
Expected price = (0.55 × $91) + (0.30 × $88) + (0.15 × $89)
               = $90.00

Range: $88 – $91
Primary driver: Geopolitical Tension (55%)
```

### Step 5 — Direction Correction

The SARIMAX model was trained on 2006–2022 data, a period where high-VIX episodes coincided with demand collapses (2008 crash, 2014–2016 glut). This means the model sometimes produces counterintuitive price directions for supply shock scenarios.

A post-processing direction correction is applied after each scenario run:

```python
BULLISH_SCENARIOS = {"opec_cut", "geopolitical_tension", "demand_boom"}
BEARISH_SCENARIOS = {"global_recession", "rate_hike"}

# If a bullish scenario produced a price DROP, mirror it:
# corrected = 2 × baseline - shocked
# This preserves the model's magnitude estimate
# while enforcing the correct economic direction.
```

### Step 6 — LLM Explanation

gpt-oss-120b reads the actual model numbers and generates an institutional-style explanation covering the expected price, what drives the low vs high end of the range, and how macro signals shifted the probability weights.

---

## 🧮 Why SARIMAX? (Modeling Decision)

| Model | Strength | Why Not Used |
|---|---|---|
| **SARIMA** | Clean time series structure | No external variable support — cannot inject macro shocks |
| **XGBoost** | Powerful pattern recognition | Treats observations as independent, no temporal structure, no counterfactual mechanism |
| **Prophet** | Calendar seasonality | Built for daily business metrics, weak exogenous support, no economic interpretability |
| ✅ **SARIMAX** | Time structure + external variables + interpretable coefficients | **Chosen** |

The **X in SARIMAX** (eXogenous variables) is the critical design choice. It allows five macro drivers to be injected directly into the forecast — without it, scenario simulation is impossible.

**Key model coefficient:**
```
dollar_return coefficient = -107.80  (p < 0.001)

Interpretation:
  A 1% appreciation in the US dollar index
  corresponds to a $1.08/barrel drop in Brent crude.
  Economically interpretable. Statistically significant.
  Directionally consistent with oil being USD-denominated.
```

---

## 📊 Model Specification

```
Model:         SARIMAX(1, 1, 1)(1, 0, 1, 52)
                │  │  │   │  │  │   52 = weekly seasonality
                │  │  │   └──┴──┘ seasonal AR, I, MA terms
                └──┴──┘ non-seasonal AR, differencing, MA terms

Production fit: all 1,043 weekly observations (2006 – Jan 2026)
Forecasts start: the week after the latest observation in the dataset
Evaluation:    walk-forward backtest, 198 weekly origins (2022 – 2025),
               parameters re-estimated yearly (see "Model Evaluation")

Exogenous variables (5 macro drivers):
  dollar_return   → DXY US Dollar Index weekly return
  indpro_return   → US Industrial Production weekly change
  inventory_pct   → US crude oil inventory % change (EIA)
  fed_funds_diff  → Federal Funds Rate weekly difference
  vix_diff        → CBOE VIX (fear index) weekly difference
```

**Note on accuracy:** an earlier version of this README quoted a 15.22% test MAPE. That number came from a single 209-week forecast that was given the *actual future* macro values, and a flat "no change" forecast scored better on the same weeks (12.39%). The honest walk-forward results are in [Model Evaluation](#-model-evaluation). The system's value is scenario comparison, not point forecasting.

---

## 🔄 Probabilistic Simulation Pipeline

```
Step 1 — LLM Probability Parsing
  Natural language query →
  gpt-oss-120b (Groq) →
  { scenario_key: probability_weight } across relevant scenarios

Step 2 — Macro Signal Adjustment
  Read live VIX, dollar, inventory, Fed funds from dataset →
  Adjust LLM probabilities using real market signals →
  Renormalise weights to sum to 1.0

Step 3 — Baseline Forecast
  SARIMAX → forecast with ZERO shocks →
  "What happens if nothing changes"

Step 4 — Per-Scenario SARIMAX Forecasts
  Drop scenarios under 5% weight, renormalise the rest to sum to 1.0
  Compute the no-shock baseline once, with its 80% / 95% statistical band
  For each remaining scenario:
    Build exogenous shock matrix →
    SARIMAX → shocked forecast →
    Apply direction correction if needed

Step 5 — Weighted Distribution Assembly
  expected_price[t] = Σ (probability_i × price_i[t]) for each week
  price_range = [min scenario week-12, max scenario week-12]
  Safety clamp: expected price always sits inside range

Step 6 — LLM Explanation
  Actual model numbers + scenario breakdown →
  gpt-oss-120b →
  Institutional research note: expected price, range
  interpretation, macro signal effects

Step 7 — Response to UI
  Expected price, range, scenario breakdown,
  probability shifts, fan chart data, explanation
```

---

## 🗂️ Project Structure

```
crude-oil-forecaster/
│
├── app.py                  # FastAPI backend — all REST endpoints
├── run.py                  # Local launcher for the API (uvicorn)
├── scenario_engine.py      # Core engine — SARIMAX + direction correction
│                           # + macro probability adjustment
├── llm_explainer.py        # Groq (gpt-oss-120b) — parsing + explanation
├── utils.py                # Model loading, data helpers
├── scenario_library.py     # Event study: Brent moves after past events of each type
├── train.py                # Fits the production model on all data (Render build)
├── streamlit_ui/
│   ├── app_ui.py           # Streamlit frontend — probabilistic UI
│   └── requirements.txt    # UI-only dependencies (isolated from the API's)
├── models/
│   └── sarimax_model.pkl   # Generated at build time — not in GitHub
│
├── data/
│   ├── oil_macro_weekly.csv        # Weekly data (2006 – Jan 2026)
│   ├── oil_macro_transformed.csv  # Stationary transformed features
│   └── scenario_events.csv        # 31 dated historical events per scenario type
│
├── evaluation/             # Walk-forward backtest, challengers, conformal calibration
├── reports/
│   ├── backtest/report.md     # Accuracy vs baselines, promotion decisions
│   ├── calibration/report.md  # Interval coverage and calibration
│   └── scenarios/report.md    # Scenario calibration on historical events
├── tests/                  # pytest suite (API, engine, LLM, evaluation)
│
├── requirements.txt        # Pinned dependencies (Python 3.11 stable)
├── .python-version         # Pins Python 3.11.8 — fixes Render build
└── .env                    # Local secrets (never committed)
```

---

## 🚀 API Endpoints

Base URL: `https://crudeoil-forecaster.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Keepalive ping endpoint |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/scenarios` | List all predefined scenario keys |
| `GET` | `/current-price` | Latest Brent price from dataset |
| `POST` | `/simulate` | Single-scenario simulation (deterministic) |
| `POST` | `/simulate-probabilistic` | **Main endpoint** — probabilistic multi-scenario simulation |
| `POST` | `/simulate-direct` | Run by scenario key — bypasses LLM parsing |

### Example — Probabilistic Simulation Request

```bash
curl -X POST "https://crudeoil-forecaster.onrender.com/simulate-probabilistic" \
  -H "Content-Type: application/json" \
  -d '{"query": "What if Middle East tensions rise while demand also grows?", "forecast_weeks": 12}'
```

### Example Response (abbreviated)

```json
{
  "price_expected": 87.00,
  "price_low": 80.12,
  "price_high": 93.45,
  "primary_driver": "geopolitical_tension",
  "primary_driver_name": "Major Geopolitical Tension",
  "scenario_breakdown": {
    "geopolitical_tension": { "probability": 0.55, "week12_price": 91.20 },
    "demand_boom":           { "probability": 0.30, "week12_price": 88.40 },
    "opec_cut":              { "probability": 0.15, "week12_price": 89.10 }
  },
  "adjusted_probabilities": { ... },
  "macro_adjustments": {
    "vix_high_geo": "VIX elevated → geopolitical weight +30%"
  },
  "weighted_forecast": [84.2, 85.1, 85.8, ...],
  "baseline_forecast": [82.6, 82.6, 82.6, ...],
  "explanation": "Brent crude is expected to reach $87.00/barrel...",
  "uncertainty_note": "Scenario probabilities are LLM-estimated..."
}
```

---

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.11
- [Groq API key](https://console.groq.com) (free)
- [FRED API key](https://fredaccount.stlouisfed.org) (free)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jaywestty/CrudeOil-Forecaster.git
cd CrudeOil-Forecaster

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
GROQ_API_KEY=your_groq_key_here
FRED_API_KEY=your_fred_key_here

# 5. Train the model (once, ~1 minute)
python train.py

# 6. Start the FastAPI backend
uvicorn app:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# 7. In a new terminal, install the Streamlit UI's dependencies and start it
#    (use a separate virtual environment: the UI pins its own versions)
pip install -r streamlit_ui/requirements.txt
streamlit run streamlit_ui/app_ui.py
# UI: http://localhost:8501

# 8. Run the tests and (optionally) the evaluation
pytest
python -m evaluation.run_backtest      # ~8 min  -> reports/backtest/report.md
python -m evaluation.run_calibration   # ~4 min  -> reports/calibration/report.md
python -m evaluation.run_scenario_study  # ~1 min -> reports/scenarios/report.md
```

---

## 📦 Data Sources

| Variable | Source | Frequency | Description |
|---|---|---|---|
| Brent Crude Price | EIA / FRED | Weekly | USD per barrel |
| US Dollar Index | FRED (DTWEXBGS) | Weekly | DXY trade-weighted index |
| Industrial Production | FRED (INDPRO) | Monthly → Weekly | US manufacturing output |
| Crude Inventories | EIA | Weekly | US commercial crude stocks |
| Federal Funds Rate | FRED (FEDFUNDS) | Monthly → Weekly | US benchmark interest rate |
| VIX | FRED (VIXCLS) | Daily → Weekly | CBOE volatility / fear index |

---

## 🧪 Model Evaluation

Every forecast below was made using only data available at the time (walk-forward,
198 weekly origins from Jan 2022 to Oct 2025, parameters re-estimated yearly).
Full reports: [`reports/backtest/report.md`](reports/backtest/report.md) and
[`reports/calibration/report.md`](reports/calibration/report.md).

```
Point accuracy: MAE relative to a "price stays the same" forecast (< 1.00 = better)
                          1 week   4 weeks   12 weeks
  SARIMAX (as deployed)    1.012    1.024     1.045
  Plain SARIMA             1.009    1.020     1.039
  Ridge (lagged features)  1.015    1.044     0.998
  Gradient-boosted trees   1.003    1.117     1.273
  -> No model beats the no-change forecast significantly
     (Diebold-Mariano, Holm-corrected, rule fixed before testing).

Why the macro variables don't help:
  - Dollar/oil weekly correlation: -0.39 in 2006-21, -0.06 in 2022-25
    (VIX: -0.33 -> -0.07). The relationship broke after 2021.
  - In this specification a macro shock shifts the price once;
    it does not build up over the horizon.

Statistical band (95%) coverage: ~85% at 1 week, ~92% at 4, ~97% at 12,
  but only ~71-85% during the 2022 crisis (too narrow in turmoil).
```

**What this means:** weekly Brent prices are close to unpredictable from these
inputs, which matches the forecasting literature. The app is therefore framed as a
**scenario analysis** tool: compare what-ifs against the no-shock baseline, and
read the statistical band as a typical-conditions range.

### Scenario calibration on historical events

Could scenarios be calibrated on what Brent actually did after real events, instead of
hand-tuned macro shocks? [`reports/scenarios/report.md`](reports/scenarios/report.md) tests
this on 31 dated events (OPEC cuts, geopolitical shocks, recessions, Fed hikes, demand
surges), with the catalogue and decision rule fixed before any outcome was computed.

```
Leave-one-event-out, 12-week cumulative move (mean absolute error, log points)
  Naive (no move)                        0.188
  Current engine (SARIMAX shocks)        0.191
  Historical median of other events      0.216   -> rule says keep the current engine

12-week Brent moves after past events (median, range)
  Geopolitical tension (9)   -0.3%   (-63% to +17%)
  OPEC cut (8)               +8.3%   (-36% to +68%)
  Global recession (4)      -15.3%   (-61% to  -4%)
```

Events with the same label had very different effects, so the median is a poor forecast.
The app therefore shows this history as **context** ("after 9 similar events, Brent moved
between -63% and +17% within 12 weeks"), never as the scenario forecast itself.

---

## 📖 Reading the Output Numbers

When the system returns a result, four numbers appear. Here is exactly what each one means and how they relate to each other.

**Example output:**
```
Live Price (sidebar):   $94.00
Baseline Wk 12:         $82.63
Expected Price Wk 12:   $87.23    (+$4.60 vs baseline)
Forecast Range:         $81 – $90
```

### What each number is

```
$94.00  Live Price
        The real current Brent crude market price,
        fetched live from FRED (DCOILBRENTEU series).
        This is display only — it does NOT feed into
        the model's forecast calculations.

$82.63  Baseline Week-12 Price
        What SARIMAX forecasts at week 12 if absolutely
        nothing changes — no scenario applied.
        The model starts from its last training observation
        ($61.35, the final row of the dataset) and projects
        forward using current macro conditions.

$87.23  Expected Price Week-12
        What SARIMAX forecasts at week 12 WITH your scenario
        applied — the probability-weighted average across all
        relevant scenarios.

+$4.60  The difference between expected ($87.23) and
        baseline ($82.63). Your scenario adds $4.60 to
        what would have happened naturally.

$81–$90 The spread across all scenarios considered.
        $81 = lowest-priced scenario outcome (worst case)
        $90 = highest-priced scenario outcome (best case)
        $87.23 sits inside this range — the weighted midpoint.
```

### How all four numbers connect

```
Real market today          Model baseline            Model expected
     $94.00        →           $82.63        →           $87.23
  (FRED live,              (no shock applied,        (scenario applied,
  display only)             from dataset)            +$4.60 uplift)

                                    ↑
                     Range: $81 ────┼──── $90
                                 $87.23
                           (expected sits here)
```

### Why the live price ($94) and baseline ($82.63) are different

This is the most common question and the most important one to understand.

The SARIMAX model was trained on a dataset whose last observation is **$61.35** (early 2026). The live FRED price of $94 is fetched separately for display purposes only — it does not update the model's starting point. So the model forecasts forward from $61.35, arriving at a baseline of $82.63 at week 12. The real market is at $94 because prices moved after the dataset cutoff.

```
Dataset ends:     $61.35  ← model's last known price
Model baseline:   $82.63  ← model's week-12 forecast from $61.35
FRED live price:  $94.00  ← real market today (display only)

The $94 and $82.63 are on different reference points.
They are not directly comparable.
```

The system's value is in **relative scenario comparison** — how much a specific shock moves prices relative to the no-shock baseline — not in absolute price level accuracy. In a production deployment, the model would be retrained periodically so its starting observation stays close to the current market price.

### Scenario spread vs statistical band

The UI shows two different ranges, and they answer different questions:

```
Scenario spread      lowest -> highest scenario price at the final week.
                     "How far apart are the what-ifs?" Not a probability range.

80% / 95% band       the model's own uncertainty around the no-shock baseline.
                     "Where would the price typically land if nothing special
                     happens?" Close to nominal coverage on average, but too
                     narrow in crises (see Model Evaluation).
```

### Historical context

The Scenarios tab also shows what Brent did after past events of each type, one dot per
event. This is history, not a forecast: it shows how wide real outcomes were, and the
step-2a test found its median did not predict better than assuming no change.

### What the forecast range tells you

```
Range $81–$90 with expected $87.23 means:

  If the bearish scenarios dominate → price lands near $81
  If the bullish scenarios dominate → price lands near $90
  Probability-weighted best estimate → $87.23

  A narrow range (e.g. $85–$88) = high scenario agreement,
  low uncertainty

  A wide range (e.g. $75–$100) = scenarios strongly disagree,
  high uncertainty — the query likely involves multiple
  competing forces
```

---

## ⚙️ Deployment Architecture

### Backend — Render (FastAPI)

```
Platform:       Render Web Service (Free Tier)
Runtime:        Python 3.11.8 (pinned via .python-version)
Build Command:  pip install -r requirements.txt && python train.py
Start Command:  uvicorn app:app --host 0.0.0.0 --port $PORT
Model storage:  Trained fresh on every deploy — no binary in GitHub
Auto-deploy:    Yes — triggers on every push to main branch

Why train on deploy:
  SARIMAX model = ~300MB → above GitHub 100MB hard limit
  Retrained on each deploy → model always reflects latest code
  Trade-off: adds 5–10 minutes to build time

Free Tier behaviour:
  ✅ 512MB RAM — sufficient for SARIMAX in memory
  ⚠️  Spins down after 15 min of inactivity (cold start)
  ⚠️  Cold start: 30–60 seconds to wake up
```

### Frontend — Streamlit Cloud

```
Platform:    Streamlit Community Cloud (Free)
Entry point: streamlit_ui/app_ui.py
Secrets:     API_URL set via Streamlit Cloud dashboard
Auto-deploy: Yes — triggers on every push to main branch
```

---

## 🔐 Environment Variables

| Variable | Where to Set | Description |
|---|---|---|
| `GROQ_API_KEY` | Render → Environment | gpt-oss-120b access via Groq |
| `GROQ_MODEL` | Render → Environment (optional) | Override the Groq model (default `openai/gpt-oss-120b`) |
| `FRED_API_KEY` | Render → Environment | FRED macroeconomic data |
| `API_URL` | Streamlit Cloud → Secrets | Render backend URL |

**Never commit `.env` to GitHub.** It is in `.gitignore`.

---

## 💬 Predefined Scenarios

| Scenario Key | Name | Direction | Primary Shock |
|---|---|---|---|
| `opec_cut` | OPEC Production Cut | 📈 Bullish | Dollar weakens, inventories draw |
| `geopolitical_tension` | Major Geopolitical Tension | 📈 Bullish | Supply disruption, hoarding |
| `demand_boom` | Global Demand Boom | 📈 Bullish | Industrial surge, inventories draw |
| `global_recession` | Global Recession | 📉 Bearish | Industrial collapse, inventory build |
| `rate_hike` | Aggressive Fed Rate Hike | 📉 Bearish | Dollar strengthens, demand slows |

---

## 🧩 Technical Decisions & Trade-offs

**Why probabilistic instead of single-scenario?**
Real oil market queries often involve multiple competing forces simultaneously — "OPEC cuts while demand grows" involves both supply tightening and demand pull. A single scenario picks one and ignores the other. A probability distribution weights both, produces a meaningful expected price, and exposes the uncertainty range. This is how energy trading desks actually think.

**Why LLM probability assignment + macro signal adjustment?**
The LLM interprets the user's language and assigns base probabilities. But language alone misses what the market is currently pricing — a geopolitical question asked during a calm low-VIX period should carry different weights than the same question during market panic. The macro adjustment layer bridges that gap using real data, making the system partially data-driven rather than purely LLM-dependent.

**Why post-processing direction correction?**
The SARIMAX model learned from 2006–2022 data. During that window, supply disruption signals often preceded demand collapses (2008 global crisis, 2014 glut). This means the model associates supply shock signals with price decreases — the opposite of economic theory. Rather than retraining with different data (which would lose 17 years of calibration), a direction correction mirrors the price impact around the baseline for scenarios where the model output contradicts economic fundamentals. The model's magnitude estimate is preserved; only the direction is corrected when wrong.

**Why SARIMAX over XGBoost or Prophet?**
XGBoost treats observations as independent rows — it has no concept of time order, cannot model seasonality, and has no native mechanism for injecting future exogenous shocks into a forecast. Prophet is built for daily business metrics with calendar patterns. SARIMAX was specifically designed for multivariate time series with external regressors — exactly what macro-driven oil price simulation requires.

**Why Groq for the LLM?**
Groq's LPU inference hardware returns responses in a few seconds, which matters for a real-time UI that calls the LLM several times per simulation. The model is `openai/gpt-oss-120b`, an open-weight reasoning model on Groq's free tier (same API key). It runs with low reasoning effort and extra token headroom so its reasoning never crowds out the answer.

**Why train the model on Render instead of committing it?**
The trained SARIMAX model is ~300MB — above GitHub's 100MB hard limit. Rather than using Git LFS or external storage, the model is retrained on every Render deploy via the build command. This keeps the repository clean, ensures the model always reflects the latest code and data, and avoids external storage dependencies. Trade-off: adds 5–10 minutes to build time per deploy.

---

## 📝 Known Limitations & Future Improvements

```
Current limitations:
  - Scenario probabilities are LLM-estimated, not derived from
    options market implied volatility or historical scenario frequencies
  - No model beats a no-change forecast for weekly prices (see Model Evaluation)
  - Macro relationships learned before 2022 largely broke afterwards
  - Statistical bands are too narrow during crises
  - Data is a static snapshot (through Jan 2026); forecasts start from its last week
  - 12-week forecast horizon — the range the backtests validated
  - Five fixed scenarios — cannot model novel events without mapping
    to one of the five (e.g. carbon taxes, SPR releases)
  - Direction correction is a heuristic, not a structural fix. Replacing it with
    event-calibrated paths was tested (step 2a) and did not forecast better, so it
    stays, documented as a known limitation

Natural next upgrades:
  - Duration profiles: short-war vs prolonged-war shock decay
  - Options-implied probability calibration
  - Regime detection model for automatic macro context
  - Scheduled data refresh and retraining
  - Scenario shocks that build up over the horizon, calibrated on past events
```

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*End-to-end ML system covering data ingestion, econometric modeling, probabilistic simulation, LLM integration, macro signal adjustment, REST API, and cloud deployment.*
