# 🛢️ Crude Oil Price Scenario Forecaster

> **Macroeconomic simulation engine for Brent crude oil pricing — powered by SARIMAX econometrics and LLaMA 3.3 via Groq.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://crude-forecaster.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://crudeoil-forecaster.onrender.com/docs)

---

## 🌐 Live Deployments

| Service | URL | Description |
|---|---|---|
| 🖥️ Frontend (Streamlit) | [crude-forecaster.streamlit.app](https://crude-forecaster.streamlit.app/) | Interactive scenario simulation UI |
| ⚙️ Backend (FastAPI) | [crudeoil-forecaster.onrender.com](https://crudeoil-forecaster.onrender.com/) | REST API + auto-generated docs at `/docs` |

> ⚠️ **Note on cold starts:** The API is hosted on Render's free tier, which **spins down after 15 minutes of inactivity** to save resources. The first request after a period of sleep may take **30–60 seconds** to wake up — this is expected behaviour, not a bug. Just wait a moment and it will respond normally.

---

## 📌 What This Project Does

This system answers questions like:

- *"What happens to oil prices if OPEC cuts production by 10%?"*
- *"Simulate a global recession scenario"*
- *"What if the US Federal Reserve raises interest rates aggressively?"*
- *"What if there's a geopolitical conflict in the Middle East?"*

You type a natural language question. The system:
1. Uses **LLaMA 3.3 70B** (via Groq) to parse your intent into structured economic shock parameters
2. Injects those shocks into a **SARIMAX econometric model** trained on 20 years of weekly oil market data
3. Generates a **baseline forecast** (what happens with no shock) vs a **counterfactual forecast** (what happens with your scenario)
4. Uses the LLM again to explain the results in plain economic language

---

## 🏗️ System Architecture

```
User types scenario query (natural language)
            │
            ▼
   ┌─────────────────┐
   │  Streamlit UI   │  ← crude-forecaster.streamlit.app
   │   (Frontend)    │
   └────────┬────────┘
            │  HTTP POST /simulate
            ▼
   ┌─────────────────┐
   │   FastAPI API   │  ← crudeoil-forecaster.onrender.com
   │   (Backend)     │
   └────────┬────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌──────────┐
│ SARIMAX │  │  Groq    │
│  Model  │  │ LLaMA 3.3│
│ (.pkl)  │  │  (LLM)   │
└─────────┘  └──────────┘
     │             │
     └──────┬──────┘
            ▼
   Forecast + Explanation
   returned to Streamlit UI
```

---

## 🧠 Why SARIMAX? (Modeling Decision)

Three models were considered. Here's why SARIMAX won:

| Model | Strength | Why Not Used |
|---|---|---|
| **SARIMA** | Clean time series structure | No external variable support — can't inject macro shocks |
| **XGBoost** | Powerful, wins Kaggle competitions | Treats observations as independent, no temporal structure, no counterfactual mechanism |
| **Prophet** | Great for calendar seasonality | Built for daily business metrics, poor exogenous variable support, no economic interpretability |
| ✅ **SARIMAX** | Time structure + external variables + interpretable coefficients | **Chosen** |

The **X in SARIMAX** (eXogenous variables) is the entire reason it was chosen. It allows injecting macro shocks — dollar returns, inventory changes, VIX spikes — directly into the forecast. Without the X, scenario simulation is impossible.

**Key model coefficient (defend this in interview):**
```
dollar_return = -107.80 (p < 0.001)
→ "A 1% appreciation in the US dollar
   corresponds to a $1.08/barrel drop in Brent crude"
→ This is economically interpretable and statistically significant
```

---

## 📊 Model Specification

```
Model:          SARIMAX(1, 1, 1)(1, 0, 1, 52)
                 │  │  │   │  │  │   52 = weekly seasonality
                 │  │  │   └──┴──┘ seasonal AR, I, MA terms
                 └──┴──┘ non-seasonal AR, differencing, MA terms

Training data:  2006 – 2022  (80% of 1,043 weekly observations)
Test period:    2022 – 2024  (20% holdout)
Validation:     Rolling window cross-validation (10 folds)
Test MAPE:      15.22%
AIC:            Minimized via order selection

Exogenous variables (the 5 macro drivers):
  dollar_return   → DXY US Dollar Index weekly return
  indpro_return   → US Industrial Production weekly change
  inventory_pct   → US crude oil inventory % change (EIA)
  fed_funds_diff  → Federal Funds Rate weekly difference
  vix_diff        → CBOE VIX (fear index) weekly difference
```

---

## 🔄 Simulation Pipeline (Step by Step)

When you click **RUN SIMULATION**, here is exactly what happens under the hood:

```
Step 1 — LLM Query Parsing
  Your natural language query →
  Groq API (LLaMA 3.3 70B) →
  Structured JSON: { scenario_key, magnitude_modifier, confidence, reasoning }

Step 2 — Shock Construction
  Predefined scenario shocks × magnitude_modifier =
  calibrated shock vector for each of the 5 macro variables

Step 3 — Baseline Forecast
  SARIMAX model → forecast with ZERO shocks applied →
  "what happens if nothing changes"

Step 4 — Counterfactual Forecast
  SARIMAX model → same forecast WITH shocks injected →
  "what happens under your scenario"

Step 5 — LLM Explanation
  Both forecasts + scenario metadata →
  Groq API →
  Plain English economic explanation + uncertainty note

Step 6 — Response Assembly
  Weekly forecast table, impact at Week 1 and Week 12,
  shock breakdown, explanation → returned to Streamlit UI
```

---

## 🗂️ Project Structure

```
crude-oil-forecaster/
│
├── api.py                  # FastAPI backend — all REST endpoints
├── streamlit_app.py        # Streamlit frontend — the UI
├── scenario_engine.py      # Core simulation logic — runs SARIMAX
├── llm_explainer.py        # Groq/LLaMA interface — parsing + explanation
├── utils.py                # Model loading, data helpers
├── train.py                # Model training script (runs on Render build)
│
├── models/
│   └── sarimax_model.pkl   # Generated at build time — not in GitHub
│
├── data/
│   ├── oil_macro_weekly.csv        # Raw weekly data (2004–2024)
│   └── oil_macro_transformed.csv  # Stationary transformed features
│
├── requirements.txt        # Python dependencies (pinned for stability)
├── .python-version         # Pins Python 3.11.8 — fixes Render build issues
└── .env                    # Local secrets (never committed)
```

---

## 🚀 API Endpoints

Base URL: `https://crudeoil-forecaster.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/scenarios` | List all predefined scenarios |
| `GET` | `/current-price` | Latest Brent crude price from dataset |
| `POST` | `/simulate` | **Main endpoint** — run full NL simulation |
| `POST` | `/simulate-direct` | Run simulation by scenario key (bypasses LLM) |

### Example Request

```bash
curl -X POST "https://crudeoil-forecaster.onrender.com/simulate" \
  -H "Content-Type: application/json" \
  -d '{"query": "What if OPEC cuts production by 10%?", "forecast_weeks": 12}'
```

### Example Response (abbreviated)

```json
{
  "scenario_name":     "OPEC Production Cut",
  "current_price":     74.23,
  "parsed_confidence": "high",
  "impact_week1":  { "difference": 3.42, "pct_change": 4.61 },
  "impact_week12": { "difference": 7.18, "pct_change": 9.67 },
  "explanation":   "An OPEC production cut of this magnitude...",
  "weekly_forecasts": [...]
}
```

---

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.11
- A [Groq API key](https://console.groq.com) (free)
- A [FRED API key](https://fredaccount.stlouisfed.org) (free)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jaywestty/CrudeOil-Forecaster.git
cd CrudeOil-Forecaster

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the project root:
GROQ_API_KEY=your_groq_key_here
FRED_API_KEY=your_fred_key_here

# 5. Train the model (only needed once)
python train.py

# 6. Start the FastAPI backend
uvicorn api:app --reload
# API now running at http://localhost:8000
# Docs at http://localhost:8000/docs

# 7. In a new terminal, start the Streamlit frontend
streamlit run streamlit_app.py
# UI now running at http://localhost:8501
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

## 🧪 Model Validation

```
Validation strategy: Rolling window cross-validation
  10 folds across the test period (2022–2024)
  Each fold trains on all data up to that point
  Forecasts 12 weeks ahead out-of-sample

Results:
  Test MAPE:  15.22%
  Benchmark:  ARIMA baseline (univariate, no macro variables)
  SARIMAX outperforms ARIMA baseline on directional accuracy
  and responds meaningfully to macro variable shocks

Note on MAPE: 15.22% on weekly oil prices is reasonable.
Oil is one of the most volatile commodities in the world —
affected by geopolitical events, OPEC decisions, and
macroeconomic shocks that no statistical model can fully anticipate.
```

---

## ⚙️ Deployment Architecture

### Backend — Render (FastAPI)

```
Platform:       Render Web Service (Free Tier)
Runtime:        Python 3.11
Build Command:  pip install -r requirements.txt && python train.py
Start Command:  uvicorn api:app --host 0.0.0.0 --port $PORT
Model Storage:  Trained fresh on every deploy — no binary in GitHub
Auto-deploy:    Yes — triggers on every push to main branch

Build Command explained:
  pip install -r requirements.txt   → install dependencies
  && python train.py                → train SARIMAX, save .pkl to disk
  &&                                → only trains if install succeeded

Free Tier Behaviour:
  ✅ 512MB RAM — sufficient for SARIMAX model in memory
  ✅ Shared CPU — adequate for inference
  ⚠️  Spins down after 15 min of inactivity
  ⚠️  Cold start takes 30–60 seconds to wake up
  ⚠️  Build takes ~5–10 min longer due to training step
```

### Frontend — Streamlit Cloud

```
Platform:       Streamlit Community Cloud (Free)
Runtime:        Python 3.11
Entry point:    streamlit_app.py
Secrets:        API_URL set via Streamlit Cloud dashboard
Auto-deploy:    Yes — triggers on every push to main branch
```

### Model Training on Render

```
Problem:  SARIMAX model = 300MB — too large to store in GitHub
Solution: Train the model fresh on every Render deploy

How it works:
  Build Command runs two steps in sequence:
    pip install -r requirements.txt   ← install all dependencies
    && python train.py                ← train SARIMAX and save .pkl

  The && means: "only run train.py if pip install succeeded"

  Render trains the model once per deploy, saves it to disk,
  and the running API loads it from there.
  No large files in GitHub. No LFS needed.

Trade-off:
  ✅ Clean — no binary files in version control
  ✅ Model always reflects the latest data and code
  ⚠️  Adds ~5–10 minutes to build time on each deploy
  ⚠️  Depends on FRED/EIA API availability at build time
```

---

## 🔐 Environment Variables

| Variable | Where to Set | Description |
|---|---|---|
| `GROQ_API_KEY` | Render dashboard → Environment | LLaMA 3.3 access via Groq |
| `FRED_API_KEY` | Render dashboard → Environment | FRED macroeconomic data |
| `API_URL` | Streamlit Cloud → Secrets | Render backend URL |

**Never commit `.env` to GitHub.** It is in `.gitignore`.

---

## 💬 Predefined Scenarios

| Scenario Key | Name | Description |
|---|---|---|
| `opec_cut` | OPEC Production Cut | OPEC reduces output, supply tightens |
| `us_recession` | US Recession | Demand collapses, industrial output drops |
| `dollar_surge` | Dollar Surge | USD strengthens, oil becomes expensive to import |
| `middle_east_conflict` | Middle East Conflict | Supply disruption risk premium spikes |
| `fed_hike` | Fed Rate Hike | Higher rates slow growth, reduce demand |
| `shale_boom` | US Shale Boom | Supply surge from US production |

---

## 🧩 Technical Decisions & Trade-offs

**Why not XGBoost or Prophet?**
XGBoost treats observations as independent rows — it has no concept of time order or seasonality. Prophet is designed for daily business metrics with calendar patterns (Black Friday, Christmas). Neither provides a native mechanism for injecting economic shocks into future forecasts. SARIMAX was built for exactly this problem.

**Why separate frontend and backend?**
Decoupling the API from the UI means the API can be called by any client — another frontend, a mobile app, a Jupyter notebook, or a direct curl command. This is standard production architecture. The Streamlit UI is one consumer of the API, not tightly coupled to it.

**Why Groq for the LLM?**
Groq's inference hardware (LPU) is significantly faster than standard GPU inference — LLaMA 3.3 70B responses come back in under 2 seconds. For a real-time UI, this latency matters. OpenAI GPT-4 at similar capability costs more and responds slower on free tiers.

**Why train the model on Render instead of committing it to GitHub?**
The trained SARIMAX model file is 300MB — well above GitHub's 100MB hard limit. Rather than using Git LFS or S3, the model is retrained fresh on every Render deploy via `pip install -r requirements.txt && python train.py` in the build command. This keeps the repository clean (no binary files in version control), ensures the model always reflects the latest code, and avoids any external storage dependency. The trade-off is a longer build time (~5–10 minutes), which is acceptable for a deployment that rarely changes.

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built as a technical assessment — demonstrating end-to-end ML system design from data ingestion through econometric modeling, LLM integration, API development, and cloud deployment.*
