import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page Configuration 
st.set_page_config(
    page_title="Oil Price Scenario Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration 
API_BASE = os.getenv("API_URL_BASE", "http://localhost:8000")


# Custom CSS 
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    /* Root variables */
    :root {
        --oil-black:   #0d0d0d;
        --oil-dark:    #1a1a1a;
        --oil-card:    #1f1f1f;
        --amber:       #e8a020;
        --amber-light: #f5c842;
        --red-down:    #e05252;
        --green-up:    #4caf82;
        --text-main:   #f0ede8;
        --text-muted:  #8a8580;
        --border:      #2e2e2e;
    }

    /* Global background */
    .stApp {
        background-color: var(--oil-black);
        font-family: 'DM Sans', sans-serif;
        color: var(--text-main);
    }

    header {
    visibility: visible !important;
    background: #0d0d0d !important;   /* keep black */
}

[data-testid="stHeader"] {
    background: #0d0d0d !important;
}

[data-testid="stToolbar"] {
    background: #0d0d0d !important;
}

    /* Header */
    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        font-weight: 400;
        color: var(--text-main);
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .main-subheader {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: var(--amber);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* Price display card */
    .price-card {
        background: var(--oil-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .price-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .price-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: var(--amber-light);
    }
    .price-unit {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-left: 0.3rem;
    }

    /* Metric cards */
    .metric-card {
        background: var(--oil-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .metric-value-up {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: var(--green-up);
    }
    .metric-value-down {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: var(--red-down);
    }
    .metric-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Section labels */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--amber);
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* Explanation box */
    .explanation-box {
        background: var(--oil-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.75;
        color: var(--text-main);
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
    }

    /* Uncertainty box */
    .uncertainty-box {
        background: #1a1510;
        border: 1px solid #3a2e1a;
        border-left: 3px solid var(--amber);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #c4b08a;
        font-family: 'DM Sans', sans-serif;
        line-height: 1.6;
    }

    /* Confidence badge */
    .badge-high   { background:#1a3a2a; color:#4caf82; padding:2px 10px;
                    border-radius:20px; font-size:0.7rem; font-family:'DM Mono',monospace;
                    letter-spacing:1px; text-transform:uppercase; }
    .badge-medium { background:#2a2a1a; color:#e8a020; padding:2px 10px;
                    border-radius:20px; font-size:0.7rem; font-family:'DM Mono',monospace;
                    letter-spacing:1px; text-transform:uppercase; }
    .badge-low    { background:#3a1a1a; color:#e05252; padding:2px 10px;
                    border-radius:20px; font-size:0.7rem; font-family:'DM Mono',monospace;
                    letter-spacing:1px; text-transform:uppercase; }

    /* Shocks table */
    .shock-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--border);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
    }
    .shock-var   { color: var(--text-muted); }
    .shock-pos   { color: var(--red-down); }
    .shock-neg   { color: var(--green-up); }

    /* Sidebar */
    .stSidebar { background: var(--oil-dark) !important; }
    [data-testid="stSidebar"] { background: var(--oil-dark); }

    /* Input styling */
    .stTextArea textarea {
        background: var(--oil-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton button {
        background: var(--amber) !important;
        color: var(--oil-black) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background: var(--amber-light) !important;
    }

    /* Selectbox */
    .stSelectbox select, [data-baseweb="select"] {
        background: var(--oil-card) !important;
        color: var(--text-main) !important;
        border-color: var(--border) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] { color: var(--amber) !important; }

    /* Divider */
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# Helper Functions 

def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.status_code == 200
    except:
        return False


def get_current_price():
    """Fetch current Brent price from API."""
    try:
        r = requests.get(f"{API_BASE}/current-price", timeout=5)
        return r.json()
    except:
        return {"price": "N/A", "date": "N/A", "unit": "USD/barrel"}


def get_scenarios():
    """Fetch available scenarios from API."""
    try:
        r = requests.get(f"{API_BASE}/scenarios", timeout=5)
        return r.json()["scenarios"]
    except:
        return []


def run_simulation(query, forecast_weeks):
    """Call the /simulate endpoint with user query."""
    try:
        r = requests.post(
            f"{API_BASE}/simulate",
            json={"query": query, "forecast_weeks": forecast_weeks},
            timeout=60   # LLM calls can take a moment
        )
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, r.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is uvicorn running?"
    except Exception as e:
        return None, str(e)


def make_forecast_chart(weekly_forecasts, current_price, scenario_name):
    """Build the plotly forecast chart."""
    weeks    = [w["week"] for w in weekly_forecasts]
    baseline = [w["baseline"] for w in weekly_forecasts]
    scenario = [w["scenario"] for w in weekly_forecasts]

    fig = go.Figure()

    # Current price reference line
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="#8a8580",
        line_width=1,
        annotation_text=f"  Current ${current_price:.2f}",
        annotation_font_color="#8a8580",
        annotation_font_size=10
    )

    # Baseline forecast
    fig.add_trace(go.Scatter(
        x=weeks, y=baseline,
        mode="lines",
        name="Baseline (no shock)",
        line=dict(color="#8a8580", width=2, dash="dash"),
        hovertemplate="Week %{x}<br>Baseline: $%{y:.2f}<extra></extra>"
    ))

    # Scenario forecast
    scenario_color = "#e05252" if scenario[-1] < baseline[-1] else "#4caf82"
    fig.add_trace(go.Scatter(
        x=weeks, y=scenario,
        mode="lines+markers",
        name=scenario_name,
        line=dict(color=scenario_color, width=2.5),
        marker=dict(size=5, color=scenario_color),
        hovertemplate="Week %{x}<br>Scenario: $%{y:.2f}<extra></extra>"
    ))

    # Fill between baseline and scenario
    fig.add_trace(go.Scatter(
        x=weeks + weeks[::-1],
        y=scenario + baseline[::-1],
        fill="toself",
        fillcolor=f"rgba({','.join(str(int(scenario_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Impact range",
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#8a8580", size=10),
        legend=dict(
            font=dict(size=10, color="#8a8580"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#2e2e2e",
            borderwidth=1
        ),
        xaxis=dict(
            title="Forecast Week",
            gridcolor="#1f1f1f",
            linecolor="#2e2e2e",
            tickcolor="#2e2e2e",
            title_font=dict(size=10),
        ),
        yaxis=dict(
            title="Brent Price (USD/barrel)",
            gridcolor="#1f1f1f",
            linecolor="#2e2e2e",
            tickcolor="#2e2e2e",
            title_font=dict(size=10),
            tickprefix="$"
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        height=350,
    )

    return fig


def make_delta_chart(weekly_forecasts):
    """Bar chart showing price delta per week."""
    weeks  = [f"W{w['week']}" for w in weekly_forecasts]
    deltas = [w["change"] for w in weekly_forecasts]
    colors = ["#e05252" if d < 0 else "#4caf82" for d in deltas]

    fig = go.Figure(go.Bar(
        x=weeks, y=deltas,
        marker_color=colors,
        hovertemplate="Week %{x}<br>Δ Price: $%{y:.2f}<extra></extra>"
    ))

    fig.add_hline(y=0, line_color="#2e2e2e", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#8a8580", size=10),
        xaxis=dict(gridcolor="#1f1f1f", linecolor="#2e2e2e"),
        yaxis=dict(
            gridcolor="#1f1f1f",
            linecolor="#2e2e2e",
            tickprefix="$",
            title="Price Change vs Baseline"
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=220,
        showlegend=False
    )

    return fig


# Main App 

def main():

    # API Health Check 
    api_online = check_api_health()

    #  Sidebar 
    with st.sidebar:
        st.markdown("""
        <div style='font-family: DM Serif Display, serif; 
                    font-size:1.4rem; color:#f0ede8; margin-bottom:0.2rem;'>
            🛢️ Oil Forecaster
        </div>
        <div style='font-family: DM Mono, monospace; font-size:0.6rem;
                    letter-spacing:2px; color:#e8a020; 
                    text-transform:uppercase; margin-bottom:1.5rem;'>
            Scenario Simulation System
        </div>
        """, unsafe_allow_html=True)

        # API status
        status_color = "#4caf82" if api_online else "#e05252"
        status_text  = "API ONLINE" if api_online else "API OFFLINE"
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace; font-size:0.65rem;
                    color:{status_color}; letter-spacing:2px;
                    margin-bottom:1rem;'>
            ● {status_text}
        </div>
        """, unsafe_allow_html=True)

        if not api_online:
            st.error("Start the API first:\n```\nuvicorn api:app --reload\n```")

        st.markdown("---")

        # Current price
        price_data = get_current_price()
        st.markdown(f"""
        <div class='price-card'>
            <div class='price-label'>Brent Crude — Latest</div>
            <div class='price-value'>${price_data['price']}
                <span class='price-unit'>/ barrel</span>
            </div>
            <div style='font-family:DM Mono,monospace; font-size:0.65rem;
                        color:#8a8580; margin-top:0.3rem;'>
                as of {price_data['date']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick scenario buttons
        st.markdown("""
        <div class='section-label'>Quick Scenarios</div>
        """, unsafe_allow_html=True)

        scenarios = get_scenarios()
        scenario_map = {s["name"]: s["key"] for s in scenarios}

        quick_scenario = st.selectbox(
            "Pick a predefined scenario",
            ["— Select —"] + [s["name"] for s in scenarios],
            label_visibility="collapsed"
        )

        if quick_scenario != "— Select —":
            st.session_state["prefill"] = quick_scenario

        st.markdown("---")

        # Forecast horizon
        forecast_weeks = st.slider(
            "Forecast Horizon (weeks)",
            min_value=4,
            max_value=24,
            value=12,
            step=4
        )

        st.markdown("---")

        # Model info
        st.markdown("""
        <div class='section-label'>Model Info</div>
        <div style='font-family:DM Mono,monospace; font-size:0.7rem;
                    color:#8a8580; line-height:1.8;'>
            Model: SARIMAX(1,1,1)(1,0,1,52)<br>
            Training: 2006 – 2022<br>
            Test MAPE: 15.22%<br>
            LLM: LLaMA 3.3 70B (Groq)<br>
            Variables: 5 macro drivers
        </div>
        """, unsafe_allow_html=True)


    # Main Content 

    # Header
    st.markdown("""
    <div class='main-header'>Crude Oil<br>Scenario Forecaster</div>
    <div class='main-subheader'>
        Macroeconomic Simulation · SARIMAX · LLaMA 3.3
    </div>
    """, unsafe_allow_html=True)

    # Query input
    prefill = st.session_state.get("prefill", "")
    default_text = prefill if prefill else \
        "What happens to oil prices if OPEC cuts production by 10%?"

    query = st.text_area(
        "Enter your scenario in plain English",
        value=default_text,
        height=90,
        placeholder="e.g. What if there's a major war in the Middle East?",
        label_visibility="collapsed"
    )

    # Clear prefill after use
    if "prefill" in st.session_state:
        del st.session_state["prefill"]

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶  RUN SIMULATION", disabled=not api_online)
    with col_hint:
        st.markdown("""
        <div style='font-family:DM Mono,monospace; font-size:0.7rem;
                    color:#8a8580; padding-top:0.6rem;'>
            Ask anything — the AI will map your query to the right scenario
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Results 
    if run_clicked:
        if not query.strip():
            st.warning("Please enter a scenario query.")
            return

        with st.spinner("⚙️  Running simulation..."):
            data, error = run_simulation(query, forecast_weeks)

        if error:
            st.error(f"Simulation failed: {error}")
            return

        # Parsed Query Info 
        confidence = data["parsed_confidence"]
        badge_class = f"badge-{confidence}"
        st.markdown(f"""
        <div style='margin-bottom:1.2rem;'>
            <span class='{badge_class}'>{confidence} confidence</span>
            <span style='font-family:DM Mono,monospace; font-size:0.72rem;
                         color:#8a8580; margin-left:0.8rem;'>
                Mapped to: <strong style='color:#f0ede8;'>
                {data['scenario_name']}</strong>
            </span>
            <div style='font-family:DM Sans,sans-serif; font-size:0.8rem;
                        color:#8a8580; margin-top:0.3rem;'>
                {data['parsed_reasoning']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Impact Metrics Row 
        w1  = data["impact_week1"]
        w12 = data["impact_week12"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            css_class = "metric-value-down" if w1["difference"] < 0 \
                        else "metric-value-up"
            sign = "+" if w1["difference"] >= 0 else ""
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Week 1 Impact</div>
                <div class='{css_class}'>{sign}${w1['difference']:.2f}</div>
                <div class='metric-sub'>{sign}{w1['pct_change']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            css_class = "metric-value-down" if w12["difference"] < 0 \
                        else "metric-value-up"
            sign = "+" if w12["difference"] >= 0 else ""
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Week 12 Impact</div>
                <div class='{css_class}'>{sign}${w12['difference']:.2f}</div>
                <div class='metric-sub'>{sign}{w12['pct_change']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Baseline (Wk 12)</div>
                <div class='metric-value-up'>${w12['baseline']:.2f}</div>
                <div class='metric-sub'>no shock applied</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Scenario (Wk 12)</div>
                <div class='metric-value-up'>${w12['shocked']:.2f}</div>
                <div class='metric-sub'>with shock applied</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 
        col_chart, col_delta = st.columns([3, 2])

        with col_chart:
            st.markdown("""
            <div class='section-label'>Forecast — Baseline vs Scenario</div>
            """, unsafe_allow_html=True)
            fig_forecast = make_forecast_chart(
                data["weekly_forecasts"],
                data["current_price"],
                data["scenario_name"]
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        with col_delta:
            st.markdown("""
            <div class='section-label'>Weekly Price Delta</div>
            """, unsafe_allow_html=True)
            fig_delta = make_forecast_chart(
                data["weekly_forecasts"],
                data["current_price"],
                data["scenario_name"]
            )
            fig_delta = make_delta_chart(data["weekly_forecasts"])
            st.plotly_chart(fig_delta, use_container_width=True)

            # Shocks applied
            st.markdown("""
            <div class='section-label' style='margin-top:1rem;'>
                Shocks Applied
            </div>
            """, unsafe_allow_html=True)

            shock_labels = {
                "dollar_return":  "USD Return",
                "indpro_return":  "Indust. Prod",
                "inventory_pct":  "Inventories",
                "fed_funds_diff": "Fed Funds",
                "vix_diff":       "VIX"
            }

            for var, val in data["shocks_applied"].items():
                if val != 0:
                    label = shock_labels.get(var, var)
                    sign  = "+" if val > 0 else ""
                    css   = "shock-pos" if val > 0 else "shock-neg"
                    st.markdown(f"""
                    <div class='shock-row'>
                        <span class='shock-var'>{label}</span>
                        <span class='{css}'>{sign}{val:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Explanation 
        col_exp, col_unc = st.columns([3, 2])

        with col_exp:
            st.markdown("""
            <div class='section-label'>Economic Explanation</div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='explanation-box'>{data['explanation']}</div>
            """, unsafe_allow_html=True)

        with col_unc:
            st.markdown("""
            <div class='section-label'>Uncertainty Note</div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='uncertainty-box'>⚠️ {data['uncertainty_note']}</div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Raw forecast table
            st.markdown("""
            <div class='section-label'>Raw Forecast Table</div>
            """, unsafe_allow_html=True)

            df = pd.DataFrame(data["weekly_forecasts"])
            df.columns = ["Week", "Baseline ($)", "Scenario ($)", "Δ ($)"]
            df = df.set_index("Week")
            st.dataframe(
                df.style.format("${:.2f}").applymap(
                    lambda v: "color: #e05252" if v < 0 else "color: #4caf82",
                    subset=["Δ ($)"]
                ),
                use_container_width=True,
                height=250
            )

        # Footer 
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace; font-size:0.65rem;
                    color:#8a8580; text-align:center; padding:0.5rem;'>
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
            SARIMAX(1,1,1)(1,0,1,52) · LLaMA 3.3 70B via Groq ·
            Training data: 2006–2022 · Test MAPE: 15.22%
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state — show instructions
        st.markdown("""
        <div style='text-align:center; padding:3rem 0; color:#8a8580;'>
            <div style='font-family:DM Serif Display,serif; 
                        font-size:1.1rem; margin-bottom:0.8rem;
                        color:#2e2e2e; font-size:4rem;'>🛢️</div>
            <div style='font-family:DM Mono,monospace; font-size:0.75rem;
                        letter-spacing:2px; text-transform:uppercase;
                        margin-bottom:0.8rem;'>
                Ready to simulate
            </div>
            <div style='font-family:DM Sans,sans-serif; font-size:0.9rem;
                        max-width:400px; margin:0 auto; line-height:1.7;'>
                Type any oil market scenario above in plain English.
                The system will parse your intent, run the econometric
                model, and explain the results.
            </div>
            <div style='margin-top:1.5rem; font-family:DM Mono,monospace;
                        font-size:0.7rem; color:#3a3a3a;'>
                "What if OPEC cuts production?"<br>
                "Simulate a global recession"<br>
                "What happens if the Fed raises rates?"<br>
                "What if there's a war in the Middle East?"
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()