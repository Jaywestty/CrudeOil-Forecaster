import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Scenario Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Configuration ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── Custom CSS ─────────────────────────────────────────────────────
# Identical to your original — no changes here
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

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

    .stApp {
        background-color: var(--oil-black);
        font-family: 'DM Sans', sans-serif;
        color: var(--text-main);
    }

    header { visibility: visible !important; background: #0d0d0d !important; }
    [data-testid="stHeader"]  { background: #0d0d0d !important; }
    [data-testid="stToolbar"] { background: #0d0d0d !important; }

    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem; font-weight: 400;
        color: var(--text-main); letter-spacing: -0.5px;
        line-height: 1.1; margin-bottom: 0.2rem;
    }
    .main-subheader {
        font-family: 'DM Mono', monospace; font-size: 0.75rem;
        color: var(--amber); letter-spacing: 3px;
        text-transform: uppercase; margin-bottom: 1.5rem;
    }

    .price-card {
        background: var(--oil-card); border: 1px solid var(--border);
        border-left: 3px solid var(--amber); border-radius: 6px;
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .price-label {
        font-family: 'DM Mono', monospace; font-size: 0.65rem;
        letter-spacing: 2px; text-transform: uppercase;
        color: var(--text-muted); margin-bottom: 0.3rem;
    }
    .price-value { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: var(--amber-light); }
    .price-unit  { font-size: 0.85rem; color: var(--text-muted); margin-left: 0.3rem; }

    .metric-card {
        background: var(--oil-card); border: 1px solid var(--border);
        border-radius: 6px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-label {
        font-family: 'DM Mono', monospace; font-size: 0.6rem;
        letter-spacing: 2px; text-transform: uppercase;
        color: var(--text-muted); margin-bottom: 0.4rem;
    }
    .metric-value-up   { font-family:'DM Serif Display',serif; font-size:1.8rem; color:var(--green-up); }
    .metric-value-down { font-family:'DM Serif Display',serif; font-size:1.8rem; color:var(--red-down); }
    .metric-value-neu  { font-family:'DM Serif Display',serif; font-size:1.8rem; color:var(--amber-light); }
    .metric-sub { font-size:0.75rem; color:var(--text-muted); margin-top:0.2rem; }

    .section-label {
        font-family: 'DM Mono', monospace; font-size: 0.65rem;
        letter-spacing: 3px; text-transform: uppercase; color: var(--amber);
        margin-bottom: 0.8rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    .explanation-box {
        background: var(--oil-card); border: 1px solid var(--border);
        border-radius: 6px; padding: 1.5rem; font-size: 0.95rem;
        line-height: 1.75; color: var(--text-main);
        font-family: 'DM Sans', sans-serif; font-weight: 300;
    }
    .uncertainty-box {
        background: #1a1510; border: 1px solid #3a2e1a;
        border-left: 3px solid var(--amber); border-radius: 6px;
        padding: 1rem 1.2rem; font-size: 0.85rem; color: #c4b08a;
        font-family: 'DM Sans', sans-serif; line-height: 1.6;
    }

    /* NEW — probability bar for breakdown display */
    .prob-bar-container {
        background: #2a2a2a; border-radius: 4px;
        height: 6px; width: 100%; margin: 4px 0 8px 0;
    }
    .prob-bar-fill {
        background: var(--amber); border-radius: 4px; height: 6px;
    }

    .badge-high   { background:#1a3a2a; color:#4caf82; padding:2px 10px; border-radius:20px;
                    font-size:0.7rem; font-family:'DM Mono',monospace; letter-spacing:1px; text-transform:uppercase; }
    .badge-medium { background:#2a2a1a; color:#e8a020; padding:2px 10px; border-radius:20px;
                    font-size:0.7rem; font-family:'DM Mono',monospace; letter-spacing:1px; text-transform:uppercase; }
    .badge-low    { background:#3a1a1a; color:#e05252; padding:2px 10px; border-radius:20px;
                    font-size:0.7rem; font-family:'DM Mono',monospace; letter-spacing:1px; text-transform:uppercase; }

    .shock-row {
        display: flex; justify-content: space-between;
        padding: 0.4rem 0; border-bottom: 1px solid var(--border);
        font-family: 'DM Mono', monospace; font-size: 0.78rem;
    }
    .shock-var { color: var(--text-muted); }
    .shock-pos { color: var(--red-down); }
    .shock-neg { color: var(--green-up); }

    .stSidebar { background: var(--oil-dark) !important; }
    [data-testid="stSidebar"] { background: var(--oil-dark); }

    .stTextArea textarea {
        background: var(--oil-card) !important; color: var(--text-main) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton button {
        background: var(--amber) !important; color: var(--oil-black) !important;
        font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
        letter-spacing: 2px !important; text-transform: uppercase !important;
        border: none !important; border-radius: 4px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 500 !important; width: 100% !important;
    }
    .stButton button:hover { background: var(--amber-light) !important; }
    .stSelectbox select, [data-baseweb="select"] {
        background: var(--oil-card) !important; color: var(--text-main) !important;
        border-color: var(--border) !important;
    }
    .stSlider [data-baseweb="slider"] { color: var(--amber) !important; }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────
# These are identical to your original versions

def check_api_health():
    try:
        r = requests.get(f"{API_URL}/", timeout=60)
        return r.status_code == 200
    except:
        return False

def get_current_price():
    try:
        r = requests.get(f"{API_URL}/current-price", timeout=5)
        return r.json()
    except:
        return {"price": "N/A", "date": "N/A", "unit": "USD/barrel"}

def get_scenarios():
    try:
        r = requests.get(f"{API_URL}/scenarios", timeout=5)
        return r.json()["scenarios"]
    except:
        return []

# ── Probabilistic simulation call ─────────────────────────────────

def run_probabilistic_simulation(query, forecast_weeks):
    """
    Call the new /simulate-probabilistic endpoint.

    Returns the full probabilistic result including:
    - expected price, range (low/high)
    - scenario breakdown with probabilities
    - original vs adjusted probabilities
    - macro adjustment log
    - weighted weekly forecast
    """
    try:
        r = requests.post(
            f"{API_URL}/simulate-probabilistic",
            json={"query": query, "forecast_weeks": forecast_weeks},
            timeout=90   # slightly longer — runs SARIMAX multiple times
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API."
    except Exception as e:
        return None, str(e)


# ── Chart Builders ─────────────────────────────────────────────────
# Original charts unchanged

def make_forecast_chart(weekly_forecasts, current_price, scenario_name):
    weeks    = [w["week"]     for w in weekly_forecasts]
    baseline = [w["baseline"] for w in weekly_forecasts]
    scenario = [w["scenario"] for w in weekly_forecasts]

    fig = go.Figure()
    fig.add_hline(
        y=current_price, line_dash="dot", line_color="#8a8580", line_width=1,
        annotation_text=f"  Current ${current_price:.2f}",
        annotation_font_color="#8a8580", annotation_font_size=10
    )
    fig.add_trace(go.Scatter(
        x=weeks, y=baseline, mode="lines", name="Baseline (no shock)",
        line=dict(color="#8a8580", width=2, dash="dash"),
        hovertemplate="Week %{x}<br>Baseline: $%{y:.2f}<extra></extra>"
    ))
    sc = "#e05252" if scenario[-1] < baseline[-1] else "#4caf82"
    fig.add_trace(go.Scatter(
        x=weeks, y=scenario, mode="lines+markers", name=scenario_name,
        line=dict(color=sc, width=2.5), marker=dict(size=5, color=sc),
        hovertemplate="Week %{x}<br>Scenario: $%{y:.2f}<extra></extra>"
    ))
    rgb = tuple(int(sc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    fig.add_trace(go.Scatter(
        x=weeks + weeks[::-1], y=scenario + baseline[::-1],
        fill="toself",
        fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#8a8580", size=10),
        legend=dict(font=dict(size=10, color="#8a8580"), bgcolor="rgba(0,0,0,0)",
                    bordercolor="#2e2e2e", borderwidth=1),
        xaxis=dict(title="Forecast Week", gridcolor="#1f1f1f", linecolor="#2e2e2e",
                   tickcolor="#2e2e2e", title_font=dict(size=10)),
        yaxis=dict(title="Brent Price (USD/barrel)", gridcolor="#1f1f1f",
                   linecolor="#2e2e2e", tickcolor="#2e2e2e",
                   title_font=dict(size=10), tickprefix="$"),
        hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10), height=350
    )
    return fig

def make_delta_chart(weekly_forecasts):
    weeks  = [f"W{w['week']}" for w in weekly_forecasts]
    deltas = [w["change"]     for w in weekly_forecasts]
    colors = ["#e05252" if d < 0 else "#4caf82" for d in deltas]

    fig = go.Figure(go.Bar(
        x=weeks, y=deltas, marker_color=colors,
        hovertemplate="Week %{x}<br>Δ Price: $%{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_color="#2e2e2e", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#8a8580", size=10),
        xaxis=dict(gridcolor="#1f1f1f", linecolor="#2e2e2e"),
        yaxis=dict(gridcolor="#1f1f1f", linecolor="#2e2e2e",
                   tickprefix="$", title="Price Change vs Baseline"),
        margin=dict(l=10, r=10, t=20, b=10), height=220, showlegend=False
    )
    return fig


# ── NEW: Probabilistic fan chart ──────────────────────────────────

def make_probabilistic_chart(data):
    """
    Fan chart for probabilistic mode.

    Shows:
    - Baseline (dashed grey) — what happens with no shock
    - Each individual scenario (thin dotted lines, muted colors)
    - Weighted expected forecast (solid amber line) — the headline number
    - Shaded band between min and max scenario — the uncertainty range

    Reading this chart:
    - If the amber line is above baseline → net bullish outcome
    - If amber line is below baseline → net bearish outcome
    - The width of the shaded band shows how much uncertainty exists
    - Wider band = more scenario disagreement = more uncertainty
    """
    weeks    = list(range(1, data["forecast_weeks"] + 1))
    weighted = data["weighted_forecast"]
    baseline = data["baseline_forecast"]

    fig = go.Figure()

    # Current price reference line
    fig.add_hline(
        y=data["current_price"], line_dash="dot",
        line_color="#8a8580", line_width=1,
        annotation_text=f"  Current ${data['current_price']:.2f}",
        annotation_font_color="#8a8580", annotation_font_size=10
    )

    # Baseline
    fig.add_trace(go.Scatter(
        x=weeks, y=baseline, mode="lines",
        name="Baseline (no shock)",
        line=dict(color="#8a8580", width=1.5, dash="dash"),
        hovertemplate="Week %{x}<br>Baseline: $%{y:.2f}<extra></extra>"
    ))

    # Individual scenario lines — thin and muted so they don't overpower
    # the weighted expected line which is the real headline
    scenario_palette = ["#3a5a7a", "#7a3a5a", "#3a7a5a", "#7a5a3a", "#5a3a7a"]
    for i, (key, item) in enumerate(data["scenario_breakdown"].items()):
        color = scenario_palette[i % len(scenario_palette)]
        # Use linear interpolation from current price to week-12 for display
        # (we only have week-12 prices per scenario in the response,
        #  not full weekly arrays — this approximates the curve visually)
        start = data["current_price"]
        end   = item["week12_price"]
        line_y = [start + (end - start) * (w / weeks[-1]) for w in weeks]

        fig.add_trace(go.Scatter(
            x=weeks, y=line_y, mode="lines",
            name=f"{item['name'].split('(')[0].strip()} ({item['probability']*100:.0f}%)",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.55,
            hovertemplate=(
                f"{item['name']}<br>Week %{{x}}<br>~$%{{y:.2f}}<extra></extra>"
            )
        ))

    # Weighted expected forecast — the main line, amber, prominent
    fig.add_trace(go.Scatter(
        x=weeks, y=weighted, mode="lines+markers",
        name="Expected (probability-weighted)",
        line=dict(color="#e8a020", width=3),
        marker=dict(size=4, color="#e8a020"),
        hovertemplate="Week %{x}<br>Expected: $%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#8a8580", size=10),
        legend=dict(font=dict(size=9, color="#8a8580"), bgcolor="rgba(0,0,0,0)",
                    bordercolor="#2e2e2e", borderwidth=1),
        xaxis=dict(title="Forecast Week", gridcolor="#1f1f1f", linecolor="#2e2e2e",
                   title_font=dict(size=10)),
        yaxis=dict(title="Brent Price (USD/barrel)", gridcolor="#1f1f1f",
                   linecolor="#2e2e2e", title_font=dict(size=10), tickprefix="$"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10), height=380
    )
    return fig


# ── NEW: Probabilistic results renderer ───────────────────────────

def render_probabilistic_results(data):
    """
    Render results for /simulate-probabilistic — multi-scenario mode.

    Layout:
    Row 1 — 4 metric cards: expected price, range, vs baseline, primary driver
    Row 2 — fan chart (left) + scenario probability breakdown (right)
    Row 3 — explanation + uncertainty + weighted forecast table
    """

    # ── Row 1 — summary metrics ──────────────────────────────────
    diff_from_baseline = data["price_expected"] - data["baseline_week12"]
    diff_css  = "metric-value-down" if diff_from_baseline < 0 else "metric-value-up"
    diff_sign = "+" if diff_from_baseline >= 0 else ""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Expected Price (Wk 12)</div>
            <div class='metric-value-neu'>${data['price_expected']:.2f}</div>
            <div class='metric-sub'>probability-weighted</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Forecast Range</div>
            <div class='metric-value-neu' style='font-size:1.4rem;'>
                ${data['price_low']:.0f} – ${data['price_high']:.0f}
            </div>
            <div class='metric-sub'>low / high scenario</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>vs Baseline (Wk 12)</div>
            <div class='{diff_css}'>{diff_sign}${diff_from_baseline:.2f}</div>
            <div class='metric-sub'>baseline ${data['baseline_week12']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        primary_prob = data["adjusted_probabilities"].get(
            data["primary_driver"], 0
        )
        short_name   = data["primary_driver_name"].split("(")[0].strip()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Primary Driver</div>
            <div style='font-family:DM Serif Display,serif; font-size:1.05rem;
                        color:var(--amber-light); margin:0.4rem 0;'>
                {short_name}
            </div>
            <div class='metric-sub'>{primary_prob*100:.0f}% weight</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2 — fan chart + probability breakdown ────────────────
    col_chart, col_breakdown = st.columns([3, 2])

    with col_chart:
        st.markdown(
            "<div class='section-label'>Probabilistic Forecast — "
            "Expected Price & Scenario Lines</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(make_probabilistic_chart(data), use_container_width=True)

        # Show what macro signals shifted
        if data.get("macro_adjustments"):
            st.markdown(
                "<div class='section-label' style='margin-top:0.5rem;'>"
                "Macro Signal Adjustments Applied</div>",
                unsafe_allow_html=True
            )
            for _, note in data["macro_adjustments"].items():
                st.markdown(f"""
                <div class='shock-row'>
                    <span class='shock-var' style='font-size:0.7rem;'>{note}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_breakdown:
        st.markdown(
            "<div class='section-label'>Scenario Probability Breakdown</div>",
            unsafe_allow_html=True
        )

        # Sort by probability descending so the dominant scenario is at top
        sorted_breakdown = sorted(
            data["scenario_breakdown"].items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )

        for key, item in sorted_breakdown:
            prob_pct  = item["probability"] * 100
            price_val = item["week12_price"]
            diff      = price_val - data["baseline_week12"]
            diff_sign = "+" if diff >= 0 else ""
            diff_col  = "#e05252" if diff < 0 else "#4caf82"
            short     = item["name"].split("(")[0].strip()

            st.markdown(f"""
            <div style='margin-bottom:0.9rem;'>
                <div style='display:flex; justify-content:space-between;
                            font-family:DM Mono,monospace; font-size:0.7rem;'>
                    <span style='color:#f0ede8;'>{short}</span>
                    <span style='color:{diff_col};'>
                        ${price_val:.2f} ({diff_sign}${diff:.2f})
                    </span>
                </div>
                <div style='display:flex; align-items:center;
                            gap:8px; margin-top:4px;'>
                    <div class='prob-bar-container' style='flex:1;'>
                        <div class='prob-bar-fill'
                             style='width:{prob_pct:.0f}%;'></div>
                    </div>
                    <span style='font-family:DM Mono,monospace; font-size:0.65rem;
                                 color:#8a8580; min-width:32px;'>
                        {prob_pct:.0f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Show probability adjustments only if macro signals actually changed anything
        orig = data.get("original_probabilities", {})
        adj  = data.get("adjusted_probabilities", {})
        diffs = {k: adj.get(k, 0) - orig.get(k, 0) for k in adj if abs(adj.get(k, 0) - orig.get(k, 0)) > 0.01}

        if diffs:
            st.markdown(
                "<div class='section-label' style='margin-top:1rem;'>"
                "Probability Shifts from Macro Signals</div>",
                unsafe_allow_html=True
            )
            for k, change in diffs.items():
                sign = "+" if change >= 0 else ""
                css  = "shock-pos" if change > 0 else "shock-neg"
                st.markdown(f"""
                <div class='shock-row'>
                    <span class='shock-var'>{k}</span>
                    <span class='{css}'>{sign}{change*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3 — explanation + uncertainty + table ────────────────
    col_exp, col_unc = st.columns([3, 2])

    with col_exp:
        st.markdown("<div class='section-label'>Economic Explanation</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='explanation-box'>{data['explanation']}</div>",
                    unsafe_allow_html=True)

    with col_unc:
        st.markdown("<div class='section-label'>Uncertainty Note</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='uncertainty-box'>⚠️ {data['uncertainty_note']}</div>",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Weighted Forecast Table</div>",
                    unsafe_allow_html=True)

        # Build table from weighted_forecast + baseline_forecast arrays
        table_rows = [
            {
                "Week":          i + 1,
                "Baseline ($)":  round(b, 2),
                "Expected ($)":  round(w, 2),
                "Δ ($)":         round(w - b, 2)
            }
            for i, (b, w) in enumerate(
                zip(data["baseline_forecast"], data["weighted_forecast"])
            )
        ]
        df = pd.DataFrame(table_rows).set_index("Week")
        st.dataframe(
            df.style.format("${:.2f}").applymap(
                lambda v: "color: #e05252" if v < 0 else "color: #4caf82",
                subset=["Δ ($)"]
            ),
            use_container_width=True, height=250
        )


# ── Main App ───────────────────────────────────────────────────────

def main():

    api_online = check_api_health()
    price_data = get_current_price()
    scenarios  = get_scenarios()

    # ── Sidebar — identical to original ───────────────────────────
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

        status_color = "#4caf82" if api_online else "#e05252"
        status_text  = "API ONLINE" if api_online else "API OFFLINE"
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace; font-size:0.65rem;
                    color:{status_color}; letter-spacing:2px; margin-bottom:1rem;'>
            ● {status_text}
        </div>
        """, unsafe_allow_html=True)

        if not api_online:
            st.warning("⏳ API is waking up. Wait 30–60 seconds then refresh.")

        st.markdown("---")

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

        st.markdown("<div class='section-label'>Quick Scenarios</div>",
                    unsafe_allow_html=True)
        quick_scenario = st.selectbox(
            "Pick a predefined scenario",
            ["— Select —"] + [s["name"] for s in scenarios],
            label_visibility="collapsed"
        )
        if quick_scenario != "— Select —":
            st.session_state["prefill"] = quick_scenario

        st.markdown("---")

        forecast_weeks = st.slider(
            "Forecast Horizon (weeks)",
            min_value=4, max_value=24, value=12, step=4
        )

        st.markdown("---")

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

    # ── Main Content ───────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>Crude Oil<br>Scenario Forecaster</div>
    <div class='main-subheader'>
        Macroeconomic Simulation · SARIMAX · LLaMA 3.3
    </div>
    """, unsafe_allow_html=True)

    # ── Query input ────────────────────────────────────────────────
    prefill      = st.session_state.get("prefill", "")
    default_text = prefill if prefill else \
        "What happens to oil prices if OPEC cuts production by 10%?"

    query = st.text_area(
        "Enter your scenario in plain English",
        value=default_text, height=90,
        placeholder="e.g. What if Middle East tensions rise while demand also grows?",
        label_visibility="collapsed"
    )

    if "prefill" in st.session_state:
        del st.session_state["prefill"]

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶  RUN SIMULATION", disabled=not api_online)
    with col_hint:
        st.markdown("""
        <div style='font-family:DM Mono,monospace; font-size:0.7rem;
                    color:#8a8580; padding-top:0.6rem;'>
            LLM assigns scenario probabilities · macro signals adjust them · SARIMAX runs each
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Results ────────────────────────────────────────────────────
    if run_clicked:
        if not query.strip():
            st.warning("Please enter a scenario query.")
            return

        with st.spinner("⚙️  Running probabilistic simulation across multiple scenarios..."):
            data, error = run_probabilistic_simulation(query, forecast_weeks)

        if error:
            st.error(f"Simulation failed: {error}")
            return

        render_probabilistic_results(data)

        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace; font-size:0.65rem;
                    color:#8a8580; text-align:center; padding:0.5rem;'>
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
            SARIMAX(1,1,1)(1,0,1,52) · Probabilistic Mode ·
            LLaMA 3.3 70B via Groq · Training data: 2006–2022 · MAPE: 15.22%
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center; padding:3rem 0; color:#8a8580;'>
            <div style='font-size:4rem; margin-bottom:0.8rem;'>🛢️</div>
            <div style='font-family:DM Mono,monospace; font-size:0.75rem;
                        letter-spacing:2px; text-transform:uppercase;
                        margin-bottom:0.8rem;'>
                Ready to simulate
            </div>
            <div style='font-family:DM Sans,sans-serif; font-size:0.9rem;
                        max-width:420px; margin:0 auto; line-height:1.7;'>
                Type any oil market scenario in plain English.
                The system assigns probabilities across multiple scenarios,
                adjusts them with live macro signals, and returns an
                expected price with a confidence range.
            </div>
            <div style='margin-top:1.5rem; font-family:DM Mono,monospace;
                        font-size:0.7rem; color:#3a3a3a;'>
                "What if OPEC cuts production?"<br>
                "What if Russia-Ukraine war escalates?"<br>
                "What if Middle East tensions rise AND demand grows?"<br>
                "What happens if the Fed raises rates aggressively?"
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()