import html
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Scenario Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Configuration ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Horizons the walk-forward backtest validated (1–12 weeks).
HORIZONS = [4, 8, 12]

EXAMPLE_QUERIES = [
    "What if OPEC cuts production by 10%?",
    "What if Russia-Ukraine tensions escalate?",
    "What if Middle East tensions rise and demand grows?",
    "What if the Fed raises rates aggressively?",
]

# Facts from the walk-forward evaluation (reports/backtest, reports/calibration).
EVALUATION_FACTS = [
    "198 weekly forecasts, 2022–2025, made using only data available at the time",
    "No model (SARIMAX, Ridge, gradient boosting) beat a no-change forecast",
    "95% band coverage: ~85% at 1 week, ~92% at 4, ~97% at 12; lower in the 2022 crisis",
]

# ── Theme ──────────────────────────────────────────────────────────
AMBER = "#e8a020"
AMBER_LIGHT = "#f5c842"
RED = "#e05252"
GREEN = "#4caf82"
MUTED = "#8a8580"
GRID = "#1f1f1f"
LINE = "#2e2e2e"
SCENARIO_PALETTE = ["#5b8db8", "#b85b8d", "#5bb88d", "#b88d5b", "#8d5bb8"]

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

    .stApp { background-color: var(--oil-black); font-family: 'DM Sans', sans-serif; color: var(--text-main); }
    header, [data-testid="stHeader"], [data-testid="stToolbar"] { background: #0d0d0d !important; }

    .main-header {
        font-family: 'DM Serif Display', serif; font-size: 2.4rem; font-weight: 400;
        color: var(--text-main); letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 0.2rem;
    }
    .main-subheader {
        font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--amber);
        letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.2rem;
    }

    .price-card {
        background: var(--oil-card); border: 1px solid var(--border);
        border-left: 3px solid var(--amber); border-radius: 6px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem;
    }
    .price-label, .metric-label {
        font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 2px;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;
    }
    .price-value { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: var(--amber-light); }
    .price-unit  { font-size: 0.85rem; color: var(--text-muted); margin-left: 0.3rem; }
    .small-mono  { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-muted); }

    .metric-card {
        background: var(--oil-card); border: 1px solid var(--border); border-radius: 6px;
        padding: 0.95rem 1.1rem; text-align: center; height: 100%;
    }
    .metric-value { font-family: 'DM Serif Display', serif; font-size: 1.75rem; color: var(--amber-light); }
    .metric-value.up   { color: var(--green-up); }
    .metric-value.down { color: var(--red-down); }
    .metric-value.text { font-size: 1.1rem; margin: 0.35rem 0; }
    .metric-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; }

    .section-label {
        font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 3px;
        text-transform: uppercase; color: var(--amber); margin: 0.4rem 0 0.8rem 0;
        padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
    }

    .explanation-box {
        background: var(--oil-card); border: 1px solid var(--border); border-radius: 6px;
        padding: 1.4rem; font-size: 0.95rem; line-height: 1.75; color: var(--text-main);
        font-weight: 300; overflow-wrap: break-word;
    }
    .note-box {
        background: #1a1510; border: 1px solid #3a2e1a; border-left: 3px solid var(--amber);
        border-radius: 6px; padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #c4b08a; line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    .honesty-banner {
        background: #14181c; border: 1px solid #263039; border-left: 3px solid #5b8db8;
        border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.82rem; color: #a9b7c4; margin-bottom: 1rem;
    }

    .prob-row { margin-bottom: 0.85rem; }
    .prob-head { display: flex; justify-content: space-between; font-family: 'DM Mono', monospace; font-size: 0.72rem; }
    .prob-bar-container { background: #2a2a2a; border-radius: 4px; height: 6px; flex: 1; }
    .prob-bar-fill { background: var(--amber); border-radius: 4px; height: 6px; }

    .info-row {
        display: flex; justify-content: space-between; gap: 1rem; padding: 0.4rem 0;
        border-bottom: 1px solid var(--border); font-family: 'DM Mono', monospace; font-size: 0.74rem;
    }
    .info-key { color: var(--text-muted); }

    [data-testid="stSidebar"] { background: var(--oil-dark); }
    .stTextArea textarea {
        background: var(--oil-card) !important; color: var(--text-main) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    .stButton button[kind="primary"] {
        background: var(--amber) !important; color: var(--oil-black) !important;
        font-family: 'DM Mono', monospace !important; letter-spacing: 2px !important;
        text-transform: uppercase !important; border: none !important; font-weight: 500 !important;
    }
    .stButton button[kind="secondary"] {
        background: transparent !important; color: var(--text-muted) !important;
        border: 1px solid var(--border) !important; font-size: 0.78rem !important;
    }
    .stButton button[kind="secondary"]:hover { color: var(--amber-light) !important; border-color: var(--amber) !important; }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────

def safe_text(text) -> str:
    """Escape model/API text before placing it inside HTML.

    LLM output is untrusted: escaping stops it injecting markup. '$' becomes an
    HTML entity because Streamlit would otherwise treat $...$ as LaTeX.
    """
    escaped = html.escape(str(text or ""))
    return escaped.replace("$", "&#36;").replace("\n", "<br>")


def set_query(text: str) -> None:
    """Button callback. Callbacks run before the rerun, so they may change a
    widget's value; assigning after the text box is drawn would raise."""
    st.session_state["query"] = text


def short_name(name: str) -> str:
    return name.split("(")[0].strip()


def money(x: float, decimals: int = 2) -> str:
    return f"&#36;{x:,.{decimals}f}"


def metric_card(label: str, value: str, sub: str = "", css: str = "") -> str:
    return (
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value {css}'>{value}</div><div class='metric-sub'>{sub}</div></div>"
    )


# ── API calls ──────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_api_status():
    """Root endpoint: online flag plus model info (cached briefly)."""
    try:
        r = requests.get(f"{API_URL}/", timeout=60)
        if r.status_code == 200:
            return True, r.json().get("model_info", {})
    except requests.RequestException:
        pass
    return False, {}


@st.cache_data(ttl=300, show_spinner=False)
def get_current_price():
    """Current Brent price (live from FRED via the API, or dataset fallback)."""
    try:
        r = requests.get(f"{API_URL}/current-price", timeout=30)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {"price": None, "date": "N/A", "unit": "USD/barrel", "source": "unavailable"}


@st.cache_data(ttl=3600, show_spinner=False)
def get_scenarios():
    try:
        r = requests.get(f"{API_URL}/scenarios", timeout=10)
        if r.status_code == 200:
            return r.json()["scenarios"]
    except requests.RequestException:
        pass
    return []


def run_probabilistic_simulation(query, forecast_weeks):
    try:
        r = requests.post(
            f"{API_URL}/simulate-probabilistic",
            json={"query": query, "forecast_weeks": forecast_weeks},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json(), None
        try:
            return None, r.json().get("detail", f"HTTP {r.status_code}")
        except ValueError:
            return None, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. It may be waking up; try again in a minute."
    except requests.RequestException as e:
        return None, str(e)


# ── Data shaping ───────────────────────────────────────────────────

def x_axis(data):
    """Calendar dates when the API provides them, otherwise week numbers."""
    dates = data.get("forecast_dates")
    if dates:
        return pd.to_datetime(dates), "Forecast week (date)"
    return list(range(1, data["forecast_weeks"] + 1)), "Forecast week"


def scenario_path(data, item):
    """Full weekly path from the API; older APIs only sent the final price."""
    if item.get("forecast"):
        return item["forecast"]
    n, start, end = data["forecast_weeks"], data["current_price"], item["week12_price"]
    return [start + (end - start) * (w / n) for w in range(1, n + 1)]


def final_probabilities(data):
    return data.get("final_probabilities") or {k: v["probability"] for k, v in data["scenario_breakdown"].items()}


def forecast_table(data) -> pd.DataFrame:
    x, _ = x_axis(data)
    df = pd.DataFrame({"Week": range(1, data["forecast_weeks"] + 1)})
    if data.get("forecast_dates"):
        df["Date"] = [d.date() for d in x]
    df["Baseline"] = data["baseline_forecast"]
    band = data.get("statistical_band")
    if band:
        df["80% low"], df["80% high"] = band["level_80"]["lower"], band["level_80"]["upper"]
        df["95% low"], df["95% high"] = band["level_95"]["lower"], band["level_95"]["upper"]
    df["Expected"] = data["weighted_forecast"]
    df["Expected − baseline"] = df["Expected"] - df["Baseline"]
    for key, item in data["scenario_breakdown"].items():
        df[short_name(item["name"])] = scenario_path(data, item)
    return df.set_index("Week")


# ── Charts ─────────────────────────────────────────────────────────

def _base_layout(fig, x_title, height):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=MUTED, size=10),
        legend=dict(font=dict(size=9, color=MUTED), bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=x_title, gridcolor=GRID, linecolor=LINE, title_font=dict(size=10)),
        yaxis=dict(title="Brent (USD/barrel)", gridcolor=GRID, linecolor=LINE,
                   title_font=dict(size=10), tickprefix="$"),
        hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10), height=height,
    )
    return fig


def make_forecast_chart(data):
    """Baseline with statistical bands, each scenario's path, and the weighted expectation."""
    x, x_title = x_axis(data)
    xs = list(x)
    fig = go.Figure()

    band = data.get("statistical_band")
    if band:
        for level, alpha in (("level_95", 0.10), ("level_80", 0.20)):
            lo, hi = band[level]["lower"], band[level]["upper"]
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1], y=hi + lo[::-1], fill="toself",
                fillcolor=f"rgba(138,133,128,{alpha})", line=dict(color="rgba(0,0,0,0)"),
                name=f"{level[-2:]}% statistical band", hoverinfo="skip",
            ))

    fig.add_hline(
        y=data["current_price"], line_dash="dot", line_color=MUTED, line_width=1,
        annotation_text=f"  Latest ${data['current_price']:.2f}",
        annotation_font_color=MUTED, annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=xs, y=data["baseline_forecast"], mode="lines", name="Baseline (no shock)",
        line=dict(color=MUTED, width=1.5, dash="dash"),
        hovertemplate="Baseline: $%{y:.2f}<extra></extra>",
    ))

    ordered = sorted(data["scenario_breakdown"].items(), key=lambda kv: -kv[1]["probability"])
    for i, (_, item) in enumerate(ordered):
        fig.add_trace(go.Scatter(
            x=xs, y=scenario_path(data, item), mode="lines",
            name=f"{short_name(item['name'])} ({item['probability'] * 100:.0f}%)",
            line=dict(color=SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)], width=1.4, dash="dot"),
            hovertemplate=f"{short_name(item['name'])}: $%{{y:.2f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=xs, y=data["weighted_forecast"], mode="lines+markers", name="Expected (probability-weighted)",
        line=dict(color=AMBER, width=3), marker=dict(size=4, color=AMBER),
        hovertemplate="Expected: $%{y:.2f}<extra></extra>",
    ))
    return _base_layout(fig, x_title, 430)


def make_history_chart(history, names):
    """One dot per past event (hollow = overlapped a bigger shock), diamond = median."""
    fig = go.Figure()
    fig.add_vline(x=0, line_color=LINE, line_width=1)
    for i, (key, ctx) in enumerate(history.items()):
        label = names.get(key, key)
        color = SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        events = ctx["events"]
        fig.add_trace(go.Scatter(
            x=[e["move_pct"] for e in events], y=[label] * len(events), mode="markers",
            name=label, showlegend=False,
            marker=dict(size=11, color=[("rgba(0,0,0,0)" if e["confounded"] else color) for e in events],
                        line=dict(color=color, width=1.5)),
            customdata=[[e["date"], e["event"]] for e in events],
            hovertemplate="%{customdata[1]}<br>%{customdata[0]}: %{x:+.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[ctx["median_pct"]], y=[label], mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=14, color=AMBER, line=dict(color="#0d0d0d", width=1)),
            hovertemplate=f"Median: %{{x:+.1f}}% ({ctx['n_events']} events)<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=MUTED, size=10),
        xaxis=dict(title="Brent move after the event (%)", gridcolor=GRID, linecolor=LINE, ticksuffix="%",
                   zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor=LINE),
        margin=dict(l=10, r=10, t=10, b=10), height=90 + 60 * max(len(history), 1),
    )
    return fig


def make_probability_chart(data):
    """LLM estimate → after macro adjustment → final weights, per scenario."""
    names = {k: short_name(v["name"]) for k, v in data["scenario_breakdown"].items()}
    keys = list(dict.fromkeys(
        list(data.get("original_probabilities", {})) + list(data.get("adjusted_probabilities", {}))
    ))
    final = final_probabilities(data)
    stages = [
        ("LLM estimate", data.get("original_probabilities", {}), "#5a5652"),
        ("After macro signals", data.get("adjusted_probabilities", {}), "#a07018"),
        ("Final (used)", final, AMBER),
    ]
    fig = go.Figure()
    total_orig = sum(data.get("original_probabilities", {}).values()) or 1.0
    for label, probs, color in stages:
        scale = total_orig if label == "LLM estimate" else 1.0
        fig.add_trace(go.Bar(
            y=[names.get(k, k.replace("_", " ").title()) for k in keys],
            x=[100 * probs.get(k, 0) / scale for k in keys],
            name=label, orientation="h", marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=MUTED, size=10),
        legend=dict(font=dict(size=9, color=MUTED), orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Probability (%)", gridcolor=GRID, linecolor=LINE, ticksuffix="%"),
        yaxis=dict(gridcolor=GRID, linecolor=LINE, autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10), height=130 + 55 * max(len(keys), 1),
    )
    return fig


# ── Results ────────────────────────────────────────────────────────

def render_summary(data):
    n = data["forecast_weeks"]
    dates = data.get("forecast_dates")
    when = f"week {n}" + (f" · {dates[-1]}" if dates else "")
    diff = data["price_expected"] - data["baseline_week12"]
    sign = "+" if diff >= 0 else "−"
    final = final_probabilities(data)
    primary = data["primary_driver"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card(f"Expected price", money(data["price_expected"]), when), unsafe_allow_html=True)
    c2.markdown(metric_card(
        "Scenario spread", f"{money(data['price_low'], 0)} – {money(data['price_high'], 0)}",
        "lowest / highest scenario"), unsafe_allow_html=True)
    band = data.get("statistical_band")
    if band:
        lo, hi = band["level_80"]["lower"][-1], band["level_80"]["upper"][-1]
        c3.markdown(metric_card("80% statistical band", f"{money(lo, 0)} – {money(hi, 0)}",
                                f"around the baseline, {when}"), unsafe_allow_html=True)
    else:
        c3.markdown(metric_card("vs baseline", f"{sign}{money(abs(diff))}",
                                f"baseline {money(data['baseline_week12'])}",
                                "up" if diff >= 0 else "down"), unsafe_allow_html=True)
    c4.markdown(metric_card("Primary driver", safe_text(short_name(data["primary_driver_name"])),
                            f"{final.get(primary, 0) * 100:.0f}% weight · {sign}{money(abs(diff))} vs baseline",
                            "text"), unsafe_allow_html=True)


def render_forecast_tab(data):
    st.plotly_chart(make_forecast_chart(data), width="stretch")
    st.markdown(
        "<div class='small-mono'>Grey bands: the model's own uncertainty around the no-shock baseline. "
        "Dotted lines: each scenario. Amber: probability-weighted expectation. "
        "The bands describe typical conditions and were too narrow in the 2022 crisis.</div>",
        unsafe_allow_html=True,
    )


def render_scenarios_tab(data):
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='section-label'>How the weights were set</div>", unsafe_allow_html=True)
        st.plotly_chart(make_probability_chart(data), width="stretch")
        dropped = data.get("dropped_scenarios") or {}
        if dropped:
            names = ", ".join(k.replace("_", " ") for k in dropped)
            st.markdown(f"<div class='small-mono'>Dropped (under 5% weight, rest rescaled): {safe_text(names)}</div>",
                        unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='section-label'>Week-{data['forecast_weeks']} price by scenario</div>",
                    unsafe_allow_html=True)
        for _, item in sorted(data["scenario_breakdown"].items(), key=lambda kv: -kv[1]["probability"]):
            pct = item["probability"] * 100
            price = item["week12_price"]
            d = price - data["baseline_week12"]
            color = GREEN if d >= 0 else RED
            st.markdown(f"""
            <div class='prob-row'>
              <div class='prob-head'>
                <span>{safe_text(short_name(item['name']))}</span>
                <span style='color:{color};'>{money(price)} ({'+' if d >= 0 else '−'}{money(abs(d))})</span>
              </div>
              <div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>
                <div class='prob-bar-container'><div class='prob-bar-fill' style='width:{pct:.0f}%;'></div></div>
                <span class='small-mono' style='min-width:34px;'>{pct:.0f}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        notes = data.get("macro_adjustments") or {}
        st.markdown("<div class='section-label'>Macro signals applied</div>", unsafe_allow_html=True)
        if notes:
            for note in notes.values():
                st.markdown(f"<div class='info-row'><span>{safe_text(note)}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='small-mono'>No macro signal was strong enough to shift the weights.</div>",
                        unsafe_allow_html=True)

    render_history_section(data)


def render_history_section(data):
    """What Brent did after past events of each scenario type (descriptive only)."""
    history = data.get("historical_context") or {}
    if not history:
        return
    names = {k: short_name(v["name"]) for k, v in data["scenario_breakdown"].items()}
    week = next(iter(history.values()))["week"]
    st.markdown(f"<div class='section-label' style='margin-top:1.2rem;'>What happened after similar events "
                f"(week {week})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='note-box'>📚 {safe_text(data.get('historical_note', ''))}</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_history_chart(history, names), width="stretch")
    for key, ctx in history.items():
        label = (f"{names.get(key, key)}: {ctx['n_events']} events, {ctx['low_pct']:+.1f}% to "
                 f"{ctx['high_pct']:+.1f}% (median {ctx['median_pct']:+.1f}%) · today's price → "
                 f"${ctx['price_low']:,.2f} to ${ctx['price_high']:,.2f}")
        with st.expander(label.replace("$", "\\$")):
            rows = pd.DataFrame(ctx["events"]).rename(columns={
                "date": "Date", "event": "Event", "move_pct": f"Move by week {ctx['week']} (%)",
                "confounded": "Overlaps a bigger shock"})
            rows["Overlaps a bigger shock"] = rows["Overlaps a bigger shock"].map({True: "yes", False: ""})
            st.dataframe(rows, hide_index=True, width="stretch")


def render_explanation_tab(data):
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='section-label'>Analyst explanation</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='explanation-box'>{safe_text(data['explanation'])}</div>",
                    unsafe_allow_html=True)
        if data.get("reasoning"):
            st.markdown(f"<div class='small-mono' style='margin-top:0.6rem;'>How the query was read: "
                        f"{safe_text(data['reasoning'])}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-label'>Uncertainty</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='note-box'>⚠️ {safe_text(data['uncertainty_note'])}</div>", unsafe_allow_html=True)
        band = data.get("statistical_band")
        if band:
            st.markdown(f"<div class='note-box'>📏 {safe_text(band['note'])}</div>", unsafe_allow_html=True)


def render_data_tab(data):
    df = forecast_table(data)
    money_cols = [c for c in df.columns if c != "Date"]
    st.dataframe(
        df.style.format("${:.2f}", subset=money_cols).map(
            lambda v: f"color: {RED}" if v < 0 else f"color: {GREEN}", subset=["Expected − baseline"]
        ),
        width="stretch", height=min(80 + 35 * len(df), 480),
    )
    st.download_button(
        "Download forecast (CSV)", df.to_csv().encode("utf-8"),
        file_name=f"brent_scenarios_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
    )


def render_results(data, query):
    st.markdown(f"<div class='small-mono' style='margin-bottom:0.6rem;'>Results for: "
                f"<span style='color:#f0ede8;'>“{safe_text(query)}”</span></div>", unsafe_allow_html=True)
    render_summary(data)
    st.markdown("<br>", unsafe_allow_html=True)
    tab_fc, tab_sc, tab_ex, tab_data = st.tabs(["📈 Forecast", "🎯 Scenarios", "📝 Explanation", "🔢 Data"])
    with tab_fc:
        render_forecast_tab(data)
    with tab_sc:
        render_scenarios_tab(data)
    with tab_ex:
        render_explanation_tab(data)
    with tab_data:
        render_data_tab(data)


# ── Sidebar ────────────────────────────────────────────────────────

def render_sidebar(api_online, model_info, price_data, scenarios):
    with st.sidebar:
        st.markdown("""
        <div style='font-family:DM Serif Display,serif; font-size:1.4rem; color:#f0ede8;'>🛢️ Oil Forecaster</div>
        <div style='font-family:DM Mono,monospace; font-size:0.6rem; letter-spacing:2px; color:#e8a020;
                    text-transform:uppercase; margin-bottom:1.2rem;'>Scenario Simulation System</div>
        """, unsafe_allow_html=True)

        color, label = (GREEN, "API ONLINE") if api_online else (RED, "API OFFLINE")
        st.markdown(f"<div class='small-mono' style='color:{color}; letter-spacing:2px; margin-bottom:0.8rem;'>"
                    f"● {label}</div>", unsafe_allow_html=True)
        if not api_online:
            st.warning("The API may be waking up (free hosting sleeps). Wait 30–60 seconds, then refresh.")
            if st.button("Retry connection"):
                get_api_status.clear()
                st.rerun()

        price = price_data.get("price")
        # Exact match: the fallback label "dataset (FRED unavailable)" also contains "FRED".
        is_live = price_data.get("source") == "FRED live"
        price_html = f"${price:,.2f}" if isinstance(price, (int, float)) else "—"
        st.markdown(f"""
        <div class='price-card'>
            <div class='price-label'>Brent crude — latest</div>
            <div class='price-value'>{price_html}<span class='price-unit'>/ barrel</span></div>
            <div class='small-mono'>as of {safe_text(price_data.get('date'))}</div>
            <div class='small-mono' style='color:{GREEN if is_live else MUTED};'>
                ● {'live (FRED)' if is_live else 'dataset fallback'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Quick scenarios</div>", unsafe_allow_html=True)
        names = [s["name"] for s in scenarios]
        choice = st.selectbox("Predefined scenario", ["— Select —"] + names, label_visibility="collapsed")
        if choice != "— Select —":
            st.button("Use this scenario", width="stretch", on_click=set_query,
                      args=(f"What happens to oil prices under: {choice}?",))

        st.markdown("<div class='section-label'>Horizon</div>", unsafe_allow_html=True)
        weeks = st.select_slider("Forecast horizon (weeks)", options=HORIZONS, value=12,
                                 label_visibility="collapsed")
        st.caption("Capped at 12 weeks: the horizon the backtests validated.")

        with st.expander("About the model", expanded=False):
            rows = {
                "Forecaster": model_info.get("forecaster", "SARIMAX(1,1,1)(1,0,1,52)"),
                "LLM": model_info.get("llm", "—"),
                "Data through": model_info.get("data_as_of", "—"),
            }
            for k, v in rows.items():
                st.markdown(f"<div class='info-row'><span class='info-key'>{k}</span>"
                            f"<span>{safe_text(v)}</span></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-label' style='margin-top:0.8rem;'>Evaluation</div>",
                        unsafe_allow_html=True)
            for fact in EVALUATION_FACTS:
                st.markdown(f"<div class='small-mono' style='margin-bottom:0.35rem;'>• {safe_text(fact)}</div>",
                            unsafe_allow_html=True)
    return weeks


# ── Main ───────────────────────────────────────────────────────────

def main():
    api_online, model_info = get_api_status()
    price_data = get_current_price() if api_online else {"price": None, "date": "N/A", "source": "unavailable"}
    scenarios = get_scenarios() if api_online else []
    weeks = render_sidebar(api_online, model_info, price_data, scenarios)

    llm_label = model_info.get("llm", "gpt-oss-120b").split(" ")[0].split("/")[-1]
    st.markdown(f"""
    <div class='main-header'>Crude Oil Scenario Forecaster</div>
    <div class='main-subheader'>Macroeconomic scenarios · SARIMAX · {safe_text(llm_label)}</div>
    <div class='honesty-banner'>This is a <b>scenario analysis</b> tool, not a price predictor. In walk-forward
    backtests (2022–2025) no model beat a simple “price stays the same” forecast. Use it to compare
    <i>what-ifs</i>, and read the bands as typical-conditions ranges.</div>
    """, unsafe_allow_html=True)

    st.session_state.setdefault("query", EXAMPLE_QUERIES[0])
    query = st.text_area("Describe a scenario in plain English", key="query", height=90,
                         placeholder="e.g. What if Middle East tensions rise while demand also grows?")

    st.markdown("<div class='small-mono' style='margin:-0.3rem 0 0.4rem 0;'>Try an example:</div>",
                unsafe_allow_html=True)
    for col, example in zip(st.columns(len(EXAMPLE_QUERIES)), EXAMPLE_QUERIES):
        col.button(example, width="stretch", key=f"ex_{example}", on_click=set_query, args=(example,))

    run_clicked = st.button("▶  Run simulation", type="primary", disabled=not api_online)
    st.markdown("---")

    if run_clicked:
        if not query.strip():
            st.warning("Please describe a scenario first.")
        else:
            with st.spinner("Weighting scenarios, applying macro signals and running SARIMAX..."):
                data, error = run_probabilistic_simulation(query.strip(), weeks)
            if error:
                st.error(f"Simulation failed: {error}")
            else:
                # Keep results across reruns: Streamlit re-executes the script on every
                # click (tabs, downloads, sidebar), which would otherwise clear them.
                st.session_state["result"] = {"data": data, "query": query.strip(), "at": datetime.now()}

    result = st.session_state.get("result")
    if result:
        render_results(result["data"], result["query"])
        st.markdown(f"""
        <div class='small-mono' style='text-align:center; padding:1rem 0 0.5rem 0;'>
            Generated {result['at']:%Y-%m-%d %H:%M} · {safe_text(model_info.get('forecaster', 'SARIMAX'))}
            · {safe_text(model_info.get('llm', ''))} · data through {safe_text(model_info.get('data_as_of', '—'))}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:2.5rem 0; color:#8a8580;'>
            <div style='font-size:3.2rem; margin-bottom:0.6rem;'>🛢️</div>
            <div style='font-family:DM Mono,monospace; font-size:0.75rem; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:0.6rem;'>Ready to simulate</div>
            <div style='max-width:460px; margin:0 auto; line-height:1.7; font-size:0.9rem;'>
                Describe an oil market event. The LLM spreads probability across five scenarios,
                live macro signals adjust the weights, and SARIMAX projects each path with a
                statistical band around the no-shock baseline.
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
