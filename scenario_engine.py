import numpy as np
import pandas as pd
from utils import load_data, load_transformed_data, load_model
from utils import get_latest_values, format_price_change


# ── Scenario Definitions ───────────────────────────────────────────
# Each scenario maps to specific shocks on our 5 exogenous variables.
#
# How do we choose shock magnitudes?
# We look at what actually happened historically during similar events
# and use those as calibration points.
#
# The variables we can shock:
#   dollar_return   → weekly % change in USD index
#   indpro_return   → weekly % change in industrial production
#   inventory_pct   → weekly % change in crude inventories
#   fed_funds_diff  → weekly change in fed funds rate (percentage points)
#   vix_diff        → weekly change in VIX index

# ── Why these shock values? ───────────────────────────────────────────────────
# The SARIMAX model was trained on 2006–2022 data. During that period:
#   - High VIX episodes coincided with the 2008 demand COLLAPSE → model learned
#     high VIX = lower prices (even when the cause is a supply shock)
#   - Large inventory draws preceded the 2014–2016 glut unwinding → model
#     learned inventory draws = lower prices in some regimes
#
# So we work WITH the model's learned relationships rather than against them:
#   - indpro_return is the most reliable bullish/bearish signal
#   - fed_funds_diff drives dollar and rate expectations cleanly
#   - dollar_return is respected by the model (oil is USD-denominated)
#   - VIX shocks are kept moderate to avoid triggering the 2008 crash memory
#   - inventory shocks are kept as secondary confirmation signals
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = {

    "opec_cut": {
        "name":        "OPEC Production Cut (10%)",
        "description": "OPEC announces a coordinated 10% production cut. "
                       "Supply tightens, inventories draw down, "
                       "market expects higher prices ahead.",
        "shocks": {
            # Primary signal: industrial production holds but supply shrinks
            # Dollar weakens as oil exporters earn more → bullish for oil
            "dollar_return":  -0.006,  # dollar weakens on supply cut news
            "indpro_return":  +0.005,  # slight industrial uptick on price expectations
            "fed_funds_diff":  0.0,
            # Secondary: inventory draw confirms tightening, VIX modest
            "inventory_pct":  -0.04,   # inventories draw down
            "vix_diff":       +1.0,    # slight uncertainty (not a crash signal)
        }
    },

    "global_recession": {
        "name":        "Global Recession",
        "description": "A global recession takes hold. Industrial activity "
                       "contracts sharply, demand for oil collapses, "
                       "financial markets enter panic mode.",
        "shocks": {
            # Primary: industrial collapse is the clearest bearish signal
            "indpro_return":  -0.025,  # sharp industrial contraction
            "fed_funds_diff": -0.50,   # Fed cuts rates in response to recession
            "dollar_return":  +0.004,  # safe haven dollar bid
            # Secondary: inventory builds, VIX spikes (consistent with 2008 pattern)
            "inventory_pct":  +0.03,   # demand collapse → inventory build
            "vix_diff":       +8.0,    # fear spikes (model knows this = recession)
        }
    },

    "rate_hike": {
        "name":        "Aggressive Fed Rate Hike (+75bps)",
        "description": "The Federal Reserve raises rates aggressively "
                       "by 75 basis points. Economic growth slows, "
                       "dollar strengthens, oil demand weakens.",
        "shocks": {
            # Primary: rate hike → strong dollar → oil bearish
            "fed_funds_diff": +0.75,   # 75 basis points — direct signal
            "dollar_return":  +0.010,  # strong dollar appreciation
            "indpro_return":  -0.008,  # economic slowdown hits industry
            # Secondary: mild fear, inventories stable
            "vix_diff":       +2.0,    # mild uncertainty
            "inventory_pct":  +0.01,   # slight build as demand cools
        }
    },

    "geopolitical_tension": {
        "name":        "Major Geopolitical Tension (Supply Disruption)",
        "description": "Significant geopolitical conflict disrupts oil "
                       "supply routes. Supply uncertainty drives prices up "
                       "as markets price in a risk premium.",
        "shocks": {
            # Primary: dollar weakens on risk-off for oil exporters
            # industrial production unaffected initially (supply shock not demand)
            "dollar_return":  -0.005,  # oil exporters currencies strengthen
            "indpro_return":  +0.003,  # production continues, just rerouted
            "fed_funds_diff":  0.0,
            # Secondary: inventory draws on hoarding, VIX moderate
            "inventory_pct":  -0.05,   # hoarding and supply uncertainty
            "vix_diff":       +4.0,    # elevated but not 2008-level crash signal
        }
    },

    "demand_boom": {
        "name":        "Global Demand Boom (China Reopening)",
        "description": "A major emerging market (e.g. China) reopens "
                       "strongly after a period of restriction. "
                       "Industrial demand surges, oil consumption rises.",
        "shocks": {
            # Primary: strong industrial signal — clearest bullish driver
            "indpro_return":  +0.020,  # strong industrial surge
            "dollar_return":  -0.005,  # risk-on weakens dollar
            "fed_funds_diff":  0.0,
            # Secondary: inventories draw, VIX calm
            "inventory_pct":  -0.03,   # demand surge draws down inventories
            "vix_diff":       -2.0,    # calm optimistic market
        }
    }
}


class ScenarioEngine:
    """
    The main engine that runs scenario simulations.

    Think of this class as a machine with two settings:
      1. Baseline mode    → forecast using current macro conditions
      2. Scenario mode    → forecast using shocked macro conditions

    The difference between those two forecasts = scenario impact.

    NEW in v2: Probabilistic mode
      3. Probabilistic    → run each scenario separately, then combine
                            results using probability weights into a
                            single expected price + confidence range.
    """

    def __init__(self):
        print("⚙️  Initializing Scenario Engine...")
        self.weekly_data = load_data()
        self.transformed = load_transformed_data()
        self.model       = load_model()
        self.latest      = get_latest_values(self.weekly_data, self.transformed)
        # Columns the model expects as exogenous inputs
        # Must match exactly what we trained with
        self.exog_cols = [
            'dollar_return',
            'indpro_return',
            'inventory_pct',
            'fed_funds_diff',
            'vix_diff'
        ]
        print("✅ Scenario Engine ready\n")


    def _build_exog_matrix(self, shocks, forecast_weeks):
        """
        Build the exogenous variable matrix for forecasting.

        For each week in the forecast horizon, we need values for all 5
        exogenous variables. We start from current observed values and
        add the scenario shocks on top.

        Think of it like:
          current conditions + scenario shock = new conditions
          feed new conditions into SARIMAX → get shocked forecast

        Args:
            shocks (dict): shock values for each variable
            forecast_weeks (int): how many weeks to forecast

        Returns:
            DataFrame of shape (forecast_weeks, 5)
        """
        base = {
            'dollar_return':  self.latest['dollar_return'],
            'indpro_return':  self.latest['indpro_return'],
            'inventory_pct':  self.latest['inventory_pct'],
            'fed_funds_diff': self.latest['fed_funds_diff'],
            'vix_diff':       self.latest['vix_diff'],
        }

        shocked = {col: base[col] + shocks.get(col, 0.0) for col in self.exog_cols}

        # Repeat the shocked values for every forecast week.
        # This assumes the shock PERSISTS throughout the forecast period.
        return pd.DataFrame([shocked] * forecast_weeks, columns=self.exog_cols)


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHOD — baseline forecast
    # Unchanged from your working version.
    # ─────────────────────────────────────────────────────────────────────

    def run_baseline(self, forecast_weeks=12):
        """
        Generate a baseline forecast — no shocks, just current conditions.

        Returns:
            dict with forecast values and metadata
        """
        zero_shocks = {col: 0.0 for col in self.exog_cols}
        exog        = self._build_exog_matrix(zero_shocks, forecast_weeks)
        forecast    = self.model.get_forecast(steps=forecast_weeks, exog=exog)

        return {
            'type':          'baseline',
            'forecast_mean': forecast.predicted_mean,
            'conf_int':      forecast.conf_int(alpha=0.05),
            'weeks':         forecast_weeks,
            'current_price': self.latest['brent_price']
        }


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHOD — single named scenario
    # Unchanged from your working version. Used by /simulate endpoint.
    # ─────────────────────────────────────────────────────────────────────

    def run_scenario(self, scenario_key, forecast_weeks=12):
        """
        Run a specific named scenario and compare to baseline.

        Args:
            scenario_key (str): key from SCENARIOS dict
            forecast_weeks (int): how many weeks ahead to forecast

        Returns:
            dict with baseline, shocked forecast, and impact analysis
        """
        if scenario_key not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_key}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )

        scenario = SCENARIOS[scenario_key]
        shocks   = scenario['shocks']

        print(f"▶  Running scenario: {scenario['name']}")
        print(f"   {scenario['description']}\n")
        print("   Shocks being applied:")
        for var, magnitude in shocks.items():
            if magnitude != 0:
                sign = "+" if magnitude > 0 else ""
                print(f"     {var:<20} {sign}{magnitude}")
        print()

        # Baseline
        baseline_result   = self.run_baseline(forecast_weeks)
        baseline_forecast = baseline_result['forecast_mean']

        # Shocked forecast
        shocked_exog         = self._build_exog_matrix(shocks, forecast_weeks)
        shocked_forecast_obj = self.model.get_forecast(
            steps=forecast_weeks, exog=shocked_exog
        )
        shocked_forecast = shocked_forecast_obj.predicted_mean
        shocked_conf_int = shocked_forecast_obj.conf_int(alpha=0.05)

        # Impact calculations
        baseline_end = baseline_forecast.iloc[-1]
        shocked_end  = shocked_forecast.iloc[-1]
        impact       = format_price_change(baseline_end, shocked_end)

        baseline_w1 = baseline_forecast.iloc[0]
        shocked_w1  = shocked_forecast.iloc[0]
        impact_w1   = format_price_change(baseline_w1, shocked_w1)

        return {
            'scenario_key':      scenario_key,
            'scenario_name':     scenario['name'],
            'scenario_desc':     scenario['description'],
            'shocks':            shocks,
            'current_price':     self.latest['brent_price'],
            'baseline_forecast': baseline_forecast,
            'shocked_forecast':  shocked_forecast,
            'shocked_conf_int':  shocked_conf_int,
            'impact_week1':      impact_w1,
            'impact_week12':     impact,
            'weeks':             forecast_weeks
        }


    # ─────────────────────────────────────────────────────────────────────
    # NEW METHOD — macro signal probability adjustment
    # Called inside run_probabilistic_scenario() before the weighted
    # forecast is computed.
    #
    # WHY THIS EXISTS:
    #   The LLM assigns probabilities based on the user's words alone.
    #   But real markets are shaped by what's actually happening RIGHT NOW.
    #   This method reads live macro conditions from our dataset and
    #   shifts the probabilities to reflect current reality.
    #
    #   Example: user asks about geopolitical tension but VIX is at 12
    #   (very calm). Our adjustment will reduce geopolitical weight slightly
    #   because the market itself isn't pricing in that fear.
    #
    #   Example: user asks about demand boom but inventories are RISING.
    #   Rising inventories = oversupply = market doesn't support the boom.
    #   We reduce demand_boom weight and boost recession weight.
    #
    # Returns adjusted probabilities + a log of what changed and why
    # (the log is shown in the UI and in the explanation).
    # ─────────────────────────────────────────────────────────────────────

    def _adjust_probabilities_with_macro(self, probabilities):
        """
        Adjust LLM-assigned base probabilities using live macro signals.

        This is what separates the system from pure LLM guessing —
        we use REAL data (VIX level, dollar direction, inventory trend)
        to nudge the probabilities toward what current market conditions
        actually suggest.

        After adjustment, probabilities are renormalized to sum to 1.0.

        Args:
            probabilities (dict): LLM-assigned {scenario_key: float}

        Returns:
            tuple: (adjusted_probabilities dict, adjustments_log dict)
        """
        probs       = probabilities.copy()
        adjustments = {}

        # Read current macro conditions from our dataset
        current_vix       = self.latest.get('vix_diff', 0)
        current_dollar    = self.latest.get('dollar_return', 0)
        current_inventory = self.latest.get('inventory_pct', 0)
        current_fed       = self.latest.get('fed_funds_diff', 0)

        # ── VIX signal ─────────────────────────────────────────────
        # VIX above 25 = elevated fear in the market.
        # Boost fear-driven scenarios (geopolitical, recession).
        if current_vix > 25:
            if "geopolitical_tension" in probs:
                probs["geopolitical_tension"] *= 1.30
                adjustments["vix_high_geo"] = (
                    "VIX elevated → geopolitical weight +30%"
                )
            if "global_recession" in probs:
                probs["global_recession"] *= 1.20
                adjustments["vix_high_rec"] = (
                    "VIX elevated → recession weight +20%"
                )
        # VIX below 15 = very calm market. Boost optimistic scenarios.
        elif current_vix < 15:
            if "demand_boom" in probs:
                probs["demand_boom"] *= 1.20
                adjustments["vix_low"] = (
                    "VIX low → demand boom weight +20%"
                )

        # ── Dollar signal ───────────────────────────────────────────
        # Dollar strengthening: oil becomes more expensive in other currencies.
        # Historically negative for oil demand — boost rate_hike + recession.
        if current_dollar > 0.005:
            if "rate_hike" in probs:
                probs["rate_hike"] *= 1.25
                adjustments["dollar_strong_rate"] = (
                    "Strong dollar → rate hike weight +25%"
                )
            if "global_recession" in probs:
                probs["global_recession"] *= 1.10
                adjustments["dollar_strong_rec"] = (
                    "Strong dollar → recession weight +10%"
                )
        # Dollar weakening: bullish for oil (cheaper for non-USD buyers).
        elif current_dollar < -0.005:
            if "demand_boom" in probs:
                probs["demand_boom"] *= 1.20
                adjustments["dollar_weak_dem"] = (
                    "Weak dollar → demand boom weight +20%"
                )
            if "opec_cut" in probs:
                probs["opec_cut"] *= 1.10
                adjustments["dollar_weak_opec"] = (
                    "Weak dollar → OPEC cut weight +10%"
                )

        # ── Inventory signal ────────────────────────────────────────
        # Inventories falling = supply is tightening = bullish.
        # Boost supply-side scenarios.
        if current_inventory < -0.02:
            if "opec_cut" in probs:
                probs["opec_cut"] *= 1.25
                adjustments["inventory_draw_opec"] = (
                    "Inventory draw → OPEC cut weight +25%"
                )
            if "geopolitical_tension" in probs:
                probs["geopolitical_tension"] *= 1.15
                adjustments["inventory_draw_geo"] = (
                    "Inventory draw → geopolitical weight +15%"
                )
        # Inventories rising = oversupply = bearish.
        # Boost demand-collapse scenarios.
        elif current_inventory > 0.02:
            if "global_recession" in probs:
                probs["global_recession"] *= 1.20
                adjustments["inventory_build_rec"] = (
                    "Inventory build → recession weight +20%"
                )
            if "demand_boom" in probs:
                probs["demand_boom"] *= 0.85
                adjustments["inventory_build_dem"] = (
                    "Inventory build → demand boom weight -15%"
                )

        # ── Fed funds signal ────────────────────────────────────────
        if current_fed > 0.25:
            if "rate_hike" in probs:
                probs["rate_hike"] *= 1.30
                adjustments["fed_hiking"] = (
                    "Fed actively hiking → rate hike weight +30%"
                )

        # ── Renormalize ─────────────────────────────────────────────
        # After adjustments the weights no longer sum to 1.0.
        # Divide each by the new total so they sum to exactly 1.0 again.
        total    = sum(probs.values())
        adjusted = {k: round(v / total, 4) for k, v in probs.items()}

        return adjusted, adjustments


    # ─────────────────────────────────────────────────────────────────────
    # NEW METHOD — probabilistic multi-scenario simulation
    # Called by /simulate-probabilistic endpoint.
    #
    # HOW IT WORKS (plain English):
    #   1. Take LLM probability weights from parse_query_probabilistic()
    #   2. Adjust them using live macro signals (VIX, dollar, inventories)
    #   3. Run the SARIMAX model once per scenario (same model, no retraining)
    #   4. Multiply each scenario's weekly forecast by its probability
    #   5. Sum all the weighted forecasts → one expected price per week
    #   6. Compute price range: min and max across all scenario week-12 prices
    #   7. Return everything for the API to package into a response
    #
    # Why no retraining?
    #   The SARIMAX model is already trained. run_scenario() just feeds it
    #   different exogenous variable values. We call it N times (once per
    #   scenario) with different inputs — like using a calculator multiple
    #   times with different numbers.
    # ─────────────────────────────────────────────────────────────────────

    def run_probabilistic_scenario(self, probabilities, forecast_weeks=12):
        """
        Run a probability-weighted multi-scenario simulation.

        Args:
            probabilities (dict): {scenario_key: probability_float}
                                  from LLM parse step, sums to 1.0
            forecast_weeks (int): how many weeks ahead to forecast

        Returns:
            dict with weighted forecast, range, and full breakdown
        """
        print(f"▶  Running probabilistic simulation "
              f"across {len(probabilities)} scenarios")

        # Step 1 — Adjust LLM probabilities with live macro signals
        adjusted_probs, macro_adjustments = \
            self._adjust_probabilities_with_macro(probabilities)

        print("   Adjusted probabilities:")
        for k, v in adjusted_probs.items():
            print(f"     {k:<25} {v*100:.1f}%")

        # Step 2 — Run baseline once (same for all scenarios)
        baseline_result   = self.run_baseline(forecast_weeks)
        baseline_forecast = baseline_result['forecast_mean'].values
        baseline_week12   = float(baseline_forecast[-1])

        # Step 3 — Run each scenario separately, store results
        scenario_results = {}
        scenario_week12  = {}
        scenario_names   = {}

        for key, prob in adjusted_probs.items():
            # Skip scenarios with negligible weight — not worth computing
            if prob < 0.05:
                continue

            result = self.run_scenario(key, forecast_weeks)

            scenario_results[key] = {
                "probability":   prob,
                "forecast":      result["shocked_forecast"].values,
                "week12_price":  float(result["shocked_forecast"].values[-1]),
                "week1_price":   float(result["shocked_forecast"].values[0]),
                "scenario_name": result["scenario_name"],
                "impact_week12": result["impact_week12"]
            }
            scenario_week12[key] = float(result["shocked_forecast"].values[-1])
            scenario_names[key]  = result["scenario_name"]

        # Step 4 — Weighted average forecast across all weeks
        #
        # This is the core math — very simple once you see it:
        #
        #   expected_price[week_t] = Σ (probability_i × price_i[week_t])
        #
        # For example with 3 scenarios at week 12:
        #   (0.45 × $90) + (0.30 × $87) + (0.25 × $79) = $86.90
        #
        # We do this for every week, not just week 12,
        # which gives us a full weighted forecast curve.
        weighted_forecast = np.zeros(forecast_weeks)
        for key, data in scenario_results.items():
            weighted_forecast += data["probability"] * data["forecast"]

        # Step 5 — Price range and summary statistics
        #
        # Range is computed from individual scenario week-12 prices.
        # This shows the spread across possible outcomes.
        # Expected price is the probability-weighted average of those prices.
        # We then clamp expected to always sit within [low, high] as a
        # mathematical guarantee — it should be there by definition but
        # floating point and probability normalization can cause tiny violations.
        all_week12     = [d["week12_price"] for d in scenario_results.values()]
        price_low      = round(min(all_week12), 2)
        price_high     = round(max(all_week12), 2)
        price_expected = round(float(weighted_forecast[-1]), 2)

        # Safety clamp: expected price must sit inside the range
        price_expected = round(
            max(price_low, min(price_high, price_expected)), 2
        )

        # Primary driver = scenario with the highest adjusted probability
        primary_driver = max(adjusted_probs, key=adjusted_probs.get)

        print(f"   Expected Week-12 price: ${price_expected}")
        print(f"   Range: ${price_low} – ${price_high}")
        print(f"   Primary driver: {primary_driver}")

        return {
            "weighted_forecast":      weighted_forecast,
            "baseline_forecast":      baseline_forecast,
            "baseline_week12":        baseline_week12,
            "scenario_results":       scenario_results,
            "scenario_week12":        scenario_week12,
            "scenario_names":         scenario_names,
            "price_expected":         price_expected,
            "price_low":              price_low,
            "price_high":             price_high,
            "primary_driver":         primary_driver,
            "current_price":          self.latest["brent_price"],
            "scenario_probabilities": adjusted_probs,
            "original_probabilities": probabilities,
            "macro_adjustments":      macro_adjustments,
            "forecast_weeks":         forecast_weeks
        }


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHODS — list_scenarios and print_results
    # ─────────────────────────────────────────────────────────────────────

    def list_scenarios(self):
        """Print all available scenarios."""
        print("\n" + "=" * 50)
        print("AVAILABLE SCENARIOS")
        print("=" * 50)
        for key, scenario in SCENARIOS.items():
            print(f"\n  [{key}]")
            print(f"  {scenario['name']}")
            print(f"  {scenario['description'][:80]}...")
        print()

    def print_results(self, result):
        """Print a clean summary of single-scenario results."""
        print("\n" + "=" * 55)
        print(f"SCENARIO RESULTS: {result['scenario_name']}")
        print("=" * 55)
        print(f"\nCurrent Brent Price:  ${result['current_price']:.2f}/barrel")
        print(f"\n{'Week':<8} {'Baseline':>12} {'Scenario':>12} {'Change':>15}")
        print("-" * 55)

        for i, (b, s) in enumerate(zip(
            result['baseline_forecast'],
            result['shocked_forecast']
        ), 1):
            diff   = s - b
            sign   = "+" if diff >= 0 else ""
            marker = " ← Week 1"  if i == 1             else (
                     " ← Week 12" if i == result['weeks'] else "")
            print(f"Week {i:<3} ${b:>10.2f}  ${s:>10.2f}  "
                  f"{sign}${diff:>8.2f}{marker}")

        print("-" * 55)
        print(f"\nImmediate impact (Week 1):  {result['impact_week1']['formatted']}")
        print(f"Full impact (Week 12):      {result['impact_week12']['formatted']}")
        print("=" * 55)