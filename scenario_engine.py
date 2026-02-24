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

SCENARIOS = {

    "opec_cut": {
        "name":        "OPEC Production Cut (10%)",
        "description": "OPEC announces a coordinated 10% production cut. "
                       "Supply tightens, inventories draw down, "
                       "risk sentiment improves.",
        "shocks": {
            # Supply cut → inventories fall (negative inventory change)
            "inventory_pct":   -0.05,   # 5% weekly inventory drawdown
            # Market interprets cut as bullish → VIX falls (less fear)
            "vix_diff":        -2.0,    # VIX drops 2 points
            # Dollar slightly weakens as oil-exporting currencies strengthen
            "dollar_return":   -0.002,  # 0.2% dollar weakening
            # Industrial production and fed funds unchanged
            "indpro_return":    0.0,
            "fed_funds_diff":   0.0,
        }
    },

    "global_recession": {
        "name":        "Global Recession",
        "description": "A global recession takes hold. Industrial activity "
                       "contracts sharply, demand for oil collapses, "
                       "financial markets enter panic mode.",
        "shocks": {
            # Recession → industrial activity falls sharply
            "indpro_return":  -0.02,    # 2% weekly industrial contraction
            # Fear spikes dramatically
            "vix_diff":       +8.0,     # VIX jumps 8 points (panic)
            # Dollar strengthens as safe haven
            "dollar_return":  +0.005,   # 0.5% dollar strengthening
            # Demand collapse → inventories build up
            "inventory_pct":  +0.03,    # 3% inventory build
            "fed_funds_diff":  0.0,     # rates unchanged initially
        }
    },

    "rate_hike": {
        "name":        "Aggressive Fed Rate Hike (+75bps)",
        "description": "The Federal Reserve raises rates aggressively "
                       "by 75 basis points. Economic growth slows, "
                       "dollar strengthens, oil demand weakens.",
        "shocks": {
            # Rate hike → dollar strengthens
            "dollar_return":  +0.008,   # 0.8% dollar appreciation
            # Higher rates → growth slowdown expectations
            "vix_diff":       +3.0,     # mild fear increase
            # Economic slowdown → slightly less industrial activity
            "indpro_return":  -0.005,   # 0.5% industrial slowdown
            # Rate hike itself
            "fed_funds_diff": +0.75,    # 75 basis points
            "inventory_pct":   0.0,     # inventories unchanged immediately
        }
    },

    "geopolitical_tension": {
        "name":        "Major Geopolitical Tension (Supply Disruption)",
        "description": "Significant geopolitical conflict disrupts oil "
                       "supply routes. Markets panic, safe havens rally, "
                       "supply uncertainty drives prices up.",
        "shocks": {
            # Supply disruption → inventories draw down fast
            "inventory_pct":  -0.08,    # 8% sharp inventory drop
            # Extreme fear spike
            "vix_diff":       +12.0,    # VIX surges (war-level fear)
            # Dollar as safe haven
            "dollar_return":  +0.003,   # 0.3% safe haven dollar bid
            # Industrial production initially unaffected
            "indpro_return":   0.0,
            "fed_funds_diff":  0.0,
        }
    },

    "demand_boom": {
        "name":        "Global Demand Boom (China Reopening)",
        "description": "A major emerging market (e.g. China) reopens "
                       "strongly after a period of restriction. "
                       "Industrial demand surges globally.",
        "shocks": {
            # Strong industrial activity surge
            "indpro_return":  +0.015,   # 1.5% industrial surge
            # Inventories draw down on demand
            "inventory_pct":  -0.04,    # 4% inventory drawdown
            # Risk-on sentiment → VIX falls
            "vix_diff":       -3.0,     # VIX drops on optimism
            # Dollar weakens on risk-on
            "dollar_return":  -0.003,   # 0.3% dollar weakening
            "fed_funds_diff":  0.0,
        }
    }
}


class ScenarioEngine:
    """
    The main engine that runs scenario simulations.
    
    Think of this class as a machine with two settings:
      1. Baseline mode  → forecast using current macro conditions
      2. Scenario mode  → forecast using shocked macro conditions
    
    The difference between those two forecasts = scenario impact
    """

    def __init__(self):
        print(" Initializing Scenario Engine...")
        self.weekly_data     = load_data()
        self.transformed     = load_transformed_data()
        self.model           = load_model()
        self.latest          = get_latest_values(
                                   self.weekly_data, 
                                   self.transformed
                               )
        # Columns the model expects as exogenous inputs
        # Must match exactly what we trained with in Colab
        self.exog_cols = [
            'dollar_return',
            'indpro_return',
            'inventory_pct',
            'fed_funds_diff',
            'vix_diff'
        ]
        print(" Scenario Engine ready\n")


    def _build_exog_matrix(self, shocks, forecast_weeks):
        """
        Build the exogenous variable matrix for forecasting.
        
        For each week in our forecast horizon, we need values
        for all 5 exogenous variables. We start from current
        values and apply the shocks on top.
        
        Think of it like: 
          current conditions + scenario shock = new conditions
          feed new conditions into model → get shocked forecast
        
        Args:
            shocks (dict): shock values for each variable
            forecast_weeks (int): how many weeks to forecast
            
        Returns:
            DataFrame of shape (forecast_weeks, 5)
        """
        # Start from the latest observed values
        base = {
            'dollar_return':  self.latest['dollar_return'],
            'indpro_return':  self.latest['indpro_return'],
            'inventory_pct':  self.latest['inventory_pct'],
            'fed_funds_diff': self.latest['fed_funds_diff'],
            'vix_diff':       self.latest['vix_diff'],
        }

        # Apply shocks on top of baseline values
        shocked = {}
        for col in self.exog_cols:
            shocked[col] = base[col] + shocks.get(col, 0.0)

        # Repeat the shocked values across all forecast weeks
        # This assumes the shock persists throughout the forecast horizon
        exog_matrix = pd.DataFrame(
            [shocked] * forecast_weeks,
            columns=self.exog_cols
        )

        return exog_matrix


    def run_baseline(self, forecast_weeks=12):
        """
        Generate a baseline forecast — no shocks, just current conditions.
        
        Args:
            forecast_weeks (int): how many weeks ahead to forecast
            
        Returns:
            dict with forecast values and metadata
        """
        # Zero shocks = baseline conditions
        zero_shocks = {col: 0.0 for col in self.exog_cols}
        exog        = self._build_exog_matrix(zero_shocks, forecast_weeks)

        forecast    = self.model.get_forecast(
                          steps=forecast_weeks,
                          exog=exog
                      )

        mean_forecast = forecast.predicted_mean
        conf_int      = forecast.conf_int(alpha=0.05)

        return {
            'type':          'baseline',
            'forecast_mean': mean_forecast,
            'conf_int':      conf_int,
            'weeks':         forecast_weeks,
            'current_price': self.latest['brent_price']
        }


    def run_scenario(self, scenario_key, forecast_weeks=12):
        """
        Run a specific named scenario and compare to baseline.
        
        Args:
            scenario_key (str): key from SCENARIOS dict
                                e.g. 'opec_cut', 'global_recession'
            forecast_weeks (int): how many weeks ahead to forecast
            
        Returns:
            dict with baseline, shocked forecast, and impact analysis
        """
        if scenario_key not in SCENARIOS:
            available = list(SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario_key}'. "
                f"Available: {available}"
            )

        scenario = SCENARIOS[scenario_key]
        shocks   = scenario['shocks']

        print(f" Running scenario: {scenario['name']}")
        print(f"   {scenario['description']}\n")
        print("   Shocks being applied:")
        for var, magnitude in shocks.items():
            if magnitude != 0:
                sign = "+" if magnitude > 0 else ""
                print(f"     {var:<20} {sign}{magnitude}")
        print()

        # Generate baseline forecast 
        baseline_result  = self.run_baseline(forecast_weeks)
        baseline_forecast = baseline_result['forecast_mean']

        #  Generate shocked forecast 
        shocked_exog     = self._build_exog_matrix(shocks, forecast_weeks)
        shocked_forecast_obj = self.model.get_forecast(
                                   steps=forecast_weeks,
                                   exog=shocked_exog
                               )
        shocked_forecast  = shocked_forecast_obj.predicted_mean
        shocked_conf_int  = shocked_forecast_obj.conf_int(alpha=0.05)

        #  Calculate impact 
        # Compare end-of-horizon prices (week 12 vs week 12)
        baseline_end = baseline_forecast.iloc[-1]
        shocked_end  = shocked_forecast.iloc[-1]
        impact       = format_price_change(baseline_end, shocked_end)

        # Week 1 impact (immediate effect)
        baseline_w1  = baseline_forecast.iloc[0]
        shocked_w1   = shocked_forecast.iloc[0]
        impact_w1    = format_price_change(baseline_w1, shocked_w1)

        return {
            'scenario_key':       scenario_key,
            'scenario_name':      scenario['name'],
            'scenario_desc':      scenario['description'],
            'shocks':             shocks,
            'current_price':      self.latest['brent_price'],
            'baseline_forecast':  baseline_forecast,
            'shocked_forecast':   shocked_forecast,
            'shocked_conf_int':   shocked_conf_int,
            'impact_week1':       impact_w1,
            'impact_week12':      impact,
            'weeks':              forecast_weeks
        }


    def list_scenarios(self):
        """Print all available scenarios."""
        print("\n" + "="*50)
        print("AVAILABLE SCENARIOS")
        print("="*50)
        for key, scenario in SCENARIOS.items():
            print(f"\n  [{key}]")
            print(f"  {scenario['name']}")
            print(f"  {scenario['description'][:80]}...")
        print()


    def print_results(self, result):
        """
        Print a clean, readable summary of scenario results.
        Called after run_scenario() to display findings.
        """
        print("\n" + "="*55)
        print(f"SCENARIO RESULTS: {result['scenario_name']}")
        print("="*55)
        print(f"\nCurrent Brent Price:  ${result['current_price']:.2f}/barrel")
        print(f"\n{'Week':<8} {'Baseline':>12} {'Scenario':>12} {'Change':>15}")
        print("-"*55)

        for i, (b, s) in enumerate(zip(
            result['baseline_forecast'],
            result['shocked_forecast']
        ), 1):
            diff   = s - b
            sign   = "+" if diff >= 0 else ""
            marker = " ← Week 1" if i == 1 else (
                     " ← Week 12" if i == result['weeks'] else "")
            print(f"Week {i:<3} ${b:>10.2f}  ${s:>10.2f}  "
                  f"{sign}${diff:>8.2f}{marker}")

        print("-"*55)
        w1  = result['impact_week1']
        w12 = result['impact_week12']
        print(f"\nImmediate impact (Week 1):  {w1['formatted']}")
        print(f"Full impact (Week 12):      {w12['formatted']}")
        print("="*55)