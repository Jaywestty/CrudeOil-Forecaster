"""Turn raw series into the weekly and transformed datasets (pure functions).

RECIPE_V1 reproduces the original Colab notebook exactly:
  - restrict each series to [start, end]
  - resample to weeks ending Sunday: prices/stocks take the week's last value,
    indices and rates take the week's mean
  - forward-fill gaps (monthly series and holidays), then drop rows that are
    still incomplete (before every series has started)
  - transform to stationary: log returns for brent / dollar / industrial
    production, % change for inventories, first differences for fed funds / VIX

Release lags (``release_lag_days``) are an optional fix for look-ahead in
monthly data: FRED dates a monthly value at the START of the month it
describes, but it is only published weeks later (industrial production around
the 15th of the following month). With a lag, each value is moved to the date
it became public before resampling, so a backtest never sees it early.
RECIPE_V1 uses no lags, to match the data the existing models were built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEKLY_COLUMNS = ["brent", "inventories", "industrial_prod", "dollar_index", "fed_funds", "vix"]
TRANSFORMED_COLUMNS = [
    "brent_return", "dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff",
]


@dataclass(frozen=True)
class Recipe:
    name: str
    start: str = "2000-01-01"
    weekly_rule: str = "W"  # weeks ending Sunday
    aggregation: dict = field(default_factory=lambda: {
        "brent": "last", "inventories": "last", "industrial_prod": "mean",
        "dollar_index": "mean", "fed_funds": "mean", "vix": "mean",
    })
    release_lag_days: dict = field(default_factory=dict)
    # Drop the final week if it has not ended by `end` (a mid-week cutoff
    # would otherwise produce a half-filled week that later changes).
    complete_weeks_only: bool = True


# Exactly what the Colab notebook did, including its partial final week
# (cutoff 2025-12-31, a Wednesday). Used to verify reproducibility.
RECIPE_V1 = Recipe(name="v1-notebook", complete_weeks_only=False)
# The notebook recipe, but only fully ended weeks.
RECIPE_V1_1 = Recipe(name="v1.1-complete-weeks")
# Also moves monthly values to roughly when they were published:
# industrial production ~mid next month (+45 days), fed funds monthly average
# at the start of next month (+32 days).
RECIPE_V2 = Recipe(name="v2-release-aware", release_lag_days={"industrial_prod": 45, "fed_funds": 32})
RECIPES = {r.name: r for r in (RECIPE_V1, RECIPE_V1_1, RECIPE_V2)}

# Step 3b decision, fixed before comparing results: production uses the
# release-aware recipe because it only uses information that was public at
# each date. The v1.1 vs v2 backtest measures how much the old timing leak
# inflated results; accuracy does not decide whether to keep a leak.
PRODUCTION_RECIPE = RECIPE_V2


def build_weekly(raw: dict[str, pd.Series], end: str, recipe: Recipe = RECIPE_V1) -> pd.DataFrame:
    """Weekly levels for every series, using only observations dated <= end."""
    cols = {}
    for name in WEEKLY_COLUMNS:
        s = raw[name].sort_index()
        lag = recipe.release_lag_days.get(name, 0)
        if lag:
            s = s.copy()
            s.index = s.index + pd.Timedelta(days=lag)
        s = s[(s.index >= pd.Timestamp(recipe.start)) & (s.index <= pd.Timestamp(end))]
        cols[name] = s.resample(recipe.weekly_rule).agg(recipe.aggregation[name])
    weekly = pd.DataFrame(cols)[WEEKLY_COLUMNS]
    if recipe.complete_weeks_only:
        weekly = weekly[weekly.index <= pd.Timestamp(end)]  # week label = its last day (Sunday)
    weekly = weekly.ffill().dropna()
    weekly.index.name = None
    return weekly


def build_transformed(weekly: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=weekly.index)
    t["brent_return"] = np.log(weekly["brent"] / weekly["brent"].shift(1))
    t["dollar_return"] = np.log(weekly["dollar_index"] / weekly["dollar_index"].shift(1))
    t["indpro_return"] = np.log(weekly["industrial_prod"] / weekly["industrial_prod"].shift(1))
    t["inventory_pct"] = weekly["inventories"].pct_change()
    t["fed_funds_diff"] = weekly["fed_funds"].diff()
    t["vix_diff"] = weekly["vix"].diff()
    return t.dropna()[TRANSFORMED_COLUMNS]
