"""Fit the production SARIMAX model on ALL available data.

Evaluation lives in ``evaluation/`` (walk-forward backtest, challengers,
calibration). This script only builds the model the API serves, so it uses
every week of data: holding back a test set here would just make production
forecasts start from an older state and learn from less history.
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG_COLS = ["dollar_return", "indpro_return", "inventory_pct", "fed_funds_diff", "vix_diff"]

print("Loading data...")
weekly = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True)
transformed = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True)

common = weekly.index.intersection(transformed.index)
target = weekly["brent"].loc[common]
exog = transformed[EXOG_COLS].loc[common]

print(f"Training on all {len(target)} weeks ({target.index[0].date()} to {target.index[-1].date()})...")
model = SARIMAX(
    target,
    exog=exog,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # maxiter=200: the default 50 iterations stopped before convergence;
    # the walk-forward backtest converged within ~75.
    fitted = model.fit(disp=False, maxiter=200)

converged = fitted.mle_retvals.get("converged", False)
print(f"   Converged: {converged} ({fitted.mle_retvals.get('iterations')} iterations)")
if not converged:
    print("   WARNING: optimiser did not converge; check the data before deploying.")

Path("models").mkdir(exist_ok=True)
with open("models/sarimax_model.pkl", "wb") as f:
    pickle.dump(fitted, f)

print(" Model saved to models/sarimax_model.pkl")
print(f"   AIC: {fitted.aic:.2f}")
print(f"   Forecasts will start the week after {target.index[-1].date()}")
