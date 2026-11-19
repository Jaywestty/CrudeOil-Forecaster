"""Fit the production SARIMAX model on ALL available data in data/.

Evaluation lives in ``evaluation/`` (walk-forward backtest, challengers,
calibration) and data promotion in ``pipeline/promote.py``. This script only
builds the model the API serves, so it uses every week of the promoted data.
The specification lives in modeling.py, shared with the promotion gate.
"""

import pickle
from pathlib import Path

import pandas as pd

import modeling

print("Loading data...")
weekly = pd.read_csv("data/oil_macro_weekly.csv", index_col=0, parse_dates=True)
transformed = pd.read_csv("data/oil_macro_transformed.csv", index_col=0, parse_dates=True)

y, _ = modeling.align(weekly, transformed)
print(f"Training {modeling.SPEC} on all {len(y)} weeks ({y.index[0].date()} to {y.index[-1].date()})...")
fitted = modeling.fit(weekly, transformed)
summary = modeling.fit_summary(fitted)

print(f"   Converged: {summary['converged']} ({summary['iterations']} iterations)")
if not summary["converged"]:
    print("   WARNING: optimiser did not converge; check the data before deploying.")

# Save a serving copy: same parameters and forecasts, but only the final filter
# state (~3 MB instead of ~490 MB), so the API fits in Render's 512 MB.
serving = modeling.serving_model(fitted, weekly, transformed)
Path("models").mkdir(exist_ok=True)
with open("models/sarimax_model.pkl", "wb") as f:
    pickle.dump(serving, f)

size_mb = Path("models/sarimax_model.pkl").stat().st_size / 1e6
print(f" Model saved to models/sarimax_model.pkl ({size_mb:.1f} MB)")
print(f"   AIC: {summary['aic']:.2f}")
print(f"   Forecasts will start the week after {y.index[-1].date()}")
