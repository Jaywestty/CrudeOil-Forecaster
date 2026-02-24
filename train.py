import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("Loading data...")
weekly     = pd.read_csv('data/oil_macro_weekly.csv', 
                          index_col=0, parse_dates=True)
transformed = pd.read_csv('data/oil_macro_transformed.csv',
                           index_col=0, parse_dates=True)

exog_cols = ['dollar_return', 'indpro_return',
             'inventory_pct', 'fed_funds_diff', 'vix_diff']

# Align indices
common = weekly.index.intersection(transformed.index)
target = weekly['brent'].loc[common]
exog   = transformed[exog_cols].loc[common]

# 80/20 split
split  = int(len(target) * 0.80)
train_target = target.iloc[:split]
train_exog   = exog.iloc[:split]

print(f"Training on {len(train_target)} weeks...")
model = SARIMAX(
    train_target,
    exog=train_exog,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted = model.fit(disp=False)

Path('models').mkdir(exist_ok=True)
with open('models/sarimax_model.pkl', 'wb') as f:
    pickle.dump(fitted, f)

print(" Model saved to models/sarimax_model.pkl")
print(f"   AIC: {fitted.aic:.2f}")