import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Path configuration 
BASE_DIR    = Path(__file__).parent          
DATA_DIR    = BASE_DIR / 'data'              
MODELS_DIR  = BASE_DIR / 'models'           
OUTPUTS_DIR = BASE_DIR / 'outputs'          


def load_data():
    """
    Load the cleaned weekly dataset from Colab.
    Returns a pandas DataFrame indexed by date.
    """
    path = DATA_DIR / 'oil_macro_weekly.csv'
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f" Data loaded: {df.shape[0]} weeks, {df.shape[1]} columns")
    print(f"   Range: {df.index.min().date()} → {df.index.max().date()}")
    return df


def load_transformed_data():
    """
    Load the stationary (transformed) dataset.
    This is what the model was actually trained on as exogenous inputs.
    """
    path = DATA_DIR / 'oil_macro_transformed.csv'
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f" Transformed data loaded: {df.shape[0]} weeks")
    return df


def load_model():
    """
    Load the trained SARIMAX model from pickle file.
    """
    path = MODELS_DIR / 'sarimax_model.pkl'
    with open(path, 'rb') as f:       
        model = pickle.load(f)
    print(f" SARIMAX model loaded successfully")
    return model

def get_latest_values(weekly_data, transformed_data):
    """
    Extract the most recent values of all variables.
    These serve as the baseline for scenario simulation —
    we start from current conditions and apply shocks on top.
    
    Returns a dictionary of current macro variable values.
    """
    latest = {}

    # Latest raw values (for display and context)
    latest['brent_price']      = weekly_data['brent'].iloc[-1]
    latest['inventories']      = weekly_data['inventories'].iloc[-1]
    latest['industrial_prod']  = weekly_data['industrial_prod'].iloc[-1]
    latest['dollar_index']     = weekly_data['dollar_index'].iloc[-1]
    latest['fed_funds']        = weekly_data['fed_funds'].iloc[-1]
    latest['vix']              = weekly_data['vix'].iloc[-1]

    # Latest transformed values (for model input)
    latest['brent_return']     = transformed_data['brent_return'].iloc[-1]
    latest['dollar_return']    = transformed_data['dollar_return'].iloc[-1]
    latest['indpro_return']    = transformed_data['indpro_return'].iloc[-1]
    latest['inventory_pct']    = transformed_data['inventory_pct'].iloc[-1]
    latest['fed_funds_diff']   = transformed_data['fed_funds_diff'].iloc[-1]
    latest['vix_diff']         = transformed_data['vix_diff'].iloc[-1]

    return latest


def format_price_change(baseline, shocked):
    """
    Given a baseline forecast and a shocked forecast,
    compute and format the price difference clearly.
    """
    diff    = shocked - baseline
    pct     = (diff / baseline) * 100
    arrow   = "📈" if diff > 0 else "📉"
    sign    = "+" if diff > 0 else ""

    return {
        'baseline':    round(baseline, 2),
        'shocked':     round(shocked, 2),
        'difference':  round(diff, 2),
        'pct_change':  round(pct, 2),
        'arrow':       arrow,
        'formatted':   f"{arrow} {sign}${diff:.2f}/barrel ({sign}{pct:.1f}%)"
    }


def save_output(content, filename):
    """Save a text report to the outputs folder."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / filename
    with open(path, 'w') as f:
        f.write(content)
    print(f" Report saved: {path}")
    return path