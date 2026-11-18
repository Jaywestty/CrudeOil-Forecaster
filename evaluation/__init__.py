"""Walk-forward evaluation toolkit for the Brent forecaster.

Modules
-------
data         load the aligned target / exogenous series
forecasters  baselines (naive, seasonal naive, drift) and SARIMAX variants
backtest     the walk-forward engine (the only place that slices time)
metrics      point, interval and significance metrics
report       markdown report generation
run_backtest command-line entry point: ``python -m evaluation.run_backtest``
"""
