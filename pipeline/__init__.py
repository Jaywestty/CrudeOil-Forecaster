"""Reproducible data pipeline: fetch raw series, build the modelling dataset, validate it.

Modules
-------
sources   fetch raw series from FRED (public CSV endpoint) and EIA, save/load snapshots
build     pure functions turning raw series into the weekly and transformed datasets
validate  checks every dataset must pass before it can be used
refresh   command-line entry point: ``python -m pipeline.refresh``
"""
