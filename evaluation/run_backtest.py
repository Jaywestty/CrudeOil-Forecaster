"""Run the walk-forward backtest and write results to ``reports/backtest``.

Usage:
    python -m evaluation.run_backtest                 # full run (~10 minutes)
    python -m evaluation.run_backtest --step 13       # faster smoke run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from evaluation import metrics, report
from evaluation.backtest import BacktestConfig, run_backtest
from evaluation.data import data_fingerprint, load_dataset, train_size
from evaluation.forecasters import SeasonalNaive, _DirectMultiHorizon, default_forecasters

ROOT = Path(__file__).resolve().parent.parent
HEADLINE_HORIZONS = [1, 4, 12]


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _progress(done: int, total: int) -> None:
    if done == total or done % 20 == 0:
        print(f"  origins {done}/{total}", flush=True)


def legacy_evaluation(y, X, n_train, forecasters) -> pd.DataFrame:
    """Reproduce the notebook's single-origin evaluation over the whole test set."""
    # Seasonal naive needs horizon <= 52; direct ML models are trained for 12 weeks only.
    usable = [f for f in forecasters if not isinstance(f, (SeasonalNaive, _DirectMultiHorizon))]
    cfg = BacktestConfig(first_origin=n_train - 1, horizon=len(y) - n_train, step=len(y), refit_every=None)
    fc = run_backtest(y, X, usable, cfg)
    rows = []
    for model, g in fc.groupby("model", sort=False):
        rows.append(
            {
                "Model": model + (" ⚠ oracle" if g["oracle"].iloc[0] else ""),
                "MAE ($)": metrics.mae(g["y_true"], g["y_pred"]),
                "RMSE ($)": metrics.rmse(g["y_true"], g["y_pred"]),
                "MAPE (%)": metrics.mape(g["y_true"], g["y_pred"]),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE (%)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--step", type=int, default=1, help="weeks between forecast origins")
    parser.add_argument("--refit-every", type=int, default=52, help="weeks between refits; 0 = fit once")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "backtest")
    args = parser.parse_args(argv)

    y, X = load_dataset()
    n_train = train_size(len(y))
    forecasters = default_forecasters()
    t0 = time.perf_counter()

    print(f"Reproducing original single-origin evaluation (fit on {n_train} weeks)...", flush=True)
    legacy = legacy_evaluation(y, X, n_train, forecasters)

    cfg = BacktestConfig(
        first_origin=n_train - 1,
        horizon=args.horizon,
        step=args.step,
        refit_every=args.refit_every or None,
    )
    origins = cfg.origins(len(y))
    print(f"Walk-forward backtest: {len(origins)} origins x {len(forecasters)} models...", flush=True)
    forecasts = run_backtest(y, X, forecasters, cfg, progress=_progress)

    scale = metrics.mase_scale(y.iloc[:n_train].to_numpy())
    summary = metrics.summarize(forecasts, scale=scale)
    by_year = metrics.mae_by_year(forecasts, horizon=args.horizon).round(2)

    # Collect each distinct fit log once (SARIMAX variants share an estimator's log).
    logs: dict[int, list] = {}
    for f in forecasters:
        log = getattr(f, "fit_log", None)
        if log is None and hasattr(f, "estimator"):
            log = f.estimator.fit_log
        if log is not None:
            logs[id(log)] = log
    fit_log = pd.DataFrame([row for log in logs.values() for row in log])

    candidates = [f.name for f in forecasters if not f.is_baseline and not f.uses_future_exog]
    promotion = metrics.promotion_decisions(summary, candidates, HEADLINE_HORIZONS)

    meta = {
        "n_obs": len(y),
        "data_start": str(y.index[0].date()),
        "data_end": str(y.index[-1].date()),
        "data_fingerprint": json.dumps(data_fingerprint()),
        "git_commit": _git_commit(),
        "n_origins": len(origins),
        "step": cfg.step,
        "first_origin": str(y.index[origins[0]].date()),
        "last_origin": str(y.index[origins[-1]].date()),
        "horizon": cfg.horizon,
        "refit_every": cfg.refit_every,
        "mase_scale": scale,
        "legacy_horizon": len(y) - n_train,
        "by_year_horizon": cfg.horizon,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "python": sys.version.split()[0],
    }

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    summary.to_csv(out / "metrics.csv", index=False)
    by_year.to_csv(out / "mae_by_year.csv")
    legacy.to_csv(out / "legacy_evaluation.csv", index=False)
    fit_log.to_csv(out / "fit_log.csv", index=False)
    promotion.to_csv(out / "promotion.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = report.render(
        meta=meta,
        summary=summary,
        by_year=by_year,
        legacy=legacy,
        fit_log=fit_log,
        promotion=promotion,
        horizons=HEADLINE_HORIZONS,
    )
    (out / "report.md").write_text(md, encoding="utf-8")

    print(f"Done in {meta['runtime_seconds']}s -> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
