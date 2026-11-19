"""Step 2a: calibrate scenarios on historical events and validate them.

Usage:
    python -m evaluation.run_scenario_study      # ~1 minute (needs models/sarimax_model.pkl)

Rules fixed before any event outcome was computed:
  * events and their dates come from data/scenario_events.csv;
  * a scenario's path is the MEDIAN cumulative log move across its events;
  * validation is leave-one-event-out within each scenario type;
  * rivals: naive (no move) and a replica of the v2.1 engine (SARIMAX
    macro shocks + direction correction) using the production model,
    which has seen all data (look-ahead that can only help that rival);
  * decision: the analogue method replaces the v2.1 engine if its mean
    absolute error of the 12-week cumulative move across all events is lower.
Events flagged as confounded are kept in the primary analysis and dropped in
a pre-declared sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import report
from evaluation.data import EXOG_COLS, data_fingerprint, load_dataset
from evaluation.run_backtest import ROOT, _git_commit
from scenario_library import HORIZON, analogue_path, event_paths, load_events, origin_position, week_cols

# The v2.1 engine's scenario shocks and direction sets, copied verbatim so this
# comparison stays valid after step 2b changes scenario_engine.py.
V21_SHOCKS = {
    "opec_cut": {"dollar_return": -0.024, "indpro_return": 0.020, "fed_funds_diff": 0.0,
                 "inventory_pct": -0.16, "vix_diff": 2.0},
    "global_recession": {"indpro_return": -0.080, "fed_funds_diff": -1.50, "dollar_return": 0.016,
                         "inventory_pct": 0.10, "vix_diff": 20.0},
    "rate_hike": {"fed_funds_diff": 0.75, "dollar_return": 0.032, "indpro_return": -0.024,
                  "vix_diff": 6.0, "inventory_pct": 0.04},
    "geopolitical_tension": {"dollar_return": -0.020, "indpro_return": 0.012, "fed_funds_diff": 0.0,
                             "inventory_pct": -0.20, "vix_diff": 10.0},
    "demand_boom": {"indpro_return": 0.060, "dollar_return": -0.020, "fed_funds_diff": 0.0,
                    "inventory_pct": -0.12, "vix_diff": -4.0},
}
V21_BULLISH = {"opec_cut", "geopolitical_tension", "demand_boom"}

METHODS = ["Analogue (median of other events)", "Naive (no move)", "v2.1 engine (SARIMAX shocks + flip)"]
ANALOGUE, NAIVE, V21 = METHODS


def v21_engine_moves(model, y: pd.Series, X: pd.DataFrame, scenario: str, origin: int, horizon: int) -> np.ndarray:
    """Cumulative log moves the v2.1 engine would have predicted at this origin."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.apply(y.iloc[: origin + 1], exog=X.iloc[: origin + 1])
    base_x = np.tile(X.iloc[origin][EXOG_COLS].to_numpy(float), (horizon, 1))
    shock = np.array([V21_SHOCKS[scenario].get(c, 0.0) for c in EXOG_COLS])
    baseline = np.asarray(res.get_forecast(horizon, exog=base_x).predicted_mean)
    shocked = np.asarray(res.get_forecast(horizon, exog=base_x + shock).predicted_mean)
    wrong_way = (shocked[-1] <= baseline[-1]) if scenario in V21_BULLISH else (shocked[-1] >= baseline[-1])
    if wrong_way:  # the v2.1 "direction correction": mirror around the baseline
        shocked = 2 * baseline - shocked
    return np.log(np.maximum(shocked, 1e-6) / y.iloc[origin])


def loo_predictions(paths: pd.DataFrame, y, X, model, horizon: int, include_confounded: bool) -> pd.DataFrame:
    cols = week_cols(horizon)
    rows = []
    for ev in paths.itertuples(index=False):
        actual = np.array([getattr(ev, c) for c in cols])
        preds = {
            ANALOGUE: analogue_path(paths, ev.scenario, horizon, exclude_date=ev.date,
                                    include_confounded=include_confounded)["median"],
            NAIVE: np.zeros(horizon),
        }
        if model is not None:
            preds[V21] = v21_engine_moves(model, y, X, ev.scenario, origin_position(y.index, ev.date), horizon)
        for method, pred in preds.items():
            rows.append({
                "scenario": ev.scenario, "date": ev.date, "event": ev.event, "confounded": ev.confounded,
                "method": method, **{f"pred_{c}": float(p) for c, p in zip(cols, pred)},
                **{f"actual_{c}": float(a) for c, a in zip(cols, actual)},
            })
    return pd.DataFrame(rows)


def score(loo: pd.DataFrame, week: int) -> pd.DataFrame:
    err = (loo[f"pred_w{week}"] - loo[f"actual_w{week}"]).abs()
    hit = np.sign(loo[f"pred_w{week}"]) == np.sign(loo[f"actual_w{week}"])
    hit = hit.where(loo[f"pred_w{week}"] != 0)  # a zero forecast calls no direction
    d = loo.assign(abs_err=err, hit=hit)
    return d.groupby("method", sort=False).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                                               direction_hit=("hit", "mean")).reset_index()


def bootstrap_diff(loo: pd.DataFrame, a: str, b: str, week: int, reps: int = 10_000, seed: int = 0):
    """95% bootstrap CI (resampling events) of MAE(a) - MAE(b)."""
    def errs(m):
        g = loo[loo["method"] == m].sort_values(["scenario", "date"])
        return (g[f"pred_w{week}"] - g[f"actual_w{week}"]).abs().to_numpy()
    ea, eb = errs(a), errs(b)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float((ea < eb).mean())


def pct(x: float) -> str:
    return f"{100 * (np.exp(x) - 1):+.1f}%"


def render(meta, events_tbl, paths_tbl, overall, by_scen, sens, decision) -> str:
    return "\n".join([
        "# Step 2a — Scenario calibration on historical events",
        "",
        "_Generated by `python -m evaluation.run_scenario_study`. Do not edit by hand; rerun instead._",
        "",
        "## Setup",
        "",
        f"- {meta['n_events']} events in `data/scenario_events.csv` ({meta['n_confounded']} flagged as confounded)",
        "- Origin = last weekly price strictly before the event date; path = cumulative log move over the next 1–12 weeks",
        "- Scenario path = median across that scenario's events; validated leave-one-event-out",
        "- Rivals: naive (no move) and a replica of the v2.1 engine (SARIMAX shocks + direction flip) "
        "using the production model, which has seen all data (look-ahead that only helps that rival)",
        f"- Data fingerprint `{meta['data_fingerprint']}` · code `{meta['git_commit']}`",
        "",
        "Moves are shown as % price change: exp(log move) − 1.",
        "",
        "## Decision (rule fixed before running)",
        "",
        "Replace the v2.1 engine with analogue paths if the analogue method's mean absolute error of the "
        "12-week cumulative move, across all events, is lower.",
        "",
        decision,
        "",
        "## Validation: leave-one-event-out",
        "",
        "Error is in log points (0.10 ≈ a 10% price move).",
        "",
        md(overall),
        "",
        "### 12-week MAE by scenario",
        "",
        md(by_scen),
        "",
        "### Sensitivity: confounded events removed",
        "",
        md(sens),
        "",
        "## Calibrated scenario paths (all events)",
        "",
        md(paths_tbl),
        "",
        "## Event catalogue and what happened",
        "",
        md(events_tbl),
        "",
    ])


def md(df: pd.DataFrame) -> str:
    return report.md_table(df, {c: 3 for c in df.columns if df[c].dtype.kind == "f"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "scenarios")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "sarimax_model.pkl")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    y, X = load_dataset()
    events = load_events()
    paths = event_paths(y, events, HORIZON)
    model = None
    if args.model.exists():
        with open(args.model, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"WARNING: {args.model} not found; skipping the v2.1 engine comparison (run train.py first)")

    print(f"Leave-one-out over {len(paths)} events...", flush=True)
    loo = loo_predictions(paths, y, X, model, HORIZON, include_confounded=True)
    loo_clean = loo_predictions(paths[~paths["confounded"]], y, X, model, HORIZON, include_confounded=False)

    overall = score(loo, 12).rename(columns={"MAE": "MAE 12w", "direction_hit": "direction hit 12w"})
    overall.insert(3, "MAE 4w", score(loo, 4)["MAE"].to_numpy())
    by_scen = (
        loo.assign(abs_err=(loo["pred_w12"] - loo["actual_w12"]).abs())
        .pivot_table(index="scenario", columns="method", values="abs_err", aggfunc="mean", sort=False)
        .reset_index()
    )
    by_scen.insert(1, "n", paths.groupby("scenario").size().reindex(by_scen["scenario"]).to_numpy())
    sens = score(loo_clean, 12).rename(columns={"MAE": "MAE 12w", "direction_hit": "direction hit 12w"})

    lines = []
    if model is not None:
        diff, lo, hi, win = bootstrap_diff(loo, ANALOGUE, V21, 12)
        a_mae = overall.loc[overall["method"] == ANALOGUE, "MAE 12w"].item()
        v_mae = overall.loc[overall["method"] == V21, "MAE 12w"].item()
        verdict = "**Adopt analogue paths.**" if a_mae < v_mae else "**Keep the v2.1 engine.**"
        lines.append(f"{verdict} 12-week MAE: analogue {a_mae:.3f} vs v2.1 engine {v_mae:.3f} "
                     f"(difference {diff:+.3f}, bootstrap 95% CI {lo:+.3f} to {hi:+.3f}; "
                     f"analogue closer on {win:.0%} of events).")
    dn, dlo, dhi, dwin = bootstrap_diff(loo, ANALOGUE, NAIVE, 12)
    lines.append(f"Versus naive: difference {dn:+.3f} (95% CI {dlo:+.3f} to {dhi:+.3f}); "
                 f"analogue closer on {dwin:.0%} of events. Below zero means the analogue method is better.")
    decision = "\n\n".join(lines)

    cols = week_cols(HORIZON)
    paths_rows = []
    for s in sorted(paths["scenario"].unique()):
        a = analogue_path(paths, s, HORIZON)
        paths_rows.append({"scenario": s, "n events": a["n_events"],
                           **{f"median wk{k}": pct(a["median"][k - 1]) for k in (1, 4, 8, 12)},
                           "range wk12": f"{pct(a['low'][-1])} to {pct(a['high'][-1])}"})
    events_tbl = paths.assign(
        **{"wk4": paths["w4"].map(pct), "wk12": paths["w12"].map(pct),
           "date": paths["date"].dt.date, "origin": paths["origin"].dt.date,
           "confounded": paths["confounded"].map(lambda c: "yes" if c else "")}
    )[["scenario", "date", "event", "origin", "origin_price", "wk4", "wk12", "confounded"]]

    meta = {"n_events": len(paths), "n_confounded": int(paths["confounded"].sum()),
            "data_fingerprint": json.dumps(data_fingerprint()), "git_commit": _git_commit(),
            "runtime_seconds": round(time.perf_counter() - t0, 1)}
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    paths.to_csv(out / "event_paths.csv", index=False)
    loo.to_csv(out / "loo_predictions.csv", index=False)
    overall.to_csv(out / "summary.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        render(meta, events_tbl, pd.DataFrame(paths_rows), overall, by_scen, sens, decision), encoding="utf-8"
    )
    print(f"Done in {meta['runtime_seconds']}s -> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
