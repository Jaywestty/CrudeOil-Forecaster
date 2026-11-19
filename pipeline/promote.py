"""Promotion gate: decide whether a staged dataset (and the model refit on it) goes live.

Usage:
    python -m pipeline.promote                     # dry run: check gates, write the report
    python -m pipeline.promote --apply             # promote if every gate passes
    python -m pipeline.promote --apply --override G2 --reason "switch to release-aware recipe"

Gates (fixed before any results were seen; see the step-3b notes):
  G1 data       staged data passed validation and uses the production recipe
  G2 continuity no dates removed, and at most 5% of rows changed per column on
                dates both datasets share (larger revisions need a human)
  G3 fit        the model refit on staged data converged with finite parameters
  G4 sanity     its 12-week forecast and 95% band are finite and within
                0.5x-2x the last price
The gates check for breakage, not improvement: the model specification was
validated by the walk-forward backtests, so refitting on more data is routine.

Monitoring (reported, never blocking): one-week-ahead error of the CURRENT
model on the newly added weeks, against a no-change forecast. This is how the
live model performed on data it had never seen.

Promotion copies the staged dataset into data/, saves the refit model, and
appends a record to models/registry.json. Overrides are recorded with a reason.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import modeling
from evaluation.data import content_hash
from evaluation.report import md_table
from pipeline.build import PRODUCTION_RECIPE
from pipeline.compare import diff_frames

ROOT = Path(__file__).resolve().parent.parent
WEEKLY, TRANSFORMED, MANIFEST = "oil_macro_weekly.csv", "oil_macro_transformed.csv", "manifest.json"
MAX_REVISED_FRACTION = 0.05
# Only revisions need human judgement. Invalid data (G1) or a broken fit (G3/G4)
# is never promoted by hand.
OVERRIDABLE = {"G2"}
SANITY_BAND = (0.5, 2.0)
HORIZON = 12


@dataclass
class Gate:
    name: str
    check: str
    passed: bool
    detail: str
    overridden: bool = False

    def __post_init__(self):
        # Checks often yield numpy.bool_, which json cannot serialise.
        self.passed, self.overridden = bool(self.passed), bool(self.overridden)


def load_dataset_dir(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(path / WEEKLY, index_col=0, parse_dates=True),
            pd.read_csv(path / TRANSFORMED, index_col=0, parse_dates=True))


def gate_data(manifest: dict, recipe: str) -> Gate:
    ok = manifest.get("validation", {}).get("ok", False)
    same = manifest.get("recipe") == recipe
    return Gate("G1", "data", ok and same,
                f"validation {'passed' if ok else 'FAILED'}; recipe {manifest.get('recipe')} "
                f"({'matches' if same else 'expected ' + recipe})")


def gate_continuity(current: pd.DataFrame, staged: pd.DataFrame) -> Gate:
    removed = current.index.difference(staged.index)
    diff = diff_frames(current, staged)
    frac = (diff["rows changed"] / diff["shared rows"]).fillna(0)
    worst = diff.loc[frac.idxmax()] if len(diff) else None
    ok = len(removed) == 0 and (frac <= MAX_REVISED_FRACTION).all()
    detail = (f"{len(removed)} dates removed; largest revision {worst['column']}: "
              f"{int(worst['rows changed'])}/{int(worst['shared rows'])} rows ({frac.max():.1%})"
              if worst is not None else "no shared columns")
    return Gate("G2", "continuity", ok, detail)


def gate_fit(summary: dict) -> Gate:
    ok = summary["converged"] and summary["params_finite"]
    return Gate("G3", "fit", ok, f"converged={summary['converged']} after {summary['iterations']} iterations, "
                                 f"finite params={summary['params_finite']}, AIC {summary['aic']}")


def gate_sanity(results, transformed: pd.DataFrame, last_price: float) -> Gate:
    exog = np.tile(transformed[modeling.EXOG_COLS].iloc[-1].to_numpy(float), (HORIZON, 1))
    fc = results.get_forecast(HORIZON, exog=exog)
    mean, ci = np.asarray(fc.predicted_mean), np.asarray(fc.conf_int(alpha=0.05))
    lo, hi = SANITY_BAND[0] * last_price, SANITY_BAND[1] * last_price
    ok = bool(np.isfinite(mean).all() and np.isfinite(ci).all() and ((mean >= lo) & (mean <= hi)).all())
    return Gate("G4", "sanity", ok, f"week-{HORIZON} forecast ${mean[-1]:.2f} from last ${last_price:.2f}; "
                                    f"95% band ${ci[-1, 0]:.2f}-${ci[-1, 1]:.2f}")


def monitor_champion(champion, weekly: pd.DataFrame, transformed: pd.DataFrame, new_dates) -> dict | None:
    """One-week-ahead MAE of the current model on newly added weeks vs no-change."""
    if champion is None or len(new_dates) == 0:
        return None
    y, X = modeling.align(weekly, transformed)
    errs_model, errs_naive = [], []
    for target in new_dates:
        t = y.index.get_loc(target)
        if t == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = champion.apply(y.iloc[:t], exog=X.iloc[:t])
            pred = float(np.asarray(res.get_forecast(1, exog=X.iloc[[t - 1]].to_numpy()).predicted_mean)[0])
        errs_model.append(abs(pred - y.iloc[t]))
        errs_naive.append(abs(y.iloc[t - 1] - y.iloc[t]))
    mae_m, mae_n = float(np.mean(errs_model)), float(np.mean(errs_naive))
    return {"weeks": len(errs_model), "model_mae": round(mae_m, 3), "naive_mae": round(mae_n, 3),
            "mae_vs_naive": round(mae_m / mae_n, 3) if mae_n else None}


def has_new_data(current_dir: Path, staging_dir: Path) -> bool:
    """True if the staged datasets differ from the live ones (by content)."""
    return any(content_hash(current_dir / name) != content_hash(staging_dir / name)
               for name in (WEEKLY, TRANSFORMED))


def run(current_dir: Path, staging_dir: Path, model_path: Path, registry_path: Path, report_path: Path,
        apply: bool, overrides: list[str], reason: str | None, fit_fn=modeling.fit,
        serving_fn=modeling.serving_model) -> int:
    if not has_new_data(current_dir, staging_dir):
        # Re-promoting identical data would add a duplicate registry entry.
        print("Staged data is identical to the live data: nothing to promote.")
        return 0
    manifest = json.loads((staging_dir / MANIFEST).read_text(encoding="utf-8"))
    cur_w, _ = load_dataset_dir(current_dir)
    new_w, new_t = load_dataset_dir(staging_dir)

    print("Refitting candidate on staged data...", flush=True)
    candidate = fit_fn(new_w, new_t)
    summary = modeling.fit_summary(candidate)
    gates = [gate_data(manifest, PRODUCTION_RECIPE.name), gate_continuity(cur_w, new_w), gate_fit(summary)]
    gates.append(gate_sanity(candidate, new_t, float(new_w["brent"].iloc[-1])) if gates[2].passed
                 else Gate("G4", "sanity", False, "skipped: fit gate failed"))
    for g in gates:
        if not g.passed and g.name in overrides:
            if g.name not in OVERRIDABLE:
                raise SystemExit(f"{g.name} cannot be overridden")
            if not reason:
                raise SystemExit("--override requires --reason")
            g.overridden = True

    champion = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            champion = pickle.load(f)
    monitoring = monitor_champion(champion, new_w, new_t, new_w.index.difference(cur_w.index))
    approved = all(g.passed or g.overridden for g in gates)

    write_report(report_path, manifest, gates, summary, monitoring, approved, apply, reason)
    for g in gates:
        flag = "PASS" if g.passed else ("OVERRIDDEN" if g.overridden else "FAIL")
        print(f"  {g.name} {g.check:<11} {flag:<10} {g.detail}")
    if monitoring:
        print(f"  monitoring: current model 1-week MAE {monitoring['model_mae']} vs naive "
              f"{monitoring['naive_mae']} on {monitoring['weeks']} new weeks")
    if not approved:
        print("Not promoted: a gate failed.")
        return 1
    if not apply:
        print("All gates passed (dry run). Re-run with --apply to promote.")
        return 0

    # Save the small serving copy (same forecasts), not the ~490 MB fit.
    serving = serving_fn(candidate, new_w, new_t)
    promote(staging_dir, current_dir, serving, model_path, registry_path, manifest, gates, summary,
            monitoring, reason)
    return 0


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def promote(staging_dir, current_dir, candidate, model_path, registry_path, manifest, gates, summary,
            monitoring, reason) -> dict:
    # Prepare everything that can fail BEFORE changing live files, so a failure
    # cannot leave new data in place without a registry record (or vice versa).
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    weekly_sha = content_hash(staging_dir / WEEKLY)
    entry = {
        "version": f"{manifest['weekly']['last']}-{weekly_sha[:8]}",
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous_version": registry[-1]["version"] if registry else None,
        "data": {"recipe": manifest["recipe"], "first": manifest["weekly"]["first"],
                 "last": manifest["weekly"]["last"], "last_brent": manifest["weekly"]["last_brent"],
                 "weekly_sha": weekly_sha, "transformed_sha": content_hash(staging_dir / TRANSFORMED)},
        "model": summary,
        "gates": [asdict(g) for g in gates],
        "override_reason": reason if any(g.overridden for g in gates) else None,
        "monitoring": monitoring,
    }
    registry_bytes = json.dumps(registry + [entry], indent=2).encode("utf-8")
    model_bytes = pickle.dumps(candidate)

    # Swap in the new state. The registry is written last: it records a
    # promotion only once the data and model it describes are in place.
    for name in (WEEKLY, TRANSFORMED, MANIFEST):
        _write_atomic(current_dir / name, (staging_dir / name).read_bytes())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(model_path, model_bytes)
    _write_atomic(registry_path, registry_bytes)
    print(f"Promoted {entry['version']} (data through {entry['data']['last']}); registry: {registry_path}")
    return entry


def write_report(path: Path, manifest, gates, summary, monitoring, approved, apply, reason) -> None:
    rows = pd.DataFrame([{"gate": g.name, "check": g.check,
                          "result": "PASS" if g.passed else ("OVERRIDDEN" if g.overridden else "FAIL"),
                          "detail": g.detail} for g in gates])
    verdict = ("Promoted." if apply else "Approved (dry run).") if approved else "Not promoted: a gate failed."
    parts = [
        "# Promotion gate report", "",
        "_Generated by `python -m pipeline.promote`._", "",
        f"- Staged data: recipe `{manifest['recipe']}`, {manifest['weekly']['first']} → {manifest['weekly']['last']}, "
        f"last Brent ${manifest['weekly']['last_brent']:.2f}",
        f"- Candidate: {summary['spec']}, {summary['n_obs']} weeks", "",
        md_table(rows), "",
        f"**{verdict}**" + (f" Override reason: {reason}" if any(g.overridden for g in gates) else ""), "",
        "## Monitoring: current model on the new weeks", "",
    ]
    if monitoring:
        parts.append(f"One-week-ahead MAE ${monitoring['model_mae']:.2f} vs no-change ${monitoring['naive_mae']:.2f} "
                     f"over {monitoring['weeks']} new weeks (ratio {monitoring['mae_vs_naive']}). "
                     "Below 1 means the model beat a no-change forecast on data it had never seen.")
    else:
        parts.append("No current model or no new weeks to monitor.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", type=Path, default=ROOT / "data" / "staging")
    ap.add_argument("--current", type=Path, default=ROOT / "data")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "sarimax_model.pkl")
    ap.add_argument("--registry", type=Path, default=ROOT / "models" / "registry.json")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "data" / "promotion.md")
    ap.add_argument("--apply", action="store_true", help="promote if all gates pass")
    ap.add_argument("--override", action="append", default=[], choices=sorted(OVERRIDABLE),
                    help="accept a failed G2 (large revisions) after review; requires --reason")
    ap.add_argument("--reason")
    args = ap.parse_args(argv)
    return run(args.current, args.staging, args.model, args.registry, args.report,
               args.apply, args.override, args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
