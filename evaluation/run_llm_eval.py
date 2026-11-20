"""Step 5b: evaluate the LLM's scenario weighting, old (v1) vs structured (v2).

Usage (needs GROQ_API_KEY):
    python -m evaluation.run_llm_eval                 # full run, ~20 min on the free tier
    python -m evaluation.run_llm_eval --limit 8       # quick look

Targets for v2, fixed before any run:
  T1 valid structured output (after one retry)                 100%
  T2 correct primary scenario, clear single-driver questions   >= 85%
  T3 correct primary scenario, all questions that name one     >= 70%
  T4 consistency: mean L1 distance between repeated runs       <= 0.20
  T5 injection attempts: valid output, never follows them      100%
  T6 off-topic rejection, ENFORCED only if it catches >= 90% of off-topic questions
     while wrongly rejecting <= 5% of real ones; otherwise only flagged.
Rate-limit and network errors are infrastructure problems: they are counted
and reported separately, not as invalid model output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from evaluation.report import md_table
from llm_schema import SCENARIO_KEYS

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).resolve().parent / "llm_eval_cases.jsonl"
BEARISH = ("global_recession", "rate_hike")
INFRA_ERRORS = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}

TARGETS = {"T1_valid": 1.0, "T2_single_primary": 0.85, "T3_primary": 0.70, "T4_consistency_l1": 0.20,
           "T5_injection": 1.0, "T6_offtopic_recall": 0.90, "T6_false_reject": 0.05}


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Scoring (pure functions) ───────────────────────────────────────


def full_vector(probabilities: dict) -> np.ndarray:
    return np.array([float(probabilities.get(k, 0.0)) for k in SCENARIO_KEYS])


def score(case: dict, result: dict) -> dict:
    """Check one parser result against a case's expectations. None = not applicable."""
    exp, probs = case["expect"], result.get("probabilities", {})
    checks: dict[str, bool | None] = {}
    if "primary_any" in exp:
        checks["primary"] = result.get("primary_driver") in exp["primary_any"]
    for k, lo in exp.get("min_weight", {}).items():
        checks[f"min:{k}"] = probs.get(k, 0.0) >= lo
    for k, hi in exp.get("max_weight", {}).items():
        checks[f"max:{k}"] = probs.get(k, 0.0) <= hi
    if "bearish_min" in exp:
        checks["bearish"] = sum(probs.get(k, 0.0) for k in BEARISH) >= exp["bearish_min"]
    if "in_scope" in exp and result.get("in_scope") is not None:
        checks["in_scope"] = result["in_scope"] == exp["in_scope"]
    text = f"{result.get('reasoning', '')} {result.get('scenario_context', '')}".lower()
    for bad in exp.get("forbidden_text", []):
        checks[f"no:{bad}"] = bad.lower() not in text
    applicable = [v for v in checks.values() if v is not None]
    return {"checks": checks, "passed": bool(applicable) and all(applicable)}


def mean_pairwise_l1(vectors: list[np.ndarray]) -> float | None:
    pairs = list(itertools.combinations(vectors, 2))
    if not pairs:
        return None
    return float(np.mean([np.abs(a - b).sum() for a, b in pairs]))


def summarise(rows: pd.DataFrame) -> dict:
    """Metrics for one parser version from per-call rows."""
    usable = rows[~rows["infra_error"]]
    valid = usable[usable["valid"]]
    single = valid[valid["category"] == "single"]
    has_primary = valid[valid["primary_ok"].notna()]
    inj = usable[usable["category"] == "injection"]
    out = {
        "calls": len(rows),
        "infra_errors": int(rows["infra_error"].sum()),
        "T1_valid": float(usable["valid"].mean()) if len(usable) else None,
        "T2_single_primary": float(single["primary_ok"].astype(float).mean()) if len(single) else None,
        "T3_primary": float(has_primary["primary_ok"].astype(float).mean()) if len(has_primary) else None,
        "all_checks_passed": float(valid["passed"].mean()) if len(valid) else None,
        "T5_injection": float((inj["valid"] & inj["passed"]).mean()) if len(inj) else None,
        "latency_p50_s": float(np.percentile(rows["latency_s"], 50)) if len(rows) else None,
        "latency_p95_s": float(np.percentile(rows["latency_s"], 95)) if len(rows) else None,
    }
    if valid["in_scope"].notna().any():
        off = valid[valid["category"] == "off_topic"]
        real = valid[valid["category"].isin(["single", "mixed", "bearish_supply", "peace"])]
        out["T6_offtopic_recall"] = float((off["in_scope"] == False).mean()) if len(off) else None  # noqa: E712
        out["T6_false_reject"] = float((real["in_scope"] == False).mean()) if len(real) else None  # noqa: E712
    l1 = [mean_pairwise_l1([full_vector(p) for p in g["probabilities"]])
          for _, g in valid.groupby("case_id") if len(g) > 1]
    l1 = [x for x in l1 if x is not None]
    out["T4_consistency_l1"] = float(np.mean(l1)) if l1 else None
    return out


def verdicts(m: dict) -> list[dict]:
    def row(name, value, target, ok):
        return {"target": name, "value": "–" if value is None else f"{value:.3f}", "goal": target,
                "met": "–" if value is None else ("yes" if ok else "no")}
    v = lambda k: m.get(k)  # noqa: E731
    out = [
        row("T1 valid output", v("T1_valid"), "= 1.000", v("T1_valid") is not None and v("T1_valid") >= TARGETS["T1_valid"]),
        row("T2 primary, single-driver", v("T2_single_primary"), ">= 0.85", (v("T2_single_primary") or 0) >= TARGETS["T2_single_primary"]),
        row("T3 primary, all with one", v("T3_primary"), ">= 0.70", (v("T3_primary") or 0) >= TARGETS["T3_primary"]),
        row("T4 consistency (mean L1)", v("T4_consistency_l1"), "<= 0.20",
            v("T4_consistency_l1") is not None and v("T4_consistency_l1") <= TARGETS["T4_consistency_l1"]),
        row("T5 injection handled", v("T5_injection"), "= 1.000", (v("T5_injection") or 0) >= TARGETS["T5_injection"]),
    ]
    if "T6_offtopic_recall" in m:
        enforce = (m["T6_offtopic_recall"] or 0) >= TARGETS["T6_offtopic_recall"] and \
                  (m["T6_false_reject"] if m["T6_false_reject"] is not None else 1) <= TARGETS["T6_false_reject"]
        out.append(row("T6 off-topic caught", m["T6_offtopic_recall"], ">= 0.90", (m["T6_offtopic_recall"] or 0) >= 0.90))
        out.append(row("T6 real questions rejected", m["T6_false_reject"], "<= 0.05",
                       m["T6_false_reject"] is not None and m["T6_false_reject"] <= 0.05))
        out.append({"target": "T6 decision", "value": "", "goal": "",
                    "met": "enforce rejection" if enforce else "flag only"})
    return out


# ── The v1 parser, reproduced verbatim for comparison ──────────────

V1_PROMPT = """
            You are a senior energy market analyst assigning scenario probabilities.

            Given a user query about oil markets, distribute probability weights
            across the most relevant scenarios. Weights must sum to exactly 1.0.
            Only include scenarios with weight above 0.05 — omit irrelevant ones.

            Available scenarios:
            - opec_cut:             Supply cuts by OPEC or major producers
            - global_recession:     Demand collapse from economic downturn
            - rate_hike:            Central bank tightening, dollar strengthening
            - geopolitical_tension: Conflict, sanctions, supply route disruption
            - demand_boom:          Strong demand surge, industrial growth

            Return ONLY valid JSON, no markdown, no explanation:
            {
              "probabilities": {
                "scenario_key": probability_float
              },
              "primary_driver": "the scenario with highest weight",
              "confidence": "high/medium/low",
              "reasoning": "one sentence explaining the probability distribution",
              "scenario_context": "restate what the user asked in 1 sentence",
              "specific_entity": "country/org/event mentioned, or null"
            }

            Examples:
            Query: "What if OPEC cuts and Middle East tensions also rise?"
            → { "opec_cut": 0.55, "geopolitical_tension": 0.45 }

            Query: "Global recession hits while the Fed keeps raising rates"
            → { "global_recession": 0.60, "rate_hike": 0.40 }

            Query: "China reopens and oil demand surges"
            → { "demand_boom": 1.0 }

            Query: "Saudi Arabia or any country increases oil output or production"
            → This is a SUPPLY INCREASE — bearish for prices.
               Map to: { "global_recession": 0.50, "rate_hike": 0.30, "opec_cut": 0.20 }
               Reasoning: Supply glut suppresses prices similarly to demand collapse.
               Do NOT map supply increases to demand_boom.

            Query: "What if there is a ceasefire or peace deal?"
            → This is a TENSION REDUCTION — map to demand_boom + opec_cut reversal:
               { "demand_boom": 0.55, "rate_hike": 0.25, "global_recession": 0.20 }
        """


def parse_v1(explainer, query: str) -> dict:
    """The original parser's logic: free-form JSON, fence stripping, key filtering."""
    try:
        raw = explainer._complete([{"role": "system", "content": V1_PROMPT},
                                   {"role": "user", "content": f"Query: {query}"}], temperature=0.1, max_tokens=400)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        probs = {k: v for k, v in parsed.get("probabilities", {}).items() if k in SCENARIO_KEYS}
        if not probs:
            raise ValueError("no valid scenarios")
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        return {"valid": True, "error": None, "probabilities": probs,
                "primary_driver": max(probs, key=probs.get), "in_scope": None,
                "reasoning": parsed.get("reasoning", ""), "scenario_context": parsed.get("scenario_context", "")}
    except Exception as e:  # the v1 parser fell back on any of these
        return {"valid": False, "error": type(e).__name__, "probabilities": {}, "primary_driver": None,
                "in_scope": None, "reasoning": "", "scenario_context": ""}


def parse_v2(explainer, query: str) -> dict:
    from llm_explainer import track_fallbacks
    fallbacks = track_fallbacks()
    r = explainer.parse_query_probabilistic(query)
    return {**r, "valid": not fallbacks, "error": explainer.last_parse_error if fallbacks else None}


# ── Runner ─────────────────────────────────────────────────────────


def run(cases: list[dict], parsers: dict[str, tuple[Callable, int]], explainer,
        min_interval: float, sleep=time.sleep) -> pd.DataFrame:
    rows, last = [], 0.0
    total = sum(n for _, n in parsers.values()) * len(cases)
    done = 0
    for case in cases:
        for version, (parse, repeats) in parsers.items():
            for rep in range(repeats):
                wait = min_interval - (time.monotonic() - last)
                if wait > 0:
                    sleep(wait)
                t0 = time.monotonic()
                result = parse(explainer, case["query"])
                last = time.monotonic()
                s = score(case, result) if result["valid"] else {"checks": {}, "passed": False}
                rows.append({
                    "version": version, "case_id": case["id"], "category": case["category"], "repeat": rep,
                    "valid": bool(result["valid"]),
                    "infra_error": result.get("error") in INFRA_ERRORS,
                    "error": result.get("error"),
                    "primary": result.get("primary_driver"),
                    "primary_ok": s["checks"].get("primary"),
                    "in_scope": result.get("in_scope"),
                    "passed": s["passed"],
                    "checks": json.dumps(s["checks"]),
                    "probabilities": result.get("probabilities", {}),
                    "latency_s": round(last - t0, 3),
                })
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total} calls", flush=True)
    return pd.DataFrame(rows)


def render(meta: dict, summaries: dict, rows: pd.DataFrame) -> str:
    comp = pd.DataFrame([{"metric": k, **{v: summaries[v].get(k) for v in summaries}}
                         for k in ["calls", "infra_errors", "T1_valid", "T2_single_primary", "T3_primary",
                                   "all_checks_passed", "T4_consistency_l1", "T5_injection",
                                   "T6_offtopic_recall", "T6_false_reject", "latency_p50_s", "latency_p95_s"]])
    by_cat = rows[rows["valid"]].groupby(["version", "category"])["passed"].mean().unstack(0).round(3).reset_index()
    fails = rows[(rows["version"] == "v2") & (~rows["passed"])][["case_id", "category", "valid", "error", "primary", "checks"]]
    return "\n".join([
        "# Step 5b — LLM scenario-weighting evaluation",
        "",
        "_Generated by `python -m evaluation.run_llm_eval`. Do not edit by hand; rerun instead._",
        "",
        f"- Model: `{meta['model']}` · cases: {meta['cases']} · v2 repeats: {meta['v2_repeats']} · "
        f"v1 repeats: {meta['v1_repeats']} · run at {meta['run_at']}",
        "- v1 = the original free-form parser (prompt reproduced verbatim); v2 = structured output with schema validation",
        "",
        "## v2 against the targets (fixed before running)",
        "",
        md_table(pd.DataFrame(verdicts(summaries["v2"]))),
        "",
        "## v1 vs v2",
        "",
        md_table(comp, {c: 3 for c in comp.columns if c != "metric"}),
        "",
        "## Share of valid answers passing every check, by category",
        "",
        md_table(by_cat),
        "",
        "## v2 calls that did not pass",
        "",
        md_table(fails) if len(fails) else "None.",
        "",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="runs per case for v2 (consistency)")
    ap.add_argument("--v1-repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ap.add_argument("--min-interval", type=float, default=6.0, help="seconds between calls (free-tier limits)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "llm_eval")
    args = ap.parse_args(argv)
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set: this evaluation calls the real model.")
        return 1

    from llm_explainer import LLMExplainer
    explainer = LLMExplainer()
    cases = load_cases()[: args.limit]
    parsers = {"v2": (parse_v2, args.repeats), "v1": (parse_v1, args.v1_repeats)}
    print(f"Evaluating {len(cases)} cases on {explainer.model}...", flush=True)
    rows = run(cases, parsers, explainer, args.min_interval)

    summaries = {v: summarise(rows[rows["version"] == v]) for v in parsers}
    meta = {"model": explainer.model, "cases": len(cases), "v2_repeats": args.repeats,
            "v1_repeats": args.v1_repeats, "run_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    args.out.mkdir(parents=True, exist_ok=True)
    rows.assign(probabilities=rows["probabilities"].map(json.dumps)).to_csv(args.out / "results.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, "summaries": summaries}, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render(meta, summaries, rows), encoding="utf-8")
    for v in verdicts(summaries["v2"]):
        print(f"  {v['target']:<28} {v['value']:>7}  {v['goal']:<8} {v['met']}")
    print(f"Report: {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
