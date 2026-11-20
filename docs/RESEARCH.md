# Research notes: what can and cannot be said about Brent a few weeks ahead

This is the technical companion to the [README](../README.md). It is written for ML
researchers and engineers who want to know what was tested, how, what came out, and how
far to trust it. Every number here comes from a generated report in [`reports/`](../reports);
the script that produces each one is named next to it.

**The short version.** The project started as a SARIMAX "forecaster" with a reported 15.22%
MAPE. That number came from an evaluation that gave the model the realised future macro
data, and a flat line scored better (12.39%) on the same weeks. Rebuilt with walk-forward
evaluation, baselines and significance tests, **no model beats a no-change forecast at 1, 4
or 12 weeks**. With honest data timing, the deployed model is significantly *worse* than
no-change at 12 weeks. The project was then reframed as what it can defensibly be: a
scenario tool with checked uncertainty, historical context labelled as history, a
reproducible data pipeline, and an LLM layer evaluated against targets set before the run.

---

## Contents

1. [Method principles](#1-method-principles)
2. [Point forecasts: walk-forward evaluation](#2-point-forecasts-walk-forward-evaluation)
3. [Challengers: is it the model or the data?](#3-challengers-is-it-the-model-or-the-data)
4. [Uncertainty: are the ranges calibrated?](#4-uncertainty-are-the-ranges-calibrated)
5. [Scenarios: does history forecast what follows an event?](#5-scenarios-does-history-forecast-what-follows-an-event)
6. [Data: timing leaks and reproducibility](#6-data-timing-leaks-and-reproducibility)
7. [From experiment to production](#7-from-experiment-to-production)
8. [The LLM layer: evaluating a structured parser](#8-the-llm-layer-evaluating-a-structured-parser)
9. [Threats to validity](#9-threats-to-validity)
10. [What I would do next](#10-what-i-would-do-next)
11. [Reproduce everything](#11-reproduce-everything)
12. [References](#12-references)

---

## 1. Method principles

These rules applied to every experiment below.

| Principle | In practice |
|---|---|
| **Decide before measuring** | Each experiment's decision rule is written in its script and report *before* the first run. When a result disappointed, the rule stood. |
| **Beat the trivial baseline first** | For asset prices that means the random walk ("no change"). A model that cannot beat it has no forecasting skill, whatever its MAPE. |
| **Simulate real time** | Forecasts use only information available at the forecast date: data up to the origin, macro series on their *publication* dates, no realised future inputs unless the model is explicitly flagged as an oracle. |
| **Test differences, not just compare them** | Diebold–Mariano tests for loss differences, Holm correction across every model × horizon, Newey–West errors for overlapping forecasts. |
| **Make leakage structurally hard** | Only one module slices time. Forecasters receive read-only history; future exogenous values are opt-in and flagged in every output. Tests prove both. |
| **Record provenance** | Every report records the data fingerprints (content hashes), git commit and configuration it was produced from. |

---

## 2. Point forecasts: walk-forward evaluation

**Script:** `python -m evaluation.run_backtest` → [`reports/backtest/report.md`](../reports/backtest/report.md)

### What was wrong with the original evaluation
The notebook made **one** forecast at the end of training covering all 209 test weeks, and
supplied the *realised* dollar index, VIX, inventories, etc. as inputs. Three problems:
leakage through the exogenous variables, a single sample path (its error mostly reflects
where oil happened to go in 2022–25), and no baseline. Re-running that exact setup:

| Model (single origin, 209 weeks) | MAPE |
|---|---|
| SARIMAX exog=mean | 11.99% |
| SARIMAX exog=persist | 12.04% |
| SARIMAX exog=oracle (realised future inputs) | 12.13% |
| Naive (flat at last training price) | 12.39% |
| Original notebook's saved forecasts | 15.22% |

### Design
- **Expanding window**, 198 weekly origins (2022-01-02 → 2025-10-12), horizons 1–12.
- Parameters re-estimated every 52 weeks; between refits the Kalman filter state is updated
  with each new observation (`results.apply`), so every forecast starts from the latest price.
- **Baselines:** naive, seasonal naive (52), drift.
- **SARIMAX variants** make the unavoidable assumption about future macro inputs explicit:
  `persist` (hold last values; what the app does), `mean` (historical mean), `oracle`
  (realised values; an upper bound, not deployable).
- **Metrics:** MAE, RMSE, MAPE, MASE (Hyndman & Koehler, 2006), MAE relative to naive,
  directional accuracy, 95% coverage.
- **Significance:** Diebold–Mariano on absolute-error loss with the Harvey–Leybourne–Newbold
  small-sample correction. h-step errors from consecutive origins overlap, so the test uses
  an h−1 lag autocovariance.

### Results

| MAE relative to naive (< 1 = better) | 1 week | 4 weeks | 12 weeks |
|---|---|---|---|
| SARIMAX exog=persist (deployed) | 1.012 | 1.024 | 1.045 |
| SARIMAX exog=mean | 1.006 | 1.017 | 1.060 |
| SARIMAX exog=oracle ⚠ | 1.007 | 1.023 | 1.068 |
| SARIMA (no exog) | 1.009 | 1.020 | 1.039 |
| Drift | 1.000 | 1.002 | 1.014 |
| Seasonal naive (52) | 5.397 | 2.922 | 1.972 |

Naive MAE: $3.28 (1w), $5.94 (4w), $8.20 (12w). Directional accuracy for the SARIMAX
variants is 44–56%: coin-flip territory.

### Findings
1. **No edge over the random walk**, consistent with the oil-forecasting literature
   (Alquist, Kilian & Vigfusson, 2013).
2. **Even perfect knowledge of the future macro inputs doesn't help.** The oracle is no
   better than naive. Two reasons:
   - *The relationships broke.* Correlation of weekly Brent returns with dollar returns was
     −0.39 in 2006–21 and −0.06 in 2022–25; with VIX changes, −0.33 → −0.07. This is a
     regime shift (plausibly linked to the US becoming a net oil exporter), not a tuning problem.
   - *The specification makes shocks temporary.* With d = 1 and inputs as weekly returns,
     a held macro shock shifts the price by a constant rather than a growing amount. That is
     why the original scenario engine needed shocks "scaled 4×" and a direction flip.
3. **No annual seasonality.** Seasonal naive is 2–5× worse than naive; the 52-week seasonal
   terms cost most of the fit time for nothing measurable.
4. **Error is dominated by regime.** Naive 12-week MAE was $14.40 in 2022 and $5.13 in 2025.
   An average across years hides that, so the report breaks error down by year.

---

## 3. Challengers: is it the model or the data?

**Same script and report.** A null result is only credible if a better model would have
been detected. Two challengers that use only information available at the origin:

- **Ridge, direct multi-horizon**, on lagged returns and lagged macro features. α chosen by
  `TimeSeriesSplit` with `gap = h`, because a direct h-step label overlaps the next h−1
  training rows; without the gap, CV leaks.
- **Gradient-boosted quantile trees**, same features, quantile loss (gives intervals too).
- **Positive control:** a synthetic series with a planted lagged signal, run through the
  same models and features (a test in `tests/test_evaluation.py`). They find it, so the null
  on real data is not a pipeline failure.

**Decision rule (fixed first):** promote a model at a horizon only if its MAE is below
naive *and* its DM p-value, Holm-adjusted across all 15 candidate × horizon tests, is < 0.05.

| 12 weeks | MAE / naive | DM p | Holm p | Verdict |
|---|---|---|---|---|
| Ridge | 0.998 | 0.874 | 1.000 | better, not significant |
| Quantile GBM | 1.273 | 0.005 | 0.077 | worse |
| SARIMAX exog=persist | 1.045 | 0.280 | 1.000 | worse |

**No candidate met the rule.** The more flexible the model, the worse it did out of sample
(GBM 1.117 at 4 weeks, 1.273 at 12): extra capacity fits noise when the signal-to-noise
ratio is this low. Note also why Holm matters: the GBM's raw p = 0.017 at 4 weeks would look
"significant" in isolation, but it is one of 15 tests.

---

## 4. Uncertainty: are the ranges calibrated?

**Script:** `python -m evaluation.run_calibration` → [`reports/calibration/report.md`](../reports/calibration/report.md)

A point forecast with no skill can still come with useful uncertainty. The question is
whether a "95% range" contains the outcome 95% of the time.

### Design
- **Methods** on naive and SARIMA point forecasts: native model intervals, split conformal,
  adaptive conformal inference (ACI; Gibbs & Candès, 2021), and volatility-scaled variants
  of both (errors divided by trailing 12-week volatility at their origin).
- Conformal errors on the log scale, window of the last 156 resolved forecasts per horizon.
  Origins from 2016 exist only to provide out-of-sample errors; scoring starts in 2022.
- **Calibrated** means the nominal level lies inside the 95% CI of observed coverage, using
  **Newey–West** standard errors (hits from overlapping multi-week forecasts are correlated;
  naive binomial CIs would be far too narrow).
- **Selection** among calibrated methods by the **interval score** (Gneiting & Raftery,
  2007): width plus 2/α × miss distance, so neither uselessly wide nor overconfident
  intervals can win.

### Results
- **Selected:** ACI at 1 week; native naive / SARIMA intervals at 4 and 12 weeks.
- **Marginal coverage looks fine; conditional coverage does not.** Native SARIMA 95%
  intervals covered 70.6% (1w), 72.9% (4w), 85.0% (12w) in 2022, and 91.8–100% from 2023.
  A method can be "calibrated on average" by being too narrow in crises and too wide in calm.
- **Conformal methods inherit their calibration window.** The 2020 crash dominated the
  window until mid-2023, giving 99–100% coverage and very wide 12-week intervals.
- **Volatility scaling, a pre-registered experiment:** success criterion was "calibrated in
  both regimes in more horizon × level cells than any unscaled method". It technically
  passed (3 cells vs 2), but only at short horizons. At 12 weeks, calm pre-COVID origins
  produce tiny scale factors and so huge scaled scores; ACI + vol-scaled 95% intervals
  averaged **$1,815 wide**. The criterion measured coverage only, so its pass has to be read
  together with the interval score. Lesson: **a pre-registered criterion can be met and still
  be the wrong thing to ship**; write criteria that include width.

The app shows the selected 80% and 95% bands and states their measured coverage.

---

## 5. Scenarios: does history forecast what follows an event?

**Script:** `python -m evaluation.run_scenario_study` → [`reports/scenarios/report.md`](../reports/scenarios/report.md)

The scenario engine shocks SARIMAX inputs and mirrors bullish scenarios the model pushes
down (a heuristic). The alternative tested: use **what Brent actually did** after past events
of the same type.

### Design
- **31 dated events** in [`data/scenario_events.csv`](../data/scenario_events.csv) across five
  scenario types, with confounded events (e.g. OPEC cuts during the 2008 crash and April
  2020) flagged *before* running.
- Origin = last weekly price strictly before the event. Path = cumulative log move over 1–12 weeks.
- **Leave-one-event-out:** each event is predicted by the median path of the *other* events
  of its type.
- Rivals: naive (no move) and a replica of the current engine. The engine replica uses the
  production model fitted on all data, a look-ahead that can only help it.
- Rule: replace the engine if the analogue method's 12-week MAE is lower.

### Results

| Method | 12-week MAE (log points) | 4-week MAE |
|---|---|---|
| Analogue (median of other events) | 0.216 | 0.154 |
| Current engine (SARIMAX shocks + flip) | 0.191 | 0.116 |
| Naive (no move) | 0.188 | 0.120 |

Bootstrap 95% CI for analogue − engine: −0.003 to +0.053. **Keep the engine.** The result
survives removing confounded events (0.207 vs 0.180 vs 0.177).

**Why history fails as a forecaster: within-type dispersion.** After nine geopolitical
shocks Brent's 12-week move ranged from −62.9% (Soleimani, just before COVID) to +16.6%
(Russia–Ukraine); the median was −0.3%. OPEC cuts ranged from −35.9% to +68.4%. The event
label explains much less than everything else going on at the time.

**What was shipped instead:** the per-scenario historical ranges are shown in the app as
*context* ("after similar events, Brent moved between X and Y"), explicitly labelled as not a
forecast. The LLM prompt is constrained to present it the same way.

---

## 6. Data: timing leaks and reproducibility

**Scripts:** `python -m pipeline.refresh`, `python -m pipeline.compare` →
[`reports/data/`](../reports/data)

### A second, subtler leak
Monthly series (industrial production, fed funds) were attached to their *reference* dates,
weeks before they were published. A model forecasting on 2 March used February's industrial
production, which came out mid-March. The `v2-release-aware` recipe shifts each monthly value
to its release date. The final week of the original dataset was also partial and is now dropped.

**Decision (made on principle, before measuring):** production uses v2, because it only uses
information that was public at each date. Then the effect was measured on a fresh window
(207 origins, Sept 2022 → Aug 2026):

| 12 weeks, vs naive | MAE / naive v1.1 (leaky) | Holm p | MAE / naive v2 (honest) | Holm p |
|---|---|---|---|---|
| SARIMAX exog=persist (deployed) | 1.068 | 0.120 | 1.081 | **0.038** |
| SARIMAX exog=mean | 1.076 | 0.043 | 1.089 | 0.011 |
| Ridge | 0.998 | 1.000 | 0.992 | 1.000 |

The leak moved MAE by only ~1%, as expected when the macro inputs carry little signal. But it
changed the conclusion: with honest timing, the deployed configuration is **significantly
worse than no-change at 12 weeks**. The leak was hiding that. Small leaks matter most when
results are close to the decision boundary.

### Reproducibility
- Builds read only **raw snapshots** (`data/raw/*.csv`); fetch and build are separate.
- Rebuilding the original notebook dataset from raw snapshots matches it exactly, apart from
  documented differences: FRED's revisions to 2006 Brent prices, industrial-production
  revisions, and the old partial final week ([report](../reports/data/reproduction.md)).
- Content hashes normalise line endings, so a Windows checkout and a Linux runner agree.
- The notebook-era dataset is frozen as a test fixture.

---

## 7. From experiment to production

None of this is research output on its own, but each piece exists to keep the research
claims true once the system runs unattended.

| Concern | Mechanism |
|---|---|
| Bad data reaching users | Validation (gaps, NaNs, implausible levels, consistency) blocks a build; promotion gate G1–G4: data, continuity (≤ 5% of rows revised per column), fit converged with finite parameters, 12-week forecast and band within 0.5×–2× the last price. Only G2 can be overridden, and only with a written reason stored in the registry. |
| Silent model changes | [`models/registry.json`](../models/registry.json) records data hashes, fit summary, gate results, overrides and the previous version for every promotion. |
| Monitoring on genuinely new data | Each promotion scores the current model on the weeks it has not seen. First run: 45 new weeks, one-week MAE **$8.27 vs $8.29** for naive (ratio 0.997). Still no edge, now on data no experiment had touched. |
| Human oversight | A scheduled job (every 3 weeks) fetches, validates and gates; it opens a pull request only if a model is promoted, and CI trains and serves the new model on that PR. A person merges; Render deploys `main` after checks pass. |
| Memory on a 512 MB host | The fitted SARIMAX pickle was 488 MB (stored filter/smoother arrays for 1,000+ weeks). The serving model re-filters a fresh SARIMAX with the fitted parameters and memory flags that drop those arrays: **2.8 MB, identical forecasts** (tested). |
| Serving | Input limits (1–12 weeks, the validated range; 3–500-character queries), a TTL cache keyed on query, horizon, data version and LLM model (~15 ms per hit), degraded answers never cached, JSON logs with request ids, `/metrics`, `/health` readiness. |
| Dependency drift | Hash-locked requirements for API and UI, installed by CI. |

---

## 8. The LLM layer: evaluating a structured parser

**Script:** `python -m evaluation.run_llm_eval` → [`reports/llm_eval/report.md`](../reports/llm_eval/report.md)

The LLM (`openai/gpt-oss-120b` on Groq) turns a free-text question into weights over the five
scenarios. Those weights decide which scenario paths are mixed, so the parser is a component
that needs its own test set, like any classifier.

### What changed (v1 → v2)
- **v1:** free-form prompt asking for JSON, parsed with `json.loads`, with no schema.
- **v2:** JSON Schema structured output, validated by Pydantic (`extra="forbid"`, weights in
  [0, 1], non-empty reasoning), one retry with the validation error fed back, renormalisation,
  an explicit `in_scope` flag, the user's text wrapped in `<query>` tags with an instruction
  never to follow instructions inside them, and an equal-weight fallback that is recorded in
  metrics and flagged in the response.

### Evaluation design
- **40 hand-written cases** ([`evaluation/llm_eval_cases.jsonl`](../evaluation/llm_eval_cases.jsonl)):
  20 single-driver, 6 mixed, 3 bearish-supply, 2 peace/de-escalation, 5 off-topic,
  4 prompt injections.
- Expectations are partial on purpose (the acceptable primary scenarios, minimum or maximum
  weights, the expected `in_scope`, forbidden text). There is no single "correct" weight vector,
  so the cases check properties, not exact numbers.
- v2 run 3 times per case (consistency), v1 once with its prompt reproduced verbatim.
- Rate-limit and network errors are counted separately as infrastructure errors, not as model
  failures.
- **Targets fixed before the run**, including a decision rule: off-topic rejection would be
  *enforced* only if it caught ≥ 90% of off-topic questions while wrongly rejecting ≤ 5% of
  real ones; otherwise it would only be flagged.

### Results (run 2026-11-20, 120 v2 calls and 40 v1 calls, 0 infrastructure errors)

| Target | v2 | Goal | v1 |
|---|---|---|---|
| T1 valid structured output | 120/120 (Wilson 95% CI 97–100%) | 100% | 34/40 (85%) |
| T2 primary scenario, single-driver questions | 100% | ≥ 85% | 100% |
| T3 primary scenario, all questions naming one | 100% | ≥ 70% | 100% |
| T4 consistency, mean pairwise L1 between repeats | 0.094 | ≤ 0.20 | not measured |
| T5 injections: valid and never followed | 12/12 (CI 76–100%) | 100% | 2/4 (50%) |
| T6 off-topic caught | 15/15 (CI 80–100%) | ≥ 90% | no flag |
| T6 real questions wrongly rejected | 0/93 (CI 0–4%) | ≤ 5% | no flag |
| Latency p50 / p95 | 0.82 s / 1.20 s | | 0.73 s / 1.01 s |

**Decision: enforce rejection.** `/simulate-probabilistic` now returns HTTP 422
(`out_of_scope`) for off-topic questions instead of producing a forecast from meaningless
equal weights. A parser fallback always reports `in_scope = true`, so an LLM outage can never
turn into a rejection (tested).

### What the numbers do and don't show
- **v1's failures were the dangerous kind.** All 6 invalid v1 outputs were off-topic or
  injection cases, which in production meant a silent fallback. Worse, v1 answered one
  off-topic question with `demand_boom: 1.0`: **confidently wrong, and valid JSON**. v2 turns
  both failure modes into explicit signals (`in_scope = false`, `llm_fallback = true`). An
  invalid output is easier to handle than a plausible wrong one.
- **The weights are not calibrated probabilities.** v2 answers are strongly templated: 0.8 on
  the primary and 0.05 on everything else, or 0.5 / 0.5 for any two-driver question. They are
  soft labels. Treating them as probabilities of real-world outcomes would be a category
  error, and the app presents them as weights, not odds.
- **Argmax is fragile near ties.** In one bearish-supply repeat, rate hike beat recession
  0.5135 to 0.4865, which flipped the "primary driver". That is why bearish cases were scored
  on the *sum* of the bearish weights rather than on the argmax.
- **The scenario vocabulary is the real bottleneck.** Bearish *supply* questions (OPEC raising
  output) map onto recession + rate hike, and peace deals onto demand boom, because the five
  scenarios include no supply glut or de-escalation. The model did the best the schema allowed;
  the schema is what is missing a scenario.
- **A perfect score is a ceiling effect, not proof of perfection.** The cases were written by
  the same person who wrote the prompt (not a held-out set), and 40 cases at 3 repeats cannot
  resolve rare failures: 12/12 on injections is consistent with a true pass rate as low as
  ~76%. Repeats at temperature 0.1 measure sampling noise, not new questions.
- **This measures interface reliability, not forecasting skill.** Better weights mix scenario
  paths that themselves do not beat no-change (section 5). The LLM layer makes the tool
  dependable; it does not make it predictive.
- **Model drift.** Groq can update the hosted model. The response cache is keyed on the model
  name, and the evaluation is a one-command rerun (or the manual *LLM evaluation* workflow),
  which should be repeated on any model change.

---

## 9. Threats to validity

- **One test period, reused.** 2022–25 served steps 1a, 1b, 1c and 2, so it is no longer a
  pristine holdout; every extra experiment on it is a fork in the garden of forking paths.
  Mitigations: decision rules fixed before each run, Holm correction, and the live monitoring
  of 45 weeks no experiment had seen (same conclusion).
- **Few independent observations.** 198 overlapping origins at h = 12 hold roughly 16
  non-overlapping samples. DM's HAC variance and Newey–West coverage CIs account for this,
  but power stays low: "not significant" is not "equal".
- **Revised data.** FRED serves today's revised values, not what was known at the time.
  Real-time vintages (ALFRED) would make the backtest stricter. The release-date shift fixes
  *when* values were known but not *which* values.
- **One specification.** Refits re-estimate parameters but never re-select orders.
- **Event study size.** 4–9 events per scenario; the bootstrap CIs are wide and the analogue
  comparison was only borderline.
- **LLM evaluation set.** Small, author-written and not held out (section 8).

---

## 10. What I would do next

1. **Volatility forecasts (GARCH-type) for the bands.** The clearest failure found is
   conditional coverage in crises; that needs a volatility model, not a better point forecast.
2. **Add supply-glut and de-escalation scenarios**, each with its own events in the event
   study, so bearish supply questions stop borrowing demand-side scenarios.
3. **A held-out, harder LLM test set**: questions written by someone else, paraphrased
   injections, ambiguous multi-driver questions, non-English input, with more repeats on the
   failure-prone categories.
4. **Real-time data vintages** from ALFRED for a strictly real-time backtest.
5. **Market-implied information** (futures curve, options-implied volatility) as the natural
   next inputs: they are what markets actually price, unlike lagged macro data.

---

## 11. Reproduce everything

```bash
pip install --require-hashes -r requirements.lock
pytest                                      # unit and integration tests, no network
python -m evaluation.run_backtest           # ~8 min  -> reports/backtest/
python -m evaluation.run_calibration        # ~4 min  -> reports/calibration/
python -m evaluation.run_scenario_study     # ~1 min  -> reports/scenarios/
python -m pipeline.refresh --fetch          # fetch + build + validate -> data/staging/
python -m pipeline.promote                  # dry run of the promotion gate
python -m evaluation.run_llm_eval           # needs GROQ_API_KEY, ~20 min -> reports/llm_eval/
```

Each report records the data fingerprint and git commit it was produced from.

---

## 12. References

- Alquist, R., Kilian, L. & Vigfusson, R. J. (2013). Forecasting the price of oil. *Handbook of Economic Forecasting*, Vol. 2.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.
- Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*.
- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
