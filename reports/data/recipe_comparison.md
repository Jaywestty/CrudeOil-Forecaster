# Recipe comparison: v1.1 (monthly data on reference dates) vs v2 (on release dates)

_Walk-forward backtest (`evaluation.run_backtest`) run on both builds of the same raw snapshots._

- Origins: 207 weekly, 2022-09-11 → 2026-08-23; horizons 1–12 weeks; yearly refits
- v1.1 data fingerprint `{"oil_macro_weekly.csv": "9d65948d4dd7ecbd", "oil_macro_transformed.csv": "c9c899d9ecd45f48"}`
- v2 data fingerprint `{"oil_macro_weekly.csv": "37de0d09fe239015", "oil_macro_transformed.csv": "eb0ed461331fb486"}`
- Only industrial production and fed funds differ between the two builds

## Decision (fixed before running)

Production uses v2 because it only uses information that was public at each date. This comparison measures how much the old timing leak changed the results; accuracy does not decide whether to keep a leak.

## Accuracy

| horizon | model | MAE v1.1 ($) | MAE v2 ($) | MAE / naive v1.1 | MAE / naive v2 |
|---|---|---|---|---|---|
| 1 | SARIMA (no exog) | 3.59 | 3.59 | 1.015 | 1.015 |
| 1 | SARIMAX exog=persist | 3.59 | 3.56 | 1.013 | 1.006 |
| 1 | Ridge direct (lagged features) | 3.59 | 3.57 | 1.013 | 1.008 |
| 1 | Quantile GBM direct (lagged features) | 3.65 | 3.67 | 1.032 | 1.036 |
| 4 | SARIMA (no exog) | 6.99 | 6.99 | 1.032 | 1.032 |
| 4 | SARIMAX exog=persist | 7.02 | 7.08 | 1.036 | 1.045 |
| 4 | Ridge direct (lagged features) | 6.96 | 6.98 | 1.028 | 1.030 |
| 4 | Quantile GBM direct (lagged features) | 7.23 | 7.31 | 1.067 | 1.079 |
| 12 | SARIMA (no exog) | 11.79 | 11.79 | 1.054 | 1.054 |
| 12 | SARIMAX exog=persist | 11.95 | 12.10 | 1.068 | 1.081 |
| 12 | Ridge direct (lagged features) | 11.16 | 11.09 | 0.998 | 0.992 |
| 12 | Quantile GBM direct (lagged features) | 12.82 | 13.50 | 1.146 | 1.207 |

Naive and plain SARIMA do not use the macro data, so they are identical under both recipes.

## 12 weeks: Diebold–Mariano vs naive, Holm-adjusted

| model | MAE / naive v1.1 | Holm p v1.1 | MAE / naive v2 | Holm p v2 |
|---|---|---|---|---|
| SARIMA (no exog) | 1.054 | 0.532 | 1.054 | 0.488 |
| SARIMAX exog=persist | 1.068 | 0.120 | 1.081 | 0.038 |
| SARIMAX exog=mean | 1.076 | 0.043 | 1.089 | 0.011 |
| Ridge direct (lagged features) | 0.998 | 1.000 | 0.992 | 1.000 |
| Quantile GBM direct (lagged features) | 1.146 | 0.091 | 1.207 | 0.021 |

## Reading

- The leak's effect on accuracy is small (about 1% of MAE), as expected given the macro inputs carry little signal.
- It matters for the conclusion: with honest timing (v2), the deployed SARIMAX setup (exog=persist) is significantly worse than a no-change forecast at 12 weeks after the Holm correction (p = 0.038); with the leak (v1.1) the same comparison was not significant (p = 0.12). The leak was flattering the model enough to hide that.
- This test window (Sept 2022 – Aug 2026) differs from the step-1 window because the data now extends to Nov 2026; results are not directly comparable with reports/backtest.
