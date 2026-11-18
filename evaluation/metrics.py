"""Forecast accuracy, interval calibration and significance testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs((y_true - np.asarray(y_pred)) / y_true)) * 100)


def mase_scale(y_train: np.ndarray) -> float:
    """In-sample MAE of the one-step naive forecast (Hyndman & Koehler, 2006)."""
    return float(np.mean(np.abs(np.diff(np.asarray(y_train, dtype=float)))))


def directional_accuracy(y_origin, y_true, y_pred) -> float:
    """Share of forecasts that call the direction of the move correctly.

    Rows where the forecast or the outcome shows no change are excluded, so a
    flat forecast (Naive) has no defined direction and returns NaN.
    """
    actual = np.sign(np.asarray(y_true) - np.asarray(y_origin))
    called = np.sign(np.asarray(y_pred) - np.asarray(y_origin))
    mask = (actual != 0) & (called != 0)
    return float(np.mean(actual[mask] == called[mask])) if mask.any() else float("nan")


def interval_coverage(y_true, lower, upper) -> float:
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if np.all(np.isnan(lower)):
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def diebold_mariano(e_model, e_bench, horizon: int, loss: str = "abs") -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    H0: both forecasts have equal expected loss. A negative statistic means the
    model's loss is lower than the benchmark's. The variance uses autocovariances
    up to lag ``horizon - 1`` because h-step errors from consecutive origins
    overlap and are therefore correlated.

    Returns (statistic, two-sided p-value).
    """
    e1, e2 = np.asarray(e_model, dtype=float), np.asarray(e_bench, dtype=float)
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1**2 - e2**2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")

    n = len(d)
    if n < 2 * horizon + 1:
        return float("nan"), float("nan")
    d_bar = d.mean()
    dc = d - d_bar
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:  # the unweighted autocovariance sum can go negative; fall back to lag 0
        var = gamma[0] / n
    if var == 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p


def summarize(
    forecasts: pd.DataFrame,
    scale: float,
    benchmark: str = "Naive",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """One row per (horizon, model) with accuracy, calibration and DM-vs-benchmark."""
    df = forecasts if horizons is None else forecasts[forecasts["horizon"].isin(horizons)]
    rows = []
    for h, g in df.groupby("horizon"):
        bench = g[g["model"] == benchmark].set_index("origin")
        bench_err = bench["y_true"] - bench["y_pred"]
        bench_mae = mae(bench["y_true"], bench["y_pred"])
        for model, m in g.groupby("model", sort=False):
            m = m.set_index("origin").loc[bench.index]  # identical origins for a paired test
            err = m["y_true"] - m["y_pred"]
            stat, p = (np.nan, np.nan) if model == benchmark else diebold_mariano(err, bench_err, int(h))
            rows.append(
                {
                    "horizon": int(h),
                    "model": model,
                    "oracle": bool(m["oracle"].iloc[0]),
                    "n": len(m),
                    "MAE": mae(m["y_true"], m["y_pred"]),
                    "RMSE": rmse(m["y_true"], m["y_pred"]),
                    "MAPE": mape(m["y_true"], m["y_pred"]),
                    "MASE": mae(m["y_true"], m["y_pred"]) / scale,
                    "rel_MAE": mae(m["y_true"], m["y_pred"]) / bench_mae,
                    "DM_stat": stat,
                    "DM_p": p,
                    "dir_acc": directional_accuracy(m["y_origin"], m["y_true"], m["y_pred"]),
                    "coverage_95": interval_coverage(m["y_true"], m["lower"], m["upper"]),
                    "mean_width_95": float((m["upper"] - m["lower"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def mae_by_year(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """MAE per model per target-date year, to show how error depends on regime."""
    g = forecasts[forecasts["horizon"] == horizon].copy()
    g["abs_err"] = (g["y_true"] - g["y_pred"]).abs()
    g["year"] = pd.to_datetime(g["target_date"]).dt.year
    return g.pivot_table(index="year", columns="model", values="abs_err", aggfunc="mean", sort=False)
