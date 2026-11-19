"""Historical-analogue scenario paths (event study).

Instead of inventing macro shocks and pushing them through SARIMAX, each
scenario is calibrated on what Brent actually did after real events of that
type (data/scenario_events.csv). For every event we measure the cumulative log
price move over the following 1..H weeks; a scenario's path is the median of
those moves across its events.

Timing rule (no look-ahead): an event's *origin* is the last weekly
observation strictly before the event date, so the starting price never
includes the event's own reaction. The path at week k is
log(price[origin + k] / price[origin]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EVENTS_FILE = Path(__file__).resolve().parent / "data" / "scenario_events.csv"
HORIZON = 12


def load_events(path: Path = EVENTS_FILE) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["date"], keep_default_na=False)
    required = {"scenario", "date", "event", "why_it_qualifies", "confounded_by"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event catalogue missing columns: {sorted(missing)}")
    if events.duplicated(["scenario", "date"]).any():
        raise ValueError("Duplicate (scenario, date) rows in event catalogue")
    events["confounded"] = events["confounded_by"].str.strip() != ""
    return events.sort_values(["scenario", "date"]).reset_index(drop=True)


def origin_position(index: pd.DatetimeIndex, event_date: pd.Timestamp) -> int:
    """Position of the last observation strictly before the event date."""
    pos = int(index.searchsorted(pd.Timestamp(event_date), side="left")) - 1
    if pos < 0:
        raise ValueError(f"No observation before {pd.Timestamp(event_date).date()}")
    return pos


def event_paths(y: pd.Series, events: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per event: origin date/price and the cumulative log move at weeks 1..horizon.

    Events with no observation before them, or without ``horizon`` weeks of
    data after the origin, are skipped (they cannot be measured).
    """
    prices = y.to_numpy(float)
    if (prices <= 0).any():
        raise ValueError("Prices must be positive to compute log moves")
    logy = np.log(prices)
    rows = []
    for ev in events.itertuples(index=False):
        if pd.Timestamp(ev.date) <= y.index[0]:
            continue
        o = origin_position(y.index, ev.date)
        if o + horizon >= len(y):
            continue
        moves = logy[o + 1 : o + horizon + 1] - logy[o]
        rows.append(
            {
                "scenario": ev.scenario,
                "date": ev.date,
                "event": ev.event,
                "confounded": bool(ev.confounded),
                "origin": y.index[o],
                "origin_price": float(y.iloc[o]),
                **{f"w{k}": float(m) for k, m in enumerate(moves, start=1)},
            }
        )
    return pd.DataFrame(rows)


def week_cols(horizon: int = HORIZON) -> list[str]:
    return [f"w{k}" for k in range(1, horizon + 1)]


def analogue_path(
    paths: pd.DataFrame,
    scenario: str,
    horizon: int = HORIZON,
    exclude_date: pd.Timestamp | None = None,
    include_confounded: bool = True,
) -> dict:
    """Median cumulative log move per week, plus the min/max across events.

    ``exclude_date`` drops one event (leave-one-out validation). Returns
    {"median", "low", "high"} arrays and the number of events used.
    """
    g = paths[paths["scenario"] == scenario]
    if exclude_date is not None:
        g = g[g["date"] != pd.Timestamp(exclude_date)]
    if not include_confounded:
        g = g[~g["confounded"]]
    if g.empty:
        raise ValueError(f"No events left for scenario '{scenario}'")
    m = g[week_cols(horizon)].to_numpy(float)
    return {
        "median": np.median(m, axis=0),
        "low": m.min(axis=0),
        "high": m.max(axis=0),
        "n_events": len(g),
    }
