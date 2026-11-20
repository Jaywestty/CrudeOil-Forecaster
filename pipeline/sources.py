"""Fetch raw series and store them as snapshots.

Raw snapshots (data/raw/<name>.csv) are the pipeline's single source of truth:
the build step reads only these files, so a dataset can always be rebuilt
offline, byte-for-byte, from what was fetched.

Sources
  FRED  official API (api.stlouisfed.org) when FRED_API_KEY is set. Without a
        key, falls back to the website's CSV download (fredgraph.csv), which
        is not meant for automation and timed out from GitHub's runners.
  EIA   API v2; uses EIA_API_KEY if set, else EIA's public DEMO_KEY
        (rate-limited, fine for a weekly refresh)

Error messages never include query strings, so API keys cannot leak into logs.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# column name -> (source, series id)
SERIES = {
    "brent":           ("fred", "DCOILBRENTEU"),  # Brent spot, USD/bbl, daily
    "inventories":     ("eia",  "WCRSTUS1"),      # US crude ending stocks incl. SPR, kbbl, weekly
    "industrial_prod": ("fred", "INDPRO"),        # Industrial production index, monthly
    "dollar_index":    ("fred", "DTWEXBGS"),      # Broad trade-weighted dollar, daily
    "fed_funds":       ("fred", "FEDFUNDS"),      # Effective fed funds rate, monthly average
    "vix":             ("fred", "VIXCLS"),        # CBOE VIX close, daily
}

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EIA_API = "https://api.eia.gov/v2/seriesid/PET.{series}.W"
# FRED's website endpoint drops connections from some non-browser agents.
USER_AGENT = "Mozilla/5.0 (compatible; data-pipeline)"
TIMEOUT = (10, 60)  # (connect, read) seconds: fail fast on unreachable hosts
RETRIES = 4  # attempts per request, with exponential backoff (2, 4, 8 s)


def _get(session: requests.Session, url: str, params: dict) -> requests.Response:
    """GET with retries: public data APIs fail transiently (timeouts, dropped connections, 5xx).

    Errors name the host and path only: params may hold an API key, and
    requests' own messages include the full URL.
    """
    last_error = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
            if r.status_code < 400:
                return r
            if r.status_code < 500 and r.status_code != 429:
                # Other 4xx (bad key, unknown series) will not fix itself: fail now.
                raise requests.HTTPError(f"HTTP {r.status_code} from {url}")
            last_error = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = type(e).__name__  # e.g. ReadTimeout; str(e) could contain the URL
        if attempt < RETRIES - 1:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts (last error: {last_error})")


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """FRED CSV: a date column and a value column; '.' marks missing values."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2 or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED response for {series_id}: columns {list(df.columns)}")
    s = pd.Series(df[series_id].to_numpy(float), index=pd.to_datetime(df.iloc[:, 0]), name=series_id)
    return s.dropna().sort_index()


def parse_fred_api_json(payload: dict, series_id: str) -> pd.Series:
    """FRED API: {"observations": [{"date": ..., "value": ...}]}; '.' marks missing values."""
    obs = payload.get("observations")
    if not isinstance(obs, list) or not obs:
        raise ValueError(f"FRED API returned no observations for {series_id}: "
                         f"{payload.get('error_message', 'empty response')}")
    values = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy(float)
    s = pd.Series(values, index=pd.to_datetime([o["date"] for o in obs]), name=series_id)
    return s.dropna().sort_index()


def parse_eia_json(payload: dict, series_id: str) -> pd.Series:
    rows = payload["response"]["data"]
    if not rows:
        raise ValueError(f"EIA returned no data for {series_id}")
    s = pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([r["period"] for r in rows]),
        name=series_id,
    )
    return s[~s.index.duplicated()].sort_index()


def fetch_fred(series_id: str, session: requests.Session | None = None) -> pd.Series:
    session = session or requests.Session()
    key = os.getenv("FRED_API_KEY")
    if key:
        params = {"series_id": series_id, "api_key": key, "file_type": "json"}
        return parse_fred_api_json(_get(session, FRED_API, params).json(), series_id)
    r = _get(session, FRED_CSV, {"id": series_id})
    return parse_fred_csv(r.text, series_id)


def fred_source() -> str:
    return "FRED API" if os.getenv("FRED_API_KEY") else "FRED website CSV (no FRED_API_KEY set)"


EIA_DEMO_KEY = "DEMO_KEY"
EIA_KEY_HELP = ("Register a free key at https://www.eia.gov/opendata/register.php and set it as "
                "EIA_API_KEY (a repository secret for the weekly GitHub Action).")


def fetch_eia(series_id: str, session: requests.Session | None = None) -> pd.Series:
    session = session or requests.Session()
    key = os.getenv("EIA_API_KEY") or EIA_DEMO_KEY
    params = {
        "api_key": key,
        "data[]": "value",
        "facets[series][]": series_id,
        "frequency": "weekly",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    try:
        r = _get(session, EIA_API.format(series=series_id), params)
    except (requests.HTTPError, RuntimeError) as e:
        if key == EIA_DEMO_KEY:
            # The shared demo key is limited per IP; cloud runners share IPs, so
            # it is refused there (weekly refresh run #2: HTTP 403).
            raise RuntimeError(f"EIA refused the public DEMO_KEY ({e}). {EIA_KEY_HELP}") from None
        raise
    return parse_eia_json(r.json(), series_id)


def fetch_all() -> dict[str, pd.Series]:
    """Fetch the full history of every series."""
    session = requests.Session()
    print(f"  FRED source: {fred_source()}", flush=True)
    out = {}
    for name, (source, series_id) in SERIES.items():
        fetch = fetch_fred if source == "fred" else fetch_eia
        out[name] = fetch(series_id, session).rename(name)
        print(f"  {name:<16} {series_id:<13} {len(out[name]):>6} obs, "
              f"{out[name].index[0].date()} -> {out[name].index[-1].date()}", flush=True)
    return out


def save_snapshots(raw: dict[str, pd.Series], raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, s in raw.items():
        s.rename("value").rename_axis("date").to_csv(raw_dir / f"{name}.csv", lineterminator="\n")


def load_snapshots(raw_dir: Path = RAW_DIR) -> dict[str, pd.Series]:
    out = {}
    for name in SERIES:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw snapshot {path}; run `python -m pipeline.refresh` first")
        df = pd.read_csv(path, index_col="date", parse_dates=["date"])
        out[name] = df["value"].astype(float).rename(name)
    return out
