"""Open the Streamlit app in a headless browser so it stays (or becomes) awake.

Streamlit Community Cloud puts apps to sleep after a period without visitors,
and a plain HTTP request does not count as a visit: the app only runs when a
browser opens a session. This script loads the page like a visitor, clicks
"Yes, get this app back up!" if the app is asleep, and waits for it to render.

Usage:
    python ops/wake_streamlit.py https://crude-forecaster.streamlit.app

Always exits 0 (problems are printed as GitHub warnings): a failed keep-awake
run would otherwise attach a failing check to the latest main commit and hold
back Render's deploy.
"""

from __future__ import annotations

import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = "Yes, get this app back up"
APP_READY = "Crude Oil Scenario Forecaster"  # main header rendered by the app
TIMEOUT_MS = 180_000  # waking a sleeping app can take a couple of minutes


def wake(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        status = "awake"
        try:
            # The sleep page is top-level; give it a moment to show the button.
            page.get_by_role("button", name=WAKE_BUTTON).click(timeout=15_000)
            status = "was asleep, woken"
        except PlaywrightTimeout:
            pass  # no wake button: the app was already awake
        # The app itself renders inside an iframe on Streamlit Cloud.
        found = False
        for frame in [page.main_frame, *page.frames]:
            try:
                frame.get_by_text(APP_READY).first.wait_for(timeout=TIMEOUT_MS // 3)
                found = True
                break
            except PlaywrightTimeout:
                continue
        browser.close()
        return status if found else status + ", but the app did not finish rendering"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 0
    try:
        result = wake(argv[1])
        level = "notice" if "did not finish" not in result else "warning"
        print(f"::{level}::Streamlit app {result}")
    except Exception as e:  # never fail the run (see module docstring)
        print(f"::warning::Could not load the Streamlit app: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
