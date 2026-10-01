"""Optional browser integration check for the standalone GUI.

Run with Playwright and NumPy installed. Pass --browser for a local Chromium executable.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from web.server import SpectraConsoleHandler  # noqa: E402


def check(browser_path: str | None, screenshots: Path | None) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SpectraConsoleHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=browser_path, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            base = f"http://127.0.0.1:{server.server_port}"
            page.goto(base)
            page.wait_for_function("document.querySelector('#run-status').textContent === 'Paused'")
            assert page.locator("#run-title").inner_text() == "Frequency agile"
            assert page.locator("#waterfall-axis-y .band-label-cell").first.inner_text() == "Band 7"
            page.locator("#btn-play").click()
            page.wait_for_function("Number(document.querySelector('#timeline').value) > 3")
            page.locator("#btn-play").click()
            assert page.locator("#btn-play").inner_text() == "Play"
            page.locator("#timeline").evaluate(
                "e => {e.value = 90; e.dispatchEvent(new Event('input', {bubbles:true}));}"
            )
            page.locator("#btn-next").click()
            assert page.locator("#timeline").input_value() == "91"
            for view in ["baseline", "dual", "active"]:
                page.locator(f'[data-view="{view}"]').click()
                assert page.locator(f'[data-view="{view}"]').get_attribute("aria-pressed") == "true"
            page.locator("#fog-toggle").check()
            assert page.locator("#truth-legend").is_visible()
            page.locator("#waterfall-hud").hover(position={"x": 120, "y": 50})
            assert page.locator("#waterfall-tooltip").is_visible()
            page.locator("#fog-toggle").uncheck()
            page.locator("#waterfall-hud").hover(position={"x": 121, "y": 51})
            assert "Simulation truth" not in page.locator("#waterfall-tooltip").inner_text()
            page.locator("#fog-toggle").check()
            page.locator(".policy-details summary").click()
            page.wait_for_function(
                "document.querySelector('#belief-bars-container').textContent"
                ".includes('does not expose')"
            )
            assert "No UCB scores" in page.locator("#ucb-stack-container").inner_text()
            page.locator(".policy-details summary").click()
            if screenshots:
                screenshots.mkdir(parents=True, exist_ok=True)
                page.mouse.move(0, 0)
                page.screenshot(path=str(screenshots / "demonstration.png"), full_page=True)
            page.locator('[data-tab="tab-fom"]').click()
            assert page.locator(".results-table tbody tr").count() >= 14
            assert "Unavailable" in page.locator(".results-table").inner_text()
            if screenshots:
                page.screenshot(path=str(screenshots / "results.png"), full_page=True)
            page.locator('[data-tab="tab-feature-space"]').click()
            assert page.locator("#feature-space-canvas").evaluate("e => e.width") > 0
            if screenshots:
                page.screenshot(path=str(screenshots / "tracks.png"), full_page=True)
            page.locator('[data-tab="tab-cockpit"]').click()
            with page.expect_download() as download:
                page.locator("#btn-export").click()
            payload = json.loads(Path(download.value.path()).read_text())
            assert payload["scenario"]["seed"] == 42
            # Alternate six/eight-band buffers and restore scenario receiver defaults.
            for preset, bands in [
                ("mode-switch", 6),
                ("periodic-scan", 8),
                ("crowded-battlefield", 8),
            ]:
                page.locator("#select-preset").select_option(preset)
                page.wait_for_function(
                    "document.querySelector('#run-status').textContent === 'Paused' && "
                    "!document.querySelector('#btn-run-sim').disabled"
                )
                assert page.locator("#waterfall-axis-y .band-label-cell").count() == bands
            duration = int(page.locator("#timeline").get_attribute("max"))
            page.locator("#timeline").evaluate(
                "(e, v) => {e.value = v; e.dispatchEvent(new Event('input', {bubbles:true}));}",
                duration - 1,
            )
            page.locator("#btn-play").click()
            page.wait_for_function("document.querySelector('#btn-play').textContent === 'Replay'")
            page.locator("#btn-play").click()
            page.wait_for_function("Number(document.querySelector('#timeline').value) < 20")
            page.locator("#btn-play").click()
            page.route(
                "**/api/run",
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body='{"error":"Intentional browser check failure"}',
                ),
            )
            previous = page.locator("#run-title").inner_text()
            page.locator("#btn-run-sim").click()
            page.wait_for_function("!document.querySelector('#error-banner').hidden")
            assert page.locator("#run-title").inner_text() == previous
            assert "previous run" in page.locator("#error-banner").inner_text()
            assert not page.locator("#btn-play").is_disabled()
            page.unroute("**/api/run")
            for width in [1024, 768, 390]:
                page.set_viewport_size({"width": width, "height": 900})
                page.wait_for_timeout(100)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                if screenshots:
                    page.screenshot(path=str(screenshots / f"layout-{width}.png"), full_page=True)
            # First-load failure is readable and does not start an empty simulation.
            failed = browser.new_page()
            failed.route("**/api/scenarios", lambda route: route.abort())
            failed.goto(base)
            failed.wait_for_function("!document.querySelector('#error-banner').hidden")
            assert failed.locator("#btn-play").is_disabled()
            assert not errors, errors
            browser.close()
        print(
            "Browser checks passed: playback, presets, views, truth, metrics, "
            "tracks, export, failures, responsive layouts."
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Chromium-compatible executable")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    check(args.browser, args.screenshots)
