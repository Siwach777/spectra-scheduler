"""Optional browser integration check for the standalone GUI.

Run with Playwright and NumPy installed. Pass --browser for a local Chromium executable.
"""

from __future__ import annotations

import argparse
import json
import os
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
            page.goto(base + "/?preset=video-demo")
            page.wait_for_function("document.querySelector('#run-status').textContent === 'Paused'")
            assert page.locator("#run-title").inner_text() == "Behavior change"
            assert page.locator("#waterfall-axis-y .band-label-cell").first.inner_text() == "Band 5"
            assert page.locator("#btn-prev").is_disabled()
            page.locator("#btn-play").click()
            page.wait_for_function("Number(document.querySelector('#timeline').value) > 3")
            page.locator("#btn-play").click()
            assert page.locator("#btn-play").inner_text() == "Play"
            page.locator("#timeline").evaluate(
                "e => {e.value = 20; e.dispatchEvent(new Event('input', {bubbles:true}));}"
            )
            page.locator("#btn-next").click()
            assert page.locator("#timeline").input_value() == "21"
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
                "document.querySelectorAll('#belief-bars-container .belief-row').length === 6"
            )
            assert page.locator("#belief-bars-container .belief-row").count() == 6
            assert not page.locator("#ucb-stack-container").is_visible()
            page.locator(".policy-details summary").click()
            if screenshots:
                screenshots.mkdir(parents=True, exist_ok=True)
                page.mouse.move(0, 0)
                page.screenshot(path=str(screenshots / "demonstration.png"), full_page=True)
            page.locator('[data-tab="tab-fom"]').click()
            assert page.locator("#fom-grid-container tr").count() >= 8
            assert "Prediction accuracy" not in page.locator("#fom-grid-container").inner_text()
            assert "Unavailable" not in page.locator("#fom-grid-container").inner_text()
            page.locator(".results-extra summary").click()
            assert "Longest band gap" in page.locator("#details-grid-container").inner_text()
            if screenshots:
                page.screenshot(path=str(screenshots / "results.png"), full_page=True)
            page.locator('[data-tab="tab-feature-space"]').click()
            assert page.locator("#feature-space-canvas").evaluate("e => e.width") > 0
            if screenshots:
                page.screenshot(path=str(screenshots / "tracks.png"), full_page=True)
            page.locator("#btn-theme").click()
            assert page.locator("html").get_attribute("data-theme") == "dark"
            if screenshots:
                page.screenshot(path=str(screenshots / "tracks-dark.png"), full_page=True)
            page.locator("#btn-theme").click()
            page.locator('[data-tab="tab-cockpit"]').click()
            with page.expect_download() as download:
                page.locator("#btn-export").click()
            payload = json.loads(Path(download.value.path()).read_text())
            assert payload["scenario"]["seed"] == 7
            maximum = int(page.locator("#timeline").get_attribute("max"))
            page.locator("#timeline").evaluate(
                "(e,t) => {e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));}", maximum
            )
            assert page.locator('[data-stat="0-baseline"]').inner_text() == "6 of 30"
            assert page.locator('[data-stat="0-active"]').inner_text() == "8 of 30"
            assert page.locator("#btn-next").is_disabled()
            page.locator("#timeline").evaluate(
                "e => {e.value=0;e.dispatchEvent(new Event('input',{bubbles:true}));}"
            )
            assert page.locator('[data-stat="0-active"]').inner_text() == "0 of 0"
            # Running from Results returns to the scan and starts a fresh replay.
            page.locator('[data-tab="tab-fom"]').click()
            page.locator("#btn-run-sim").click()
            page.wait_for_function(
                "document.querySelector('#run-status').textContent === 'Playing'"
            )
            assert page.locator("#tab-cockpit").is_visible()
            page.locator("#btn-play").click()
            # Alternate six/eight-band buffers and restore scenario receiver defaults.
            for preset, bands in [
                ("mode-switch", 6),
                ("periodic-scan", 8),
                ("crowded-battlefield", 8),
            ]:
                page.locator("#select-preset").select_option(preset)
                page.wait_for_function(
                    "document.querySelector('#run-status').textContent === 'Playing' && "
                    "!document.querySelector('#btn-run-sim').disabled"
                )
                assert page.locator("#waterfall-axis-y .band-label-cell").count() == bands
                page.locator("#btn-play").click()
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
            page.locator("#btn-retry").click()
            page.wait_for_function(
                "document.querySelector('#run-status').textContent === 'Playing'"
            )
            assert not page.locator("#error-banner").is_visible()
            page.locator("#btn-play").click()
            # A malformed successful response must also preserve the previous run.
            previous = page.locator("#run-meta").inner_text()
            page.route(
                "**/api/run",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"scenario":{"duration":2,"num_bands":8},"truth_grid":[]}',
                ),
            )
            page.locator("#btn-run-sim").click()
            page.wait_for_function("!document.querySelector('#error-banner').hidden")
            assert page.locator("#run-meta").inner_text() == previous
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
            failed.unroute("**/api/scenarios")
            failed.locator("#btn-retry").click()
            failed.wait_for_function(
                "document.querySelector('#run-status').textContent === 'Paused'"
            )
            assert not failed.locator("#btn-play").is_disabled()
            if any(s["id"] == "timing-trained" and s["available"]
                   for s in page.request.get(base + "/api/schedulers").json()):
                neural = browser.new_page(viewport={"width": 1920, "height": 1080})
                neural.on("pageerror", lambda error: errors.append(str(error)))
                neural.goto(base + "/?preset=periodic-timing-video&view=dual")
                neural.wait_for_function(
                    "document.querySelector('#run-status').textContent === 'Paused'"
                )
                assert neural.locator(".policy-details").get_attribute("open") is not None
                assert neural.locator("#belief-bars-container .belief-row").count() == 8
                assert neural.locator(".policy-details").evaluate(
                    "e => e.getBoundingClientRect().bottom <= innerHeight"
                ), "The trained demo must show forecasts and receiver counters on one desktop screen"
                assert "before receiver feedback" in neural.locator("#policy-state-caption").inner_text()
                with neural.expect_download() as download:
                    neural.locator("#btn-export").click()
                learned = json.loads(Path(download.value.path()).read_text())
                assert learned["active"]["data"]["model"]["device"] == "cuda"
                end = learned["scenario"]["duration"] - 1
                neural.locator("#timeline").evaluate(
                    "(e,t) => {e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));}", end
                )
                for key in ("baseline", "active"):
                    metrics = learned[key]["data"]["metrics"]
                    assert neural.locator(f'[data-stat="0-{key}"]').inner_text() == (
                        f'{metrics["detected_transmissions"]} of {metrics["total_transmissions"]}'
                    )
                # Repeat a six-band request after the eight-band graph was cached.
                neural.locator("#select-scenario").select_option("change")
                for _ in range(2):
                    neural.locator("#btn-run-sim").click()
                    neural.wait_for_function(
                        "document.querySelector('#run-status').textContent === 'Playing' && "
                        "!document.querySelector('#btn-run-sim').disabled"
                    )
                    neural.locator("#btn-play").click()
                    assert neural.locator("#belief-bars-container .belief-row").count() == 6
                if screenshots:
                    neural.goto(base + "/?preset=periodic-timing-video&view=dual")
                    neural.wait_for_function(
                        "document.querySelector('#run-status').textContent === 'Paused'"
                    )
                    neural.locator("#timeline").evaluate(
                        "e => {e.value=300;e.dispatchEvent(new Event('input',{bubbles:true}));}"
                    )
                    neural.screenshot(path=str(screenshots / "trained-comparison.png"), full_page=True)
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
    parser.add_argument("--timing-model", type=Path)
    parser.add_argument("--planner-library", type=Path)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.timing_model:
        os.environ["SPECTRA_TIMING_CHECKPOINT"] = str(args.timing_model.resolve())
    if args.planner_library:
        os.environ["SPECTRA_PLANNER_LIBRARY"] = str(args.planner_library.resolve())
    if args.cuda_graph:
        os.environ["SPECTRA_CUDA_GRAPH"] = "1"
    check(args.browser, args.screenshots)
