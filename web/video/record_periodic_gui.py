"""Record the actual browser GUI, controlling only clicks, pacing and captions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/demo-video"
STYLE = """
body { zoom: 1.15; padding-bottom: 180px; }
.workspace { padding-top: 16px; gap: 20px; }
.waterfall-stage-wrapper { height: 204px; }
.summary-card { padding: 12px; }
.stat-policy-row span { font-size: 11px; }
#recording-caption { position: fixed; left: 0; bottom: 0; width: 100%;
  background: #f5f6f8; border-top: 1px solid #ccd5df; color: #202733;
  padding: 14px 28px 9px; z-index: 10000; text-align: center;
  font: 24px/1.5 'Segoe UI', Arial, sans-serif; box-sizing: border-box; pointer-events: none; }
#recording-caption .scope { margin-top: 5px; font-size: 14px; color: #657080; }
#recording-pointer { position: fixed; pointer-events: none; z-index: 10001;
  width: 12px; height: 12px; border-radius: 50%; border: 2px solid white;
  background: #236b61; box-shadow: 0 1px 4px #0005; transform: translate(-50%, -50%); }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083/?preset=periodic-timing-video&view=dual")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    events = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/brave-bin/brave", headless=True)
        options = {"viewport": {"width": 1920, "height": 1080}, "color_scheme": "light"}
        if not args.preview:
            options.update(record_video_dir=str(OUT / "capture"), record_video_size={"width": 1920, "height": 1080})
        context = browser.new_context(**options)
        context.add_init_script("localStorage.setItem('spectra-theme', 'light')")
        origin = time.monotonic()
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("document.querySelector('#run-status').textContent === 'Paused'", timeout=60000)
        page.add_style_tag(content=STYLE)
        page.evaluate("""() => {
          const caption = document.createElement('div'); caption.id = 'recording-caption'; document.body.append(caption);
          const pointer = document.createElement('div'); pointer.id = 'recording-pointer'; document.body.append(pointer);
          document.addEventListener('mousemove', e => {
            pointer.style.left = `${e.clientX / 1.15}px`; pointer.style.top = `${e.clientY / 1.15}px`;
          });
        }""")
        page.mouse.move(25, 25)
        page.wait_for_timeout(500)
        start = time.monotonic()

        def caption(line1, line2):
            events.append({"time": time.monotonic() - start, "lines": [line1, line2]})
            page.locator("#recording-caption").evaluate("""(e, lines) => {
              e.replaceChildren(...lines.map(line => {const d = document.createElement('div'); d.textContent = line; return d;}));
              const scope = document.createElement('div'); scope.className = 'scope';
              scope.textContent = 'Periodic scenario, seed 42 · selected simulation example · results vary by scenario'; e.append(scope);
            }""", [line1, line2])

        def click(selector):
            box = page.locator(selector).bounding_box()
            if box["y"] + box["height"] / 2 > 900:
                page.locator(selector).evaluate("e => e.scrollIntoView({block:'center',behavior:'instant'})")
                box = page.locator(selector).bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=12)
            page.wait_for_timeout(220)
            page.locator(selector).click()
            page.wait_for_timeout(250)
            page.mouse.move(25, 25, steps=12)

        def seek(tick):
            page.locator("#timeline").evaluate(
                "(e,t) => {e.value=t; e.dispatchEvent(new Event('input', {bubbles:true}));}", tick)

        caption("Spectra Scheduler: a working passive receiver scheduling demo.",
                "Trained timing scheduler: seed 0, epoch 24 — versus a 50-tick fixed sweep.")
        if args.preview:
            seek(511)
            page.screenshot(path=str(OUT / "gui-preview.png"))
            click('[data-tab="tab-fom"]')
            page.screenshot(path=str(OUT / "gui-results-preview.png"))
            print("GUI previews saved")
            browser.close()
            return
        page.wait_for_timeout(5000)
        caption("Click Run comparison: both strategies see the same simulated signal activity.",
                "One receiver, eight bands, identical sensitivity and retuning model.")
        click("#btn-run-sim")
        page.wait_for_function("document.querySelector('#run-status').textContent === 'Playing'", timeout=60000)
        assert page.locator("#btn-play").inner_text() == "Pause"
        page.wait_for_timeout(7000)
        caption("Round robin follows a fixed band order with a 50-tick listening dwell.",
                "The gold trace shows its observations; teal shows the trained timing scheduler.")
        page.wait_for_timeout(8000)
        caption("The temporal model forecasts signal opportunities from past hits and misses.",
                "Forecast-guided scheduling selects the listening band and dwell duration.")
        page.wait_for_timeout(8000)
        caption("Listening at the predicted time increases the chance of capturing short signals.",
                "Counters show true detections so far; false alarms are excluded.")
        page.wait_for_function("document.querySelector('#run-status').textContent === 'Complete'", timeout=60000)
        assert page.locator('[data-stat="0-baseline"]').inner_text() == "13 of 86"
        assert page.locator('[data-stat="0-active"]').inner_text() == "62 of 86"
        caption("Run complete: the trained scheduler captures 62 signals; the fixed sweep captures 13.",
                "62 ÷ 13 = 4.77× capture in Periodic scenario, seed 42.")
        page.screenshot(path=str(OUT / "gui-complete.png"))
        page.wait_for_timeout(6000)
        click('[data-tab="tab-fom"]')
        caption("The Results tab reports the measurements from the actual complete run.",
                "Capture rate: 72.1% for the trained scheduler versus 15.1% for the fixed sweep.")
        page.wait_for_timeout(7000)
        discovery = page.locator('.results-table tbody tr').filter(has=page.locator('td').filter(has_text='Emitter discovery'))
        assert discovery.count() == 1
        assert discovery.locator('td').nth(1).inner_text().startswith('100.0%')
        assert discovery.locator('td').nth(2).inner_text().startswith('100.0%')
        discovery.evaluate("e => e.scrollIntoView({block:'center',behavior:'instant'})")
        caption("Both strategies discover 100% of the emitters in this periodic run.",
                "The improvement is in captured transmissions: 62 of 86 versus 13 of 86.")
        page.screenshot(path=str(OUT / 'gui-discovery.png'))
        page.wait_for_timeout(5000)
        page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
        click('[data-tab="tab-cockpit"]')
        seek(300)
        click(".policy-details summary")
        page.locator(".policy-details").scroll_into_view_if_needed()
        page.evaluate("window.scrollBy(0,180)")
        caption("The policy state exposes forecasts saved before the scan action.",
                "Only receiver history enters the model; future simulation truth is used for evaluation.")
        assert page.locator("#belief-bars-container").bounding_box()["y"] + page.locator("#belief-bars-container").bounding_box()["height"] < 940
        page.screenshot(path=str(OUT / "gui-forecast.png"))
        page.wait_for_timeout(6500)
        click(".policy-details summary")
        seek(511)
        page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
        caption("Trained timing scheduler schedules across both frequency and time.",
                "Periodic scenario, seed 42: 72.1% vs 15.1% capture — 4.77×; both discover all emitters.")
        page.screenshot(path=str(OUT / "periodic-gui-thumbnail.png"))
        page.wait_for_timeout(6000)
        assert not errors, errors
        length = time.monotonic() - start
        metadata = {"trim_start_seconds": start - origin, "duration_seconds": length,
                    "captions": events, "viewport": options["viewport"],
                    "actual_counts": {"baseline": 13, "adaptive": 62, "truth": 86}}
        video = page.video
        page.close()
        context.close()
        video.save_as(str(OUT / "actual-periodic-gui-recording.webm"))
        browser.close()
        (OUT / "periodic-recording.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"recording": str(OUT / "actual-periodic-gui-recording.webm"), "duration": length}, indent=2))


if __name__ == "__main__":
    main()
