# Demonstration interface

This standalone browser interface compares passive receiver schedulers on the existing synthetic simulator. It imports the project through a web-only adapter; the CLI, training pipelines, and core modules do not depend on it.

From the repository root:

```bash
.venv-rl/bin/python web/server.py \
  --timing-model artifacts/timing-refine-v1/seed-0/best.pt
```

Open `http://127.0.0.1:8080`. Install the CUDA environment using the [runbook](../docs/runbook.md#1-set-up-the-environment). The trained timing policy requires CUDA and a local checkpoint; weights are excluded from Git. Statistical policies can also run with `.venv/bin/python web/server.py`. The server uses the standard library, serves local assets, and requires no frontend build. Use `--port 8081` if the default port is occupied.

The `timing-trained` policy uses the selected timing forecaster and its saved planning settings. `--timing-model` overrides `SPECTRA_TIMING_CHECKPOINT`; otherwise the adapter checks `artifacts/timing-console-v1/best.pt`, then `artifacts/timing-refine-v1/seed-0/best.pt`. Missing weights or CUDA make the policy unavailable, without a substitute. The periodic video preset compares it with `dwell-sweep-50`; see [recording instructions](video/README.md) for the selected example and its limits.

## HTTP API

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/scenarios` | Scenario IDs and receiver defaults |
| GET | `/api/schedulers` | Policy IDs and availability, including model provenance |
| GET | `/api/presets` | Named comparison configurations |
| GET | `/api/reports` | Available local JSON report names and sizes |
| GET | `/api/report/<name>` | One local report; paths outside the report directory are rejected |
| POST | `/api/run` | Complete comparison, truth grid, traces, metrics and deltas |

Send a JSON object to `/api/run`, for example:

```json
{"scenario":"periodic-scan","baseline":"dwell-sweep-50","active":"timing-trained","seed":42}
```

Defaults are `frequency-agile`, `round-robin`, `track-aware`, and seed `0`. Optional receiver fields are `sensitivity_dbm` (-120 to -40), `noise_std_db` (0 to 30), `false_alarm_prob` (0 to 1), and `retune_steps` (0 to 10). Seeds are integers from 0 to 99999 and use the raw scenario seed, unlike the CLI's scenario-derived reporting seeds. `perturbation` accepts `null`, `frequency-hop`, or `popup-threat`; it changes shared truth before either policy runs. Request bodies are limited to 16 KiB.

Errors return JSON with `error` and `status`: invalid inputs use HTTP 400, oversized requests 413, and a busy timing comparison 503. Only one timing comparison runs at a time. The export payload is separate from the [CLI report format](../docs/report-format.md).

## Browser controls

Choose a demonstration or adjust the scenario, policies, and seed, then run a comparison. Presets start a new comparison automatically. Receiver overrides apply to the next run; presets restore the receiver defaults for their scenario. Environment variations alter simulated activity for both policies before the run starts.

Playback starts paused. Use Play, the timeline, or the arrow controls to inspect the receiver observations. Space toggles playback, and the left/right arrow keys step through observations when a form control is not focused. Changing views or hiding the browser tab pauses playback.

The scan includes simulation truth for the demonstration. Disable **Show simulation truth** to inspect measured observations alone. Truth is never passed to a policy. The Results view shows complete-run measurements, with explicit missing values for prediction evaluation that this adapter does not implement. Signal tracks are the comparison policy's final snapshot, not a live track history. Export run downloads the full GUI payload as JSON; this is separate from the CLI report schema.

The DQN and supervised hit policies are experimental frozen baselines. They require compatible artifacts in `artifacts/` and are marked unavailable when loading fails. The GUI does not train models or substitute another policy if an artifact is missing.

Focused backend verification:

```bash
PYTHONPATH=src .venv-rl/bin/python -m pytest web/test_server.py -q \
  -k 'not neural_mlp'
```

Optional browser integration checks use Playwright, independently of project dependencies:

```bash
uv run --no-project --with playwright --with numpy python web/check_browser.py \
  --browser /opt/brave-bin/brave --screenshots /tmp/spectra-gui-review
node web/check_frontend.mjs
```

Supply the path to any installed Chromium-compatible browser, or omit `--browser` to use Playwright's installed Chromium. The checks cover playback, preset changes, seek, both receiver views, truth visibility, results, tracks, exports, loading failures, and narrow layouts.
