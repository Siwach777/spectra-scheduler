# Actual GUI screen recording

The current periodic video uses the normal GUI server and the verified timing forecaster checkpoint at training seed 0, selected epoch 24. It records the `periodic-timing-video` preset: periodic scanning, seed 42, default receiver, no variation, versus a 50-tick fixed-order sweep. Actual CUDA evaluation yields 62/86 (72.1%) versus 13/86 (15.1%) captures, a 4.77× ratio, and 100% emitter discovery for both. This is a selected example; performance varies across scenarios.

```bash
.venv-rl/bin/python web/video/verify_periodic.py
.venv-rl/bin/python web/server.py --port 8083 \
  --timing-model artifacts/timing-refine-v1/seed-0/best.pt
uv run --offline --no-project --with playwright --with numpy python web/video/record_periodic_gui.py
.venv/bin/python web/video/encode_periodic_gui.py
```

The server stays running in its terminal while the recorder runs separately. Outputs are `artifacts/demo-video/spectra-periodic-demo.mp4`, matching SRT captions, `periodic-gui-thumbnail.png`, `periodic-youtube-description.txt`, and the actual API export `periodic-run.json`. The on-screen label is **Periodic scenario, seed 42**. The normal GUI is also available at `http://127.0.0.1:8083/?preset=periodic-timing-video&view=dual`.

The following instructions describe the earlier, separate spatial RL example.

This standalone recording server connects the existing browser GUI to a genuine favorable example from the frozen grouped-policy report. Clicking Run recomputes the episode on CUDA. It imports the project without changing the CLI, schedulers, training code or evaluation metrics.

The replay uses the selected `grouped-mpc-grid-report-v1/seed-1` checkpoint: a causal timing forecaster with a trajectory-trained RLOO action network. The comparison is a fixed-order round-robin sweep with a 50-tick listening dwell. Reporting seed 16007 in the spatial-scan scenario yields 37 versus 10 true captures of 133 transmissions. That 3.7× result describes this selected example, not aggregate performance.

Reproduce the measured data and start the recording server on CUDA:

```bash
.venv-rl/bin/python web/video/collect_demo.py
.venv-rl/bin/python web/video/serve_demo.py --port 8082
```

Open `http://127.0.0.1:8082/?preset=temporal-rl-video&view=dual`. This dedicated recording server exposes only the verified preset. It checks every native scan metric against the reproduced data; its full GUI export is `artifacts/demo-video/gui-run.json`. Neural inference fails explicitly without CUDA.

Record the actual interface with Playwright and convert it with FFmpeg:

```bash
uv run --offline --no-project --with playwright --with numpy python web/video/record_gui.py
.venv/bin/python web/video/encode_gui.py
```

The recorder uses the installed Brave executable. Playwright's video encoder must be installed (`playwright install ffmpeg`). It controls real GUI buttons, timeline position, and explanatory caption overlays. It does not replace counters, forecasts, scheduler outputs or API responses. The presentation zoom and scan height keep the controls and counters visible at 1920×1080.

Outputs in `artifacts/demo-video/` include `spectra-working-demo.mp4` (1080p, 30 fps, H.264/AAC), burned-in explanatory text, separate SRT subtitles, `gui-thumbnail.png`, and a suggested YouTube description. The audio is an original quiet instrumental pad generated locally. Encoding uses four threads. `render_video.py` is the earlier standalone explanatory animation renderer and is separate from the actual GUI recording.
