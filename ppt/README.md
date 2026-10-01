# SIH idea presentation

The six-slide deck follows the supplied SIH template: title, idea, technical
approach, feasibility, impact and references. It preserves Team Axiom X,
ID 157808 and the existing cover identity. Requirements come from
[project scope](../docs/project_scope.md).

- [Editable PowerPoint](Spectra-Scheduler-SIH2026.pptx): native diagrams, text,
  clickable links and technical speaker notes.
- [Matching PDF](Spectra-Scheduler-SIH2026.pdf): six-page review/submission copy.
- `build_presentation.py`: builds both formats from the published result summaries.
- `SIH2026-IDEA-Presentation-Format.pptx` and `assets/`: reference layout, official
  artwork and preserved cover fields.

The architecture shows 288 ms of causal receiver history, an 80-ms temporal
forecast and retune-aware planning over band plus 1/10/50-ms listening dwell.
The forecaster is the main learned component; trajectory RL remains experimental.
Synthetic simulation and external PDW replay share the frozen planner. Browser
playback currently shows synthetic comparisons.

The feasibility slide reports **4.16× mean capture versus fixed sweep** on 32
unseen external **synthetic TSRD** validation recordings: 43.49% versus 10.45%,
with 93.35% emitter discovery. RateProbe remains stronger in capture at 51.66%,
with lower discovery at 84.39%. This is software replay evidence, not real RF or
hardware validation. The PDW fine-tune did not improve unseen performance, so the
selected checkpoint remains `timing-refine-v1/seed-0/best.pt`, epoch 24.

Speaker notes retain the 300-world synthetic benchmark, paired confidence
intervals, discovery trade-offs and checkpoint provenance. The seed-42 GUI video
captures 62/86 versus 13/86 (4.77×); that ratio describes one selected example.
The latency slide reports warmed serial CUDA p99 of 0.464–0.532 ms, excluding
cold startup, receiver I/O and GUI serialization. Full evidence is in
[synthetic results](../reports/timing-selected-summary.json),
[external replay results](../reports/timing-pdw-summary.json) and
[latency measurements](../reports/timing-latency-summary.json).

## Rebuild

```bash
uv pip install --python .venv/bin/python -r ppt/requirements.txt
.venv/bin/python ppt/build_presentation.py
```

The builder requires Noto Sans Regular/Bold at the Linux paths declared in the
script. It checks text widths, slide bounds and source result consistency before
writing the PPTX and PDF. Diagrams use editable PowerPoint shapes. The PDF uses
the same layout with embedded fonts through ReportLab; it is not a PowerPoint
export. All six PDF pages were visually reviewed. PowerPoint rendering can vary
with fonts; install Noto Sans when opening the editable deck. Manual deck edits
do not automatically update the builder or PDF.

Local backups, superseded presentations and development guidance remain outside
Git. Rebuilding replaces the current generated PPTX/PDF.
