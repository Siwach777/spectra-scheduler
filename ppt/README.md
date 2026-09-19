# SIH idea presentation

[PROJECT-BRIEF.md](PROJECT-BRIEF.md) is the detailed project reference for preparing
slides and speaker notes. The current deck includes dataset integration and the
first fitted-model evaluation; the brief predates that learning milestone.

- `SIH2026-IDEA-Presentation-Format.pptx`: supplied template, unchanged.
- `Spectra-Scheduler-SIH2026.pptx`: editable six-slide presentation with speaker notes.
- `Spectra-Scheduler-SIH2026.pdf`: matching six-page presentation for review and submission.

The presentation follows the template's title, idea, technical approach, feasibility,
impact and references sections. The instructions-only slide is omitted, as the
template requires a maximum of six slides. Diagrams and slide text remain editable
in PowerPoint. The PDF uses the same content and layout with embedded fonts.

## Diagram-led version

The current deck uses editable vector elements only: a spectrum/bandwidth
schematic, closed-loop control flowchart, two-path architecture, association-result
bars and a receiver-resource trade-off diagram. No generated artwork, stock photos
or raster illustrations are included. White space, restrained blue accents and
short labels replace the previous dense card layout.

A refinement pass adds an instantaneous-bandwidth annotation, explicit control-loop
signals and dwell/coverage constraints, an evaluation-only truth path, a calibrated
0–1 association-score axis, labelled trade-offs and subtle slide-progress markers.
These clarify the same content without adding slides or introducing image assets.

Dataset and learned-policy results are explicitly scoped. Planned calibration and
API/dashboard work remain labelled as planned. Speaker notes contain experimental
settings, limitations and source details; the template's six sections are retained.

Rebuild both outputs with:

```bash
uv pip install --python .venv/bin/python -r ppt/requirements.txt
.venv/bin/python ppt/build_presentation.py
```

The builder requires Noto Sans regular and bold at the Linux font paths declared in
the script. Presentation dependencies are separate from the application lockfile.
Running it replaces the generated PPTX and PDF. The PDF is rendered from the same
layout using ReportLab, not exported through PowerPoint. All six PDF pages were
visually checked; final PowerPoint rendering should also be checked in the target
presentation application. The supplied template is untouched.

## Before submission

Fill the team ID and registered team name. Confirm the exact theme on the official
SIH portal: public statement mirrors disagree, so the field is intentionally marked
for confirmation. Re-export the PDF after editing the PowerPoint; the two files do
not update each other automatically. Check formatting in the presentation app used
for the final export.

## Proposal focus

The presentation positions Spectra Scheduler as a confidence-aware, closed-loop
spectrum-management proposal. It emphasises contextual learning, pulse association,
coverage constraints, concept-drift response and receiver-resource allocation.

The engineering foundation comprises receiver simulation, heuristic/statistical
policies, anonymous tracking, parallel seeded evaluation and structured reports.

The delivery plan progresses through dataset ingestion, association benchmarking,
contextual policy learning, profiling-led optimisation and an API/dashboard.
A periodic-scan reference and complete reward/prediction metrics are included in
the validation scope. Native acceleration remains conditional on profiling.

Targets include a better interception–latency trade-off than the strongest baseline,
preserved discovery coverage, bounded-memory processing and measured decision
latency. These are acceptance objectives, not fabricated results or field claims.

## Validation methodology

Existing methods include unit and integration tests, schema checks, deterministic
regression checks, sequential/parallel equivalence tests and seeded Monte Carlo
comparisons. Planned validation adds distribution-shift stress tests, sensitivity
analysis, policy ablations and dataset-scale memory/latency profiling.

The deck describes validation methods rather than test counts. Speaker notes
distinguish existing infrastructure from planned experiments and capabilities.

Verification commands:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src python3 -m spectra_scheduler --scenario mixed --runs 100 --association
```

## Sources

- [Turing Synthetic Radar Dataset paper](https://arxiv.org/abs/2602.03856)
- [Official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
- [Official challenge code and evaluation](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
- [Transformer-based deep metric learning paper](https://arxiv.org/abs/2503.13476)
- [Official SIH problem portal](https://sih.gov.in/sih2026PS), unavailable during preparation.
- [Accessible statement copy, community-maintained](https://sih2026.vuce.in/ps/SIH26055)

Research and dataset links are clickable in both presentation formats. Published
research scores are not represented as project results.
