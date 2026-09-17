# SIH idea presentation

- `SIH2026-IDEA-Presentation-Format.pptx`: supplied template, unchanged.
- `Spectra-Scheduler-SIH2026.pptx`: editable six-slide presentation with speaker notes.
- `Spectra-Scheduler-SIH2026.pdf`: matching six-page presentation for review and submission.

The presentation follows the template's title, idea, technical approach, feasibility,
impact and references sections. The instructions-only slide is omitted, as the
template requires a maximum of six slides. Diagrams and slide text remain editable
in PowerPoint. The PDF uses the same content and layout with embedded fonts.

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
PYTHONPATH=src python3 -m unittest discover -s tests -q
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
