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

## What the slides claim

Implemented: the simulator, receiver model, heuristic/statistical strategies,
anonymous tracking, parallel seeded evaluation and JSON/CSV reports. The suite
passes 123 tests; the runner compares 13 strategies across five scenario presets.

Proposed: dataset loading/evaluation, a contextual ML scheduler, complete reward and
prediction metrics, and a periodic-scan reference. Rust, an API and a GUI remain
later options. The prototype is not presented as field-validated hardware or as a
trained neural model. Benefits are objectives, not measured operational gains.

Verification commands:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
PYTHONPATH=src python3 -m spectra_scheduler --scenario mixed --runs 100 --association
```

The mixed comparison gives about 8.2% interception for adaptive dwell and 8.1% for
track-aware scheduling. This does not establish an advantage for the more complex
policy; the slide notes explain why future models must beat the strongest baseline.

## Sources

- [Turing Synthetic Radar Dataset paper](https://arxiv.org/abs/2602.03856)
- [Official dataset card](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)
- [Official challenge code and evaluation](https://github.com/alan-turing-institute/turing-deinterleaving-challenge)
- [Transformer-based deep metric learning paper](https://arxiv.org/abs/2503.13476)
- [Official SIH problem portal](https://sih.gov.in/sih2026PS), unavailable during preparation.
- [Accessible statement copy, community-maintained](https://sih2026.vuce.in/ps/SIH26055)

Research and dataset links are clickable in both presentation formats. Published
research scores are not represented as project results.
