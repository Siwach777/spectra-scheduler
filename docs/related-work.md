# Related work notes

This note records the parts of radar pulse deinterleaving research that directly
affect the simulator, evaluation, and next implementation stages. It is intentionally
short; it is not a general literature survey.

## Main findings

Pulse deinterleaving is normally framed as clustering pulse descriptor words when the
number of emitters is unknown. Common measured fields include time of arrival, carrier
frequency, pulse width, pulse amplitude, and sometimes angle of arrival. Pulse
repetition interval is derived from arrival-time differences and is often combined
with the other fields rather than used alone.

The Turing Synthetic Radar Dataset is especially close to this project. It includes
both wideband stare and frequency-scanning receivers, drops pulses outside the tuned
receiver band, supplies emitter truth only for evaluation, and deliberately overlaps
emitters in parameter space. Its benchmark uses clustering metrics such as V-measure.

Metric-learning research treats each pulse as a feature vector in the context of the
full pulse train, then clusters learned embeddings. This is useful as a later baseline,
but it is not the next implementation step: the current simulator first needs richer
receiver-visible measurements and dataset-compatible evaluation.

## Mapping to this project

| Research concept | Current equivalent | Gap |
|---|---|---|
| Time of arrival | Simulation step on each observation | Temporal patterns are not yet part of association |
| Carrier frequency | Coarse receiver band | No within-band frequency measurement |
| Pulse width | Noisy pulse-width measurement | Implemented |
| Pulse amplitude | Noisy received power | Implemented |
| Unknown emitter count | Tracks are created and expired online | Implemented without a fixed class count |
| Scan receiver | Retuning narrow-band receiver | Implemented |
| Ground-truth clustering score | Pairwise scores, purity, and fragmentation | Add V-measure for benchmark compatibility |

## Decisions

- Use homogeneity, completeness, and V-measure in association evaluation and reports.
- Keep pairwise precision/recall/F1 because it makes false joins and fragmentation
  independently visible.
- Treat time-of-arrival-derived repetition structure or a finer frequency measurement
  as the next major signal feature. Do not expose simulator emitter identity.
- Build a small adapter for a manageable scan-mode dataset subset before considering a
  neural model.
- Keep the current heuristic tracker as an interpretable baseline. A transformer or
  metric-learning model belongs after the data and evaluation contracts are stable.

## Sources

- [The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving](https://arxiv.org/abs/2602.03856)
- [Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning](https://arxiv.org/abs/2503.13476)
- [A New Radar Signal Multiparameter-Based Deinterleaving Method](https://arxiv.org/abs/2208.09786)
- [Pulse deinterleaving based on fusing PDWs and PRI extraction process](https://link.springer.com/article/10.1186/s13638-021-01985-5)
