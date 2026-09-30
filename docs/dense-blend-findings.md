# Frozen dense-model blending

A small learned correction to the latest observed pulse rate improved mean
recording capture for two of three training seeds. Pulse-weighted capture gains
were small and their paired confidence intervals included zero for every seed.
These results support complementary model signal, while leaving robust pooled
capture improvement unestablished.

A subsequent learned gate was trained for eight CUDA epochs with optimizer
batch size 512 over 24,391 exploitation states. Its count/ranking training
objective improved while selection capture declined. Selection retained epoch
zero, so its reporting results are the initial fixed blend and provide no
learned-gate improvement. The run took 119.2 seconds with peak CUDA allocation
423 MB; artifacts are under `artifacts/dense-gate-study`. The gate remains an
unsuccessful experiment, not the recommended dense policy.

## Protocol

The original 24-center MLP checkpoints were trained on 128 recordings. With the
models frozen, weights `0, 0.25, 0.5, 0.75, 1` were compared on the existing 16
selection recordings using seed-zero's checkpoint. Maximum mean recording
capture selected weight `0.25`: 0.63821 versus 0.62837 for rate-probe, with pooled
capture 0.63325 versus 0.62530. The weight and content hashes were saved before
evaluating the separate 64 reporting recordings. The same weight was used for
all three independently trained checkpoints. The test split remains unused.

Both controls use the same short coverage probes and long exploitation dwells.
The original rate-probe ranks the latest measured rates; the zero-model-weight
control also accounts for retuning time in action ranking. The blend mixes
observed and predicted rates in `log1p(rate)` space with the existing retune-aware
ranking. Its inputs remain causal receiver summaries.

## Separate reporting results

| Policy | Mean capture | Pooled capture | Mean discovery | Paired mean capture gain, 95% interval | Paired pooled gain, 95% interval |
| --- | ---: | ---: | ---: | --- | --- |
| Rate-probe | 0.66502 | 0.65843 | 0.88386 | reference | reference |
| Retune-aware measured rate | 0.66529 | 0.65897 | 0.88228 | +0.00027 [-0.00096, +0.00164] | +0.00054 [-0.00034, +0.00156] |
| Blend, training seed 0 | 0.68401 | 0.66194 | 0.90548 | +0.01899 [+0.00582, +0.03579] | +0.00351 [-0.00161, +0.00880] |
| Blend, training seed 1 | 0.66797 | 0.65740 | 0.88474 | +0.00295 [-0.00738, +0.01517] | -0.00103 [-0.00640, +0.00477] |
| Blend, training seed 2 | 0.67727 | 0.66092 | 0.91294 | +0.01225 [+0.00343, +0.02196] | +0.00249 [-0.00418, +0.00979] |

Intervals use 2,000 paired bootstrap resamples of independent recordings.
Mean capture has 63 supported recording groups: one reporting recording contains
zero truth pulses. Pooled capture sums intercepted pulses and truth pulses over
all 64 recordings. Compared with the retune-aware control, mean capture gains
remain positive with intervals excluding zero for seeds 0 and 2; all three pooled
intervals still include zero. Discovery and mean discovery-delay differences
have intervals including zero for each model seed.

The full selection and reporting run took 65.8 seconds with batch size 16 under
concurrent CUDA training. Peak process RSS was 1.10 GB, peak CUDA allocation
12.9 MB, and peak CUDA reservation 27.3 MB. Mean per-recording decision latency,
averaged over recordings, was 1.17–1.42 ms for the blended models, including the
full batch inference wait. Rate-probe used approximately 0.047 ms.

## Reproduction

```sh
.venv-rl/bin/python -m spectra_scheduler.experiments.dense_blend_study \
  --checkpoints artifacts/dense-value-24/mlp-0/best.pt \
    artifacts/dense-value-24/mlp-1/best.pt \
    artifacts/dense-value-24/mlp-2/best.pt \
  --run-dir artifacts/dense-blend-frozen --inference-batch-size 16
```

The command requires CUDA for neural inference, checks split and training
provenance, preserves checkpoint bytes, and saves `selection.json`,
`frozen.json`, and `comparison.json`. `--selection-only` freezes the selected
weight; `--report-only` validates that the frozen selection, plans, checkpoints,
and relevant source hashes agree before reporting. The default dense-policy
weight remains unchanged. Further weight tuning belongs on selection data.
