# Dense MLP and planning findings

This is a record of development experiments, not a claim of final performance.
The stare replay results below use disjoint training, checkpoint-selection, and
reporting recordings. The test split has not been used. Generated checkpoints,
manifests, and full per-recording reports live under ignored `artifacts/` paths.

## What has been tried

| Technique | Observation | Interpretation |
| --- | --- | --- |
| More tune centers | On 64 validation recordings, rate-probe pooled capture rose from 0.595 with eight centers to 0.671 with twelve and 0.689 with 24 (`artifacts/action-grid/validation-64.json`). | Receiver passband placement is a substantial physical bottleneck. The 24-center dense study uses 72 band/dwell actions, rather than the older 24-action grid. |
| Original 24-center dense MLP | Three seeds trained for 100 epochs on 128 training recordings. Separate 64-recording reporting mean capture was 0.675/0.656/0.666 versus 0.665 for rate-probe; pooled capture was 0.653/0.647/0.650 versus 0.658 (`artifacts/dense-value-24/comparison.json`). | Mean recording-level capture sometimes improves, but none of the three models beats the control on pulse-weighted capture. No reliable learned gain is established. |
| Spectral-neighborhood MLP | A shared-weight encoder now sees five adjacent centers in every causal history frame, current tune distance, and global context. On the 128-recording selection run, its best model captured 0.620 versus 0.628 for rate-probe (`artifacts/dense-spectral-probe`). | Local frequency structure is plausible, but this architecture change alone did not solve scheduling. |
| Larger counterfactual training set | An additional 384 disjoint training recordings expanded the pool to 512 recordings and 664,107 labeled decisions. A bounded, shuffled CUDA block stream kept the large cache off GPU. The best 40-epoch spectral model selected at epoch 32. On separate reporting recordings it reached mean capture 0.668 versus 0.665, pooled capture **0.647 versus 0.658**, and mean discovery 0.913 versus 0.884 (`artifacts/dense-spectral-expanded/comparison.json`). | More data improves discovery but still does not win the primary pulse-weighted capture comparison. One model seed is insufficient to establish robustness. |
| Observed-rate/model blend | Weight 0.25 selected on 16 selection recordings was frozen across three original MLP seeds. Separate reporting mean capture was 0.684/0.668/0.677 versus 0.665 for rate-probe; pooled capture was 0.662/0.657/0.661 versus 0.658 (`artifacts/dense-blend-frozen/comparison.json`). | Mean paired intervals exclude zero for two seeds, but every pooled interval includes zero. Some model signal complements measured rates; a robust pooled capture win remains unestablished. See [frozen blend findings](dense-blend-findings.md). The default remains pure model ranking. |
| Predicted variable dwell | Choosing the highest model-predicted pulse rate over all 72 actions reduced selection capture: original MLP 0.624 to 0.613; spectral MLP 0.620 to 0.608. These were in-memory ablations, not saved policies. | A one-step rate optimum is not automatically a good sequential schedule. Short dwells are vulnerable to noisy count predictions and may waste future opportunities on repeated retunes. |
| Opportunity-weighted ranking | A new optional loss weights high-count decisions sublinearly and can select checkpoints using both mean and pooled capture. Warm-start fine-tuning from the 512-recording spectral checkpoint was interrupted after 21 of 24 epochs at the user's request. Its best checkpoint was still epoch zero; the latest scored epoch had selection pooled capture 0.605 versus 0.625 for rate-probe (`artifacts/dense-spectral-opportunity/mlp-0/progress.json`). | No improvement was demonstrated. The implementation remains available for a controlled later experiment; do not present it as a successful model. |

The dense labels enumerate exact next-action outcomes from uncensored training
recordings, while behavior trajectories provide only causal observations. The
training loss currently fits all-action pulse counts and ranks long-dwell bands.
The runtime policy uses short coverage probes and model-ranked long dwells.
The count target is useful supervision, but a one-step hindsight label does not
measure the effect of an action on future observation quality, coverage, or
receiver state.

## What the large-action MPC criticism gets right

Search with hundreds of actions and a small simulation budget could become
shallow. Frequency/dwell structure, candidate filtering, and bounded branching
are reasonable design options if a **working integrated** Dense-MPC planner is
built. These are conditional engineering ideas, not measured improvements here.
The current neural MPC uses a separate synthetic interface; its 24 actions are
eight bands by three dwells. It is not the 24-center/72-action dense replay model.
The physical replay default has a fixed retune delay and no distance-dependent
slew, so large-distance slew pruning is not currently a demonstrated bottleneck.
A hard revisit rule bounds the policy's nominal revisit interval; it cannot
guarantee emitter discovery when pulses are missed or not observable. Shared
band weights are already present in the dense model; the main failure is
decision quality, not growth of the output layer with band count.

Fresh physical-MPC experiments now include a Gumbel top-k/sequential-halving
search option. It improved discovery over PUCT but reduced capture on paired
development worlds; the trained Gumbel model did not reliably beat its own
untrained search. The separate synthetic MPC therefore still has a more basic
learning problem than branching alone. See [MPC assessment](mpc-repair.md).
Additional search depth or a different root allocation can amplify model error.

## Highest-value next hypotheses to test

1. **Train on states induced by the learned scheduler.** The present dense cache
   mixes uniform and rate-probe trajectories. Its predictions are then used on
   the model's own trajectory, where observation histories can differ. Collect
   additional *train-only* learned-policy rollouts, label all candidate actions
   with the existing counterfactual index, and retrain with the original data
   retained. Compare paired capture and discovery with the same frozen control.
   This targets the state-distribution mismatch directly.
2. **Represent belief and timing, then test whether it changes decisions.** The
   current 16-frame MLP receives aggregate causal features but no explicit
   emitter-level phase belief. A causal timing/track representation could help
   predict scanning or periodic emitters through silent intervals. Measure
   calibration and decision regret by emitter regime before committing to a
   heavier model; never feed file-local truth labels to the runtime policy.
3. **Optimize sequential outcomes instead of one-step count alone.** A learned
   value for future capture and coverage, trained from causal rollouts, would
   account for information gained by probing. If integrated MPC is revisited,
   first make elapsed time, retune state, and coverage state explicit and verify
   that search beats direct inference with the same model. Only then test
   factorized dwell/candidate selection under measured latency and capture.

For any candidate, freeze the training and checkpoint-selection protocol before
examining separate reporting recordings. Report per-recording and pooled pulse
capture, discovery, paired intervals, multiple training seeds, and inference
latency. A larger run is justified by a clear selection-set gain and a specific
failure mode it addresses, not by training loss decreasing alone.
