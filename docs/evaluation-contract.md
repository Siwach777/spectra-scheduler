# Metrics and prediction evaluation contract

`evaluation_contract.py` defines version 1 of the model-independent contract.
`replay_evaluation.evaluate_policy` emits it under `evaluation` in each policy report.
`synthetic_evaluation.evaluate_scheduler` emits the same contract for discrete worlds.
Synthetic policies using `SyntheticAction` emit version 2 for the complete selected
listening dwell, including retuning and horizon clipping.
This implements measurement infrastructure; it supplies neither a trained predictor
nor evidence of learned scheduling performance. Requirements come exclusively from
[project_scope.md](project_scope.md).

## Evaluation target and information boundary

One sample is the **selected action's elapsed window**, including retuning and
clipping at episode end. Forecasts must be produced before executing that action.
They predict true receiver captures before observation-buffer overflow, not raw
occupancy or delivered detections that might include false alarms. This distinction
makes the target reflect receiver sensitivity, missed detections and scheduling.
The truth denominator includes all in-spectrum arrivals during that same elapsed
window, including other bands and retuning time.

The contract evaluates the actions actually taken, not unexecuted counterfactual
actions, arbitrary future policies, or emitter-specific forecasts. Longer-horizon
and per-emitter targets would require separate versioned contracts. Comparisons
must use the same receiver, available actions, episode horizon and evaluation data.

`TruthOutcome` stays in the evaluator. It is not included in observations,
transitions or rewards. Replay policies receive only causal feature arrays and
the public environment specification. Synthetic policies receive their existing
observations; an optional `forecast(time_step, band)` method runs before feedback
for tick policies. Macro policies implement `choose_action(time_step)` and may
implement `forecast(time_step, action)`; both run before any action feedback.
These are API boundaries, not a security sandbox against malicious Python code.

## Seven required figures of merit

Let N be all in-spectrum truth events, E the events inside the listening window
and passband that satisfy pulse-boundary rules, D the subset above the sensitivity
threshold, C the true captures, A the number of actions, and T elapsed seconds.
Counts are pooled over the episode before computing ratios.

| Figure | Contract definition and units |
| --- | --- |
| Probability of Detection | C / D. Conditional receiver detection probability; scheduling losses are excluded from this denominator. |
| Probability of False Alarm | Listening windows with no detectable signal but a false alarm / listening windows with no detectable signal. Retuning is excluded. |
| Sensitivity | Configured received-amplitude threshold, with explicit units and enabled/disabled status; also report sensitivity loss (E − D) / E. |
| Average Intercept Rate | C / T, true captures per simulated second, including retuning time. Interception ratio C / N is reported separately. |
| Average Reward / Cost | Sum of observable reward / A, plus sum of reward / T and total reward. Variable dwell makes the time-normalized value necessary for comparison. |
| Percentage of Correct Predictions | 100 × (TP + TN) / forecasted windows, classifying true capture occurrence with a fixed probability threshold of 0.5. |
| Average Intercept Time Error | Mean absolute error in seconds on windows with a true capture and a finite time prediction; always accompanied by timing coverage and the restricted error below. |

Receiver sensitivity is a model setting here, not an empirically measured hardware
minimum detectable signal. Replay amplitude uses dataset dB; synthetic amplitude
uses simulated dBm. Neither is evidence of calibrated hardware sensitivity. A
sensitivity study must run the same recordings and policies at multiple declared
thresholds and compare detection and capture metrics; these units cannot be pooled.

Replay does not simulate false alarms: its PFA is `null` with status `not_modelled`,
never an invented zero. Use the synthetic backend to evaluate this requirement.
The contract can accept false-alarm outcomes from future receiver implementations.

## Prediction fields and censoring

Every `Forecast` contains:

- `hit_probability`: probability of at least one true capture in this action window.
- `intercept_delay_seconds`: predicted first true capture arrival delay from action
  start, including retuning; `None` predicts no capture within the window.
- `interception_ratio`: predicted C / N over the selected action's elapsed window.

The probability and point-time forecasts are separate outputs; they need not imply
the same binary classification. Arrival time, rather than pulse-completion time,
is used for timing, consistently with replay's existing discovery-delay metric.
Actual captured pulses must still finish inside the listening window.

No capture means the first-intercept time is **right-censored at the window end**.
It does not mean an intercept happened there. The report therefore includes:

- Conditional time MAE, finite timing-pair count and event-window count.
- Timing-event coverage: finite predicted delays / windows with actual captures.
- Restricted time MAE over every forecasted window: compare min(predicted delay, H)
  with min(actual delay, H), where H is that window's elapsed duration. A missing
  predicted or actual intercept is represented as H only for this restricted target.
- Forecast coverage, censored-window count, confusion matrix and Brier score.
- Interception-ratio MAE across forecasted windows with N > 0, and its sample count.

Restricted time MAE penalizes missed events and premature forecasts on censored
windows without inventing unobserved future events. It is not an estimate of the
uncensored time error. Conditional MAE must never be presented without coverage;
an always-abstaining model has no conditional MAE, not perfect timing accuracy.
Per-window scores are macro averages across forecasted windows. Because a policy
chooses its windows, compare them alongside episode capture/discovery/coverage
outcomes, not as the sole measure of which scheduling policy is better.

Zero denominators produce JSON `null`. No forecasts produce zero forecast count
and undefined prediction scores. Empty truth windows remain classification samples
but do not contribute to ratio MAE. Invalid probabilities, counts, nonfinite values
and impossible timing outcomes fail explicitly. The accumulator retains only
sufficient statistics, so its memory does not grow with trace length.

## Replay integration

Existing policies returning an integer action remain compatible. A forecasting
policy returns an immutable decision with the forecast attached:

```python
from spectra_scheduler.evaluation_contract import Decision, Forecast

def act(self, observation):
    action, probability, delay_seconds, ratio = self.model(observation)
    return Decision(action, Forecast(probability, delay_seconds, ratio))
```

The evaluator saves this decision before stepping. It obtains the last action's
truth through `ReplayEnv.evaluation_outcome()` only after the action completes.
The replay feature/action specification and checkpoint compatibility remain unchanged;
the additive evaluation block has its own `contract_version`.

Run the existing reference check to generate the new report structure:

```bash
.venv/bin/python -m spectra_scheduler.replay_evaluation \
  --split train --max-files 3 --workers 2 \
  --output reports/generated/evaluation-contract-smoke.json
```

Sweep/random intentionally emit no forecasts. Their prediction scores are null.
This command is a pipeline check, not a learned-model benchmark.

## Synthetic integration

```python
from spectra_scheduler.synthetic_evaluation import evaluate_scheduler

report = evaluate_scheduler(
    simulation, scheduler,
    step_seconds=0.001,
    reward=lambda observation: float(observation.hit),
    reward_description="One per observed hit, including false alarms",
)
```

The step duration is required; discrete steps have no implicit physical duration.
The example's millisecond duration is an explicit experimental choice. Supply the
same reward definition to all competitors; recreate stateful reward objects per run.
This adapter accepts optional shared truth for paired runs. Existing legacy report
formats retain their historical definitions and are not silently reinterpreted.

Version 1 tick actions place discrete events at step boundaries, so a captured
event's within-action delay is zero. For a multi-tick target, return
`SyntheticAction(band, dwell_steps)` from `choose_action(time_step)`. The requested
duration counts listening ticks; retuning consumes additional elapsed ticks. The
evaluator clips the window at episode end and records the first true capture as
`(capture_step - action_start_step) * step_seconds`. It pools all-spectrum truth,
eligible events, detections, reward and false alarms over that identical window.
No capture remains right-censored at its actual clipped end. Version 2 false-alarm
support counts negative listening ticks, including those inside a single action.
The `discovery` block separately reports first acquisition and reacquisition;
those fields are episode outcomes, not action-window forecast labels.

`Simulation.run` and `evaluate_scheduler` use the same `SyntheticAction` execution
path and configure public receiver retune durations and episode horizon before
resetting the scheduler. Sparse `(time_step, band)` events represent `S(k,t)=1`;
an absent key represents `S(k,t)=0`. No dense occupancy array is required.

## Acceptance and subsequent model work

Infrastructure acceptance requires independently hand-calculated metric checks,
invalid-input rejection, censoring and empty-window tests, stream/chunk parity,
overflow and retune handling, and forecast-before-feedback tests. These checks are
implemented in `test_evaluation_contract.py`, `test_replay_env.py` and
`test_pulse_replay.py`.

Model selection must use validation data only. Final test recordings remain outside
training, tuning and reference smoke checks. Compare multiple seeds and paired
recordings with uncertainty intervals. Numerical performance targets and acceptable
capture/coverage tradeoffs are experiment decisions still to be set before final
model selection; this contract does not fabricate hackathon acceptance thresholds.
