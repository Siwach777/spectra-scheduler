"""Causal finite-horizon planning from forecasts and public receiver timing.

The planner receives predicted capture counts, never future truth or emitter
parameters. It accounts for opportunities lost while retuning and returns only
the first native action; the policy replans after subsequent receiver feedback.
"""

from __future__ import annotations

import numpy as np

from .evaluation_contract import Forecast
from .experiments.calibrated_timing import CalibratedBeliefPolicy
from .simulation import SyntheticAction
from .timing_belief import TimingBeliefPolicy


class ForecastPlannerWorkspace:
    """Bounded reusable DP arrays for a known batch, band count and horizon."""

    def __init__(self, batch, bands, future, dwells=(1, 10, 50)):
        if min(batch, bands, future) < 1 or not dwells or min(dwells) < 1:
            raise ValueError("planner dimensions and dwells must be positive")
        self.batch_size = batch
        self.batch, self.bands, self.future = batch, bands, future
        self.dwells = tuple(dwells)
        previous = bands + 1
        shape = (batch, previous, bands, len(dwells))
        self.prefix = np.empty((batch, bands, future + 1), np.float64)
        self.state_values = np.empty((batch, future + 1, previous), np.float64)
        self.action_values = np.empty((batch, bands, len(dwells)), np.float64)
        self.delays = np.zeros((batch, previous, bands, 1), np.int64)
        self.starts = np.empty((batch, previous, bands, 1), np.int64)
        self.ends = np.empty(shape, np.int64)
        self.start_indices = np.empty(shape, np.int64)
        self.end_indices = np.empty(shape, np.int64)
        self.value_indices = np.empty(shape, np.int64)
        self.capture = np.empty(shape, np.float64)
        self.start_capture = np.empty(shape, np.float64)
        self.q_values = np.empty(shape, np.float64)
        self.prefix_offsets = np.arange(batch)[:, None, None, None] * bands * (
            future + 1
        ) + np.arange(bands)[None, None, :, None] * (future + 1)
        self.value_offsets = (
            np.arange(batch)[:, None, None, None] * (future + 1) * previous
            + np.arange(bands)[None, None, :, None]
        )
        self.dwell_array = np.asarray(dwells, np.int64)[None, None, None]


def first_actions(
    predictions, current_bands, retune_tables, horizons, dwells=(1, 10, 50), workspace=None
):
    """Return (selected bands, dwell ticks) for a batch of causal forecasts.

    Inputs have shapes ``[batch,bands,future]``, ``[batch]``,
    ``[batch,bands,bands]`` and ``[batch]``. Current band -1 means initial
    reception without a retune. Horizons are remaining physical ticks and are
    clipped to the supplied forecast length. ``workspace.action_values`` retains
    the integrated return for every first band/dwell action. Ties choose the
    first band and shortest supplied dwell, permitting prompt replanning.
    """
    predicted = np.asarray(predictions)
    if predicted.ndim != 3 or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("predictions must be finite nonnegative batch/band/future counts")
    batch, bands, future = predicted.shape
    if min(batch, bands, future) < 1:
        raise ValueError("predictions must have nonempty dimensions")
    dwells = tuple(dwells)
    if any(type(dwell) is not int or dwell < 1 for dwell in dwells) or not dwells:
        raise ValueError("dwells must be positive integer ticks")
    current = np.asarray(current_bands)
    tables = np.asarray(retune_tables)
    remaining = np.asarray(horizons)
    if (
        current.shape != (batch,)
        or remaining.shape != (batch,)
        or tables.shape != (batch, bands, bands)
    ):
        raise ValueError("public planning state shapes differ from the prediction batch")
    if (
        not np.isfinite(tables).all()
        or (tables < 0).any()
        or (tables != np.floor(tables)).any()
        or (current < -1).any()
        or (current >= bands).any()
        or (current != np.floor(current)).any()
        or not np.isfinite(remaining).all()
        or (remaining < 1).any()
        or (remaining != np.floor(remaining)).any()
    ):
        raise ValueError("public planning state must use valid integer ticks and bands")
    if np.diagonal(tables, axis1=1, axis2=2).any():
        raise ValueError("staying on the same band cannot require retuning")
    if tables.max() + max(dwells) > future:
        raise ValueError("forecast horizon does not cover public retuning and dwells")
    if workspace is None:
        workspace = ForecastPlannerWorkspace(batch, bands, future, dwells)
    if (workspace.batch, workspace.bands, workspace.future, workspace.dwells) != (
        batch,
        bands,
        future,
        dwells,
    ):
        raise ValueError("workspace dimensions differ from the requested planning batch")
    states = bands + 1
    terminal = np.minimum(remaining, future).astype(np.int64)[:, None, None, None]
    workspace.prefix[..., 0] = 0
    np.cumsum(predicted, axis=-1, dtype=np.float64, out=workspace.prefix[..., 1:])
    workspace.state_values.fill(0)
    workspace.delays[:, :bands, :, 0] = tables
    for tick in range(future - 1, -1, -1):
        np.add(workspace.delays, tick, out=workspace.starts)
        np.minimum(workspace.starts, terminal, out=workspace.starts)
        np.add(workspace.starts, workspace.dwell_array, out=workspace.ends)
        np.minimum(workspace.ends, terminal, out=workspace.ends)
        np.add(workspace.prefix_offsets, workspace.starts, out=workspace.start_indices)
        np.add(workspace.prefix_offsets, workspace.ends, out=workspace.end_indices)
        np.take(workspace.prefix, workspace.end_indices, out=workspace.capture)
        np.take(workspace.prefix, workspace.start_indices, out=workspace.start_capture)
        np.subtract(workspace.capture, workspace.start_capture, out=workspace.capture)
        np.multiply(workspace.ends, states, out=workspace.value_indices)
        np.add(workspace.value_indices, workspace.value_offsets, out=workspace.value_indices)
        np.take(workspace.state_values, workspace.value_indices, out=workspace.q_values)
        np.add(workspace.q_values, workspace.capture, out=workspace.q_values)
        np.max(workspace.q_values, axis=(2, 3), out=workspace.state_values[:, tick])
    previous = np.where(current < 0, bands, current).astype(np.int64)
    workspace.action_values[:] = workspace.q_values[np.arange(batch), previous]
    flat = workspace.action_values.reshape(batch, -1)
    maximum = flat.max(1, keepdims=True)
    # Cumulative summation can create tiny discrepancies between equal plans.
    choice = (flat >= maximum - 1e-10).argmax(1)
    return choice // len(dwells), np.asarray(dwells)[choice % len(dwells)]


class TimingPlannerPolicy(TimingBeliefPolicy):
    """Expected-total-capture planning with optional causal coverage probes.

    Coverage uses the inherited revisit/probe settings. The DP objective has no
    additional age bonus, retune penalty or switch margin: retuning already costs
    forecast opportunities. Those inherited scoring parameters apply only to the
    old greedy policy. This policy preserves its observation and forecast format.
    """

    def __init__(self, model, config=None, *, coverage=True):
        super().__init__(model, config)
        self.coverage = coverage
        self.workspace = None

    def reset(self, bands):
        super().reset(bands)
        self.workspace = ForecastPlannerWorkspace(
            1, bands, self.model.config.future, self.config.dwells
        )

    def coverage_action(self, time_step):
        """Return a required public-history coverage action, or None."""
        if not self.coverage:
            return None
        ages = np.where(
            self.history.last_listen >= 0,
            time_step - self.history.last_listen,
            time_step + self.config.revisit,
        )
        if ages.max() < self.config.revisit:
            return None
        band = int(ages.argmax())
        dwell = min(self.config.dwells, key=lambda value: abs(value - self.config.probe))
        return SyntheticAction(band, int(dwell))

    def select(self, time_step, predicted):
        action = self.coverage_action(time_step)
        if action is not None:
            self.probes += 1
        else:
            bands, dwells = first_actions(
                predicted[None],
                [self.history.current_band],
                self.retune[None],
                [self.horizon - time_step],
                self.config.dwells,
                self.workspace,
            )
            action = SyntheticAction(int(bands[0]), int(dwells[0]))
        return self.accept_action(time_step, predicted, action)

    def accept_action(self, time_step, predicted, action):
        """Record one batched-planner action and its causal pre-action forecast."""
        if time_step != self.history.time:
            raise ValueError("planner clock differs from causal history")
        if action.dwell_steps not in self.config.dwells or not 0 <= action.band < self.bands:
            raise ValueError("planner action differs from the declared native action menu")
        self.decisions += 1
        previous, band = self.history.current_band, action.band
        remaining = self.horizon - time_step
        delay = int(self.retune[previous, band]) if previous >= 0 else 0
        start, end = min(delay, remaining), min(delay + action.dwell_steps, remaining)
        rates = np.clip(predicted[band, start:end], 0, 1 - 1e-7)
        survival = np.concatenate(([1.0], np.cumprod(1 - rates)))
        mass = survival[:-1] * rates
        probability = float(1 - survival[-1])
        conditional_time = float(mass @ (np.arange(start, end) * 0.001) / max(probability, 1e-12))
        capture = predicted[band, start:end].sum()
        ratio = float(np.clip(capture / max(predicted[:, :end].sum(), 1e-12), 0, 1))
        self.pending = (
            time_step,
            band,
            action.dwell_steps,
            Forecast(probability, conditional_time if probability >= 0.5 else None, ratio),
        )
        return action


# The calibration mixin retains the same public pdet correction as existing
# assessments, without requiring any change to their shared policy or evaluator.
class CalibratedTimingPlannerPolicy(TimingPlannerPolicy, CalibratedBeliefPolicy):
    """Planner with the existing publicly calibrated arrival-ratio forecast."""
