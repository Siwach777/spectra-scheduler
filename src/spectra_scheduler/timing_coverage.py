"""Bounded extra acquisition from public hit history, without emitter identities."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .simulation import SyntheticAction
from .timing_planner import CalibratedTimingPlannerPolicy, first_actions


@dataclass(frozen=True)
class RecoveryConfig:
    revisit: int = 128
    dwell: int = 10
    fraction: float = 0.15
    minimum_hits: int = 2
    maximum_relative_loss: float = 0.1

    def __post_init__(self):
        for name in ("revisit", "dwell", "minimum_hits"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError("coverage fraction must be in [0, 1]")
        if not isfinite(self.maximum_relative_loss) or not 0 <= self.maximum_relative_loss <= 1:
            raise ValueError("relative forecast loss must be in [0, 1]")


class RecoveryTimingPlanner(CalibratedTimingPlannerPolicy):
    """Preserve initial probes, then revisit weak-evidence bands within a time cap.

    The cap includes retuning and listening, excludes initial mandatory coverage,
    and is relative to the public episode horizon. Repeated hits are evidence of
    occupancy, not proof that every emitter in that band has been discovered.
    """

    def __init__(self, model, config=None, *, recovery=None, coverage=True):
        super().__init__(model, config, coverage=coverage)
        self.recovery = recovery or RecoveryConfig()
        if self.recovery.dwell not in self.config.dwells:
            raise ValueError("recovery dwell must belong to the saved action menu")

    def reset(self, bands):
        super().reset(bands)
        self.recovery_ticks = 0
        self._pending_recovery = None

    def coverage_action(self, time_step):
        self._pending_recovery = None
        action = super().coverage_action(time_step)
        if action is not None or not self.coverage:
            return action
        ages = time_step - self.history.last_listen
        eligible = (self.history.last_listen >= 0) & (
            self.history.hits < self.recovery.minimum_hits
        ) & (ages >= self.recovery.revisit)
        if not eligible.any():
            return None
        band = int(np.argmax(np.where(eligible, ages, -1)))
        previous = self.history.current_band
        delay = int(self.retune[previous, band]) if previous >= 0 else 0
        cost = min(delay + self.recovery.dwell, self.horizon - time_step)
        if self.recovery_ticks + cost > self.recovery.fraction * self.horizon:
            return None
        action = SyntheticAction(band, self.recovery.dwell)
        self._pending_recovery = (time_step, action, cost)
        return action

    def filter_coverage_action(self, action, values):
        """Probe only when its forecast return is close to the best planned return."""
        if action is None or self._pending_recovery is None:
            return action
        best = float(values.max())
        probe = float(values[action.band, self.config.dwells.index(action.dwell_steps)])
        if best - probe > self.recovery.maximum_relative_loss * max(best, 1e-12):
            self._pending_recovery = None
            return None
        return action

    def select(self, time_step, predicted):
        bands, dwells = first_actions(
            predicted[None], [self.history.current_band], self.retune[None],
            [self.horizon - time_step], self.config.dwells, self.workspace,
        )
        action = self.filter_coverage_action(
            self.coverage_action(time_step), self.workspace.action_values[0]
        )
        if action is None:
            action = SyntheticAction(int(bands[0]), int(dwells[0]))
        else:
            self.probes += 1
        return self.accept_action(time_step, predicted, action)

    def accept_action(self, time_step, predicted, action):
        accepted = super().accept_action(time_step, predicted, action)
        if self._pending_recovery is not None:
            step, forced, cost = self._pending_recovery
            if step == time_step and forced == action:
                self.recovery_ticks += cost
        self._pending_recovery = None
        return accepted
