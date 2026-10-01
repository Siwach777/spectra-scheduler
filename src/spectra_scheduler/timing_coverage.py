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


@dataclass(frozen=True)
class AcquisitionConfig:
    dwell: int = 10
    fraction: float = 0.15
    minimum_hits: int = 2
    maximum_relative_loss: float = 0.05

    def __post_init__(self):
        RecoveryConfig(dwell=self.dwell, fraction=self.fraction,
                       minimum_hits=self.minimum_hits,
                       maximum_relative_loss=self.maximum_relative_loss)


class AcquisitionTimingPlanner(CalibratedTimingPlannerPolicy):
    """Extend already-selected short probes when evidence and forecast return allow.

    This adds contiguous listening on the band the planner already selected, so
    it needs no extra retune. The cap charges only added elapsed ticks, including
    clipping at the episode end. Measurements determine reliable support; emitter
    identities, true periods and future activity are never policy inputs.
    """

    def __init__(self, model, config=None, *, acquisition=None, coverage=True):
        super().__init__(model, config, coverage=coverage)
        self.acquisition = acquisition or AcquisitionConfig()
        if self.acquisition.dwell not in self.config.dwells:
            raise ValueError("acquisition dwell must belong to the saved action menu")
        if not hasattr(model, "quality_threshold"):
            raise ValueError("acquisition requires a forecaster with a measured-power gate")
        self.power_threshold = float(model.quality_threshold.detach().cpu())

    def reset(self, bands):
        super().reset(bands)
        self.reliable_hits = np.zeros(bands, np.int64)
        self.acquisition_ticks = 0
        self._pending_acquisition = None

    def observe(self, observation):
        super().observe(observation)
        if observation.listening and observation.measurements:
            normalized = np.clip((max(m.power_dbm for m in observation.measurements) + 100) / 40,
                                 0, 2)
            if normalized >= self.power_threshold:
                self.reliable_hits[observation.band] += 1

    def filter_coverage_action(self, action, values):
        self._pending_acquisition = None
        if action is not None or not self.coverage:
            return action
        # Preserve the original search until at least one reliable signal exists.
        if not self.reliable_hits.any():
            return None
        maximum = float(values.max())
        choice = int(np.flatnonzero(values.flatten() >= maximum - 1e-10)[0])
        band, index = divmod(choice, len(self.config.dwells))
        dwell = self.config.dwells[index]
        cfg = self.acquisition
        if dwell >= cfg.dwell or self.reliable_hits[band] >= cfg.minimum_hits:
            return None
        proposed = float(values[band, self.config.dwells.index(cfg.dwell)])
        if maximum - proposed > cfg.maximum_relative_loss * max(maximum, 1e-12):
            return None
        previous = self.history.current_band
        delay = int(self.retune[previous, band]) if previous >= 0 else 0
        remaining = self.horizon - self.history.time
        cost = min(delay + cfg.dwell, remaining) - min(delay + dwell, remaining)
        if cost <= 0 or self.acquisition_ticks + cost > cfg.fraction * self.horizon:
            return None
        candidate = SyntheticAction(band, cfg.dwell)
        self._pending_acquisition = (self.history.time, candidate, cost)
        return candidate

    def select(self, time_step, predicted):
        bands, dwells = first_actions(
            predicted[None], [self.history.current_band], self.retune[None],
            [self.horizon - time_step], self.config.dwells, self.workspace,
        )
        action = self.filter_coverage_action(
            self.coverage_action(time_step), self.workspace.action_values[0]
        ) or SyntheticAction(int(bands[0]), int(dwells[0]))
        return self.accept_action(time_step, predicted, action)

    def accept_action(self, time_step, predicted, action):
        accepted = super().accept_action(time_step, predicted, action)
        if self._pending_acquisition is not None:
            step, forced, cost = self._pending_acquisition
            if step == time_step and forced == action:
                self.acquisition_ticks += cost
        self._pending_acquisition = None
        return accepted
