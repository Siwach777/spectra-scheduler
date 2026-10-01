"""Causal scan acquisition before the trained timing planner takes over."""

from .timing_planner import CalibratedTimingPlannerPolicy
from .whittle_policy import ScanStrategyConfig, WhittleScanScheduler


class PhasedTimingPlanner(CalibratedTimingPlannerPolicy):
    def __init__(self, model, config, *, coverage=True, acquisition=None):
        super().__init__(model, config, coverage=coverage)
        self.acquisition = acquisition or ScanStrategyConfig(strategy="adaptive")
        if self.acquisition.strategy not in ("phased", "adaptive"):
            raise ValueError("timing handover requires a phased acquisition strategy")

    def reset(self, bands):
        super().reset(bands)
        self.sweep = WhittleScanScheduler(self.acquisition)
        self.sweep.set_retune_table(self.retune)
        self.sweep.set_episode_horizon(self.horizon)
        self.sweep.reset(bands)

    def coverage_action(self, time_step):
        if self.sweep.switch.exploring(time_step):
            return self.sweep.golden_action(time_step)
        return super().coverage_action(time_step)

    def observe(self, observation):
        self.sweep.observe(observation)
        super().observe(observation)
