"""Public receiver calibration for forecasts of detectable-signal arrival ratios.

These wrappers convert an expected capture share to captures / all arrivals by
using the receiver's public detection probability. The denominator assumes
signals are above sensitivity. It does not infer unseen below-threshold arrivals.
Scheduling decisions and capture forecasts are unchanged.
"""

from dataclasses import replace
from math import isfinite
from typing import Protocol

from ..simulation import Simulation
from ..timing_belief import TimingBeliefPolicy
from .phase_selection import BetaPhaseScheduler


class CalibratablePolicy(Protocol):
    def set_detection_probability(self, probability: float) -> None: ...


def configure_public_detection(simulation: Simulation, scheduler: CalibratablePolicy) -> None:
    """Expose public receiver response; never pass emitters or realized truth."""
    scheduler.set_detection_probability(simulation.receiver.detection_probability)


class _CalibratedRatio:
    def set_detection_probability(self, probability: float) -> None:
        if not isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("ratio calibration requires a positive detection probability")
        self.detection_probability = probability

    def forecast(self, time_step, action):
        if not hasattr(self, "detection_probability"):
            raise RuntimeError("configure the public receiver response before forecasting")
        forecast = super().forecast(time_step, action)
        if forecast is None:
            return None
        return replace(
            forecast, interception_ratio=(forecast.interception_ratio * self.detection_probability)
        )


class CalibratedBeliefPolicy(_CalibratedRatio, TimingBeliefPolicy):
    """CUDA timing policy with a publicly calibrated arrival-ratio forecast."""


class CalibratedPhasePolicy(_CalibratedRatio, BetaPhaseScheduler):
    """Structural control with the same arrival-ratio forecast calibration."""
