"""Regression check of macro listening dwell through the tick benchmark adapter."""

import pytest

pytest.importorskip("torch")

from spectra_scheduler.experiments.physical_model_compare import TickMacroAdapter  # noqa: E402
from spectra_scheduler.experiments.timing_report import reporting_world, reward  # noqa: E402
from spectra_scheduler.simulation import SyntheticAction  # noqa: E402
from spectra_scheduler.synthetic_evaluation import evaluate_scheduler  # noqa: E402


class RecordingMacro:
    def set_retune_table(self, table):
        self.retune = table

    def set_episode_horizon(self, horizon):
        self.horizon = horizon

    def reset(self, bands):
        self.actions, self.received = 0, []
        self.bands = bands

    def choose_action(self, step):
        band = self.actions % self.bands
        self.actions += 1
        return SyntheticAction(band, 8)

    def observe(self, observation):
        self.received.append(observation)


def test_adapter_executes_same_macro_observations_and_receiver_counts():
    simulation, _ = reporting_world("periodic-scan", 2000)
    macro, tick = RecordingMacro(), RecordingMacro()
    kwargs = dict(step_seconds=.001, reward=reward,
                  reward_description="observed_hit - 0.05 * retuning")
    expected = evaluate_scheduler(simulation, macro, **kwargs)
    actual = evaluate_scheduler(simulation, TickMacroAdapter(tick), **kwargs)
    assert macro.received == tick.received
    assert macro.actions == tick.actions
    assert actual["counts"] == expected["counts"]
    assert actual["reward_sum"] == pytest.approx(expected["reward_sum"])
    assert any(not o.listening for o in tick.received)
