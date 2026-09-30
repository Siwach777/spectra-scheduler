from dataclasses import dataclass, field

from spectra_scheduler.emitters import Emitter, get_emitter_changes
from spectra_scheduler.models import (
    DetectionRecord,
    EmitterChange,
    Observation,
    SimulationResult,
    Transmission,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.schedulers import Scheduler


@dataclass(frozen=True)
class SyntheticAction:
    """Listen on a band for this many ticks, after any required retuning."""

    band: int
    dwell_steps: int

    def __post_init__(self):
        if type(self.band) is not int or self.band < 0:
            raise ValueError("action band must be a nonnegative integer")
        if type(self.dwell_steps) is not int or self.dwell_steps < 1:
            raise ValueError("dwell_steps must be a positive integer")


def configure_scheduler(simulation, scheduler):
    """Pass public receiver timing and horizon before resetting episode state."""
    if hasattr(scheduler, "set_retune_table"):
        scheduler.set_retune_table(
            tuple(
                tuple(
                    simulation.receiver.retune_duration(a, b)
                    for b in range(simulation.num_bands)
                )
                for a in range(simulation.num_bands)
            )
        )
    if hasattr(scheduler, "set_episode_horizon"):
        scheduler.set_episode_horizon(simulation.duration)
    scheduler.reset(simulation.num_bands)


@dataclass(frozen=True)
class Simulation:
    num_bands: int
    duration: int
    emitters: tuple[Emitter, ...]
    receiver: Receiver = field(default_factory=Receiver)

    def __post_init__(self) -> None:
        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    def generate_truth(self) -> tuple[Transmission, ...]:
        events = [
            event
            for emitter in self.emitters
            for event in emitter.transmissions(self.duration, self.num_bands)
        ]
        return tuple(sorted(events))

    def generate_emitter_changes(self) -> tuple[EmitterChange, ...]:
        changes = [
            change
            for emitter in self.emitters
            for change in get_emitter_changes(emitter, self.duration)
        ]
        return tuple(sorted(changes))

    def run(
        self,
        scheduler: Scheduler,
        *,
        truth: tuple[Transmission, ...] | None = None,
    ) -> SimulationResult:
        episode = SimulationEpisode(self, truth=truth)
        configure_scheduler(self, scheduler)
        while episode.time_step < self.duration:
            if hasattr(scheduler, "choose_action"):
                action = scheduler.choose_action(episode.time_step)
                if not isinstance(action, SyntheticAction):
                    raise ValueError("choose_action() must return SyntheticAction")
                for observation in episode.step_action(action):
                    scheduler.observe(observation)
            else:
                band = scheduler.choose_band(episode.time_step)
                scheduler.observe(episode.step(band))
        return episode.result()


class SimulationEpisode:
    """Shared step engine for batch comparisons and interactive RL environments."""

    def __init__(self, simulation: Simulation, *, truth=None):
        self.simulation = simulation
        self.transmissions = simulation.generate_truth() if truth is None else truth
        self.events: dict[tuple[int, int], list[Transmission]] = {}
        for event in self.transmissions:
            self.events.setdefault((event.time_step, event.band), []).append(event)
        self.time_step = 0
        self.previous_band = None
        self.retune_remaining = 0
        self.records: list[DetectionRecord] = []

    def step(self, band: int) -> Observation:
        if self.time_step >= self.simulation.duration:
            raise RuntimeError("episode is finished")
        if not 0 <= band < self.simulation.num_bands:
            raise ValueError(f"scheduler chose invalid band {band} at step {self.time_step}")
        receiver = self.simulation.receiver
        if self.previous_band is not None and band != self.previous_band:
            self.retune_remaining = receiver.retune_duration(self.previous_band, band)
        self.previous_band = band
        if self.retune_remaining > 0:
            record = DetectionRecord(Observation(self.time_step, band, listening=False))
            self.retune_remaining -= 1
        else:
            record = receiver.listen(
                self.time_step, band, self.events.get((self.time_step, band), [])
            )
        self.records.append(record)
        self.time_step += 1
        return record.observation

    def step_action(self, action: SyntheticAction) -> tuple[Observation, ...]:
        """Complete a listening dwell, including retune ticks and horizon clipping."""
        if not isinstance(action, SyntheticAction):
            raise ValueError("action must be SyntheticAction")
        if not 0 <= action.band < self.simulation.num_bands:
            raise ValueError("action band outside receiver range")
        if self.time_step >= self.simulation.duration:
            raise RuntimeError("episode is finished")
        observations = []
        listens = 0
        while listens < action.dwell_steps and self.time_step < self.simulation.duration:
            observation = self.step(action.band)
            observations.append(observation)
            listens += int(observation.listening)
        return tuple(observations)

    def result(self) -> SimulationResult:
        if self.time_step != self.simulation.duration:
            raise RuntimeError("result requires a completed episode")
        return SimulationResult(
            duration=self.simulation.duration,
            num_bands=self.simulation.num_bands,
            transmissions=self.transmissions,
            emitter_changes=self.simulation.generate_emitter_changes(),
            observations=tuple(record.observation for record in self.records),
            detection_records=tuple(self.records),
        )
