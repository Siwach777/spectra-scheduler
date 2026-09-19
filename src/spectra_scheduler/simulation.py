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
        scheduler.reset(self.num_bands)
        for time_step in range(self.duration):
            band = scheduler.choose_band(time_step)
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
