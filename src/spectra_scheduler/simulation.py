from dataclasses import dataclass, field

from spectra_scheduler.emitters import Emitter
from spectra_scheduler.models import (
    DetectionRecord,
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

    def run(
        self,
        scheduler: Scheduler,
        *,
        truth: tuple[Transmission, ...] | None = None,
    ) -> SimulationResult:
        transmissions = self.generate_truth() if truth is None else truth
        events_by_time_and_band: dict[tuple[int, int], list[Transmission]] = {}
        for event in transmissions:
            events_by_time_and_band.setdefault((event.time_step, event.band), []).append(event)

        scheduler.reset(self.num_bands)
        observations: list[Observation] = []
        detection_records: list[DetectionRecord] = []

        for time_step in range(self.duration):
            band = scheduler.choose_band(time_step)
            if not 0 <= band < self.num_bands:
                raise ValueError(f"scheduler chose invalid band {band} at step {time_step}")

            visible_events = events_by_time_and_band.get((time_step, band), [])
            detection_record = self.receiver.listen(time_step, band, visible_events)
            observation = detection_record.observation
            observations.append(observation)
            detection_records.append(detection_record)
            scheduler.observe(observation)

        return SimulationResult(
            duration=self.duration,
            num_bands=self.num_bands,
            transmissions=transmissions,
            observations=tuple(observations),
            detection_records=tuple(detection_records),
        )
