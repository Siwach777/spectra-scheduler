from dataclasses import dataclass

from spectra_scheduler.emitters import Emitter
from spectra_scheduler.models import Observation, SimulationResult, Transmission
from spectra_scheduler.schedulers import Scheduler


@dataclass(frozen=True)
class Simulation:
    num_bands: int
    duration: int
    emitters: tuple[Emitter, ...]

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

    def run(self, scheduler: Scheduler) -> SimulationResult:
        transmissions = self.generate_truth()
        events_by_time_and_band: dict[tuple[int, int], list[Transmission]] = {}
        for event in transmissions:
            events_by_time_and_band.setdefault((event.time_step, event.band), []).append(event)

        scheduler.reset(self.num_bands)
        observations: list[Observation] = []

        for time_step in range(self.duration):
            band = scheduler.choose_band(time_step)
            if not 0 <= band < self.num_bands:
                raise ValueError(f"scheduler chose invalid band {band} at step {time_step}")

            visible_events = events_by_time_and_band.get((time_step, band), [])
            observation = Observation(
                time_step=time_step,
                band=band,
                detected_emitters=tuple(event.emitter_id for event in visible_events),
            )
            observations.append(observation)
            scheduler.observe(observation)

        return SimulationResult(
            duration=self.duration,
            num_bands=self.num_bands,
            transmissions=transmissions,
            observations=tuple(observations),
        )
