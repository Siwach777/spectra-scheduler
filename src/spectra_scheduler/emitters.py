from dataclasses import dataclass
from typing import Protocol

from spectra_scheduler.models import Transmission


class Emitter(Protocol):
    emitter_id: str

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]: ...


def _validate_common(
    emitter_id: str,
    period: int,
    phase: int,
    duration: int,
) -> None:
    if not emitter_id:
        raise ValueError("emitter_id cannot be empty")
    if period <= 0:
        raise ValueError("period must be positive")
    if phase < 0:
        raise ValueError("phase cannot be negative")
    if duration < 0:
        raise ValueError("duration cannot be negative")


@dataclass(frozen=True)
class PeriodicEmitter:
    """An emitter that transmits on one band at a fixed interval."""

    emitter_id: str
    band: int
    period: int
    phase: int = 0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(self.emitter_id, self.period, self.phase, duration)
        if not 0 <= self.band < num_bands:
            raise ValueError(f"band {self.band} is outside a {num_bands}-band spectrum")

        return [
            Transmission(time_step, self.band, self.emitter_id)
            for time_step in range(self.phase, duration, self.period)
        ]


@dataclass(frozen=True)
class FrequencyHoppingEmitter:
    """An emitter that moves through a repeating sequence of bands."""

    emitter_id: str
    bands: tuple[int, ...]
    period: int
    phase: int = 0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(self.emitter_id, self.period, self.phase, duration)
        if not self.bands:
            raise ValueError("bands cannot be empty")
        invalid_bands = [band for band in self.bands if not 0 <= band < num_bands]
        if invalid_bands:
            raise ValueError(f"bands outside a {num_bands}-band spectrum: {invalid_bands}")

        return [
            Transmission(time_step, self.bands[index % len(self.bands)], self.emitter_id)
            for index, time_step in enumerate(range(self.phase, duration, self.period))
        ]
