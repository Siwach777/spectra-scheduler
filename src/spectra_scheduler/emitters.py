import random
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


@dataclass(frozen=True)
class BurstEmitter:
    """Transmit several closely spaced pulses, followed by a quiet interval."""

    emitter_id: str
    band: int
    burst_period: int
    pulses_per_burst: int
    pulse_spacing: int = 1
    phase: int = 0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(self.emitter_id, self.burst_period, self.phase, duration)
        if not 0 <= self.band < num_bands:
            raise ValueError(f"band {self.band} is outside a {num_bands}-band spectrum")
        if self.pulses_per_burst <= 0:
            raise ValueError("pulses_per_burst must be positive")
        if self.pulse_spacing <= 0:
            raise ValueError("pulse_spacing must be positive")
        burst_length = (self.pulses_per_burst - 1) * self.pulse_spacing
        if burst_length >= self.burst_period:
            raise ValueError("pulses must fit before the next burst")

        events: list[Transmission] = []
        for burst_start in range(self.phase, duration, self.burst_period):
            for pulse_number in range(self.pulses_per_burst):
                time_step = burst_start + pulse_number * self.pulse_spacing
                if time_step < duration:
                    events.append(Transmission(time_step, self.band, self.emitter_id))
        return events


@dataclass(frozen=True)
class JitteredPeriodicEmitter:
    """A periodic emitter whose interval varies by a small seeded amount."""

    emitter_id: str
    band: int
    period: int
    jitter: int
    seed: int = 0
    phase: int = 0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(self.emitter_id, self.period, self.phase, duration)
        if not 0 <= self.band < num_bands:
            raise ValueError(f"band {self.band} is outside a {num_bands}-band spectrum")
        if self.jitter < 0:
            raise ValueError("jitter cannot be negative")
        if self.jitter >= self.period:
            raise ValueError("jitter must be smaller than period")

        generator = random.Random(self.seed)
        events: list[Transmission] = []
        time_step = self.phase
        while time_step < duration:
            events.append(Transmission(time_step, self.band, self.emitter_id))
            time_step += self.period + generator.randint(-self.jitter, self.jitter)
        return events
