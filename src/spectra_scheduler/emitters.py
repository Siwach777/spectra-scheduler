import random
from dataclasses import dataclass
from typing import Protocol

from spectra_scheduler.models import EmitterChange, Transmission


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


@dataclass(frozen=True)
class WindowedEmitter:
    """Limit an emitter to part of a scenario."""

    emitter: Emitter
    start_time: int = 0
    end_time: int | None = None

    @property
    def emitter_id(self) -> str:
        return self.emitter.emitter_id

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        if self.start_time < 0:
            raise ValueError("start_time cannot be negative")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")

        end_time = duration if self.end_time is None else min(duration, self.end_time)
        return [
            event
            for event in self.emitter.transmissions(duration, num_bands)
            if self.start_time <= event.time_step < end_time
        ]


@dataclass(frozen=True)
class ModeSwitchingEmitter:
    """Change from one emitter pattern to another at a chosen time."""

    first_mode: Emitter
    second_mode: Emitter
    switch_time: int

    @property
    def emitter_id(self) -> str:
        return self.first_mode.emitter_id

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        if self.switch_time < 0:
            raise ValueError("switch_time cannot be negative")
        if self.first_mode.emitter_id != self.second_mode.emitter_id:
            raise ValueError("both modes must belong to the same emitter")

        first_events = self.first_mode.transmissions(duration, num_bands)
        second_events = self.second_mode.transmissions(duration, num_bands)
        before_switch = [
            event for event in first_events if event.time_step < self.switch_time
        ]
        after_switch = [
            event for event in second_events if event.time_step >= self.switch_time
        ]
        return before_switch + after_switch


def get_emitter_changes(emitter: Emitter, duration: int) -> list[EmitterChange]:
    """Return mode changes known to the simulator but hidden from schedulers."""

    if isinstance(emitter, ModeSwitchingEmitter):
        if emitter.switch_time < duration:
            return [EmitterChange(emitter.switch_time, emitter.emitter_id)]
        return []
    if isinstance(emitter, WindowedEmitter):
        changes = get_emitter_changes(emitter.emitter, duration)
        end_time = duration if emitter.end_time is None else emitter.end_time
        return [
            change
            for change in changes
            if emitter.start_time <= change.time_step < end_time
        ]
    return []
