import random
from dataclasses import dataclass
from math import isfinite
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
    power_dbm: float,
    pulse_width_us: float,
) -> None:
    if not emitter_id:
        raise ValueError("emitter_id cannot be empty")
    if period <= 0:
        raise ValueError("period must be positive")
    if phase < 0:
        raise ValueError("phase cannot be negative")
    if duration < 0:
        raise ValueError("duration cannot be negative")
    if not isfinite(power_dbm):
        raise ValueError("power_dbm must be finite")
    if not isfinite(pulse_width_us) or pulse_width_us <= 0:
        raise ValueError("pulse_width_us must be finite and positive")


@dataclass(frozen=True)
class PeriodicEmitter:
    """An emitter that transmits on one band at a fixed interval."""

    emitter_id: str
    band: int
    period: int
    phase: int = 0
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(
            self.emitter_id,
            self.period,
            self.phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
        if not 0 <= self.band < num_bands:
            raise ValueError(f"band {self.band} is outside a {num_bands}-band spectrum")

        return [
            Transmission(
                time_step,
                self.band,
                self.emitter_id,
                self.power_dbm,
                self.pulse_width_us,
            )
            for time_step in range(self.phase, duration, self.period)
        ]


@dataclass(frozen=True)
class FrequencyHoppingEmitter:
    """An emitter that moves through a repeating sequence of bands."""

    emitter_id: str
    bands: tuple[int, ...]
    period: int
    phase: int = 0
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(
            self.emitter_id,
            self.period,
            self.phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
        if not self.bands:
            raise ValueError("bands cannot be empty")
        invalid_bands = [band for band in self.bands if not 0 <= band < num_bands]
        if invalid_bands:
            raise ValueError(f"bands outside a {num_bands}-band spectrum: {invalid_bands}")

        return [
            Transmission(
                time_step,
                self.bands[index % len(self.bands)],
                self.emitter_id,
                self.power_dbm,
                self.pulse_width_us,
            )
            for index, time_step in enumerate(range(self.phase, duration, self.period))
        ]


@dataclass(frozen=True)
class ScanningEmitter:
    """Move across adjacent bands and reverse at each edge."""

    emitter_id: str
    lowest_band: int
    highest_band: int
    period: int
    phase: int = 0
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(
            self.emitter_id,
            self.period,
            self.phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
        if self.lowest_band < 0 or self.highest_band >= num_bands:
            raise ValueError("scan range is outside the spectrum")
        if self.lowest_band > self.highest_band:
            raise ValueError("lowest_band cannot be greater than highest_band")

        upward = list(range(self.lowest_band, self.highest_band + 1))
        downward = list(range(self.highest_band - 1, self.lowest_band, -1))
        scan_cycle = upward + downward
        return [
            Transmission(
                time_step,
                scan_cycle[index % len(scan_cycle)],
                self.emitter_id,
                self.power_dbm,
                self.pulse_width_us,
            )
            for index, time_step in enumerate(range(self.phase, duration, self.period))
        ]


@dataclass(frozen=True)
class SpatialScanningEmitter:
    """Fixed-frequency pulses visible while a periodic beam crosses the receiver.

    This is a rectangular visibility gate, not an antenna/propagation model.
    phase is the scan phase; pulse phase is independent. Truth represents signals
    present at the receiver location, not emissions pointed elsewhere.
    """

    emitter_id: str
    band: int
    scan_period: int
    visible_steps: int
    pulse_period: int = 1
    phase: int = 0
    pulse_phase: int = 0
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        for name in (
            "band",
            "scan_period",
            "visible_steps",
            "pulse_period",
            "phase",
            "pulse_phase",
            "duration",
            "num_bands",
        ):
            value = locals().get(name, getattr(self, name, None))
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        _validate_common(
            self.emitter_id,
            self.pulse_period,
            self.pulse_phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
        if not 0 < self.visible_steps <= self.scan_period or self.phase < 0:
            raise ValueError("require 0 < visible_steps <= scan_period and nonnegative phase")
        if not 0 <= self.band < num_bands:
            raise ValueError("spatial scanner band outside spectrum")
        # Enumerate only visible gates; do not allocate invisible transmissions.
        events = []
        for start in range(
            self.phase % self.scan_period - self.scan_period, duration, self.scan_period
        ):
            low, high = max(0, start, self.pulse_phase), min(duration, start + self.visible_steps)
            first = low + (self.pulse_phase - low) % self.pulse_period
            events.extend(
                Transmission(t, self.band, self.emitter_id, self.power_dbm, self.pulse_width_us)
                for t in range(first, high, self.pulse_period)
            )
        return events


@dataclass(frozen=True)
class BurstEmitter:
    """Transmit several closely spaced pulses, followed by a quiet interval."""

    emitter_id: str
    band: int
    burst_period: int
    pulses_per_burst: int
    pulse_spacing: int = 1
    phase: int = 0
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(
            self.emitter_id,
            self.burst_period,
            self.phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
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
                    events.append(
                        Transmission(
                            time_step,
                            self.band,
                            self.emitter_id,
                            self.power_dbm,
                            self.pulse_width_us,
                        )
                    )
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
    power_dbm: float = -60.0
    pulse_width_us: float = 1.0

    def transmissions(self, duration: int, num_bands: int) -> list[Transmission]:
        _validate_common(
            self.emitter_id,
            self.period,
            self.phase,
            duration,
            self.power_dbm,
            self.pulse_width_us,
        )
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
            events.append(
                Transmission(
                    time_step,
                    self.band,
                    self.emitter_id,
                    self.power_dbm,
                    self.pulse_width_us,
                )
            )
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
        before_switch = [event for event in first_events if event.time_step < self.switch_time]
        after_switch = [event for event in second_events if event.time_step >= self.switch_time]
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
        return [change for change in changes if emitter.start_time <= change.time_step < end_time]
    return []
