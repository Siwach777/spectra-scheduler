import hashlib
from dataclasses import dataclass
from math import cos, log, pi, sqrt

from spectra_scheduler.models import (
    DetectionRecord,
    Observation,
    SignalMeasurement,
    Transmission,
)


@dataclass(frozen=True)
class Receiver:
    """A narrow-band receiver with simple detection errors."""

    detection_probability: float = 1.0
    false_alarm_probability: float = 0.0
    sensitivity_dbm: float = -90.0
    noise_std_db: float = 0.0
    retune_steps: int = 0
    tuning_speed_bands_per_step: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be between 0 and 1")
        if not 0.0 <= self.false_alarm_probability <= 1.0:
            raise ValueError("false_alarm_probability must be between 0 and 1")
        if self.noise_std_db < 0.0:
            raise ValueError("noise_std_db cannot be negative")
        if self.retune_steps < 0:
            raise ValueError("retune_steps cannot be negative")
        if (
            self.tuning_speed_bands_per_step is not None
            and self.tuning_speed_bands_per_step <= 0
        ):
            raise ValueError("tuning_speed_bands_per_step must be positive")

    def retune_duration(self, previous_band: int, next_band: int) -> int:
        distance = abs(next_band - previous_band)
        if distance == 0:
            return 0
        if self.tuning_speed_bands_per_step is None:
            return self.retune_steps

        speed = self.tuning_speed_bands_per_step
        distance_delay = (distance + speed - 1) // speed
        return max(self.retune_steps, distance_delay)

    def listen(
        self,
        time_step: int,
        band: int,
        visible_events: list[Transmission],
    ) -> DetectionRecord:
        measured_events = [
            (
                event,
                event.power_dbm
                + self._noise_db(time_step, band, event.emitter_id, event_number),
            )
            for event_number, event in enumerate(visible_events)
        ]
        detectable_events = [
            (event, measured_power)
            for event, measured_power in measured_events
            if measured_power >= self.sensitivity_dbm
        ]
        detected_events = [
            (event, measured_power)
            for event_number, (event, measured_power) in enumerate(detectable_events)
            if self._sample("detection", time_step, band, event.emitter_id, event_number)
            < self.detection_probability
        ]
        false_alarm = not detectable_events and (
            self._sample("false-alarm", time_step, band) < self.false_alarm_probability
        )
        measurements = tuple(
            SignalMeasurement(measured_power, event.pulse_width_us)
            for event, measured_power in detected_events
        )
        if false_alarm:
            false_alarm_power = self.sensitivity_dbm + 3.0 * self._sample(
                "false-alarm-power", time_step, band
            )
            false_alarm_width = 0.2 + 2.8 * self._sample(
                "false-alarm-width", time_step, band
            )
            measurements += (
                SignalMeasurement(false_alarm_power, false_alarm_width),
            )
        return DetectionRecord(
            observation=Observation(
                time_step=time_step,
                band=band,
                detections=len(detected_events) + int(false_alarm),
                measurements=measurements,
            ),
            detectable_emitters=tuple(event.emitter_id for event, _ in detectable_events),
            detected_emitters=tuple(event.emitter_id for event, _ in detected_events),
            false_alarm=false_alarm,
        )

    def _noise_db(self, *parts: object) -> float:
        if self.noise_std_db == 0.0:
            return 0.0
        first = max(self._sample("noise-a", *parts), 1e-15)
        second = self._sample("noise-b", *parts)
        standard_normal = sqrt(-2.0 * log(first)) * cos(2.0 * pi * second)
        return standard_normal * self.noise_std_db

    def _sample(self, *parts: object) -> float:
        key = ":".join(str(part) for part in (self.seed, *parts)).encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest())
        return value / 2**64
