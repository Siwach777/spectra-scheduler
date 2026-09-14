import hashlib
from dataclasses import dataclass

from spectra_scheduler.models import Observation, Transmission


@dataclass(frozen=True)
class Receiver:
    """A narrow-band receiver with simple detection errors."""

    detection_probability: float = 1.0
    false_alarm_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be between 0 and 1")
        if not 0.0 <= self.false_alarm_probability <= 1.0:
            raise ValueError("false_alarm_probability must be between 0 and 1")

    def listen(
        self,
        time_step: int,
        band: int,
        visible_events: list[Transmission],
    ) -> Observation:
        detected_emitters = tuple(
            event.emitter_id
            for event_number, event in enumerate(visible_events)
            if self._sample("detection", time_step, band, event.emitter_id, event_number)
            < self.detection_probability
        )
        false_alarm = not visible_events and (
            self._sample("false-alarm", time_step, band) < self.false_alarm_probability
        )
        return Observation(
            time_step=time_step,
            band=band,
            detected_emitters=detected_emitters,
            false_alarm=false_alarm,
        )

    def _sample(self, *parts: object) -> float:
        key = ":".join(str(part) for part in (self.seed, *parts)).encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest())
        return value / 2**64
