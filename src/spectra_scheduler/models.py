from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Transmission:
    """One transmission event in the simulated spectrum."""

    time_step: int
    band: int
    emitter_id: str

    def __post_init__(self) -> None:
        if self.time_step < 0:
            raise ValueError("time_step cannot be negative")
        if self.band < 0:
            raise ValueError("band cannot be negative")
        if not self.emitter_id:
            raise ValueError("emitter_id cannot be empty")


@dataclass(frozen=True)
class Observation:
    """What the receiver reports after listening to one band."""

    time_step: int
    band: int
    detections: int = 0

    def __post_init__(self) -> None:
        if self.time_step < 0:
            raise ValueError("time_step cannot be negative")
        if self.band < 0:
            raise ValueError("band cannot be negative")
        if self.detections < 0:
            raise ValueError("detections cannot be negative")

    @property
    def hit(self) -> bool:
        return self.detections > 0


@dataclass(frozen=True)
class DetectionRecord:
    """Evaluation details that are not exposed to a scheduler."""

    observation: Observation
    detected_emitters: tuple[str, ...] = ()
    false_alarm: bool = False


@dataclass(frozen=True)
class SimulationResult:
    """Truth and receiver observations from a completed simulation."""

    duration: int
    num_bands: int
    transmissions: tuple[Transmission, ...]
    observations: tuple[Observation, ...]
    detection_records: tuple[DetectionRecord, ...]
