"""Observation-only context and a frozen hit-prediction scheduling baseline."""

import json
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import exp, isfinite
from pathlib import Path

from spectra_scheduler.models import Observation

FEATURE_NAMES = (
    "band_hit_rate",
    "recent_hit_rate",
    "support",
    "observation_age",
    "hit_age",
    "last_was_hit",
    "global_hit_rate",
    "current_band_hit_rate",
    "tuning_distance",
    "same_band",
    "consecutive_listens",
    "neighbour_hit_rate",
)
HISTORY_STEPS = 24


class ObservationHistory:
    """No absolute time, band ID, scenario ID, seed or simulator truth is a feature."""

    def __init__(self, num_bands: int):
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        self.num_bands = num_bands
        self.visits = [0] * num_bands
        self.hits = [0] * num_bands
        self.last_listen = [-1] * num_bands
        self.last_hit = [-1] * num_bands
        self.recent = [deque(maxlen=8) for _ in range(num_bands)]
        self.global_recent = deque(maxlen=64)
        self.current_band = None
        self.consecutive_listens = 0
        self.last_time = -1

    def rate(self, band: int) -> float:
        return (self.hits[band] + 1) / (self.visits[band] + 2)

    def features(self, time_step: int, band: int) -> tuple[float, ...]:
        if not 0 <= band < self.num_bands or time_step <= self.last_time:
            raise ValueError("features require a valid band and a future observation step")
        recent = [hit for t, hit in self.recent[band] if time_step - t <= HISTORY_STEPS]
        global_hits = [hit for t, hit in self.global_recent if time_step - t <= HISTORY_STEPS]
        neighbours = [b for b in (band - 1, band + 1) if 0 <= b < self.num_bands]
        same = self.current_band == band

        def age(previous: int) -> float:
            return (
                min(time_step - previous, HISTORY_STEPS) / HISTORY_STEPS if previous >= 0 else 1.0
            )

        return (
            self.rate(band),
            (sum(recent) + 1) / (len(recent) + 2),
            min(self.visits[band], 20) / 20,
            age(self.last_listen[band]),
            age(self.last_hit[band]),
            float(self.recent[band][-1][1]) if self.recent[band] else 0.0,
            (sum(global_hits) + 1) / (len(global_hits) + 2),
            self.rate(self.current_band) if self.current_band is not None else 0.5,
            abs(band - self.current_band) / max(1, self.num_bands - 1)
            if self.current_band is not None
            else 0.0,
            float(same),
            min(self.consecutive_listens, 6) / 6 if same else 0.0,
            sum(self.rate(b) for b in neighbours) / len(neighbours) if neighbours else 0.5,
        )

    def update(self, observation: Observation) -> None:
        band = observation.band
        if not 0 <= band < self.num_bands or observation.time_step <= self.last_time:
            raise ValueError("observations must have valid bands and strictly increasing steps")
        if band != self.current_band:
            self.consecutive_listens = 0
        self.current_band = band
        self.last_time = observation.time_step
        if not observation.listening:
            return
        self.consecutive_listens += 1
        self.visits[band] += 1
        self.hits[band] += int(observation.hit)
        self.last_listen[band] = observation.time_step
        if observation.hit:
            self.last_hit[band] = observation.time_step
        self.recent[band].append((observation.time_step, int(observation.hit)))
        self.global_recent.append((observation.time_step, int(observation.hit)))


@dataclass(frozen=True)
class PolicySettings:
    minimum_dwell: int = 2
    maximum_dwell: int = 6
    coverage_steps: int = 18
    switching_penalty: float = 0.08

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.minimum_dwell, self.maximum_dwell, self.coverage_steps)
        ):
            raise ValueError("dwell and coverage settings must be integers")
        if self.minimum_dwell < 1 or self.maximum_dwell < self.minimum_dwell:
            raise ValueError("invalid dwell settings")
        if self.coverage_steps < 1:
            raise ValueError("coverage_steps must be positive")
        if not isfinite(self.switching_penalty) or self.switching_penalty < 0:
            raise ValueError("switching_penalty must be finite and nonnegative")


@dataclass(frozen=True)
class HitModel:
    coefficients: tuple[float, ...]
    intercept: float
    training: dict
    policy: PolicySettings = PolicySettings()

    def __post_init__(self):
        if len(self.coefficients) != len(FEATURE_NAMES):
            raise ValueError("model coefficient count does not match feature schema")
        if not all(isfinite(v) for v in (*self.coefficients, self.intercept)):
            raise ValueError("model coefficients must be finite")

    def predict(self, features: tuple[float, ...]) -> float:
        if len(features) != len(FEATURE_NAMES) or not all(isfinite(v) for v in features):
            raise ValueError("invalid prediction features")
        value = self.intercept + sum(
            w * x for w, x in zip(self.coefficients, features, strict=True)
        )
        # Stable sigmoid without an exponential overflow for large negative logits.
        if value >= 0:
            return 1 / (1 + exp(-value))
        return exp(value) / (1 + exp(value))

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "model_type": "logistic-observed-hit",
            "feature_names": list(FEATURE_NAMES),
            "history_steps": HISTORY_STEPS,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "policy": asdict(self.policy),
            "training": self.training,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        return sha256(payload).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "HitModel":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if (
                data["schema_version"] != 1
                or data["model_type"] != "logistic-observed-hit"
                or data["feature_names"] != list(FEATURE_NAMES)
                or data["history_steps"] != HISTORY_STEPS
            ):
                raise ValueError("unsupported model or feature schema")
            if not isinstance(data["training"], dict):
                raise ValueError("missing training provenance")
            return cls(
                tuple(data["coefficients"]),
                data["intercept"],
                data["training"],
                PolicySettings(**data["policy"]),
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid model artifact: {error}") from error


class LearnedScheduler:
    """A fitted predictor plus explicit dwell/coverage rules, not a full RL agent."""

    def __init__(self, model: HitModel):
        self.model = model
        self.history = None

    def reset(self, num_bands: int) -> None:
        self.history = ObservationHistory(num_bands)
        self.selected = 0
        self.listening_dwell = 0
        self.last_listening = False
        self.predictions = []
        self.targets = []
        self.pending = None

    def choose_band(self, time_step: int) -> int:
        if self.history is None:
            raise RuntimeError("scheduler must be reset before use")
        history, settings = self.history, self.model.policy
        # Complete a retune and minimum listening dwell before selecting another band.
        if self.last_listening and self.listening_dwell >= settings.minimum_dwell:
            ages = [
                time_step - t if t >= 0 else time_step + settings.coverage_steps
                for t in history.last_listen
            ]
            candidates = [b for b in range(history.num_bands) if b != self.selected]
            overdue = [b for b in candidates if ages[b] >= settings.coverage_steps]
            if overdue:
                chosen = max(overdue, key=lambda b: (ages[b], -abs(b - self.selected), -b))
            else:
                if self.listening_dwell < settings.maximum_dwell:
                    candidates.append(self.selected)
                if not candidates:
                    candidates = [self.selected]
                chosen = max(
                    candidates,
                    key=lambda b: (
                        self.model.predict(history.features(time_step, b))
                        - settings.switching_penalty
                        * abs(b - self.selected)
                        / max(1, history.num_bands - 1),
                        ages[b],
                        -b,
                    ),
                )
            if chosen != self.selected:
                self.selected = chosen
                self.listening_dwell = 0
        probability = self.model.predict(history.features(time_step, self.selected))
        self.pending = (time_step, self.selected, probability)
        return self.selected

    def observe(self, observation: Observation) -> None:
        if self.pending is None or self.pending[:2] != (observation.time_step, observation.band):
            raise ValueError("observation does not match the pending decision")
        if observation.listening:
            self.listening_dwell += 1
            self.predictions.append(self.pending[2])
            self.targets.append(int(observation.hit))
        self.last_listening = observation.listening
        self.history.update(observation)
        self.pending = None
