"""Bounded-memory, physical-time receiver replay of full-spectrum PDW recordings.

This is an arrival/whole-pulse receiver model, not a waveform simulator. Only
uncensored (stare) recordings support counterfactual scheduling evaluation.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dataset_io import inspect_header, iter_pulses
from .evaluation_contract import TruthOutcome


@dataclass(frozen=True)
class ReplayConfig:
    start_us: float = 0.0
    stop_us: float = 10_000_000.0
    min_frequency_mhz: float = 0.0
    max_frequency_mhz: float = 18_000.0
    bandwidth_mhz: float = 2250.0
    retune_us: float = 100.0
    slew_mhz_per_us: float | None = None
    detection_probability: float = 1.0
    sensitivity_db: float | None = None
    batch_rows: int = 65536
    max_observation_pulses: int = 65536
    seed: int = 0

    def __post_init__(self):
        values = (
            self.start_us,
            self.stop_us,
            self.min_frequency_mhz,
            self.max_frequency_mhz,
            self.bandwidth_mhz,
            self.retune_us,
            self.detection_probability,
        )
        if not np.isfinite(values).all():
            raise ValueError("receiver parameters must be finite")
        if not 0 <= self.start_us < self.stop_us:
            raise ValueError("require 0 <= start_us < stop_us")
        span = self.max_frequency_mhz - self.min_frequency_mhz
        if self.min_frequency_mhz < 0 or not 0 < self.bandwidth_mhz <= span:
            raise ValueError("bandwidth must fit within a positive frequency range")
        if self.retune_us < 0 or not 0 <= self.detection_probability <= 1:
            raise ValueError("invalid retune delay or detection probability")
        if self.slew_mhz_per_us is not None and (
            not np.isfinite(self.slew_mhz_per_us) or self.slew_mhz_per_us <= 0
        ):
            raise ValueError("slew rate must be finite and positive")
        if self.sensitivity_db is not None and not np.isfinite(self.sensitivity_db):
            raise ValueError("sensitivity must be finite")
        for name in ("batch_rows", "max_observation_pulses", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (name != "seed"):
                raise ValueError(f"invalid {name}")
        if self.seed >= 2**64:
            raise ValueError("seed must fit uint64")


@dataclass(frozen=True)
class DwellAction:
    center_frequency_mhz: float
    dwell_us: float


@dataclass(frozen=True)
class PulseObservation:
    start_us: float
    listening_start_us: float
    end_us: float
    center_frequency_mhz: float
    pulses: np.ndarray  # canonical five columns; amplitude is dB, NOT calibrated dBm
    overflow_pulses: int


def _uniform_rows(start: int, count: int, seed: int) -> np.ndarray:
    """Counter-based SplitMix64: same pulse draw for every schedule/chunk size."""
    with np.errstate(over="ignore"):
        x = np.arange(start, start + count, dtype=np.uint64) + np.uint64(seed)
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x ^= x >> 31
    return (x >> 11).astype(np.float64) * (1.0 / 2**53)


class PulseReplay:
    """Interactive step/reset-by-construction environment; use as a context manager.

    An action requests listening dwell *after* retuning. The first tuning is free.
    Windows and passbands are half-open. A pulse must start in the passband and
    listening window and finish by the window end. No partial-pulse detection.
    Observations own their bounded array; later steps never overwrite old data.
    Labels are used only in aggregate evaluation, never returned by step().
    """

    def __init__(self, path: Path, config: ReplayConfig, *, source_mode: str):
        if source_mode != "stare":
            raise ValueError("counterfactual replay requires uncensored stare data")
        self.path, self.config = Path(path), config
        self._before = self.path.stat()
        self._has_labels = inspect_header(self.path).has_labels
        self._stream = iter_pulses(self.path, config.batch_rows)
        self._batch = None
        self._offset = 0
        self._eof = False
        self._closed = False
        self.time_us = config.start_us
        self.center_frequency_mhz = None
        self._buffer = np.empty((config.max_observation_pulses, 5), dtype=np.float64)
        self._truth_first = {}
        self._detected_first = {}
        self._counts = dict(
            steps=0,
            truth_pulses=0,
            detectable_pulses=0,
            intercepted_pulses=0,
            delivered_pulses=0,
            overflow_pulses=0,
        )
        self._listening_us = 0.0
        self._retuning_us = 0.0
        self._last_outcome = None

    def evaluation_outcome(self) -> TruthOutcome:
        """Evaluator-only truth for the last action; never part of PulseObservation."""
        if self._last_outcome is None:
            raise RuntimeError("no completed action to evaluate")
        return self._last_outcome

    @property
    def done(self):
        return self.time_us >= self.config.stop_us

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._stream.close()
            self._closed = True
            after = self.path.stat()
            if (after.st_ino, after.st_size, after.st_mtime_ns) != (
                self._before.st_ino,
                self._before.st_size,
                self._before.st_mtime_ns,
            ):
                raise ValueError("source changed during replay")

    def _ensure_batch(self):
        while not self._eof and (self._batch is None or self._offset == len(self._batch.features)):
            self._batch = next(self._stream, None)
            self._offset = 0
            if self._batch is None:
                self._eof = True
            else:
                x = self._batch.features
                if np.any(x[:, :2] < 0) or np.any(x[:, 2] <= 0):
                    raise ValueError(
                        "arrival/frequency must be nonnegative; width must be positive"
                    )
        return not self._eof

    @staticmethod
    def _first_seen(target, labels, times):
        if labels is not None and len(labels):
            unique, first = np.unique(labels, return_index=True)
            for label, index in zip(unique, first, strict=True):
                target.setdefault(int(label), float(times[index]))

    def step(self, action: DwellAction) -> PulseObservation:
        if self._closed or self.done:
            raise RuntimeError("replay is closed or finished")
        c = self.config
        center, dwell = action.center_frequency_mhz, action.dwell_us
        if not np.isfinite([center, dwell]).all() or dwell <= 0:
            raise ValueError("action must have finite center and positive finite dwell")
        low, high = center - c.bandwidth_mhz / 2, center + c.bandwidth_mhz / 2
        if low < c.min_frequency_mhz or high > c.max_frequency_mhz:
            raise ValueError("receiver passband is outside configured frequency range")
        delay = 0.0
        if self.center_frequency_mhz is not None and center != self.center_frequency_mhz:
            distance = abs(center - self.center_frequency_mhz)
            delay = max(
                c.retune_us, distance / c.slew_mhz_per_us if c.slew_mhz_per_us is not None else 0
            )
        start = self.time_us
        listen = min(start + delay, c.stop_us)
        end = min(listen + dwell, c.stop_us)
        if end <= start:
            raise ValueError("dwell is too small to advance simulation time")
        size = intercepted = 0
        truth_count = eligible_count = detectable_count = 0
        first_intercept = None
        while self._ensure_batch():
            batch = self._batch
            times = batch.features[:, 0]
            left = max(self._offset, int(np.searchsorted(times, start, side="left")))
            right = int(np.searchsorted(times, end, side="left"))
            self._offset = max(left, right)
            x = batch.features[left:right]
            labels = batch.labels[left:right] if batch.labels is not None else None
            if len(x):
                in_range = (x[:, 1] >= c.min_frequency_mhz) & (x[:, 1] < c.max_frequency_mhz)
                detectable = in_range.copy()
                if c.sensitivity_db is not None:
                    detectable &= x[:, 4] >= c.sensitivity_db
                self._counts["truth_pulses"] += int(in_range.sum())
                truth_count += int(in_range.sum())
                self._counts["detectable_pulses"] += int(detectable.sum())
                self._first_seen(
                    self._truth_first,
                    labels[in_range] if labels is not None else None,
                    x[in_range, 0],
                )
                eligible = in_range & (x[:, 0] >= listen) & (x[:, 0] + x[:, 2] <= end)
                eligible &= (x[:, 1] >= low) & (x[:, 1] < high)
                eligible_count += int(eligible.sum())
                mask = detectable & eligible
                detectable_count += int(mask.sum())
                if c.detection_probability < 1:
                    mask &= (
                        _uniform_rows(batch.start + left, len(x), c.seed) < c.detection_probability
                    )
                indices = np.flatnonzero(mask)
                if len(indices) and first_intercept is None:
                    first_intercept = (float(x[indices[0], 0]) - start) / 1_000_000
                intercepted += len(indices)
                take = indices[: c.max_observation_pulses - size]
                self._buffer[size : size + len(take)] = x[take]
                size += len(take)
                # Discovery counts only pulses delivered to the scheduler, not dropped PDWs.
                self._first_seen(
                    self._detected_first, labels[take] if labels is not None else None, x[take, 0]
                )
            if self._offset < len(batch.features):
                break
        self.time_us, self.center_frequency_mhz = end, center
        self._counts["steps"] += 1
        self._counts["intercepted_pulses"] += intercepted
        self._counts["delivered_pulses"] += size
        self._counts["overflow_pulses"] += intercepted - size
        self._listening_us += end - listen
        self._retuning_us += listen - start
        self._last_outcome = TruthOutcome(
            elapsed_seconds=(end - start) / 1_000_000,
            truth_count=truth_count,
            eligible_count=eligible_count,
            detectable_count=detectable_count,
            captured_count=intercepted,
            first_intercept_seconds=first_intercept,
        )
        pulses = self._buffer[:size].copy()
        pulses.flags.writeable = False
        return PulseObservation(start, listen, end, center, pulses, intercepted - size)

    def report(self) -> dict:
        """Evaluation-only aggregate metrics for the elapsed episode, no truth observations."""
        elapsed = self.time_us - self.config.start_us
        n = self._counts
        delays = [time - self._truth_first[label] for label, time in self._detected_first.items()]
        return {
            "schema_version": 1,
            "source": str(self.path),
            "source_mode": "stare",
            "config": asdict(self.config),
            "complete": self.done,
            "simulated_us": elapsed,
            **n,
            "interception_fraction": n["intercepted_pulses"] / n["truth_pulses"]
            if n["truth_pulses"]
            else None,
            "delivery_fraction": n["delivered_pulses"] / n["truth_pulses"]
            if n["truth_pulses"]
            else None,
            "listening_us": self._listening_us,
            "retuning_us": self._retuning_us,
            "listening_fraction": self._listening_us / elapsed if elapsed else None,
            "emitters_present": len(self._truth_first) if self._has_labels else None,
            "emitters_discovered": len(self._detected_first) if self._has_labels else None,
            "discovery_fraction": len(self._detected_first) / len(self._truth_first)
            if self._truth_first
            else None,
            "mean_discovery_delay_us": float(np.mean(delays)) if delays else None,
        }
