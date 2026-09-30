"""Causal observation summaries for band-and-dwell planning decisions."""

from __future__ import annotations

import math

import numpy as np

from .config import MAX_BANDS, STEP_FEATURE_DIM
from .observation import ObservationEncoder


class MacroObservationEncoder:
    """Update observation history every tick and emit one feature per dwell.

    The final tick supplies band and measurement fields. Per-band history is
    encoded after that tick is incorporated, at the macro decision boundary.
    The hit and listening fields summarize every tick in the completed dwell,
    so interior detections are visible to the recurrent representation.
    """

    def __init__(self, num_bands: int, *, include_elapsed: bool = False) -> None:
        self.encoder = ObservationEncoder(num_bands)
        self.include_elapsed = include_elapsed
        self._scratch = np.empty(STEP_FEATURE_DIM, dtype=np.float32)
        self._last = np.empty(STEP_FEATURE_DIM, dtype=np.float32)
        self._ticks = 0
        self._hits = 0
        self._listens = 0

    def observe(self, observation) -> None:
        if not 0 <= observation.band < self.encoder.num_bands:
            raise ValueError("observation band outside receiver range")
        self.encoder.update(observation)
        self.encoder.encode_step(observation, out=self._scratch)
        np.copyto(self._last, self._scratch)
        self._ticks += 1
        self._hits += int(observation.hit)
        self._listens += int(observation.listening)

    def finish(self) -> np.ndarray:
        if self._ticks < 1:
            raise RuntimeError("cannot finish an empty dwell")
        if self.include_elapsed:
            result = np.empty(STEP_FEATURE_DIM + 1, dtype=np.float32)
            result[:STEP_FEATURE_DIM] = self._last
            # Macro actions have variable physical duration, including retune.
            # Without this field, silent 1/10/50-tick transitions can be
            # observationally identical despite very different time advances.
            result[-1] = min(math.log1p(self._ticks) / math.log1p(64), 1.0)
        else:
            result = self._last.copy()
        result[MAX_BANDS] = self._hits / self._ticks
        result[MAX_BANDS + 1] = self._listens / self._ticks
        self._ticks = self._hits = self._listens = 0
        return result
