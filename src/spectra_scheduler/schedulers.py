import random
from math import log, sqrt
from dataclasses import dataclass, field
from typing import Protocol

from spectra_scheduler.models import Observation


class Scheduler(Protocol):
    def reset(self, num_bands: int) -> None: ...

    def choose_band(self, time_step: int) -> int: ...

    def observe(self, observation: Observation) -> None: ...


@dataclass
class RoundRobinScheduler:
    """Visit each band in order."""

    start_band: int = 0
    _num_bands: int = field(init=False, default=0)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if not 0 <= self.start_band < num_bands:
            raise ValueError("start_band must be inside the spectrum")
        self._num_bands = num_bands

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        return (self.start_band + time_step) % self._num_bands

    def observe(self, observation: Observation) -> None:
        pass


@dataclass
class RandomScheduler:
    """Choose bands uniformly using a repeatable random seed."""

    seed: int = 0
    _num_bands: int = field(init=False, default=0)
    _random: random.Random = field(init=False)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self._num_bands = num_bands
        self._random = random.Random(self.seed)

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        return self._random.randrange(self._num_bands)

    def observe(self, observation: Observation) -> None:
        pass


@dataclass
class RevisitOnHitScheduler:
    """Sweep normally, but revisit a band immediately after detecting something."""

    _num_bands: int = field(init=False, default=0)
    _next_sweep_band: int = field(init=False, default=0)
    _revisit_band: int | None = field(init=False, default=None)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self._num_bands = num_bands
        self._next_sweep_band = 0
        self._revisit_band = None

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        if self._revisit_band is not None:
            band = self._revisit_band
            self._revisit_band = None
            return band

        band = self._next_sweep_band
        self._next_sweep_band = (self._next_sweep_band + 1) % self._num_bands
        return band

    def observe(self, observation: Observation) -> None:
        if observation.hit:
            self._revisit_band = observation.band


@dataclass
class UcbScheduler:
    """Learn a hit rate for each band while continuing to explore."""

    exploration: float = 1.0
    _num_bands: int = field(init=False, default=0)
    _visits: list[int] = field(init=False, default_factory=list)
    _hits: list[int] = field(init=False, default_factory=list)
    _selected_band: int | None = field(init=False, default=None)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.exploration < 0:
            raise ValueError("exploration cannot be negative")
        self._num_bands = num_bands
        self._visits = [0] * num_bands
        self._hits = [0] * num_bands
        self._selected_band = None

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")

        for band, visits in enumerate(self._visits):
            if visits == 0:
                self._selected_band = band
                return band

        total_visits = sum(self._visits)
        scores = [
            hits / visits + self.exploration * sqrt(log(total_visits) / visits)
            for hits, visits in zip(self._hits, self._visits, strict=True)
        ]
        self._selected_band = max(range(self._num_bands), key=scores.__getitem__)
        return self._selected_band

    def observe(self, observation: Observation) -> None:
        if self._selected_band is None:
            raise RuntimeError("choose_band must be called before observe")
        if observation.band != self._selected_band:
            raise ValueError("observation does not match the selected band")
        self._visits[observation.band] += 1
        self._hits[observation.band] += int(observation.hit)
