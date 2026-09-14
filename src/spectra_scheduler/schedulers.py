import random
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
