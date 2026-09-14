import random
from collections import deque
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
class ShuffledSweepScheduler:
    """Visit every band once per cycle, in a newly shuffled order."""

    seed: int = 0
    _num_bands: int = field(init=False, default=0)
    _order: list[int] = field(init=False, default_factory=list)
    _position: int = field(init=False, default=0)
    _random: random.Random = field(init=False)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self._num_bands = num_bands
        self._random = random.Random(self.seed)
        self._start_cycle()

    def _start_cycle(self) -> None:
        self._order = list(range(self._num_bands))
        self._random.shuffle(self._order)
        self._position = 0

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        if self._position == self._num_bands:
            self._start_cycle()
        band = self._order[self._position]
        self._position += 1
        return band

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


@dataclass
class SlidingWindowUcbScheduler:
    """Use UCB scores based only on a recent window of observations."""

    window_size: int = 20
    exploration: float = 1.0
    _num_bands: int = field(init=False, default=0)
    _visits: list[int] = field(init=False, default_factory=list)
    _hits: list[int] = field(init=False, default_factory=list)
    _history: deque[tuple[int, int]] = field(init=False, default_factory=deque)
    _selected_band: int | None = field(init=False, default=None)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.exploration < 0:
            raise ValueError("exploration cannot be negative")
        self._num_bands = num_bands
        self._visits = [0] * num_bands
        self._hits = [0] * num_bands
        self._history.clear()
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

        hit = int(observation.hit)
        self._history.append((observation.band, hit))
        self._visits[observation.band] += 1
        self._hits[observation.band] += hit

        if len(self._history) > self.window_size:
            old_band, old_hit = self._history.popleft()
            self._visits[old_band] -= 1
            self._hits[old_band] -= old_hit


@dataclass
class PeriodAwareScheduler:
    """Briefly monitor new signals, then revisit them at the observed interval."""

    probe_steps: int = 6
    max_band_gap: int = 12
    _num_bands: int = field(init=False, default=0)
    _next_sweep_band: int = field(init=False, default=0)
    _selected_band: int | None = field(init=False, default=None)
    _probe_band: int | None = field(init=False, default=None)
    _probe_remaining: int = field(init=False, default=0)
    _hit_times: list[list[int]] = field(init=False, default_factory=list)
    _periods: list[int | None] = field(init=False, default_factory=list)
    _last_visited: list[int] = field(init=False, default_factory=list)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.probe_steps < 0:
            raise ValueError("probe_steps cannot be negative")
        if self.max_band_gap <= 0:
            raise ValueError("max_band_gap must be positive")
        self._num_bands = num_bands
        self._next_sweep_band = 0
        self._selected_band = None
        self._probe_band = None
        self._probe_remaining = 0
        self._hit_times = [[] for _ in range(num_bands)]
        self._periods = [None] * num_bands
        self._last_visited = [-1] * num_bands

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")

        oldest_band = max(
            range(self._num_bands),
            key=lambda band: time_step - self._last_visited[band] - 1,
        )
        oldest_gap = time_step - self._last_visited[oldest_band] - 1
        if oldest_gap >= self.max_band_gap:
            return self._select(oldest_band, time_step)

        if self._probe_band is not None and self._probe_remaining > 0:
            self._probe_remaining -= 1
            return self._select(self._probe_band, time_step)
        self._probe_band = None

        due_bands = [
            band
            for band, period in enumerate(self._periods)
            if period is not None
            and self._hit_times[band]
            and time_step > self._hit_times[band][-1]
            and (time_step - self._hit_times[band][-1]) % period == 0
        ]
        if due_bands:
            return self._select(due_bands[0], time_step)

        selected_band = self._next_sweep_band
        self._next_sweep_band = (self._next_sweep_band + 1) % self._num_bands
        return self._select(selected_band, time_step)

    def _select(self, band: int, time_step: int) -> int:
        self._selected_band = band
        self._last_visited[band] = time_step
        return band

    def observe(self, observation: Observation) -> None:
        if self._selected_band is None:
            raise RuntimeError("choose_band must be called before observe")
        if observation.band != self._selected_band:
            raise ValueError("observation does not match the selected band")
        if not observation.hit:
            return

        hit_times = self._hit_times[observation.band]
        hit_times.append(observation.time_step)
        if len(hit_times) >= 2:
            observed_period = hit_times[-1] - hit_times[-2]
            if observed_period > 0:
                self._periods[observation.band] = observed_period
            if self._probe_band == observation.band:
                self._probe_band = None
                self._probe_remaining = 0
        elif self.probe_steps:
            self._probe_band = observation.band
            self._probe_remaining = self.probe_steps
