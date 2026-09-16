import random
from collections import deque
from math import log, sqrt
from dataclasses import dataclass, field
from typing import Protocol

from spectra_scheduler.change_detection import BinaryRateChangeDetector
from spectra_scheduler.models import Observation
from spectra_scheduler.tracking import SignalTracker


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
class DwellSweepScheduler:
    """Sweep in order while dwelling on each band for several steps."""

    dwell_steps: int = 2
    start_band: int = 0
    _num_bands: int = field(init=False, default=0)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.dwell_steps <= 0:
            raise ValueError("dwell_steps must be positive")
        if not 0 <= self.start_band < num_bands:
            raise ValueError("start_band must be inside the spectrum")
        self._num_bands = num_bands

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        band_offset = time_step // self.dwell_steps
        return (self.start_band + band_offset) % self._num_bands

    def observe(self, observation: Observation) -> None:
        pass


@dataclass
class AdaptiveDwellScheduler:
    """Sweep all bands, extending the listening dwell after a hit."""

    minimum_dwell_steps: int = 2
    hit_extension_steps: int = 2
    maximum_dwell_steps: int = 6
    start_band: int = 0
    _num_bands: int = field(init=False, default=0)
    _current_band: int = field(init=False, default=0)
    _remaining_dwell: int = field(init=False, default=0)
    _listened_on_band: int = field(init=False, default=0)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.minimum_dwell_steps <= 0:
            raise ValueError("minimum_dwell_steps must be positive")
        if self.hit_extension_steps < 0:
            raise ValueError("hit_extension_steps cannot be negative")
        if self.maximum_dwell_steps < self.minimum_dwell_steps:
            raise ValueError("maximum_dwell_steps cannot be shorter than minimum dwell")
        if not 0 <= self.start_band < num_bands:
            raise ValueError("start_band must be inside the spectrum")

        self._num_bands = num_bands
        self._current_band = self.start_band
        self._remaining_dwell = self.minimum_dwell_steps
        self._listened_on_band = 0

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        if self._remaining_dwell == 0:
            self._current_band = (self._current_band + 1) % self._num_bands
            self._remaining_dwell = self.minimum_dwell_steps
            self._listened_on_band = 0
        return self._current_band

    def observe(self, observation: Observation) -> None:
        if observation.band != self._current_band:
            raise ValueError("observation does not match the selected band")
        if not observation.listening:
            return

        self._remaining_dwell -= 1
        self._listened_on_band += 1
        if observation.hit:
            available_extension = self.maximum_dwell_steps - self._listened_on_band
            self._remaining_dwell = min(
                self._remaining_dwell + self.hit_extension_steps,
                available_extension,
            )


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
        if not observation.listening:
            return
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
        if not observation.listening:
            return

        hit = int(observation.hit)
        self._history.append((observation.band, hit))
        self._visits[observation.band] += 1
        self._hits[observation.band] += hit

        if len(self._history) > self.window_size:
            old_band, old_hit = self._history.popleft()
            self._visits[old_band] -= 1
            self._hits[old_band] -= old_hit


@dataclass
class BayesianBandScheduler:
    """Track a decaying hit-probability belief for each band."""

    prior_hits: float = 1.0
    prior_misses: float = 3.0
    forgetting_factor: float = 0.97
    exploration_bonus: float = 0.15
    switch_penalty: float = 0.02
    max_band_gap: int = 18
    _num_bands: int = field(init=False, default=0)
    _hits: list[float] = field(init=False, default_factory=list)
    _misses: list[float] = field(init=False, default_factory=list)
    _visits: list[int] = field(init=False, default_factory=list)
    _last_visited: list[int] = field(init=False, default_factory=list)
    _selected_band: int | None = field(init=False, default=None)
    _retry_band: int | None = field(init=False, default=None)

    def reset(self, num_bands: int) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.prior_hits <= 0 or self.prior_misses <= 0:
            raise ValueError("belief priors must be positive")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be between zero and one")
        if self.exploration_bonus < 0 or self.switch_penalty < 0:
            raise ValueError("exploration_bonus and switch_penalty cannot be negative")
        if self.max_band_gap <= 0:
            raise ValueError("max_band_gap must be positive")

        self._num_bands = num_bands
        self._hits = [self.prior_hits] * num_bands
        self._misses = [self.prior_misses] * num_bands
        self._visits = [0] * num_bands
        self._last_visited = [-1] * num_bands
        self._selected_band = None
        self._retry_band = None

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")
        if self._retry_band is not None:
            band = self._retry_band
            self._retry_band = None
            self._selected_band = band
            return band

        unvisited = [band for band, visits in enumerate(self._visits) if visits == 0]
        if unvisited:
            if self._selected_band is None:
                selected_band = unvisited[0]
            else:
                selected_band = min(
                    unvisited,
                    key=lambda band: abs(band - self._selected_band),
                )
            self._selected_band = selected_band
            return selected_band

        oldest_band = min(
            range(self._num_bands),
            key=self._last_visited.__getitem__,
        )
        oldest_gap = time_step - self._last_visited[oldest_band] - 1
        if oldest_gap >= self.max_band_gap:
            self._selected_band = oldest_band
            return oldest_band

        scores = [self._band_score(band) for band in range(self._num_bands)]

        self._selected_band = max(range(self._num_bands), key=scores.__getitem__)
        return self._selected_band

    def _band_score(self, band: int) -> float:
        hits = self._hits[band]
        misses = self._misses[band]
        evidence = hits + misses
        probability = hits / evidence
        exploration = self.exploration_bonus / sqrt(evidence)
        switching_cost = (
            self.switch_penalty * abs(band - self._selected_band)
            if self._selected_band is not None
            else 0.0
        )
        return probability + exploration - switching_cost

    def observe(self, observation: Observation) -> None:
        if self._selected_band is None:
            raise RuntimeError("choose_band must be called before observe")
        if observation.band != self._selected_band:
            raise ValueError("observation does not match the selected band")
        if not observation.listening:
            self._retry_band = observation.band
            return

        for band in range(self._num_bands):
            self._hits[band] = self.prior_hits + (
                self._hits[band] - self.prior_hits
            ) * self.forgetting_factor
            self._misses[band] = self.prior_misses + (
                self._misses[band] - self.prior_misses
            ) * self.forgetting_factor

        self._hits[observation.band] += int(observation.hit)
        self._misses[observation.band] += int(not observation.hit)
        self._visits[observation.band] += 1
        self._last_visited[observation.band] = observation.time_step


@dataclass
class TransitionBandScheduler(BayesianBandScheduler):
    """Add recent detected-band transitions to the probability score."""

    transition_prior: float = 1.0
    transition_weight: float = 0.35
    transition_forgetting_factor: float = 0.97
    max_transition_interval: int = 8
    _transitions: list[list[float]] = field(init=False, default_factory=list)
    _last_hit_band: int | None = field(init=False, default=None)
    _last_hit_time: int | None = field(init=False, default=None)

    def reset(self, num_bands: int) -> None:
        if self.transition_prior <= 0:
            raise ValueError("transition_prior must be positive")
        if self.transition_weight < 0:
            raise ValueError("transition_weight cannot be negative")
        if not 0.0 < self.transition_forgetting_factor <= 1.0:
            raise ValueError(
                "transition_forgetting_factor must be between zero and one"
            )
        if self.max_transition_interval <= 0:
            raise ValueError("max_transition_interval must be positive")

        super().reset(num_bands)
        self._transitions = [
            [self.transition_prior] * num_bands for _ in range(num_bands)
        ]
        self._last_hit_band = None
        self._last_hit_time = None

    def _band_score(self, band: int) -> float:
        score = super()._band_score(band)
        if self._last_hit_band is None:
            return score

        transition_row = self._transitions[self._last_hit_band]
        transition_probability = transition_row[band] / sum(transition_row)
        return score + self.transition_weight * transition_probability

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        if not observation.listening:
            return

        for row_index in range(self._num_bands):
            for column_index in range(self._num_bands):
                self._transitions[row_index][column_index] = (
                    self.transition_prior
                    + (
                        self._transitions[row_index][column_index]
                        - self.transition_prior
                    )
                    * self.transition_forgetting_factor
                )

        if not observation.hit:
            return

        if (
            self._last_hit_band is not None
            and self._last_hit_time is not None
            and observation.time_step - self._last_hit_time
            <= self.max_transition_interval
        ):
            self._transitions[self._last_hit_band][observation.band] += (
                observation.detections
            )
        self._last_hit_band = observation.band
        self._last_hit_time = observation.time_step


@dataclass
class ChangeAwareBayesianScheduler(BayesianBandScheduler):
    """Reset a band's stale belief after a sustained hit-rate change."""

    change_reference_window: int = 4
    change_recent_window: int = 3
    minimum_rate_change: float = 0.5
    _change_detector: BinaryRateChangeDetector = field(init=False)
    _detected_change_count: int = field(init=False, default=0)

    def reset(self, num_bands: int) -> None:
        super().reset(num_bands)
        self._change_detector = BinaryRateChangeDetector(
            reference_window_size=self.change_reference_window,
            recent_window_size=self.change_recent_window,
            minimum_rate_change=self.minimum_rate_change,
        )
        self._detected_change_count = 0

    @property
    def detected_change_count(self) -> int:
        return self._detected_change_count

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        if not observation.listening:
            return
        if not self._change_detector.update(observation.band, observation.hit):
            return

        self._hits[observation.band] = self.prior_hits + int(observation.hit)
        self._misses[observation.band] = self.prior_misses + int(not observation.hit)
        self._detected_change_count += 1


@dataclass
class TrackAwareScheduler(AdaptiveDwellScheduler):
    """Use adaptive sweep acquisition, then follow confirmed signal motion."""

    minimum_track_observations: int = 3
    pulse_width_tolerance_us: float = 0.2
    power_tolerance_db: float = 8.0
    track_max_age_steps: int = 10
    reconnect_pulse_width_tolerance_us: float = 0.1
    reconnect_power_tolerance_db: float = 4.0
    reconnect_max_age_steps: int = 30
    max_band_gap: int = 30
    maximum_tracking_misses: int = 2
    _tracker: SignalTracker = field(init=False)
    _retry_band: int | None = field(init=False, default=None)
    _using_track: bool = field(init=False, default=False)
    _last_visited: list[int] = field(init=False, default_factory=list)
    _active_track_id: int | None = field(init=False, default=None)
    _tracking_misses: int = field(init=False, default=0)
    _suppressed_track_ids: set[int] = field(init=False, default_factory=set)

    def reset(self, num_bands: int) -> None:
        if self.minimum_track_observations < 2:
            raise ValueError("minimum_track_observations must be at least two")
        if self.max_band_gap <= 0:
            raise ValueError("max_band_gap must be positive")
        if self.maximum_tracking_misses <= 0:
            raise ValueError("maximum_tracking_misses must be positive")
        super().reset(num_bands)
        self._tracker = SignalTracker(
            pulse_width_tolerance_us=self.pulse_width_tolerance_us,
            power_tolerance_db=self.power_tolerance_db,
            max_age_steps=self.track_max_age_steps,
            reconnect_pulse_width_tolerance_us=(
                self.reconnect_pulse_width_tolerance_us
            ),
            reconnect_power_tolerance_db=self.reconnect_power_tolerance_db,
            reconnect_max_age_steps=self.reconnect_max_age_steps,
        )
        self._retry_band = None
        self._using_track = False
        self._last_visited = [-1] * num_bands
        self._active_track_id = None
        self._tracking_misses = 0
        self._suppressed_track_ids.clear()

    @property
    def track_count(self) -> int:
        return len(self._tracker.tracks)

    def choose_band(self, time_step: int) -> int:
        if self._retry_band is not None:
            self._current_band = self._retry_band
            self._retry_band = None
            return self._current_band

        oldest_band = min(
            range(self._num_bands),
            key=self._last_visited.__getitem__,
        )
        oldest_gap = time_step - self._last_visited[oldest_band] - 1
        if oldest_gap >= self.max_band_gap:
            self._current_band = oldest_band
            self._remaining_dwell = 1
            self._listened_on_band = 0
            self._using_track = False
            self._active_track_id = None
            self._tracking_misses = 0
            return self._current_band

        confirmed_tracks = [
            track
            for track in self._tracker.tracks
            if track.observation_count >= self.minimum_track_observations
            and track.track_id not in self._suppressed_track_ids
        ]
        if confirmed_tracks:
            track = max(
                confirmed_tracks,
                key=lambda candidate: (candidate.last_time, candidate.observation_count),
            )
            if track.track_id != self._active_track_id:
                self._tracking_misses = 0
            self._active_track_id = track.track_id
            self._current_band = track.predicted_band(
                time_step=time_step + 1,
                num_bands=self._num_bands,
            )
            self._using_track = True
            return self._current_band

        if self._using_track:
            self._remaining_dwell = 0
        self._using_track = False
        self._active_track_id = None
        self._tracking_misses = 0
        return super().choose_band(time_step)

    def observe(self, observation: Observation) -> None:
        if observation.band != self._current_band:
            raise ValueError("observation does not match the selected band")
        tracks = self._tracker.update(observation)
        refreshed_track_ids = {
            track.track_id for track in tracks if track.last_time == observation.time_step
        }
        self._suppressed_track_ids.difference_update(refreshed_track_ids)
        if not observation.listening:
            self._retry_band = observation.band
            return
        self._last_visited[observation.band] = observation.time_step
        if self._using_track:
            if self._active_track_id in refreshed_track_ids:
                self._tracking_misses = 0
            else:
                self._tracking_misses += 1
                if self._tracking_misses >= self.maximum_tracking_misses:
                    self._suppressed_track_ids.add(self._active_track_id)
                    self._using_track = False
                    self._active_track_id = None
                    self._remaining_dwell = 0
            return
        super().observe(observation)


@dataclass
class PeriodAwareScheduler:
    """Briefly monitor new signals, then revisit them at the observed interval."""

    probe_steps: int = 6
    max_band_gap: int = 12
    _num_bands: int = field(init=False, default=0)
    _next_sweep_band: int = field(init=False, default=0)
    _selected_band: int | None = field(init=False, default=None)
    _retry_band: int | None = field(init=False, default=None)
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
        self._retry_band = None
        self._probe_band = None
        self._probe_remaining = 0
        self._hit_times = [[] for _ in range(num_bands)]
        self._periods = [None] * num_bands
        self._last_visited = [-1] * num_bands

    def choose_band(self, time_step: int) -> int:
        if self._num_bands == 0:
            raise RuntimeError("scheduler must be reset before use")

        if self._retry_band is not None:
            band = self._retry_band
            self._retry_band = None
            return self._select(band)

        oldest_band = max(
            range(self._num_bands),
            key=lambda band: time_step - self._last_visited[band] - 1,
        )
        oldest_gap = time_step - self._last_visited[oldest_band] - 1
        if oldest_gap >= self.max_band_gap:
            return self._select(oldest_band)

        if self._probe_band is not None and self._probe_remaining > 0:
            self._probe_remaining -= 1
            return self._select(self._probe_band)
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
            return self._select(due_bands[0])

        selected_band = self._next_sweep_band
        self._next_sweep_band = (self._next_sweep_band + 1) % self._num_bands
        return self._select(selected_band)

    def _select(self, band: int) -> int:
        self._selected_band = band
        return band

    def observe(self, observation: Observation) -> None:
        if self._selected_band is None:
            raise RuntimeError("choose_band must be called before observe")
        if observation.band != self._selected_band:
            raise ValueError("observation does not match the selected band")
        if not observation.listening:
            self._retry_band = observation.band
            return

        self._last_visited[observation.band] = observation.time_step
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
