"""Optional multiscale recurrent inputs derived only from receiver observations.

Observed rising edges require a contiguous listening miss followed by a hit.
Returning to an active band after an observation gap cannot create an onset.
Period estimates describe observed edge intervals, which may span missed cycles.
"""

import numpy as np

from spectra_scheduler.rl import Context


class PhysicalContext(Context):
    """Retain millisecond timing beyond the legacy 24-tick feature saturation."""

    def __init__(self, num_bands, coverage_steps=512):
        super().__init__(num_bands)
        if not np.isfinite(coverage_steps) or coverage_steps <= 0:
            raise ValueError("coverage timescale must be positive and finite")
        self.coverage_steps = float(coverage_steps)
        self.previous_time = np.full(num_bands, -2, dtype=np.int64)
        self.previous_hit = np.zeros(num_bands, dtype=bool)
        self.run_length = np.zeros(num_bands, dtype=np.int64)
        self.onset = np.full(num_bands, -1, dtype=np.int64)
        self.period = np.zeros(num_bands, dtype=np.float64)
        self.intervals = np.zeros(num_bands, dtype=np.int64)
        self.hit_rates = np.full((num_bands, 2), 0.5, dtype=np.float64)
        self.rate_decay = np.exp(-1.0 / np.array((64.0, 512.0)))

    def observe(self, observation):
        super().observe(observation)
        if not observation.listening:
            return
        band, step, hit = observation.band, observation.time_step, observation.hit
        contiguous = self.previous_time[band] + 1 == step
        same = contiguous and self.previous_hit[band] == hit
        self.run_length[band] = self.run_length[band] + 1 if same else 1
        if contiguous and hit and not self.previous_hit[band]:
            if self.onset[band] >= 0:
                interval = step - self.onset[band]
                self.period[band] = (
                    interval if self.intervals[band] == 0
                    else 0.8 * self.period[band] + 0.2 * interval
                )
                self.intervals[band] += 1
            self.onset[band] = step
        self.hit_rates[band] *= self.rate_decay
        self.hit_rates[band] += (1 - self.rate_decay) * float(hit)
        self.previous_time[band], self.previous_hit[band] = step, hit

    def encode(self, step):
        base = super().encode(step)
        listening_age = np.where(
            np.asarray(self.history.last_listen) >= 0,
            step - np.asarray(self.history.last_listen), step + 1,
        )
        hit_age = np.where(
            np.asarray(self.history.last_hit) >= 0,
            step - np.asarray(self.history.last_hit), step + 1,
        )
        onset_age = np.where(self.onset >= 0, step - self.onset, step + 1)
        extra = np.column_stack((
            listening_age / (listening_age + 64),
            listening_age / (listening_age + 512),
            hit_age / (hit_age + 64),
            hit_age / (hit_age + 512),
            self.run_length / (self.run_length + 64),
            self.period / (self.period + 1024),
            onset_age / (onset_age + 1024),
            np.minimum(self.intervals, 4) / 4,
            self.hit_rates[:, 0],
            self.hit_rates[:, 1],
        )).astype(np.float32)
        return np.concatenate((base, extra), axis=1)

    def coverage_debt(self):
        last = np.asarray(self.history.last_listen)
        step = self.history.last_time
        age = np.where(last >= 0, step - last, step + 1)
        return float(np.mean(age / (age + self.coverage_steps)))


def make_context(num_bands, observation_version=1, coverage_steps=512):
    if observation_version == 1:
        return Context(num_bands)
    if observation_version == 2:
        return PhysicalContext(num_bands, coverage_steps)
    raise ValueError("unsupported recurrent observation version")
