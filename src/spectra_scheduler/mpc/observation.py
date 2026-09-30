"""Causal observation encoding and the legacy demonstration reward helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectra_scheduler.models import Observation

from .config import MAX_BANDS, STEP_FEATURE_DIM


class ObservationEncoder:
    """Tracks per-band RF observation statistics and encodes GRU inputs.

    Maintains exponentially-smoothed hit rates, visit/hit ages, signal
    fingerprints and recent outcome history for each frequency band.
    Produces a fixed-size feature vector at every time step suitable for
    the representation network's GRU.
    """

    def __init__(self, num_bands: int) -> None:
        assert 1 <= num_bands <= MAX_BANDS
        self.num_bands = num_bands

        # Per-band statistics --------------------------------------------------
        self.hit_ema = np.full(num_bands, 0.25)  # optimistic prior
        self.ema_decay = 0.92
        self.visit_count = np.zeros(num_bands, dtype=np.int64)
        self.hit_count = np.zeros(num_bands, dtype=np.int64)
        self.last_visit_step = np.full(num_bands, -1, dtype=np.int64)
        self.last_hit_step = np.full(num_bands, -1, dtype=np.int64)
        self.last_was_hit = np.zeros(num_bands, dtype=np.float64)

        # Signal fingerprints (power / pulse-width running means) --------------
        self.mean_power = np.full(num_bands, -70.0)
        self.mean_pw = np.full(num_bands, 1.0)
        self.sig_count = np.zeros(num_bands, dtype=np.int64)

        # Global tracking ------------------------------------------------------
        self.current_band: int = 0
        self.total_steps: int = 0
        self.total_hits: int = 0
        self.total_listens: int = 0

    # ------------------------------------------------------------------

    def update(self, obs: Observation) -> None:
        """Incorporate a new observation into the encoder state."""
        band = obs.band
        self.total_steps += 1

        if not obs.listening:
            return  # retuning step — no information gained

        self.total_listens += 1
        self.visit_count[band] += 1
        self.last_visit_step[band] = obs.time_step
        hit = float(obs.hit)

        if obs.hit:
            self.hit_count[band] += 1
            self.last_hit_step[band] = obs.time_step
            self.last_was_hit[band] = 1.0
            self.total_hits += 1

            # Update signal fingerprint
            if obs.measurements:
                m = obs.measurements[0]
                n = self.sig_count[band]
                self.mean_power[band] = (self.mean_power[band] * n + m.power_dbm) / (n + 1)
                self.mean_pw[band] = (self.mean_pw[band] * n + m.pulse_width_us) / (n + 1)
                self.sig_count[band] = n + 1
        else:
            self.last_was_hit[band] = 0.0

        # Exponential moving average hit rate
        self.hit_ema[band] = self.ema_decay * self.hit_ema[band] + (1.0 - self.ema_decay) * hit
        self.current_band = band

    # ------------------------------------------------------------------

    def encode_step(self, obs: Observation, out=None) -> np.ndarray:
        """Encode a single observation into a fixed-size GRU input vector.

        Layout (29-dim for MAX_BANDS=8)::

            [0..7]   band one-hot
            [8]      hit indicator
            [9]      listening indicator
            [10]     normalised detection count
            [11]     normalised power (shifted & scaled)
            [12]     normalised pulse width
            [13..20] per-band smoothed hit rates (padded to MAX_BANDS)
            [21..28] per-band normalised visit ages (padded to MAX_BANDS)

        The context fields (13-28) reflect encoder state *before* this
        observation is incorporated, which is correct for causal modelling.
        """
        features = np.zeros(STEP_FEATURE_DIM, dtype=np.float64) if out is None else out
        if out is not None:
            features.fill(0)

        # Band one-hot
        features[obs.band] = 1.0

        # Observation features
        features[MAX_BANDS] = float(obs.hit)
        features[MAX_BANDS + 1] = float(obs.listening)
        features[MAX_BANDS + 2] = min(obs.detections, 5) / 5.0
        if obs.measurements:
            m = obs.measurements[0]
            features[MAX_BANDS + 3] = (m.power_dbm + 90.0) / 60.0  # → ~[0, 1]
            features[MAX_BANDS + 4] = min(m.pulse_width_us, 10.0) / 10.0
        # else: zeros (no measurement available)

        # Per-band context summary
        offset_ema = MAX_BANDS + 5
        offset_age = offset_ema + MAX_BANDS
        features[offset_ema : offset_ema + self.num_bands] = self.hit_ema
        for b in range(self.num_bands):
            if self.last_visit_step[b] >= 0:
                age = obs.time_step - self.last_visit_step[b]
            else:
                age = obs.time_step + 1  # never visited
            features[offset_age + b] = min(age, 30) / 30.0

        return features


@dataclass(frozen=True)
class RewardFunction:
    """Multi-objective reward shaping for Neural-MPC training.

    Rewards signal detections, discovery of new signal sources, and
    sustained tracking; penalises lost observation time (retuning)
    and neglected frequency bands.
    """

    hit_reward: float = 1.0
    """Base reward for a successful signal capture / hit."""

    miss_penalty: float = -0.05
    """Small penalty for listening on a band with no activity."""

    retune_penalty: float = -0.15
    """Cost of a retuning step (lost observation opportunity)."""

    discovery_bonus: float = 0.3
    """Extra reward for the first signal detection on a band."""

    tracking_bonus: float = 0.2
    """Reward for consecutive detections on the same band."""

    coverage_scale: float = -0.03
    """Per-band penalty weight when a band exceeds the coverage threshold."""

    coverage_threshold: int = 15
    """Steps before the progressive coverage penalty activates."""

    def compute(self, obs: Observation, encoder: ObservationEncoder) -> float:
        """Compute the scalar reward for a single observation step."""
        reward = 0.0

        if not obs.listening:
            return self.retune_penalty

        if obs.hit:
            reward += self.hit_reward
            # First-ever hit on this band
            if encoder.hit_count[obs.band] <= 1:
                reward += self.discovery_bonus
            # Consecutive hits (tracking continuity)
            if encoder.last_was_hit[obs.band] > 0.5:
                reward += self.tracking_bonus
        else:
            reward += self.miss_penalty

        # Progressive coverage penalty for neglected bands
        for b in range(encoder.num_bands):
            if encoder.last_visit_step[b] >= 0:
                age = obs.time_step - encoder.last_visit_step[b]
            else:
                age = obs.time_step + 1
            if age > self.coverage_threshold:
                excess = (age - self.coverage_threshold) / float(self.coverage_threshold)
                reward += self.coverage_scale * excess

        return reward
