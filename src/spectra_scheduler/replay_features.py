"""Causal, strategy-independent PDW summaries; no emitter labels or calibrated dBm."""

import numpy as np

FEATURE_NAMES = (
    "visited",
    "age_fraction",
    "hit",
    "log_rate",
    "amplitude_db_scaled",
    "log_width_us",
    "aoa_sin",
    "aoa_cos",
    "overflow_fraction",
)
FEATURE_VERSION = 1


class ReplayFeatures:
    def __init__(self, bands, start_us, stop_us):
        self.bands = bands
        self.start = start_us
        self.duration = stop_us - start_us
        self.last = np.full(bands, start_us, dtype=np.float64)
        self.values = np.zeros((bands, len(FEATURE_NAMES)), dtype=np.float32)
        self.buffer = np.zeros(bands * len(FEATURE_NAMES) + 2, dtype=np.float32)

    def update(self, band, observation):
        x = observation.pulses
        listening = observation.end_us - observation.listening_start_us
        if listening <= 0:
            return
        row = self.values[band]
        row.fill(0)
        row[0] = 1
        row[2] = bool(len(x))
        row[3] = np.log1p(len(x) * 1e6 / listening) / 20
        if len(x):
            row[4] = np.mean(x[:, 4]) / 100
            row[5] = np.mean(np.log1p(x[:, 2])) / 10
            angles = np.deg2rad(x[:, 3])
            row[6:8] = np.mean(np.sin(angles)), np.mean(np.cos(angles))
        total = len(x) + observation.overflow_pulses
        row[8] = observation.overflow_pulses / total if total else 0
        self.last[band] = observation.end_us

    def encode(self, time_us, current_band=None):
        self.values[:, 1] = (time_us - self.last) / self.duration
        self.buffer[:-2] = self.values.ravel()
        self.buffer[-2] = (time_us - self.start) / self.duration
        self.buffer[-1] = -1 if current_band is None else current_band / max(1, self.bands - 1)
        # The caller owns the output: vectorized environments may retain previous states.
        return self.buffer.copy()
