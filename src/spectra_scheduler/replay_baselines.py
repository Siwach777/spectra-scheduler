"""Observation-only adaptive controls for replay learning comparisons."""

import numpy as np


class RateProbePolicy:
    """Short coverage probes and long exploitation using the latest observed rate.

    This deliberately stronger control can vary both band and dwell without any
    learned weights. It prevents attributing short exploration dwells alone to ML.
    """

    def __init__(self, revisit_us=500_000):
        if not np.isfinite(revisit_us) or revisit_us <= 0:
            raise ValueError("revisit interval must be positive and finite")
        self.revisit_us = revisit_us

    def reset(self, specification, seed):
        receiver, interface = specification["receiver"], specification["interface"]
        self.duration = receiver["stop_us"] - receiver["start_us"]
        self.bands = interface["bands"]
        self.dwells = len(interface["dwell_us"])
        self.short = int(np.argmin(interface["dwell_us"]))
        self.long = int(np.argmax(interface["dwell_us"]))

    def act(self, observation):
        features = observation[:-2].reshape(self.bands, -1)
        ages = features[:, 1] * self.duration
        unseen = features[:, 0] == 0
        if unseen.any():
            band = int(np.argmax(np.where(unseen, ages, -np.inf)))
            dwell = self.short
        elif ages.max() >= self.revisit_us:
            band, dwell = int(ages.argmax()), self.short
        else:
            # log1p(rate) is monotonic; avoid unnecessary exp allocations.
            band, dwell = int(features[:, 3].argmax()), self.long
        return band * self.dwells + dwell
