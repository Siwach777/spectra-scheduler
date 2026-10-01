"""Observation-only adaptive band controls with complete listening dwells."""

import numpy as np

from .schedulers import BayesianBandScheduler, SlidingWindowUcbScheduler, UcbScheduler
from .simulation import SyntheticAction


class ThompsonBandScheduler:
    """Beta-Bernoulli posterior sampling; discount advances through retuning ticks."""

    def __init__(self, seed=0, discount=1.0):
        if not np.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.seed, self.discount = seed, discount

    def reset(self, bands):
        if bands < 1:
            raise ValueError("positive band count required")
        self.alpha, self.beta = np.ones(bands), np.full(bands, 3.0)
        self.rng = np.random.default_rng(self.seed)

    def choose_band(self, time_step):
        return int(self.rng.beta(self.alpha, self.beta).argmax())

    def observe(self, observation):
        self.alpha -= 1
        self.beta -= 3
        self.alpha *= self.discount
        self.beta *= self.discount
        self.alpha += 1
        self.beta += 3
        if observation.listening:
            self.alpha[observation.band] += bool(observation.hit)
            self.beta[observation.band] += not observation.hit


class AdaptiveDwellScheduler:
    """Apply a band learner at macro boundaries; observe every physical tick."""

    def __init__(self, kind, dwell=10, seed=0):
        if type(dwell) is not int or dwell < 1:
            raise ValueError("listening dwell must be a positive integer")
        factories = {
            "ucb": UcbScheduler,
            "sliding-ucb": lambda: SlidingWindowUcbScheduler(window_size=128),
            "bayesian": lambda: BayesianBandScheduler(max_band_gap=128),
            "thompson": lambda: ThompsonBandScheduler(seed),
            "discounted-thompson": lambda: ThompsonBandScheduler(seed, 0.98),
        }
        if kind not in factories:
            raise ValueError("unknown adaptive control")
        self.learner, self.dwell = factories[kind](), dwell

    def reset(self, bands):
        self.learner.reset(bands)

    def choose_action(self, time_step):
        return SyntheticAction(self.learner.choose_band(time_step), self.dwell)

    def observe(self, observation):
        self.learner.observe(observation)
