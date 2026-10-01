"""Receive-only statistical scan controls with finite-horizon subsidy indices.

Dynamics use a discrete likelihood fit to irregular noisy observations.
These controls do not train neural models or consume emitter identities or truth.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .simulation import SyntheticAction


@lru_cache(maxsize=2048)
def whittle_curve(p01, p11, discount=0.99, points=101, subsidies=64,
                  detection=1.0, false_alarm=0.0):
    """Approximate subsidy thresholds using a 256-step single-arm lookahead.

Each subsidy has a separate backward recursion. Passive reward is zero and
active reward is occupancy minus subsidy, an equivalent reward shift. This
finite-horizon approximation is not an infinite-horizon optimality claim.
"""
    if not (0 < p01 < 1 and 0 < p11 < 1 and 0 < discount < 1):
        raise ValueError("transition probabilities and discount must lie in (0,1)")
    if min(points, subsidies) < 3:
        raise ValueError("index grids must contain at least three points")
    if not 0 <= false_alarm < detection <= 1:
        raise ValueError("observation probabilities must provide informative detection")
    beliefs = np.linspace(0.0, 1.0, points)
    prices = np.linspace(0.0, 1.0, subsidies)
    drift = (1.0 - beliefs) * p01 + beliefs * p11
    hit_probability = false_alarm + (detection - false_alarm) * beliefs
    after_hit = p01 + (p11 - p01) * binary_posterior(
        beliefs, True, detection, false_alarm)
    after_miss = p01 + (p11 - p01) * binary_posterior(
        beliefs, False, detection, false_alarm)
    abstain = np.empty((subsidies, points), dtype=bool)
    for row, price in enumerate(prices):
        continuation = np.zeros(points)
        for _ in range(256):
            listen = detection * beliefs - price + discount * (
                hit_probability * np.interp(after_hit, beliefs, continuation)
                + (1 - hit_probability) * np.interp(after_miss, beliefs, continuation))
            wait = discount * np.interp(drift, beliefs, continuation)
            continuation = np.maximum(listen, wait)
        abstain[row] = wait >= listen
    thresholds = np.ones(points)
    for row in range(subsidies - 1, -1, -1):
        thresholds[abstain[row]] = prices[row]
    indexable = bool(np.all(~abstain[:-1] | abstain[1:]))
    thresholds.setflags(write=False)
    return thresholds, indexable


def binary_posterior(prior, hit, detection=1.0, false_alarm=0.0):
    """Condition occupancy on a declared hit or miss using public receiver errors."""
    busy = detection if hit else 1 - detection
    idle = false_alarm if hit else 1 - false_alarm
    prior = np.asarray(prior, dtype=float)
    numerator = prior * busy
    evidence = numerator + (1 - prior) * idle
    return np.divide(numerator, evidence, out=prior.copy(), where=evidence > 0)


def estimate_dynamics(samples, detection=1.0, false_alarm=0.0):
    """Fit signed persistence with a latent-state likelihood across physical gaps.

Stationary rate uses a smoothed marginal estimate. Conditional on that estimate,
the signed persistence grid is scored by a noisy-observation forward filter.
This is not a joint maximum-likelihood estimate of both chain parameters.
"""
    if not 0 <= false_alarm < detection <= 1:
        raise ValueError("observation probabilities must provide informative detection")
    if len(samples) < 8:
        return 0.02, 0.9
    times, hits = np.asarray(samples, dtype=float).T
    elapsed = np.diff(times)
    valid = elapsed > 0
    if valid.sum() < 2:
        return 0.02, 0.9
    base_rate = np.clip(((hits.sum() + 1.0) / (len(hits) + 2.0) - false_alarm)
                        / (detection - false_alarm), 1e-3, 1 - 1e-3)
    candidates = np.concatenate((np.linspace(-0.99, -0.01, 64), [0.0],
                                 np.exp(-np.geomspace(0.001, 8.0, 64))))
    p01 = base_rate * (1 - candidates)
    p11 = p01 + candidates
    candidates = candidates[(p01 > 0) & (p01 < 1) & (p11 > 0) & (p11 < 1)]
    filtered = np.full(len(candidates), base_rate)
    likelihood = np.zeros(len(candidates))
    gap_powers = {int(gap): candidates ** int(gap) for gap in np.unique(elapsed[valid])}
    for index, hit in enumerate(hits):
        if index:
            gap = int(times[index] - times[index - 1])
            if gap <= 0:
                continue
            filtered = base_rate + (filtered - base_rate) * gap_powers[gap]
        hit_probability = false_alarm + (detection - false_alarm) * filtered
        evidence = hit_probability if hit else 1 - hit_probability
        likelihood += np.log(np.maximum(evidence, 1e-12))
        filtered = binary_posterior(filtered, bool(hit), detection, false_alarm)
    persistence = float(candidates[likelihood.argmax()])
    idle_to_busy = base_rate * (1 - persistence)
    return float(np.clip(idle_to_busy, 1e-4, 1 - 1e-4)), float(
        np.clip(idle_to_busy + persistence, 1e-4, 1 - 1e-4))


@dataclass(frozen=True)
class ScanStrategyConfig:
    dwell: int = 1
    strategy: str = "whittle"
    patience: int = 128
    harvest_weight: float = 1.0
    coverage_weight: float = 1.0
    revisit: int = 0
    refresh: int = 64
    discount: float = 0.995
    ewma_span: int = 64
    seed: int = 0
    belief_mode: str = "beta"

    def __post_init__(self):
        if self.strategy not in ("whittle", "golden", "phased", "adaptive"):
            raise ValueError("unknown scan strategy")
        if self.belief_mode not in ("beta", "markov"):
            raise ValueError("unknown occupancy belief")
        if self.dwell not in (1, 10, 50) or min(self.patience, self.refresh, self.ewma_span) < 1:
            raise ValueError("native dwell and positive horizons required")
        if (self.revisit < 0 or not 0 < self.discount <= 1
                or self.harvest_weight < 0 or self.coverage_weight < 0):
            raise ValueError("invalid learning or coverage settings")


class DiscoverySwitch:
    """Only declared hits and listening ticks determine the phase transition."""

    def __init__(self, config, bands):
        self.config = config
        self.visited = np.zeros(bands, bool)
        self.seen = np.zeros(bands, bool)
        self.discovery = self.harvest = 0.0
        self.last_new = 0
        self.switched_at = None

    def observe(self, observation):
        if not observation.listening:
            return
        band = observation.band
        self.visited[band] = True
        new = bool(observation.hit and not self.seen[band])
        known = bool(observation.hit and self.seen[band])
        alpha = 2 / (self.config.ewma_span + 1)
        self.discovery += alpha * (new - self.discovery)
        self.harvest += alpha * (known - self.harvest)
        if new:
            self.seen[band] = True
            self.last_new = observation.time_step

    def exploring(self, step):
        if self.switched_at is not None:
            return False
        ready = self.visited.all()
        if self.config.strategy == "adaptive":
            switch = self.config.harvest_weight * self.harvest > self.discovery
        else:
            switch = step - self.last_new >= self.config.patience
        if ready and switch:
            self.switched_at = step
            return False
        return True


class WhittleScanScheduler:
    """Whittle indices or golden sweep, optionally with an observable phase switch.

    Actions use native listening dwells. Retuning pays its public duration and
    the shared observable reward's 0.05-per-tick cost.
    """

    def __init__(self, config=None):
        self.config = config or ScanStrategyConfig()

    def set_retune_table(self, table):
        self.retune = np.asarray(table, dtype=np.int64)

    def set_episode_horizon(self, horizon):
        self.horizon = horizon

    def set_observation_probabilities(self, detection, false_alarm):
        if not 0 <= false_alarm < detection <= 1:
            raise ValueError("observation probabilities must provide informative detection")
        self.detection, self.false_alarm = float(detection), float(false_alarm)

    def reset(self, bands):
        self.bands, self.time, self.current = bands, 0, -1
        self.alpha, self.beta = np.ones(bands), np.ones(bands)
        self.detection = getattr(self, "detection", 1.0)
        self.false_alarm = getattr(self, "false_alarm", 0.0)
        self.p01, self.p11 = np.full(bands, 0.02), np.full(bands, 0.9)
        self.belief = self.p01 / (1 + self.p01 - self.p11)
        self.last_listen = np.full(bands, -1, np.int64)
        self.samples = [deque(maxlen=256) for _ in range(bands)]
        self.curves = np.tile(whittle_curve(0.02, 0.9)[0], (bands, 1))
        self.next_refresh = 0
        self.indexability_violations = set()
        self.switch = DiscoverySwitch(self.config, bands)
        self.sequence = 0
        self.offset = float(np.random.default_rng(self.config.seed).random())
        self.last_decision = -1
        self.pending = None

    def golden_action(self, step):
        # Idempotent for repeated inspection of the same pre-action state.
        if step != self.last_decision:
            position = (self.offset + self.sequence * ((5**0.5 - 1) / 2)) % 1
            self.pending = SyntheticAction(min(int(self.bands * position), self.bands - 1),
                                           self.config.dwell)
            self.sequence += 1
            self.last_decision = step
        return self.pending

    def choose_action(self, step):
        if step != self.time:
            raise ValueError("scan policy requires the contiguous observation clock")
        cfg = self.config
        ages = np.where(self.last_listen >= 0, step - self.last_listen, step)
        if cfg.revisit and ages.max() >= cfg.revisit:
            return SyntheticAction(int(ages.argmax()), cfg.dwell)
        if cfg.strategy == "golden" or (
            cfg.strategy in ("phased", "adaptive") and self.switch.exploring(step)
        ):
            return self.golden_action(step)
        if step >= self.next_refresh:
            for band, samples in enumerate(self.samples):
                p01, p11 = estimate_dynamics(samples, self.detection, self.false_alarm)
                # Quantized arguments define the cached curve, avoiding order-dependent cache keys.
                p01, p11 = [float(np.clip(round(p * 50) / 50, 1e-4, 1 - 1e-4))
                            for p in (p01, p11)]
                self.p01[band], self.p11[band] = p01, p11
                markov = cfg.belief_mode == "markov"
                self.curves[band], indexable = whittle_curve(
                    p01, p11, detection=self.detection if markov else 1.0,
                    false_alarm=self.false_alarm if markov else 0.0)
                # Refilter the bounded causal history when estimated dynamics change.
                stationary = p01 / (1 + p01 - p11)
                belief, previous = stationary, None
                for tick, hit in samples:
                    if previous is not None:
                        belief = stationary + (belief - stationary) * (p11 - p01) ** (
                            tick - previous)
                    belief = float(binary_posterior(
                        belief, hit, self.detection, self.false_alarm))
                    previous = tick
                self.belief[band] = (stationary if previous is None else
                    stationary + (belief - stationary) * (p11 - p01) ** (step - previous))
                if not indexable:
                    self.indexability_violations.add(band)
            self.next_refresh = step + cfg.refresh
        occupancy = self.alpha / (self.alpha + self.beta)
        delay = self.retune[self.current] if self.current >= 0 else np.zeros(self.bands)
        if cfg.belief_mode == "markov":
            stationary = self.p01 / (1 + self.p01 - self.p11)
            occupancy = stationary + (self.belief - stationary) * (
                self.p11 - self.p01) ** delay
        grid = np.linspace(0, 1, self.curves.shape[1])
        indices = np.asarray([np.interp(p, grid, curve)
                              for p, curve in zip(occupancy, self.curves, strict=True)])
        values = indices + cfg.coverage_weight * ages / max(self.horizon, 1)
        remaining = self.horizon - step
        listening = np.minimum(cfg.dwell, np.maximum(remaining - delay, 0))
        score = (values * listening - 0.05 * np.minimum(delay, remaining)) / np.maximum(
            np.minimum(delay + cfg.dwell, remaining), 1)
        return SyntheticAction(int(score.argmax()), cfg.dwell)

    def observe(self, observation):
        if observation.time_step != self.time:
            raise ValueError("scan observations must be causal and contiguous")
        cfg = self.config
        self.alpha[:] = 1 + (self.alpha - 1) * cfg.discount
        self.beta[:] = 1 + (self.beta - 1) * cfg.discount
        if observation.listening:
            band, hit = observation.band, bool(observation.hit)
            self.alpha[band] += hit
            self.beta[band] += not hit
            self.samples[band].append((self.time, int(hit)))
            self.last_listen[band] = self.time
            self.belief[band] = binary_posterior(
                self.belief[band], hit, self.detection, self.false_alarm)
        # All arms evolve on every physical tick, including unobserved retune ticks.
        self.belief[:] = self.p01 + (self.p11 - self.p01) * self.belief
        self.switch.observe(observation)
        self.current = observation.band
        self.time += 1
