"""Framework-independent reset/step API for band-and-dwell strategies.

Rewards use receiver observations only. Metrics and labels are evaluation-only.
Variable action durations require the returned time discount, not a constant gamma.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .action_contract import DWELL_US
from .pulse_replay import DwellAction, PulseReplay, ReplayConfig
from .replay_features import FEATURE_NAMES, FEATURE_VERSION, ReplayFeatures


@dataclass(frozen=True)
class InterfaceConfig:
    bands: int = 8
    dwell_us: tuple[float, ...] = DWELL_US
    reference_us: float = 10000.0
    gamma: float = 0.99
    pulse_scale: float = 1000.0
    retune_cost: float = 0.1

    def __post_init__(self):
        if not isinstance(self.bands, int) or isinstance(self.bands, bool) or self.bands < 1:
            raise ValueError("bands must be a positive integer")
        if not self.dwell_us or not np.isfinite(self.dwell_us).all() or min(self.dwell_us) <= 0:
            raise ValueError("dwell choices must be positive and finite")
        if len(set(self.dwell_us)) != len(self.dwell_us):
            raise ValueError("dwell choices must be unique")
        if not np.isfinite(
            [self.reference_us, self.gamma, self.pulse_scale, self.retune_cost]
        ).all():
            raise ValueError("reward/discount parameters must be finite")
        if self.reference_us <= 0 or self.pulse_scale <= 0 or self.retune_cost < 0:
            raise ValueError("invalid reward scaling")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    reward: float
    terminated: bool
    discount: float
    elapsed_us: float


def tune_centers(receiver: ReplayConfig, interface: InterfaceConfig) -> np.ndarray:
    """Uniform candidate passbands spanning the receiver range, with optional overlap."""
    span = receiver.max_frequency_mhz - receiver.min_frequency_mhz
    width = receiver.bandwidth_mhz
    if interface.bands == 1:
        if not np.isclose(width, span, rtol=0, atol=1e-8):
            raise ValueError("one tune center requires full-span receiver bandwidth")
        return np.array([(receiver.min_frequency_mhz + receiver.max_frequency_mhz) / 2])
    gap = (span - width) / (interface.bands - 1)
    if gap > width + 1e-8:
        raise ValueError("tune centers must cover the receiver frequency span")
    return np.linspace(
        receiver.min_frequency_mhz + width / 2,
        receiver.max_frequency_mhz - width / 2,
        interface.bands,
        dtype=np.float64,
    )


def validate_specification(saved: dict, current: dict):
    """Strict checkpoint contract; JSON round-trips normalize tuples to lists.

    Receiver seed is episode randomness, not model semantics. All other settings
    must match; experiments with receiver shifts must explicitly validate against
    a separately approved specification rather than bypass this check silently.
    """

    def canonical(spec):
        value = json.loads(json.dumps(spec, allow_nan=False))
        value["receiver"].pop("seed", None)
        return value

    if canonical(saved) != canonical(current):
        raise ValueError("checkpoint replay specification is incompatible with this environment")


class ReplayEnv:
    def __init__(self, path: Path, receiver=None, interface=None):
        receiver = receiver if receiver is not None else ReplayConfig()
        interface = interface if interface is not None else InterfaceConfig()
        self.centers = tune_centers(receiver, interface)
        self.path, self.receiver, self.interface = Path(path), receiver, interface
        self.action_count = interface.bands * len(interface.dwell_us)
        self.observation_size = interface.bands * len(FEATURE_NAMES) + 2
        self._replay = None
        self._features = None

    def specification(self):
        return dict(
            version=1,
            feature_version=FEATURE_VERSION,
            features=list(FEATURE_NAMES),
            observation_size=self.observation_size,
            action_count=self.action_count,
            action_order="band * number_of_dwells + dwell_index",
            receiver=asdict(self.receiver),
            interface=asdict(self.interface),
        )

    def reset(self):
        self.close()
        self._replay = PulseReplay(self.path, self.receiver, source_mode="stare")
        self._features = ReplayFeatures(
            self.interface.bands, self.receiver.start_us, self.receiver.stop_us
        )
        return self._features.encode(self.receiver.start_us)

    def step(self, action: int):
        if self._replay is None:
            raise RuntimeError("call reset before step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer index")
        if not 0 <= action < self.action_count:
            raise ValueError("action out of range")
        cfg = self.interface
        band, dwell_index = divmod(int(action), len(cfg.dwell_us))
        center = self.centers[band]
        obs = self._replay.step(DwellAction(center, cfg.dwell_us[dwell_index]))
        self._features.update(band, obs)
        elapsed = obs.end_us - obs.start_us
        retune = obs.listening_start_us - obs.start_us
        reward = len(obs.pulses) / cfg.pulse_scale - cfg.retune_cost * retune / cfg.reference_us
        done = self._replay.done
        discount = 0.0 if done else cfg.gamma ** (elapsed / cfg.reference_us)
        return Transition(self._features.encode(obs.end_us, band), reward, done, discount, elapsed)

    def metrics(self):
        if self._replay is None:
            raise RuntimeError("call reset before metrics")
        return self._replay.report()

    def evaluation_outcome(self):
        """Evaluator-only last-window truth; excluded from Transition and policy inputs."""
        if self._replay is None:
            raise RuntimeError("call reset before evaluation")
        return self._replay.evaluation_outcome()

    def close(self):
        if self._replay is not None:
            self._replay.close()
            self._replay = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
