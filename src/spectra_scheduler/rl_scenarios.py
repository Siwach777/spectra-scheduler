"""Versioned procedural worlds; split namespaces separate training and evaluation."""

import json
from dataclasses import asdict
from hashlib import sha256
from random import Random

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.simulation import Simulation

GENERATOR_VERSION = 1


def procedural_scenario(seed: int, split: str, shifted: bool = False, num_bands=None) -> Simulation:
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown scenario split")
    rng = Random(f"spectra-world-v{GENERATOR_VERSION}:{split}:{seed}")
    bands, duration = (8, 180) if shifted else (6, 120)
    if num_bands is not None:
        if type(num_bands) is not int or not 2 <= num_bands <= 32:
            raise ValueError("num_bands must be an integer in 2..32")
        bands = num_bands
    emitters = []
    for i in range(rng.randint(2, 7)):
        name = f"source-{i}"
        period = rng.randint(2, 10)
        common = dict(
            phase=rng.randrange(period),
            power_dbm=rng.uniform(-88, -70),
            pulse_width_us=rng.uniform(0.3, 2.5),
        )
        band = rng.randrange(bands)
        kind = rng.randrange(5)
        if kind == 0:
            emitter = PeriodicEmitter(name, band, period, **common)
        elif kind == 1:
            emitter = FrequencyHoppingEmitter(
                name, tuple(rng.sample(range(bands), rng.randint(2, bands))), period, **common
            )
        elif kind == 2:
            low = rng.randrange(bands - 1)
            emitter = ScanningEmitter(name, low, rng.randrange(low + 1, bands), period, **common)
        elif kind == 3:
            emitter = BurstEmitter(name, band, max(6, period * 2), rng.randint(2, 4), **common)
        else:
            first = PeriodicEmitter(name, band, period, **common)
            second = PeriodicEmitter(
                name, (band + rng.randrange(1, bands)) % bands, rng.randint(2, 10), **common
            )
            emitter = ModeSwitchingEmitter(
                first, second, rng.randrange(duration // 3, 2 * duration // 3)
            )
        if rng.random() < 0.25:
            emitter = WindowedEmitter(
                emitter,
                start_time=rng.randrange(duration // 3),
                end_time=rng.randrange(2 * duration // 3, duration + 1),
            )
        emitters.append(emitter)
    return Simulation(
        bands,
        duration,
        tuple(emitters),
        Receiver(
            detection_probability=rng.uniform(0.65, 0.85) if shifted else rng.uniform(0.8, 0.98),
            false_alarm_probability=0.10 if shifted else rng.uniform(0, 0.04),
            sensitivity_dbm=-90,
            noise_std_db=4 if shifted else rng.uniform(1, 3),
            pulse_width_noise_fraction=0.08 if shifted else 0.05,
            retune_steps=2 if shifted else rng.choice((0, 1)),
            tuning_speed_bands_per_step=2 if shifted else rng.choice((2, 3)),
            seed=rng.randrange(2**31),
        ),
    )


def scenario_fingerprint(simulation: Simulation) -> str:
    return sha256(json.dumps(asdict(simulation), sort_keys=True).encode()).hexdigest()
