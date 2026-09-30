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
    SpatialScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.simulation import Simulation

GENERATOR_VERSION = 1
PHYSICAL_GENERATOR_VERSION = 1
PERIODIC_VISIBILITY_GENERATOR_VERSION = 1


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


def physical_scenario(seed: int, split: str, shifted: bool = False) -> Simulation:
    """One-millisecond worlds with required emitter families in every episode.

    This generator is separate from the legacy tick-world distribution so old
    checkpoint provenance and comparisons retain their original meaning.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown scenario split")
    rng = Random(f"spectra-physical-v{PHYSICAL_GENERATOR_VERSION}:{split}:{seed}")
    bands = 8
    duration = rng.randrange(2_000, 3_001) if shifted else rng.randrange(1_500, 2_501)
    required = ("periodic", "frequency-agile", "spatial-scan")
    choices = (*required, "burst", "change", "frequency-scan")
    kinds = [(kind, True) for kind in required]
    kinds.extend((rng.choice(choices), False) for _ in range(rng.randint(0, 4)))
    rng.shuffle(kinds)
    emitters = []
    for i, (kind, mandatory) in enumerate(kinds):
        name = f"source-{i}"
        band = rng.randrange(bands)
        period = rng.randrange(8, 81)
        phase = rng.randrange(period)
        common = dict(power_dbm=rng.uniform(-88, -70), pulse_width_us=rng.uniform(0.3, 2.5))
        if kind == "periodic":
            emitter = PeriodicEmitter(name, band, period, phase=phase, **common)
        elif kind == "frequency-agile":
            emitter = FrequencyHoppingEmitter(
                name,
                tuple(rng.sample(range(bands), rng.randint(2, bands))),
                period,
                phase=phase,
                **common,
            )
        elif kind == "spatial-scan":
            scan_period = rng.choice((80, 160)) if rng.random() < 0.25 else rng.randrange(60, 301)
            visible = rng.randrange(6, min(41, scan_period + 1))
            emitter = SpatialScanningEmitter(
                name,
                band,
                scan_period,
                visible,
                pulse_period=rng.randrange(1, min(visible, 12) + 1),
                phase=rng.randrange(scan_period),
                pulse_phase=rng.randrange(12),
                **common,
            )
        elif kind == "frequency-scan":
            low = rng.randrange(bands - 1)
            emitter = ScanningEmitter(
                name, low, rng.randrange(low + 1, bands), period, phase=phase, **common
            )
        elif kind == "burst":
            pulses = rng.randrange(2, 5)
            emitter = BurstEmitter(
                name, band, max(period, pulses + 1), pulses, phase=phase, **common
            )
        else:
            first = PeriodicEmitter(name, band, period, phase=phase, **common)
            second_period = rng.randrange(8, 81)
            second = PeriodicEmitter(
                name,
                (band + rng.randrange(1, bands)) % bands,
                second_period,
                phase=rng.randrange(second_period),
                **common,
            )
            emitter = ModeSwitchingEmitter(
                first, second, rng.randrange(duration // 3, 2 * duration // 3)
            )
        if not mandatory and rng.random() < 0.2:
            emitter = WindowedEmitter(
                emitter,
                start_time=rng.randrange(duration // 4),
                end_time=rng.randrange(3 * duration // 4, duration + 1),
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
            retune_steps=rng.choice((1, 2)) if shifted else rng.choice((0, 1)),
            tuning_speed_bands_per_step=2 if shifted else rng.choice((2, 3)),
            seed=rng.randrange(2**31),
        ),
    )


def physical_periodic_visibility_scenario(seed: int, split: str) -> Simulation:
    """Focused physical dwell world with varied beam and pulse clocks."""
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown scenario split")
    namespace = f"spectra-periodic-visibility-v{PERIODIC_VISIBILITY_GENERATOR_VERSION}"
    rng = Random(f"{namespace}:{split}:{seed}")
    period = rng.randrange(48, 241)
    visible = rng.randrange(3, min(33, period))
    pulse_period = rng.randrange(1, min(visible + 5, 17))
    return Simulation(
        8,
        1_536,
        (
            SpatialScanningEmitter(
                "periodic-beam",
                rng.randrange(8),
                period,
                visible,
                pulse_period=pulse_period,
                phase=rng.randrange(period),
                pulse_phase=rng.randrange(pulse_period),
                power_dbm=rng.uniform(-87, -73),
            ),
        ),
        Receiver(
            detection_probability=rng.uniform(0.75, 0.95),
            false_alarm_probability=rng.uniform(0.005, 0.04),
            sensitivity_dbm=-90,
            noise_std_db=rng.uniform(1, 4),
            retune_steps=rng.choice((0, 1, 2)),
            tuning_speed_bands_per_step=rng.choice((2, 3)),
            seed=rng.randrange(2**31),
        ),
    )


def scenario_fingerprint(simulation: Simulation) -> str:
    return sha256(json.dumps(asdict(simulation), sort_keys=True).encode()).hexdigest()
