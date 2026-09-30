import random

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    SpatialScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.simulation import Simulation

REQUIREMENT_SCENARIOS = ("frequency-agile", "spatial-scan", "periodic-scan")
REQUIREMENT_SCENARIO_VERSION = 2
SCENARIO_NAMES = ("mixed", "acquisition", "tracking", "change", "crowded", *REQUIREMENT_SCENARIOS)


def build_requirement_scenario(name: str, seed: int = 0) -> Simulation:
    """Focused scope scenarios with randomized phases and hidden timing parameters."""
    rng = random.Random(seed)
    if name == "frequency-agile":
        bands = list(range(8))
        rng.shuffle(bands)
        emitters = (
            FrequencyHoppingEmitter(
                "agile", tuple(bands), 3, phase=rng.randrange(3), power_dbm=-75
            ),
        )
    elif name == "spatial-scan":
        # Independent hidden clocks on two seeded, distinct fixed carriers.
        scan_bands = rng.sample(range(8), 2)
        periods = (rng.randrange(19, 32), rng.randrange(33, 48))
        emitters = tuple(
            SpatialScanningEmitter(
                f"beam-{i}",
                band,
                period,
                rng.randrange(3, 7),
                pulse_period=rng.randrange(1, 4),
                phase=rng.randrange(period),
                power_dbm=-75,
            )
            for i, (band, period) in enumerate(zip(scan_bands, periods, strict=True))
        )
    elif name == "periodic-scan":
        # Several near-commensurate revisit periods expose scan-lock failures.
        period = rng.choice((12, 16, 20, 24))
        emitters = (
            SpatialScanningEmitter(
                "periodic-beam",
                rng.randrange(8),
                period,
                rng.randrange(2, 5),
                pulse_period=rng.randrange(1, 4),
                phase=rng.randrange(period),
                power_dbm=-75,
            ),
        )
    else:
        raise ValueError(f"unknown requirement scenario: {name}")
    return Simulation(8, 512, emitters, _focused_receiver(seed))


def build_comparison_scenario(seed: int = 0) -> Simulation:
    generator = random.Random(seed)
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            WindowedEmitter(
                PeriodicEmitter(
                    "search",
                    band=1,
                    period=4,
                    phase=generator.randrange(4),
                    power_dbm=-72.0,
                    pulse_width_us=0.8,
                ),
                end_time=36,
            ),
            ModeSwitchingEmitter(
                first_mode=PeriodicEmitter(
                    "tracking",
                    band=4,
                    period=7,
                    phase=generator.randrange(7),
                    power_dbm=-76.0,
                    pulse_width_us=1.4,
                ),
                second_mode=PeriodicEmitter(
                    "tracking",
                    band=0,
                    period=4,
                    phase=generator.randrange(4),
                    power_dbm=-84.0,
                    pulse_width_us=1.4,
                ),
                switch_time=30,
            ),
            FrequencyHoppingEmitter(
                "agile",
                bands=(0, 3, 5, 2),
                period=3,
                phase=generator.randrange(3),
                power_dbm=-80.0,
                pulse_width_us=0.5,
            ),
            ScanningEmitter(
                "scanner",
                lowest_band=1,
                highest_band=4,
                period=2,
                phase=generator.randrange(2),
                power_dbm=-83.0,
                pulse_width_us=2.0,
            ),
            WindowedEmitter(
                BurstEmitter(
                    "burst",
                    band=5,
                    burst_period=12,
                    pulses_per_burst=3,
                    phase=generator.randrange(12),
                    power_dbm=-89.0,
                    pulse_width_us=0.3,
                ),
                start_time=20,
            ),
            JitteredPeriodicEmitter(
                "jittered",
                band=2,
                period=6,
                jitter=2,
                seed=seed + 1,
                phase=generator.randrange(6),
                power_dbm=-92.0,
                pulse_width_us=1.1,
            ),
        ),
        receiver=Receiver(
            detection_probability=0.85,
            false_alarm_probability=0.05,
            sensitivity_dbm=-90.0,
            noise_std_db=3.0,
            pulse_width_noise_fraction=0.05,
            retune_steps=1,
            tuning_speed_bands_per_step=2,
            seed=seed + 2,
        ),
    )


def build_acquisition_scenario(seed: int = 0) -> Simulation:
    generator = random.Random(seed)
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            PeriodicEmitter(
                "acquisition-a",
                band=0,
                period=5,
                phase=generator.randrange(5),
                power_dbm=-76.0,
                pulse_width_us=0.5,
            ),
            PeriodicEmitter(
                "acquisition-b",
                band=2,
                period=7,
                phase=generator.randrange(7),
                power_dbm=-80.0,
                pulse_width_us=0.9,
            ),
            PeriodicEmitter(
                "acquisition-c",
                band=4,
                period=9,
                phase=generator.randrange(9),
                power_dbm=-84.0,
                pulse_width_us=1.4,
            ),
            WindowedEmitter(
                PeriodicEmitter(
                    "late-arrival",
                    band=5,
                    period=4,
                    phase=generator.randrange(4),
                    power_dbm=-82.0,
                    pulse_width_us=2.0,
                ),
                start_time=20,
            ),
        ),
        receiver=_focused_receiver(seed),
    )


def build_tracking_scenario(seed: int = 0) -> Simulation:
    generator = random.Random(seed)
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            ScanningEmitter(
                "scanner",
                lowest_band=1,
                highest_band=4,
                period=2,
                phase=generator.randrange(2),
                power_dbm=-78.0,
                pulse_width_us=2.0,
            ),
        ),
        receiver=_focused_receiver(seed),
    )


def build_change_scenario(seed: int = 0) -> Simulation:
    generator = random.Random(seed)
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            ModeSwitchingEmitter(
                first_mode=PeriodicEmitter(
                    "changing-emitter",
                    band=1,
                    period=2,
                    phase=generator.randrange(2),
                    power_dbm=-78.0,
                    pulse_width_us=1.4,
                ),
                second_mode=PeriodicEmitter(
                    "changing-emitter",
                    band=4,
                    period=2,
                    phase=generator.randrange(2),
                    power_dbm=-78.0,
                    pulse_width_us=1.4,
                ),
                switch_time=30,
            ),
        ),
        receiver=_focused_receiver(seed),
    )


def build_crowded_scenario(seed: int = 0) -> Simulation:
    """Stress signal association with several deliberately similar emitters."""

    generator = random.Random(seed)
    return Simulation(
        num_bands=8,
        duration=180,
        emitters=(
            ScanningEmitter(
                "scanner-a",
                lowest_band=0,
                highest_band=7,
                period=2,
                phase=generator.randrange(2),
                power_dbm=-77.0,
                pulse_width_us=1.00,
            ),
            ScanningEmitter(
                "scanner-b",
                lowest_band=1,
                highest_band=6,
                period=2,
                phase=generator.randrange(2),
                power_dbm=-81.0,
                pulse_width_us=1.08,
            ),
            FrequencyHoppingEmitter(
                "hopper-a",
                bands=(0, 4, 2, 6),
                period=3,
                phase=generator.randrange(3),
                power_dbm=-82.0,
                pulse_width_us=0.55,
            ),
            FrequencyHoppingEmitter(
                "hopper-b",
                bands=(7, 3, 5, 1),
                period=3,
                phase=generator.randrange(3),
                power_dbm=-85.0,
                pulse_width_us=0.62,
            ),
            PeriodicEmitter(
                "fixed-a",
                band=2,
                period=4,
                phase=generator.randrange(4),
                power_dbm=-75.0,
                pulse_width_us=1.45,
            ),
            PeriodicEmitter(
                "fixed-b",
                band=6,
                period=5,
                phase=generator.randrange(5),
                power_dbm=-79.0,
                pulse_width_us=1.52,
            ),
            WindowedEmitter(
                PeriodicEmitter(
                    "late-arrival",
                    band=4,
                    period=4,
                    phase=generator.randrange(4),
                    power_dbm=-78.0,
                    pulse_width_us=1.48,
                ),
                start_time=60,
            ),
            BurstEmitter(
                "burst",
                band=7,
                burst_period=16,
                pulses_per_burst=3,
                phase=generator.randrange(16),
                power_dbm=-87.0,
                pulse_width_us=0.30,
            ),
        ),
        receiver=Receiver(
            detection_probability=0.9,
            false_alarm_probability=0.03,
            sensitivity_dbm=-90.0,
            noise_std_db=2.5,
            pulse_width_noise_fraction=0.05,
            retune_steps=1,
            tuning_speed_bands_per_step=2,
            seed=seed + 30,
        ),
    )


def build_scenario(name: str, seed: int = 0) -> Simulation:
    if name in REQUIREMENT_SCENARIOS:
        return build_requirement_scenario(name, seed)
    builders = {
        "mixed": build_comparison_scenario,
        "acquisition": build_acquisition_scenario,
        "tracking": build_tracking_scenario,
        "change": build_change_scenario,
        "crowded": build_crowded_scenario,
    }
    try:
        builder = builders[name]
    except KeyError as error:
        available = ", ".join(SCENARIO_NAMES)
        raise ValueError(f"unknown scenario {name!r}; choose from {available}") from error
    return builder(seed)


def _focused_receiver(seed: int) -> Receiver:
    return Receiver(
        detection_probability=0.9,
        false_alarm_probability=0.02,
        sensitivity_dbm=-90.0,
        noise_std_db=2.0,
        pulse_width_noise_fraction=0.05,
        retune_steps=1,
        tuning_speed_bands_per_step=2,
        seed=seed + 20,
    )
