import random

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.simulation import Simulation


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
                ),
                second_mode=PeriodicEmitter(
                    "tracking",
                    band=0,
                    period=4,
                    phase=generator.randrange(4),
                    power_dbm=-84.0,
                ),
                switch_time=30,
            ),
            FrequencyHoppingEmitter(
                "agile",
                bands=(0, 3, 5, 2),
                period=3,
                phase=generator.randrange(3),
                power_dbm=-80.0,
            ),
            ScanningEmitter(
                "scanner",
                lowest_band=1,
                highest_band=4,
                period=2,
                phase=generator.randrange(2),
                power_dbm=-83.0,
            ),
            WindowedEmitter(
                BurstEmitter(
                    "burst",
                    band=5,
                    burst_period=12,
                    pulses_per_burst=3,
                    phase=generator.randrange(12),
                    power_dbm=-89.0,
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
            ),
        ),
        receiver=Receiver(
            detection_probability=0.85,
            false_alarm_probability=0.05,
            sensitivity_dbm=-90.0,
            noise_std_db=3.0,
            retune_steps=1,
            tuning_speed_bands_per_step=2,
            seed=seed + 2,
        ),
    )
