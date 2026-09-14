import random
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from statistics import fmean, pstdev

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    WindowedEmitter,
)
from spectra_scheduler.metrics import ScanMetrics, calculate_metrics
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.schedulers import (
    RandomScheduler,
    PeriodAwareScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    Scheduler,
    ShuffledSweepScheduler,
    SlidingWindowUcbScheduler,
    UcbScheduler,
)
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
            seed=seed + 2,
        ),
    )


def run_comparison(seed: int = 0) -> dict[str, ScanMetrics]:
    simulation = build_comparison_scenario(seed)
    truth = simulation.generate_truth()
    scheduler_factories: dict[str, Callable[[], Scheduler]] = {
        "round-robin": RoundRobinScheduler,
        "random": lambda: RandomScheduler(seed=seed + 3),
        "shuffled-sweep": lambda: ShuffledSweepScheduler(seed=seed + 4),
        "revisit-on-hit": RevisitOnHitScheduler,
        "ucb-bandit": UcbScheduler,
        "sliding-ucb": SlidingWindowUcbScheduler,
        "period-aware": PeriodAwareScheduler,
    }

    return {
        name: calculate_metrics(simulation.run(factory(), truth=truth))
        for name, factory in scheduler_factories.items()
    }


def print_comparison(results: dict[str, ScanMetrics]) -> None:
    print("Spectra Scheduler - basic simulation")
    print(
        "strategy          detected  intercept  discovery  reacquired  "
        "reacq delay  sens loss  max gap  false alarms  first delay"
    )
    for name, metrics in results.items():
        print(
            f"{name:<17} "
            f"{metrics.detected_transmissions:>3}/{metrics.total_transmissions:<3} "
            f"{metrics.interception_ratio:>10.1%} "
            f"{metrics.emitter_discovery_ratio:>10.1%} "
            f"{metrics.reacquired_changes:>4}/{metrics.total_emitter_changes:<4} "
            f"{metrics.mean_reacquisition_delay:>11.1f} "
            f"{metrics.sensitivity_loss_rate:>9.1%} "
            f"{metrics.max_band_gap:>8} "
            f"{metrics.false_alarms:>13} "
            f"{metrics.mean_first_detection_delay:>12.1f}"
        )


@dataclass(frozen=True)
class ComparisonStats:
    runs: int
    mean_interception_ratio: float
    interception_ratio_stddev: float
    mean_probability_of_detection: float
    mean_emitter_discovery_ratio: float
    mean_false_alarms: float
    mean_first_detection_delay: float
    mean_reacquisition_ratio: float
    mean_reacquisition_delay: float
    mean_sensitivity_loss_rate: float
    mean_max_band_gap: float


def run_repeated_comparison(
    runs: int,
    start_seed: int = 0,
    workers: int = 1,
) -> dict[str, ComparisonStats]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    collected: dict[str, list[ScanMetrics]] = {}
    seeds = range(start_seed, start_seed + runs)

    def collect(run_results: Iterable[dict[str, ScanMetrics]]) -> None:
        for result in run_results:
            for name, metrics in result.items():
                collected.setdefault(name, []).append(metrics)

    if workers == 1:
        collect(map(run_comparison, seeds))
    else:
        worker_count = min(workers, runs)
        process_context = (
            get_context("fork") if "fork" in get_all_start_methods() else None
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context,
        ) as executor:
            chunk_size = max(1, runs // (worker_count * 4))
            collect(executor.map(run_comparison, seeds, chunksize=chunk_size))

    summaries: dict[str, ComparisonStats] = {}
    for name, samples in collected.items():
        ratios = [sample.interception_ratio for sample in samples]
        summaries[name] = ComparisonStats(
            runs=runs,
            mean_interception_ratio=fmean(ratios),
            interception_ratio_stddev=pstdev(ratios),
            mean_probability_of_detection=fmean(
                sample.probability_of_detection for sample in samples
            ),
            mean_emitter_discovery_ratio=fmean(
                sample.emitter_discovery_ratio for sample in samples
            ),
            mean_false_alarms=fmean(sample.false_alarms for sample in samples),
            mean_first_detection_delay=fmean(
                sample.mean_first_detection_delay for sample in samples
            ),
            mean_reacquisition_ratio=fmean(
                sample.reacquisition_ratio for sample in samples
            ),
            mean_reacquisition_delay=fmean(
                sample.mean_reacquisition_delay for sample in samples
            ),
            mean_sensitivity_loss_rate=fmean(
                sample.sensitivity_loss_rate for sample in samples
            ),
            mean_max_band_gap=fmean(sample.max_band_gap for sample in samples),
        )
    return summaries


def print_repeated_comparison(results: dict[str, ComparisonStats]) -> None:
    runs = next(iter(results.values())).runs if results else 0
    print(f"Spectra Scheduler - {runs} seeded simulation runs")
    print(
        "strategy          intercept ratio      discovery  reacquire  "
        "reacq delay  sens loss  max gap  false alarms  first delay"
    )
    for name, stats in results.items():
        print(
            f"{name:<17} "
            f"{stats.mean_interception_ratio:>9.1%} ± "
            f"{stats.interception_ratio_stddev:<7.1%} "
            f"{stats.mean_emitter_discovery_ratio:>10.1%} "
            f"{stats.mean_reacquisition_ratio:>10.1%} "
            f"{stats.mean_reacquisition_delay:>11.1f} "
            f"{stats.mean_sensitivity_loss_rate:>9.1%} "
            f"{stats.mean_max_band_gap:>8.1f} "
            f"{stats.mean_false_alarms:>13.1f} "
            f"{stats.mean_first_detection_delay:>12.1f}"
        )
