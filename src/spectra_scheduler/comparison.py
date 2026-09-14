from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean, pstdev

from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    JitteredPeriodicEmitter,
    PeriodicEmitter,
)
from spectra_scheduler.metrics import ScanMetrics, calculate_metrics
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.schedulers import (
    RandomScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    Scheduler,
    UcbScheduler,
)
from spectra_scheduler.simulation import Simulation


def build_comparison_scenario(seed: int = 0) -> Simulation:
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            PeriodicEmitter("search", band=1, period=4, phase=1),
            PeriodicEmitter("tracking", band=4, period=7, phase=2),
            FrequencyHoppingEmitter("agile", bands=(0, 3, 5, 2), period=3),
            BurstEmitter("burst", band=5, burst_period=12, pulses_per_burst=3, phase=4),
            JitteredPeriodicEmitter(
                "jittered",
                band=2,
                period=6,
                jitter=2,
                seed=seed + 1,
                phase=2,
            ),
        ),
        receiver=Receiver(
            detection_probability=0.85,
            false_alarm_probability=0.05,
            seed=seed + 2,
        ),
    )


def run_comparison(seed: int = 0) -> dict[str, ScanMetrics]:
    simulation = build_comparison_scenario(seed)
    scheduler_factories: dict[str, Callable[[], Scheduler]] = {
        "round-robin": RoundRobinScheduler,
        "random": lambda: RandomScheduler(seed=seed + 3),
        "revisit-on-hit": RevisitOnHitScheduler,
        "ucb-bandit": UcbScheduler,
    }

    return {
        name: calculate_metrics(simulation.run(factory()))
        for name, factory in scheduler_factories.items()
    }


def print_comparison(results: dict[str, ScanMetrics]) -> None:
    print("Spectra Scheduler - basic simulation")
    print("strategy          detected  intercept ratio  P(detect)  false alarms  first delay")
    for name, metrics in results.items():
        print(
            f"{name:<17} "
            f"{metrics.detected_transmissions:>3}/{metrics.total_transmissions:<3} "
            f"{metrics.interception_ratio:>15.1%} "
            f"{metrics.probability_of_detection:>10.1%} "
            f"{metrics.false_alarms:>13} "
            f"{metrics.mean_first_detection_delay:>12.1f}"
        )


@dataclass(frozen=True)
class ComparisonStats:
    runs: int
    mean_interception_ratio: float
    interception_ratio_stddev: float
    mean_probability_of_detection: float
    mean_false_alarms: float
    mean_first_detection_delay: float


def run_repeated_comparison(runs: int, start_seed: int = 0) -> dict[str, ComparisonStats]:
    if runs <= 0:
        raise ValueError("runs must be positive")

    collected: dict[str, list[ScanMetrics]] = {}
    for seed in range(start_seed, start_seed + runs):
        for name, metrics in run_comparison(seed).items():
            collected.setdefault(name, []).append(metrics)

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
            mean_false_alarms=fmean(sample.false_alarms for sample in samples),
            mean_first_detection_delay=fmean(
                sample.mean_first_detection_delay for sample in samples
            ),
        )
    return summaries


def print_repeated_comparison(results: dict[str, ComparisonStats]) -> None:
    runs = next(iter(results.values())).runs if results else 0
    print(f"Spectra Scheduler - {runs} seeded simulation runs")
    print("strategy          intercept ratio      P(detect)  false alarms  first delay")
    for name, stats in results.items():
        print(
            f"{name:<17} "
            f"{stats.mean_interception_ratio:>9.1%} ± "
            f"{stats.interception_ratio_stddev:<7.1%} "
            f"{stats.mean_probability_of_detection:>10.1%} "
            f"{stats.mean_false_alarms:>13.1f} "
            f"{stats.mean_first_detection_delay:>12.1f}"
        )
