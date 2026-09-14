from collections.abc import Callable

from spectra_scheduler.emitters import FrequencyHoppingEmitter, PeriodicEmitter
from spectra_scheduler.metrics import ScanMetrics, calculate_metrics
from spectra_scheduler.schedulers import (
    RandomScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    Scheduler,
)
from spectra_scheduler.simulation import Simulation


def build_demo_scenario() -> Simulation:
    return Simulation(
        num_bands=6,
        duration=60,
        emitters=(
            PeriodicEmitter("search", band=1, period=4, phase=1),
            PeriodicEmitter("tracking", band=4, period=7, phase=2),
            FrequencyHoppingEmitter("agile", bands=(0, 3, 5, 2), period=3),
        ),
    )


def run_demo() -> dict[str, ScanMetrics]:
    simulation = build_demo_scenario()
    scheduler_factories: dict[str, Callable[[], Scheduler]] = {
        "round-robin": RoundRobinScheduler,
        "random": lambda: RandomScheduler(seed=7),
        "revisit-on-hit": RevisitOnHitScheduler,
    }

    return {
        name: calculate_metrics(simulation.run(factory()))
        for name, factory in scheduler_factories.items()
    }


def main() -> None:
    results = run_demo()
    print("Spectra Scheduler - basic simulation")
    print("strategy          detected  intercept ratio  hit rate  mean first delay")
    for name, metrics in results.items():
        print(
            f"{name:<17} "
            f"{metrics.detected_transmissions:>3}/{metrics.total_transmissions:<3} "
            f"{metrics.interception_ratio:>15.1%} "
            f"{metrics.hit_rate:>9.1%} "
            f"{metrics.mean_first_detection_delay:>17.1f}"
        )


if __name__ == "__main__":
    main()
