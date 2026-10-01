"""Simulation runner and state extraction adapter for Spectra Scheduler Web GUI.

Provides clean JSON-serializable payloads for the frontend without modifying
any underlying spectra_scheduler modules.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spectra_scheduler.emitters import (
    BurstEmitter,
)
from spectra_scheduler.evaluation_contract import EvaluationAccumulator, TruthOutcome
from spectra_scheduler.learned_scheduler import HitModel, LearnedScheduler
from spectra_scheduler.metrics import calculate_metrics, calculate_track_metrics
from spectra_scheduler.models import Transmission
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.rl import RLModel, RLScheduler
from spectra_scheduler.scenarios import (
    REQUIREMENT_SCENARIOS,
    build_change_scenario,
    build_crowded_scenario,
    build_requirement_scenario,
    build_scenario,
)
from spectra_scheduler.schedulers import (
    AdaptiveDwellScheduler,
    BayesianBandScheduler,
    ChangeAwareBayesianScheduler,
    DwellSweepScheduler,
    PeriodAwareScheduler,
    RandomScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    ShuffledSweepScheduler,
    SlidingWindowUcbScheduler,
    TrackAwareScheduler,
    TransitionBandScheduler,
    UcbScheduler,
)
from spectra_scheduler.simulation import Simulation, SimulationEpisode, SyntheticAction, configure_scheduler
from spectra_scheduler.tracking import SignalTracker
from web.timing_backend import create_scheduler as create_timing_scheduler
from web.timing_backend import model_status, timing_run_slot

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_rl_model() -> tuple[RLModel, Path]:
    """Require a compatible frozen artifact; never substitute random weights."""
    for name in ("dqn-10k-seed0.json", "rl-seed0.json"):
        path = ARTIFACTS_DIR / name
        if path.exists():
            try:
                return RLModel.load(path), path
            except ValueError:
                continue
    raise ValueError("No compatible DQN artifact is available in artifacts/.")


def create_rl_scheduler() -> RLScheduler:
    model, _ = load_rl_model()
    return RLScheduler(model.network, model.reward)


def create_learned_hit_scheduler() -> Any:
    """Require the frozen supervised model instead of silently changing policy."""
    return LearnedScheduler(HitModel.load(ARTIFACTS_DIR / "hit-model.json"))


SCHEDULER_REGISTRY = {
    "round-robin": (RoundRobinScheduler, "Round robin (1-tick dwell)"),
    "timing-trained": (lambda: create_timing_scheduler(ARTIFACTS_DIR), "Trained timing scheduler"),
    "rl-mlp": (create_rl_scheduler, "DQN MLP · experimental"),
    "random": (RandomScheduler, "Random"),
    "revisit-on-hit": (RevisitOnHitScheduler, "Revisit on hit"),
    "transition-bayesian": (
        lambda: TransitionBandScheduler(transition_weight=0.35),
        "Transition Bayesian",
    ),
    "learned-hit": (create_learned_hit_scheduler, "Learned hit predictor · experimental"),
    "shuffled-sweep": (ShuffledSweepScheduler, "Shuffled sweep"),
    "dwell-sweep-8": (lambda: DwellSweepScheduler(dwell_steps=8), "Fixed sweep (8-tick dwell)"),
    "dwell-sweep-32": (lambda: DwellSweepScheduler(dwell_steps=32), "Fixed sweep (32-tick dwell)"),
    "dwell-sweep-50": (lambda: DwellSweepScheduler(dwell_steps=50), "Fixed sweep (50-tick dwell)"),
    "adaptive-dwell": (
        lambda: AdaptiveDwellScheduler(
            minimum_dwell_steps=4, hit_extension_steps=4, maximum_dwell_steps=16
        ),
        "Adaptive dwell",
    ),
    "bayesian": (
        lambda: BayesianBandScheduler(forgetting_factor=0.95),
        "Bayesian",
    ),
    "change-aware": (
        lambda: ChangeAwareBayesianScheduler(forgetting_factor=0.95),
        "Change-aware Bayesian",
    ),
    "track-aware": (
        lambda: TrackAwareScheduler(minimum_track_observations=3),
        "Track-aware",
    ),
    "ucb": (lambda: UcbScheduler(exploration=1.0), "UCB"),
    "sliding-ucb": (
        lambda: SlidingWindowUcbScheduler(window_size=20, exploration=1.0),
        "Sliding-window UCB",
    ),
    "period-aware": (PeriodAwareScheduler, "Period-aware"),
}

PRESETS = {
    "periodic-timing-video": {
        "id": "periodic-timing-video", "name": "Periodic scenario, seed 42",
        "scenario": "periodic-scan", "seed": 42,
        "baseline": "dwell-sweep-50", "active": "timing-trained",
        "description": "A selected periodic example: trained temporal forecasting versus a fixed-order sweep with 50 listening ticks per band. Both use identical receiver settings.",
    },
    "video-demo": {
        "id": "video-demo",
        "name": "Video demo · adapting to change",
        "scenario": "change",
        "seed": 7,
        "baseline": "bayesian",
        "active": "change-aware",
        "description": (
            "A seeded example of adapting to an emitter changing bands. "
            "Compare Bayesian and change-aware scheduling on identical activity."
        ),
    },
    "trained-timing": {
        "id": "trained-timing", "name": "Trained timing scheduler",
        "scenario": "frequency-agile", "seed": 42,
        "baseline": "dwell-sweep-50", "active": "timing-trained",
        "description": "Compare trained scan timing with a fixed sweep on the same received signals.",
    },
    "agile-hopper": {
        "id": "agile-hopper",
        "name": "Frequency agile",
        "scenario": "frequency-agile",
        "seed": 42,
        "baseline": "dwell-sweep-8",
        "active": "track-aware",
        "description": "Compare fixed and track-aware scanning as emitters change bands.",
    },
    "spatial-radar": {
        "id": "spatial-radar",
        "name": "Spatial scanning",
        "scenario": "spatial-scan",
        "seed": 42,
        "baseline": "dwell-sweep-8",
        "active": "period-aware",
        "description": "Observe short visibility windows from a spatially scanning emitter.",
    },
    "periodic-scan": {
        "id": "periodic-scan",
        "name": "Periodic scanning",
        "scenario": "periodic-scan",
        "seed": 42,
        "baseline": "dwell-sweep-8",
        "active": "period-aware",
        "description": "Compare fixed dwell and period-aware scheduling on recurring signals.",
    },
    "mode-switch": {
        "id": "mode-switch",
        "name": "Behavior change",
        "scenario": "change",
        "seed": 7,
        "baseline": "bayesian",
        "active": "change-aware",
        "description": "Compare Bayesian policies when an emitter changes its operating pattern.",
    },
    "crowded-battlefield": {
        "id": "crowded-battlefield",
        "name": "Crowded spectrum",
        "scenario": "crowded",
        "seed": 100,
        "baseline": "dwell-sweep-8",
        "active": "track-aware",
        "description": "Inspect scan allocation and anonymous tracks with multiple emitters.",
    },
    "neural-mlp": {
        "id": "neural-mlp",
        "name": "Trained DQN baseline",
        "scenario": "crowded",
        "seed": 42,
        "baseline": "dwell-sweep-8",
        "active": "rl-mlp",
        "description": "Compare a frozen experimental DQN with a fixed-dwell sweep on identical truth.",
    },
    "baseline-duel": {
        "id": "baseline-duel",
        "name": "Revisit on hit",
        "scenario": "frequency-agile",
        "seed": 42,
        "baseline": "dwell-sweep-8",
        "active": "revisit-on-hit",
        "description": "Compare two simple baselines: fixed sweep and revisiting a detected band.",
    },
}


def get_available_scenarios() -> list[dict[str, Any]]:
    names = {
        "frequency-agile": "Frequency agile",
        "spatial-scan": "Spatial scanning",
        "periodic-scan": "Periodic scanning",
        "change": "Behavior change",
        "crowded": "Crowded spectrum",
        "mixed": "Mixed emitters",
        "acquisition": "Acquisition",
        "tracking": "Adjacent band tracking",
    }
    scenarios = []
    for key, name in names.items():
        sim = build_scenario_instance(key)
        scenarios.append(
            {
                "id": key,
                "name": name,
                "bands": sim.num_bands,
                "duration": sim.duration,
                "sensitivity_dbm": sim.receiver.sensitivity_dbm,
                "retune_steps": sim.receiver.retune_steps,
            }
        )
    return scenarios


def get_available_schedulers() -> list[dict[str, Any]]:
    schedulers = []
    for key, (_, label) in SCHEDULER_REGISTRY.items():
        item = {"id": key, "name": label, "available": True, "artifact": None}
        try:
            if key == "timing-trained":
                item.update(model_status(ARTIFACTS_DIR))
            elif key == "rl-mlp":
                _, path = load_rl_model()
                item["artifact"] = path.name
            elif key == "learned-hit":
                create_learned_hit_scheduler()
                item["artifact"] = "hit-model.json"
        except (ValueError, OSError) as error:
            item["available"] = False
            item["reason"] = str(error)
        schedulers.append(item)
    return schedulers


def build_scenario_instance(
    name: str,
    seed: int = 0,
    sensitivity_dbm: float | None = None,
    noise_std_db: float | None = None,
    false_alarm_prob: float | None = None,
    retune_steps: int | None = None,
) -> Simulation:
    """Build a simulation instance with optional receiver overrides."""
    if name in REQUIREMENT_SCENARIOS:
        sim = build_requirement_scenario(name, seed=seed)
    elif name == "change":
        sim = build_change_scenario(seed=seed)
    elif name == "crowded":
        sim = build_crowded_scenario(seed=seed)
    else:
        sim = build_scenario(name, seed=seed)

    # Apply receiver parameter overrides if supplied
    rx = sim.receiver
    rx_pfa = false_alarm_prob if false_alarm_prob is not None else rx.false_alarm_probability
    rx_kwargs = {
        "detection_probability": rx.detection_probability,
        "false_alarm_probability": rx_pfa,
        "noise_std_db": noise_std_db if noise_std_db is not None else rx.noise_std_db,
        "sensitivity_dbm": sensitivity_dbm if sensitivity_dbm is not None else rx.sensitivity_dbm,
        "pulse_width_noise_fraction": rx.pulse_width_noise_fraction,
        "retune_steps": retune_steps if retune_steps is not None else rx.retune_steps,
        "tuning_speed_bands_per_step": rx.tuning_speed_bands_per_step,
        "seed": rx.seed,
    }
    custom_rx = Receiver(**rx_kwargs)
    return Simulation(
        num_bands=sim.num_bands,
        duration=sim.duration,
        emitters=sim.emitters,
        receiver=custom_rx,
    )


def extract_cognitive_state(scheduler: Any, num_bands: int, time_step: int) -> dict[str, Any]:
    """Extract interpretable internal belief state without leaking simulator truth."""
    state: dict[str, Any] = {
        "type": scheduler.__class__.__name__,
        "beliefs": [],
        "ucb_scores": [],
        "rate_shift_alerts": [],
        "active_tracks_count": 0,
        "decision_reason": "Policy Selection",
    }
    if hasattr(scheduler, "console_state"):
        state.update(scheduler.console_state)

    # Extract Bayesian beliefs if available
    if hasattr(scheduler, "_hits") and hasattr(scheduler, "_misses") and scheduler._hits:
        for b in range(num_bands):
            a = float(scheduler._hits[b])
            beta_val = float(scheduler._misses[b])
            mean_prob = a / (a + beta_val) if (a + beta_val) > 0 else 0.5
            variance = (a * beta_val) / (((a + beta_val) ** 2) * (a + beta_val + 1))
            state["beliefs"].append(
                {
                    "band": b,
                    "alpha": round(a, 2),
                    "beta": round(beta_val, 2),
                    "mean_probability": round(mean_prob, 3),
                    "uncertainty": round(math.sqrt(max(variance, 0.0)), 3),
                }
            )

    # Extract UCB exploitation vs exploration decomposition
    if isinstance(scheduler, (UcbScheduler, SlidingWindowUcbScheduler)) and scheduler._visits:
        total = sum(scheduler._visits)
        expl_param = getattr(scheduler, "exploration", 1.0)
        for b in range(num_bands):
            v = scheduler._visits[b]
            h = scheduler._hits[b]
            exploit = (h / v) if v > 0 else 0.0
            explore = expl_param * math.sqrt(math.log(max(total, 1)) / v) if v > 0 else 1.0
            state["ucb_scores"].append(
                {
                    "band": b,
                    "exploitation": round(exploit, 3),
                    "exploration": round(explore, 3),
                    "total_score": round(exploit + explore, 3),
                    "visits": int(v),
                }
            )

    # Extract Change-Detector alerts
    if hasattr(scheduler, "_change_detector"):
        change_count = getattr(scheduler, "detected_change_count", 0)
        state["detected_change_count"] = change_count
        if change_count > 0:
            state["rate_shift_alerts"].append(f"Shift count: {change_count}")

    # Extract active tracks from TrackAwareScheduler
    tracker = getattr(scheduler, "_tracker", getattr(scheduler, "tracker", None))
    if tracker is not None and hasattr(tracker, "tracks"):
        state["active_tracks_count"] = len(tracker.tracks)

    # Extract Neural Q-values from RLScheduler (DQN MLP)
    has_state = hasattr(scheduler, "state") and scheduler.state is not None
    if hasattr(scheduler, "network") and has_state:
        try:
            q_vals = scheduler.network.predict(scheduler.state)
            state["q_values"] = [round(float(q), 3) for q in q_vals]
            best_ch = int(q_vals.argmax())
            state["decision_reason"] = (
                f"DQN Neural Value Max (Ch-{best_ch}: Q={q_vals[best_ch]:.2f})"
            )
        except Exception:
            pass
    elif hasattr(scheduler, "_revisit_band") and scheduler._revisit_band is not None:
        state["decision_reason"] = f"Revisit-on-Hit Pursuit (Band {scheduler._revisit_band})"
    elif hasattr(scheduler, "model") and hasattr(scheduler, "selected"):
        state["decision_reason"] = f"Logistic Hit Score Max (Band {scheduler.selected})"
    elif hasattr(scheduler, "_using_track") and scheduler._using_track:
        state["decision_reason"] = "Tracking Confirmed Emitter"
    elif hasattr(scheduler, "_active_track_id") and scheduler._active_track_id is not None:
        state["decision_reason"] = "Following Emitter Kinematics"
    elif hasattr(scheduler, "_listened_on_band") and getattr(
        scheduler, "_listened_on_band", 0
    ) > getattr(scheduler, "minimum_dwell_steps", 2):
        state["decision_reason"] = "Adaptive Dwell Extension"
    elif scheduler.__class__.__name__ == "RoundRobinScheduler":
        state["decision_reason"] = "Fixed Round-Robin Sequential Step"
    elif scheduler.__class__.__name__ == "RandomScheduler":
        state["decision_reason"] = "Stochastic Uniform Dwell"

    return state


def simulate_single_scheduler(
    sim: Simulation,
    scheduler_factory: Any,
    shared_truth: tuple[Transmission, ...] | None = None,
) -> dict[str, Any]:
    """Execute a single scheduler through a SimulationEpisode and extract full step telemetry."""
    episode = SimulationEpisode(sim, truth=shared_truth)
    scheduler = scheduler_factory()
    if hasattr(scheduler, "set_detection_probability"):
        scheduler.set_detection_probability(sim.receiver.detection_probability)
    configure_scheduler(sim, scheduler)

    steps_data = []
    tracker = SignalTracker()
    evaluation = EvaluationAccumulator()
    totals = Counter(event.time_step for event in episode.transmissions)
    decisions = []

    while episode.time_step < sim.duration:
        t = episode.time_step

        macro = hasattr(scheduler, "choose_action")
        action = scheduler.choose_action(t) if macro else scheduler.choose_band(t)
        chosen_band = action.band if macro else action
        predict = getattr(scheduler, "forecast", None)
        forecast = predict(t, action) if predict is not None else None
        observations = episode.step_action(action) if macro else (episode.step(chosen_band),)
        records = episode.records[-len(observations):]
        decisions.append({"tick": t, "band": chosen_band,
                          "dwell_ticks": action.dwell_steps if macro else 1,
                          "elapsed_ticks": len(observations),
                          "forecast": asdict(forecast) if forecast is not None else None})
        for last_obs, record in zip(observations, records, strict=True):
            scheduler.observe(last_obs)
            tracker.update(last_obs)

            # The forecast stays at its pre-action decision boundary.
            cog_state = extract_cognitive_state(scheduler, sim.num_bands, last_obs.time_step)
            measured_power = (
                round(float(last_obs.measurements[0].power_dbm), 1) if last_obs.measurements else None
            )
            measured_pw = (
                round(float(last_obs.measurements[0].pulse_width_us), 2)
                if last_obs.measurements else None
            )

            steps_data.append(
                {
                "tick": last_obs.time_step,
                "rx_band": chosen_band,
                "listening": bool(last_obs.listening),
                "retuning": not bool(last_obs.listening),
                "hit_count": len(record.detected_emitters),
                "has_hit": len(record.detected_emitters) > 0,
                "false_alarms": 1 if record.false_alarm else 0,
                "missed_count": max(
                    len(record.detectable_emitters) - len(record.detected_emitters), 0
                ),
                "measured_power": measured_power,
                "measured_pw": measured_pw,
                "decision_reason": cog_state.get("decision_reason", "Policy selection"),
                "cognitive_state": cog_state,
                "decision_tick": t,
                "dwell_ticks": action.dwell_steps if macro else 1,
                "forecast": asdict(forecast) if forecast is not None else None,
                }
            )
        first = next((record.observation.time_step for record in records if record.detected_emitters), None)
        evaluation.add(TruthOutcome(
            elapsed_seconds=len(observations) * 0.001,
            truth_count=sum(totals[tick] for tick in range(t, episode.time_step)),
            eligible_count=sum(len(episode.events.get((record.observation.time_step, chosen_band), ()))
                               for record in records if record.observation.listening),
            detectable_count=sum(len(record.detectable_emitters) for record in records),
            captured_count=sum(len(record.detected_emitters) for record in records),
            first_intercept_seconds=(first - t) * 0.001 if first is not None else None,
            negative_opportunities=sum(record.observation.listening and not record.detectable_emitters
                                       for record in records),
            false_alarms=sum(record.false_alarm for record in records),
        ), sum(float(obs.hit) - 0.05 * (not obs.listening) for obs in observations), forecast)

    completed_result = episode.result()
    metrics = calculate_metrics(completed_result)
    track_metrics = calculate_track_metrics(completed_result, tracker=tracker)

    # Active tracks telemetry for feature plot
    active_tracks = []
    for trk in tracker.tracks:
        active_tracks.append(
            {
                "track_id": trk.track_id,
                "last_band": trk.last_band,
                "mean_power_dbm": round(float(trk.mean_power_dbm), 1),
                "mean_pw_us": round(float(trk.mean_pulse_width_us), 2),
                "observation_count": trk.observation_count,
                "confirmed": trk.observation_count >= 3,
                "predicted_band": trk.predicted_band(episode.time_step, sim.num_bands),
            }
        )

    # Archival ghosts
    archived_tracks = []
    for trk in getattr(tracker, "_archived_tracks", []):
        archived_tracks.append(
            {
                "track_id": trk.track_id,
                "last_band": trk.last_band,
                "mean_power_dbm": round(float(trk.mean_power_dbm), 1),
                "mean_pw_us": round(float(trk.mean_pulse_width_us), 2),
                "observation_count": trk.observation_count,
            }
        )

    return {
        "steps": steps_data,
        "metrics": {
            **asdict(metrics),
            "average_reward": (
                scheduler.total_reward / sim.duration
                if hasattr(scheduler, "total_reward")
                else None
            ),
        },
        "track_metrics": asdict(track_metrics),
        "active_tracks": active_tracks,
        "archived_tracks": archived_tracks,
        "final_cognitive_state": extract_cognitive_state(
            scheduler, sim.num_bands, episode.time_step
        ),
        "decisions": decisions,
        "evaluation": evaluation.report(),
        "model": getattr(scheduler, "model_provenance", None),
        "runtime": {"decision_count": len(decisions),
                    "inference_and_planning_seconds": getattr(scheduler, "inference_seconds", None)},
    }


def run_simulation(
    scenario_id: str,
    baseline_id: str,
    active_id: str,
    seed: int = 0,
    sensitivity_dbm: float | None = None,
    noise_std_db: float | None = None,
    false_alarm_prob: float | None = None,
    retune_steps: int | None = None,
    perturbation: str | None = None,
) -> dict[str, Any]:
    """Run synchronized head-to-head simulation on identical truth."""
    sim = build_scenario_instance(
        scenario_id,
        seed=seed,
        sensitivity_dbm=sensitivity_dbm,
        noise_std_db=noise_std_db,
        false_alarm_prob=false_alarm_prob,
        retune_steps=retune_steps,
    )

    if scenario_id not in {item["id"] for item in get_available_scenarios()}:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    if baseline_id not in SCHEDULER_REGISTRY or active_id not in SCHEDULER_REGISTRY:
        raise ValueError("Unknown scheduler")
    if perturbation not in (None, "frequency-hop", "popup-threat"):
        raise ValueError("Unknown environment variation")
    if not 0 <= seed <= 99999:
        raise ValueError("Seed must be in 0..99999")

    # Generate identical ground truth for fair comparison
    truth_transmissions = list(sim.generate_truth())

    # Apply the selected synthetic variation before either scheduler runs
    if perturbation == "frequency-hop":
        # Mutate the last 50% of transmissions for agile/switching emitters
        cutoff = sim.duration // 2
        for i, tx in enumerate(truth_transmissions):
            if tx.time_step >= cutoff:
                # Hop to an inverted band
                new_band = (tx.band + sim.num_bands // 2) % sim.num_bands
                truth_transmissions[i] = Transmission(
                    emitter_id=tx.emitter_id,
                    time_step=tx.time_step,
                    band=new_band,
                    power_dbm=tx.power_dbm,
                    pulse_width_us=tx.pulse_width_us,
                )
    elif perturbation == "popup-threat":
        # Inject sudden high-power burst on band 0
        popup = BurstEmitter(
            "additional-emitter", 0, burst_period=6, pulses_per_burst=3, power_dbm=-65.0
        )
        cutoff_tick = sim.duration // 3
        popup_txs = [
            tx
            for tx in popup.transmissions(sim.duration, sim.num_bands)
            if tx.time_step >= cutoff_tick
        ]
        truth_transmissions.extend(popup_txs)

    shared_truth = tuple(sorted(truth_transmissions))

    # Compile 2D truth grid: list of transmissions per tick
    truth_grid: list[list[dict[str, Any]]] = [[] for _ in range(sim.duration)]
    for tx in shared_truth:
        if 0 <= tx.time_step < sim.duration:
            truth_grid[tx.time_step].append(
                {
                    "emitter_id": tx.emitter_id,
                    "band": tx.band,
                    "power_dbm": round(float(tx.power_dbm), 1),
                    "pulse_width_us": round(float(tx.pulse_width_us), 2),
                }
            )

    def factory_for(key):
        factory = SCHEDULER_REGISTRY[key][0]
        if key in ("random", "shuffled-sweep"):
            return lambda: factory(seed=seed)
        return factory

    with timing_run_slot("timing-trained" in (baseline_id, active_id)):
        baseline_res = simulate_single_scheduler(sim, factory_for(baseline_id), shared_truth)
        active_res = simulate_single_scheduler(sim, factory_for(active_id), shared_truth)

    # Compute comparative deltas
    b_inter = baseline_res["metrics"]["interception_ratio"]
    a_inter = active_res["metrics"]["interception_ratio"]
    interception_delta = round((a_inter - b_inter) / b_inter * 100, 1) if b_inter else None

    b_retune = baseline_res["metrics"]["retuning_fraction"]
    a_retune = active_res["metrics"]["retuning_fraction"]
    retune_delta = round((a_retune - b_retune) * 100, 1)

    return {
        "scenario": {
            "id": scenario_id,
            "num_bands": sim.num_bands,
            "duration": sim.duration,
            "seed": seed,
            "total_transmissions": len(shared_truth),
            "sensitivity_dbm": sim.receiver.sensitivity_dbm,
            "retune_steps": sim.receiver.retune_steps,
            "perturbation": perturbation,
        },
        "truth_grid": truth_grid,
        "baseline": {
            "id": baseline_id,
            "name": SCHEDULER_REGISTRY.get(baseline_id, (None, baseline_id))[1],
            "data": baseline_res,
        },
        "active": {
            "id": active_id,
            "name": SCHEDULER_REGISTRY.get(active_id, (None, active_id))[1],
            "data": active_res,
        },
        "deltas": {
            "interception_gain_pct": interception_delta,
            "interception_delta_pp": round((a_inter - b_inter) * 100, 2),
            "retuning_overhead_delta_pct": retune_delta,
            "baseline_interception": b_inter,
            "active_interception": a_inter,
        },
    }
