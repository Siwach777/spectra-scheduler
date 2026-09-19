"""GPU recurrent PPO training, checkpoints, detached launch, and paired benchmarks."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from spectra_scheduler.learning_cli import write_json
from spectra_scheduler.rl import RewardConfig, RLModel

POLICY_KWARGS = {
    "lstm_hidden_size": 256,
    "n_lstm_layers": 1,
    "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    "shared_lstm": False,
    "enable_critic_lstm": True,
}


def checkpoint_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_checked(path, device="cpu"):
    from sb3_contrib import RecurrentPPO

    from spectra_scheduler.recurrent_env import OBSERVATION_VERSION

    metadata = json.loads(path.with_suffix(".json").read_text())
    if (
        metadata["observation_version"] != OBSERVATION_VERSION
        or metadata["sha256"] != checkpoint_digest(path)
        or metadata["algorithm"] != "recurrent-ppo"
    ):
        raise ValueError("checkpoint hash or observation schema mismatch")
    # SB3 checkpoints contain pickle-based metadata: load only trusted local artifacts.
    return RecurrentPPO.load(path, device=device), metadata


def train_run(args):
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from spectra_scheduler.recurrent_env import OBSERVATION_VERSION, RecurrentScheduler, SpectrumEnv
    from spectra_scheduler.rl import Context
    from spectra_scheduler.rl_scenarios import procedural_scenario

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable here; run outside the sandbox or explicitly use --device cpu"
        )
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / ".train.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ValueError("a training process already owns this run directory") from error
    reward = RewardConfig(coverage=args.coverage_penalty)
    config = {
        "seed": args.seed,
        "workers": args.workers,
        "rollout_steps": 128,
        "batch_size": 512,
        "epochs": 4,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coefficient": 0.01,
        "target_kl": 0.03,
        "policy_kwargs": POLICY_KWARGS,
        "reward": asdict(reward),
    }
    latest = args.run_dir / "latest.zip"
    start_episode, best_score = 0, -float("inf")
    if args.resume:
        model, prior = load_checked(latest, args.device)
        if prior["config"] != config:
            raise ValueError("resume requires the same seed, workers, architecture and reward")
        start_episode = model.num_timesteps // (120 * args.workers) + 2
        best_path = args.run_dir / "best.json"
        if best_path.exists():
            best_score = json.loads(best_path.read_text())["validation_reward_per_step"]
    else:
        if latest.exists() or (args.run_dir / "config.json").exists():
            raise ValueError(
                "run directory already has training artifacts; use --resume or a new directory"
            )
        model = None

    def environment(rank):
        return lambda: Monitor(
            SpectrumEnv(
                worker=rank, workers=args.workers, start_episode=start_episode, reward=reward
            )
        )

    # Start workers before initializing the CUDA model. CPU workers never touch CUDA.
    env = (
        DummyVecEnv([environment(0)])
        if args.workers == 1
        else SubprocVecEnv([environment(i) for i in range(args.workers)], start_method="forkserver")
    )
    try:
        if model is None:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                learning_rate=0.0003,
                n_steps=128,
                batch_size=512,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                target_kl=0.03,
                policy_kwargs=POLICY_KWARGS,
                seed=args.seed,
                device=args.device,
                verbose=0,
            )
        else:
            model.set_env(env)
        model.set_logger(configure(str(args.run_dir), ["csv"]))
        parameter_count = sum(p.numel() for p in model.policy.parameters())
        versions = {"torch": torch.__version__, "numpy": np.__version__}
        from importlib.metadata import version

        versions.update(
            {name: version(name) for name in ("sb3-contrib", "stable-baselines3", "gymnasium")}
        )
        hardware = {
            "device": str(model.device),
            "cpu_workers": args.workers,
            "torch_threads": args.torch_threads,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        }
        write_json(
            args.run_dir / "config.json",
            {
                "config": config,
                "target_steps": args.steps,
                "parameters": parameter_count,
                "versions": versions,
                "hardware": hardware,
            },
        )
        print(
            f"Recurrent PPO: {parameter_count:,} parameters; "
            f"{hardware}; target {args.steps:,} steps",
            flush=True,
        )
        started, initial_steps = perf_counter(), model.num_timesteps

        class Checkpoints(BaseCallback):
            last_saved = model.num_timesteps

            def _on_step(self):
                return True

            def save(self, name, validation_score=None):
                temp = args.run_dir / f".{name}.pending.zip"
                self.model.save(temp)
                target = args.run_dir / f"{name}.zip"
                os.replace(temp, target)
                metadata = {
                    "algorithm": "recurrent-ppo",
                    "observation_version": OBSERVATION_VERSION,
                    "sha256": checkpoint_digest(target),
                    "config": config,
                    "environment_steps": self.model.num_timesteps,
                    "versions": versions,
                    "validation_reward_per_step": validation_score,
                    "checkpoint_selection": "best observed reward on eight fixed validation worlds"
                    if name == "best"
                    else "latest completed PPO update",
                }
                write_json(target.with_suffix(".json"), metadata)

            def validate(self):
                returns = []
                for seed in range(10000, 10008):
                    simulation = procedural_scenario(seed, "validation", num_bands=8)
                    result = simulation.run(RecurrentScheduler(self.model))
                    context = Context(8)
                    total = 0.0
                    for observation in result.observations:
                        context.observe(observation)
                        total += reward.compute(observation, context)
                    returns.append(total / simulation.duration)
                return float(np.mean(returns))

            def _on_rollout_start(self):
                nonlocal best_score
                elapsed = perf_counter() - started
                progress = {
                    "status": "training",
                    "environment_steps": self.model.num_timesteps,
                    "target_steps": args.steps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": (self.model.num_timesteps - initial_steps)
                    / max(elapsed, 1e-9),
                    "parameters": parameter_count,
                    "hardware": hardware,
                }
                if self.model.ep_info_buffer:
                    progress["mean_episode_return"] = float(
                        np.mean([r["r"] for r in self.model.ep_info_buffer])
                    )
                write_json(args.run_dir / "progress.json", progress)
                if self.model.num_timesteps - self.last_saved >= args.checkpoint_steps:
                    self.save("latest")
                    score = self.validate()
                    if score > best_score:
                        best_score = score
                        self.save("best", score)
                    self.last_saved = self.model.num_timesteps
                    print(
                        f"steps {self.model.num_timesteps:,}/{args.steps:,}; "
                        f"{progress['steps_per_second']:.0f} steps/s; "
                        f"validation reward/step {score:.4f}",
                        flush=True,
                    )

        callback = Checkpoints()
        if args.steps <= model.num_timesteps:
            raise ValueError("target steps must exceed checkpoint steps")
        model.learn(
            total_timesteps=args.steps - model.num_timesteps,
            reset_num_timesteps=not args.resume,
            callback=callback,
            log_interval=1,
        )
        callback.save("latest")
        score = callback.validate()
        if score > best_score:
            callback.save("best", score)
        write_json(
            args.run_dir / "progress.json",
            {
                "status": "complete",
                "environment_steps": model.num_timesteps,
                "target_steps": args.steps,
                "elapsed_seconds": perf_counter() - started,
                "hardware": hardware,
                "validation_reward_per_step": score,
            },
        )
        print(f"Training complete: {latest}", flush=True)
    finally:
        env.close()
        lock.close()


def run_benchmark(args):
    import torch
    from sb3_contrib import RecurrentPPO

    from spectra_scheduler.recurrent_env import RecurrentScheduler, SpectrumEnv
    from spectra_scheduler.rl_benchmark import benchmark

    if args.output.resolve() in (args.model.resolve(), args.model.with_suffix(".json").resolve()):
        raise ValueError("report must not overwrite checkpoint or its metadata")
    torch.set_num_threads(1)
    model, metadata = load_checked(args.model, "cpu")
    reward = RewardConfig(**metadata["config"]["reward"])
    initial = RecurrentPPO(
        "MlpLstmPolicy",
        SpectrumEnv(),
        policy_kwargs=POLICY_KWARGS,
        seed=metadata["config"]["seed"],
        device="cpu",
        verbose=0,
    )
    report = benchmark(
        [RLModel.load(path) for path in args.dqn_models],
        runs=args.runs,
        seed=args.seed,
        split=args.split,
        reward=reward,
        num_bands=8,
        extra_policies={
            "recurrent-ppo": lambda: RecurrentScheduler(model),
            "untrained-recurrent": lambda: RecurrentScheduler(initial),
        },
        extra_metadata={"recurrent-ppo": metadata},
        progress=lambda suite, means: print(
            f"{suite}: PPO interception {means['recurrent-ppo']['interception_ratio']:.2%}",
            flush=True,
        ),
    )
    write_json(args.output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "start"):
        p = sub.add_parser(command)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--steps", type=int, default=2_000_000)
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--torch-threads", type=int, default=2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--coverage-penalty", type=float, default=0.2)
        p.add_argument("--checkpoint-steps", type=int, default=50000)
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("benchmark")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dqn-models", type=Path, nargs="*", default=[])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument(
        "--seed", type=int, default=20000, help="separate from checkpoint-selection worlds"
    )
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            if args.output.suffix != ".json":
                raise ValueError("benchmark output must be JSON")
            run_benchmark(args)
        else:
            if min(args.steps, args.workers, args.torch_threads, args.checkpoint_steps) < 1:
                raise ValueError("steps, workers, threads and checkpoint interval must be positive")
            if args.workers > 24 or args.seed < 0:
                raise ValueError("use at most 24 workers and a nonnegative seed")
            RewardConfig(coverage=args.coverage_penalty)
            if args.command == "train":
                train_run(args)
            else:
                args.run_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "spectra_scheduler.recurrent_cli", "train"]
                for key in (
                    "run_dir",
                    "steps",
                    "workers",
                    "torch_threads",
                    "seed",
                    "device",
                    "coverage_penalty",
                    "checkpoint_steps",
                ):
                    command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
                if args.resume:
                    command.append("--resume")
                environment = os.environ.copy()
                environment.update(
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
                )
                with (args.run_dir / "training.log").open("a") as output:
                    process = subprocess.Popen(
                        command,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        env=environment,
                    )
                write_json(args.run_dir / "process.json", {"pid": process.pid, "command": command})
                print(f"Launched PID {process.pid}; log: {args.run_dir / 'training.log'}")
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
