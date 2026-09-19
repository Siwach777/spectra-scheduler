# Reinforcement-learning training

The development objective is now a learned decision policy, without constraints
about making development appear student-like or artificially gradual. Existing
handwritten policies remain comparison baselines, not overrides of the RL actions.

## Two implementations

| Implementation | Purpose | Method |
| --- | --- | --- |
| `rl.py` | Lightweight reference and debugging | NumPy neural Double-DQN, replay, target network, Adam |
| `recurrent_env.py` + `recurrent_cli.py` | Main GPU training path | SB3-Contrib recurrent PPO, separate actor/critic LSTMs |

The first model has a shared 64-unit action-scoring network. The main model uses
256-unit LSTMs and 256/128-unit actor and critic heads. It uses PyTorch automatic
differentiation and the maintained SB3-Contrib optimizer/rollout implementation,
not a locally reimplemented PPO algorithm.

Neither agent has a handwritten minimum dwell, forced sweep, fallback scheduler or
overdue-band override. Repeatedly choosing the same band means dwelling. Changing
bands incurs the simulator's retuning cost, including restart of a retune when the
target changes. These consequences are learned through interaction.

## Observation, action and reward

The recurrent model observes an eight-band array of receiver-history features:
smoothed hit rates, recent feedback, visit support, observation/hit age, current-band
status, relative tuning distance and four recent listening outcomes with ages.
A validity mask distinguishes physical bands from padding on smaller legacy worlds.
The previous step's listening/retuning flag is included; exact remaining retune time
is not exposed. The LSTM maintains additional temporal memory.

Inputs exclude emitter IDs, future events, hidden detection records and scenario
identifiers. This remains a partially observed problem, not a fully observable
Markov state. Hand-designed observation features are still present; decisions are
learned, but the input is not raw RF/IQ data.

Each action selects a physical band. Training uses eight bands, with all eight
actions available on every step. When benchmarking smaller legacy scenarios, only
nonexistent bands are masked. No physically valid action is overridden.

Per-step reward:

```text
observed_hit - 0.05 * retuning - coverage_weight * mean_band_age
```

`mean_band_age` is the mean time since a band was actually listened to, capped at
24 steps and divided by 24; unseen bands age from episode start. An observed hit is
binary and may be a false alarm. The small DQN reference used coverage weight 0.05.
The recurrent run uses 0.20 to place more weight on the coverage weakness revealed
by the reference experiment. These are different training objectives, not a pure
architecture ablation. Benchmarks apply one declared reward to every evaluated
policy and also report truth-based interception, discovery and delay independently.

## GPU environment

The checked host has a Core Ultra 9 275HX (24 logical CPUs), roughly 30 GiB usable
RAM and an RTX 5070 Laptop GPU with about 8 GiB VRAM. GPU access works outside the
coding sandbox; `nvidia-smi` inside the sandbox can fail despite a working driver.
GPU training explicitly errors if CUDA is unavailable rather than silently using CPU.

Training uses a separate `.venv-rl` so CUDA packages do not alter the core project's
dependency lock. Setup from the repository root:

```bash
uv venv --python .venv/bin/python .venv-rl
uv pip install --python .venv-rl/bin/python torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-rl/bin/python -r requirements-rl.txt -e .
.venv-rl/bin/python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Skip environment creation if `.venv-rl` already exists. CUDA libraries require
several gigabytes of downloads and disk space. `requirements-rl.txt` pins the main
training libraries; actual installed versions are recorded in run/checkpoint metadata.

## Start a persistent run

```bash
.venv-rl/bin/python -m spectra_scheduler.recurrent_cli start \
  --run-dir artifacts/recurrent-main --steps 2000000 --workers 12 --device cuda
```

`start` detaches the training process; terminal/chat closure does not intentionally
stop it. The computer must remain awake. GPU access permissions must also apply to
the launched child process. Launch acknowledgement is not proof of successful
startup: check the log and progress file.

The default uses 12 subprocess environments, 128 steps per environment per rollout,
512-sample minibatches and four optimization epochs. CPU math threads are limited
to avoid multiplying a full thread pool across every worker. More workers or a
larger model do not guarantee better throughput or a better policy.

For foreground execution, replace `start` with `train`. For an explicit CPU test,
use `--device cpu --workers 1` and a separate run directory. GPU training is the
intended long-run configuration.

### Progress and artifacts

```bash
tail -f artifacts/recurrent-main/training.log
cat artifacts/recurrent-main/progress.json
nvidia-smi
```

- `process.json`: detached-process PID and command.
- `config.json`: hyperparameters, parameter count, dependencies and device.
- `progress.json`: steps, status, throughput and elapsed duration; no wall-clock stamp.
- `progress.csv`: PPO losses, entropy, KL, episode returns and throughput.
- `latest.zip` + `latest.json`: periodically saved model/optimizer and integrity metadata.
- `best.zip` + `best.json`: best observed reward on eight fixed development worlds.

Checkpoints are taken after completed PPO updates, approximately every 50,000
environment steps. Total steps may exceed the requested target by less than one
rollout. The model and JSON sidecar must be kept together; the loader checks their
SHA-256 match. Only load trusted local SB3 checkpoints: their metadata can contain
pickle serialization. An interrupted replacement between the model and sidecar
can cause a hash mismatch, which is rejected rather than silently loaded.

To stop a detached run, inspect the PID in `process.json`, confirm its command with
`ps`, and send it `SIGTERM`. Do not use broad process-name kill commands. This can
lose work since the last saved checkpoint. `Ctrl+C` in `tail` only closes the viewer.

### Resume

After confirming the old process has stopped:

```bash
.venv-rl/bin/python -m spectra_scheduler.recurrent_cli start \
  --run-dir artifacts/recurrent-main --steps 2000000 --workers 12 --device cuda --resume
```

The step target is the desired total, not an additional budget. Keep seed, worker
count and coverage penalty consistent with the saved configuration. Model and
optimizer state resume, but live environments, LSTM episode state and exact RNG
streams do not; fresh training episodes are started beyond the prior episode range.
This is a training continuation, not a bit-for-bit reproduction of an uninterrupted
run. A per-directory lock blocks concurrent trainers.

## Benchmark a checkpoint

```bash
.venv-rl/bin/python -m spectra_scheduler.recurrent_cli benchmark \
  --model artifacts/recurrent-main/best.zip --runs 30 --seed 20000 \
  --split validation --output reports/generated/recurrent-validation.json
```

The benchmark evaluates the checkpoint, an untrained recurrent-network control and
all thirteen existing policies on identical events and receiver conditions. It
resets LSTM state between episodes. Inference runs on CPU to avoid serial GPU-call
overhead in these small evaluation episodes.

Add `--dqn-models artifacts/rl-seed0.json artifacts/rl-seed1.json
artifacts/rl-seed2.json` to include the lightweight reference. The recurrent benchmark
uses eight-band procedural worlds; the original DQN benchmark used six-band worlds.
Compare models within the same report, not numbers from different configurations.

Checkpoint selection uses validation worlds 10000–10007. The default benchmark
starts at 20000 to provide separate development feedback. Both remain validation:
repeated inspection can influence model development. `--split test` selects a
separate final-evaluation namespace; do not use it for checkpoint selection.

See [rl-benchmark.md](rl-benchmark.md) for metrics, initial reference results and
the benchmark's limitations. No recurrent-PPO performance improvement is claimed
before its training and independent evaluation complete.

## Lightweight reference commands

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m spectra_scheduler.rl_cli train \
  --episodes 1500 --seed 0 --output artifacts/rl-seed0.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m spectra_scheduler.rl_cli benchmark \
  --models artifacts/rl-seed0.json artifacts/rl-seed1.json artifacts/rl-seed2.json \
  --hit-model artifacts/hit-model.json --runs 30 --split validation \
  --output reports/generated/rl-validation.json
```

Train seeds 1 and 2 with corresponding output paths before the second command.
The reference trainer saves only its final artifact, not a resumable optimizer.

## Sources

- [SB3-Contrib recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): maintained LSTM policy, recurrent-state reset and multiprocessing API.
- [PyTorch Blackwell / CUDA 12.8 support](https://pytorch.org/blog/pytorch-2-7/): basis for the installed CUDA build.
- [Double Q-learning](https://arxiv.org/abs/1509.06461): separate action selection and target evaluation in the reference learner.
- [DQN](https://arxiv.org/abs/1312.5602): replay-based neural action-value learning background.
