"""Train-only all-action labels from uncensored passive receiver recordings.

The behavior trajectory supplies causal observations. A separate index reads the
same immutable stare recording to label every possible next action; its future
pulses are never included in the policy input or behavior decision.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..dataset_io import inspect_header, iter_pulses
from ..policy_benchmark import validate_plan
from ..pulse_replay import ReplayConfig, _uniform_rows
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig, ReplayEnv, tune_centers
from ..replay_training import HistoryWindow, UniformActionPolicy


class DenseReplayIndex:
    """Bounded, ordered pulse index for exact one-step counterfactual labels.

    ``max_index_bytes`` limits resident columns before a file is read. A file
    exceeding that budget fails explicitly; it is never sampled or truncated.
    Labels use the same row-stable receiver detection draws as ``PulseReplay``.
    """

    def __init__(
        self,
        path,
        receiver: ReplayConfig,
        interface: InterfaceConfig,
        *,
        max_index_bytes=512 * 1024**2,
    ):
        self.path = Path(path)
        self.receiver, self.interface = receiver, interface
        if type(max_index_bytes) is not int or max_index_bytes < 1:
            raise ValueError("max_index_bytes must be a positive integer")
        before = self.path.stat()
        info = inspect_header(self.path)
        rows = info.rows
        # Preserve the source dtype: PulseReplay evaluates whole-pulse closure
        # in that dtype, including its rounding at the window boundary.
        source_dtype = np.dtype(info.dtype)
        bytes_per_row = 5 * source_dtype.itemsize
        if receiver.detection_probability < 1:
            bytes_per_row += np.dtype(np.float64).itemsize
        required = rows * bytes_per_row
        if required > max_index_bytes:
            raise MemoryError(
                f"dense index requires {required} bytes for {rows} rows; limit is {max_index_bytes}"
            )
        self.pulses = np.empty((rows, 5), dtype=source_dtype)
        offset = 0
        for batch in iter_pulses(self.path, receiver.batch_rows):
            count = len(batch.features)
            self.pulses[offset : offset + count] = batch.features
            offset += count
        after = self.path.stat()
        if offset != rows or (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("source changed while indexing dense labels")
        self.detection_draws = (
            _uniform_rows(0, rows, receiver.seed) if receiver.detection_probability < 1 else None
        )
        self.centers = tune_centers(receiver, interface)
        frequencies = self.pulses[:, 1]
        self._toa = self.pulses[:, 0]
        in_spectrum = (frequencies >= receiver.min_frequency_mhz) & (
            frequencies < receiver.max_frequency_mhz
        )
        self._truth_prefix = np.empty(rows + 1, dtype=np.int64)
        self._truth_prefix[0] = 0
        np.cumsum(in_spectrum, dtype=np.int64, out=self._truth_prefix[1:])
        detectable = in_spectrum
        if receiver.sensitivity_db is not None:
            detectable &= self.pulses[:, 4] >= receiver.sensitivity_db
        if self.detection_draws is not None:
            detectable &= self.detection_draws < receiver.detection_probability
        self._eligible = []
        resident = required + self._truth_prefix.nbytes
        half = receiver.bandwidth_mhz / 2
        for center in self.centers:
            indices = np.flatnonzero(
                detectable & (frequencies >= center - half) & (frequencies < center + half)
            )
            times = self._toa[indices]
            resident += indices.nbytes + times.nbytes
            if resident > max_index_bytes:
                raise MemoryError(
                    f"dense index requires at least {resident} bytes with tune centers; "
                    f"limit is {max_index_bytes}"
                )
            self._eligible.append((indices, times))
        self._bands = np.repeat(np.arange(interface.bands), len(interface.dwell_us))
        self._dwells = np.tile(np.asarray(interface.dwell_us, dtype=np.float64), interface.bands)

    def outcomes(self, start_us, current_band, time_bins=16):
        """Return labels for every band/dwell at one pre-action decision.

        ``current_band`` is the previously tuned band, or ``None`` at reset.
        The truth denominator spans the full action window, including retuning.
        Captures count intercepted pulses before observation-buffer overflow.
        """
        cfg, interface = self.receiver, self.interface
        if not np.isfinite(start_us) or not cfg.start_us <= start_us < cfg.stop_us:
            raise ValueError("start_us must be a finite pre-terminal decision time")
        if current_band is not None and (
            type(current_band) not in (int, np.int64, np.int32)
            or not 0 <= current_band < interface.bands
        ):
            raise ValueError("current_band is outside the interface")
        if type(time_bins) is not int or time_bins < 1:
            raise ValueError("time_bins must be positive")
        bands = self._bands
        delay = np.zeros(len(bands), dtype=np.float64)
        if current_band is not None:
            changing = bands != current_band
            delay[changing] = cfg.retune_us
            if cfg.slew_mhz_per_us is not None:
                distance = abs(self.centers[bands] - self.centers[current_band])
                np.maximum(delay, distance / cfg.slew_mhz_per_us, out=delay)
                delay[~changing] = 0
        listen = np.minimum(start_us + delay, cfg.stop_us)
        end = np.minimum(listen + self._dwells, cfg.stop_us)
        elapsed = end - start_us
        if np.any(elapsed <= 0):
            raise ValueError("counterfactual action does not advance time")

        pulses = self.pulses
        toa = self._toa
        window_bounds = np.searchsorted(toa, np.concatenate(([start_us], end)), side="left")
        truth = (
            self._truth_prefix[window_bounds[1:]] - self._truth_prefix[window_bounds[0]]
        ).astype(np.int32)
        captured = np.zeros(len(bands), dtype=np.int32)
        time_class = np.full(len(bands), time_bins, dtype=np.int64)
        first_intercept_us = np.full(len(bands), np.nan, dtype=np.float64)
        dwell_count = len(interface.dwell_us)
        for band in range(interface.bands):
            actions = slice(band * dwell_count, (band + 1) * dwell_count)
            indices, times = self._eligible[band]
            local_bounds = np.searchsorted(
                times, np.concatenate((listen[actions], end[actions])), side="left"
            )
            for dwell in range(dwell_count):
                action = band * dwell_count + dwell
                left, finish = int(local_bounds[dwell]), int(local_bounds[dwell + dwell_count])
                if finish <= left:
                    continue
                candidate = indices[left:finish]
                x = pulses[candidate]
                selected = np.flatnonzero(x[:, 0] + x[:, 2] <= end[action])
                captured[action] = len(selected)
                if len(selected):
                    delay_us = float(x[selected[0], 0]) - start_us
                    first_intercept_us[action] = delay_us
                    # Match the public training contract's seconds arithmetic at
                    # exact bin edges; microsecond division can round differently.
                    time_class[action] = min(
                        time_bins - 1,
                        int((delay_us / 1_000_000) / (elapsed[action] / 1_000_000) * time_bins),
                    )
        ratio = np.divide(
            captured,
            np.maximum(truth, 1),
            dtype=np.float32,
        )
        return {
            "captured": captured,
            "truth": truth,
            "elapsed_us": elapsed,
            "time_class": time_class,
            "first_intercept_us": first_intercept_us,
            "ratio": ratio,
            "ratio_valid": truth > 0,
        }


def dense_training_batches(
    root,
    plan,
    receiver=None,
    interface=None,
    *,
    batch_size=256,
    history_steps=16,
    time_bins=16,
    behavior_factories=(UniformActionPolicy, RateProbePolicy),
    max_index_bytes=512 * 1024**2,
):
    """Yield borrowed dense-label batches from train recordings only.

    Arrays are reused at the next yield; copy them before retaining. Batches
    contain one behavior and receiver seed for clear collection provenance.
    """
    if plan.get("split") != "train":
        raise ValueError("dense labels require a train-only manifest")
    if any(type(value) is not int or value < 1 for value in (batch_size, history_steps, time_bins)):
        raise ValueError("batch_size, history_steps and time_bins must be positive integers")
    if not behavior_factories:
        raise ValueError("at least one behavior policy is required")
    receiver = receiver or ReplayConfig()
    interface = interface or InterfaceConfig()
    paths = validate_plan(root, plan)
    actions = interface.bands * len(interface.dwell_us)
    features = interface.bands * 9 + 2
    arrays = {
        "history": np.empty((batch_size, history_steps, features), dtype=np.float32),
        "action": np.empty(batch_size, dtype=np.int64),
        "captured": np.empty((batch_size, actions), dtype=np.int32),
        "truth": np.empty((batch_size, actions), dtype=np.int32),
        "elapsed_us": np.empty((batch_size, actions), dtype=np.float32),
        "time_class": np.empty((batch_size, actions), dtype=np.int64),
        "ratio": np.empty((batch_size, actions), dtype=np.float32),
        "ratio_valid": np.empty((batch_size, actions), dtype=np.bool_),
    }
    for path in paths:
        for seed in plan["seeds"]:
            configured = replace(receiver, seed=seed)
            index = DenseReplayIndex(path, configured, interface, max_index_bytes=max_index_bytes)
            for behavior_factory in behavior_factories:
                # Deterministic rate probing with perfect detection would replay
                # the same trajectory under every receiver seed.
                if (
                    behavior_factory is RateProbePolicy
                    and configured.detection_probability == 1
                    and seed != plan["seeds"][0]
                ):
                    continue
                behavior_name = behavior_factory.__name__
                size = 0
                with ReplayEnv(path, configured, interface) as env:
                    observation = env.reset()
                    behavior = behavior_factory()
                    behavior.reset(env.specification(), seed)
                    history = HistoryWindow(history_steps, features)
                    time_us = configured.start_us
                    current_band = None
                    while True:
                        arrays["history"][size] = history.append(observation)
                        labels = index.outcomes(time_us, current_band, time_bins)
                        for name in (
                            "captured",
                            "truth",
                            "elapsed_us",
                            "time_class",
                            "ratio",
                            "ratio_valid",
                        ):
                            arrays[name][size] = labels[name]
                        action = int(behavior.act(observation))
                        if not 0 <= action < actions:
                            raise ValueError("behavior action outside replay interface")
                        arrays["action"][size] = action
                        transition = env.step(action)
                        outcome = env.evaluation_outcome()
                        # Verify exact labels on the trajectory already paid for.
                        observed_class = (
                            time_bins
                            if outcome.first_intercept_seconds is None
                            else min(
                                time_bins - 1,
                                int(
                                    outcome.first_intercept_seconds
                                    / outcome.elapsed_seconds
                                    * time_bins
                                ),
                            )
                        )
                        if (
                            labels["captured"][action] != outcome.captured_count
                            or labels["truth"][action] != outcome.truth_count
                            or labels["time_class"][action] != observed_class
                            or not np.isclose(
                                labels["elapsed_us"][action],
                                transition.elapsed_us,
                                rtol=0,
                                atol=1e-3,
                            )
                        ):
                            raise ValueError(
                                "dense label differs from executed replay action: "
                                f"time={time_us} action={action} "
                                f"captured={labels['captured'][action]}/{outcome.captured_count} "
                                f"truth={labels['truth'][action]}/{outcome.truth_count} "
                                f"time_class={labels['time_class'][action]}/{observed_class} "
                                f"first={labels['first_intercept_us'][action]}/"
                                f"{outcome.first_intercept_seconds} "
                                f"elapsed={labels['elapsed_us'][action]}/{transition.elapsed_us}"
                            )
                        size += 1
                        observation = transition.observation
                        time_us += transition.elapsed_us
                        current_band = action // len(interface.dwell_us)
                        if size == batch_size or transition.terminated:
                            yield {
                                **{name: value[:size] for name, value in arrays.items()},
                                "time_bins": time_bins,
                                "behavior": behavior_name,
                                "seed": seed,
                            }
                            size = 0
                        if transition.terminated:
                            break
    validate_plan(root, plan)
