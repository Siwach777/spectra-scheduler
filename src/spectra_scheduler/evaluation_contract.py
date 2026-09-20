"""Versioned, model-independent evaluation of action-window forecasts.

TruthOutcome is evaluator-only. Never include it in policy observations or rewards.
All durations use seconds here; adapters own conversion from their native units.
"""

from dataclasses import dataclass
from math import isfinite

CONTRACT_VERSION = 1


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class Forecast:
    """Pre-action predictions for the selected action, clipped to episode end.

    hit_probability predicts at least one true capture before buffer overflow.
    intercept_delay_seconds is measured from action start (including retuning).
    None predicts no intercept within the window. interception_ratio predicts
    captures / all in-spectrum arrivals over that same elapsed window.
    """

    hit_probability: float
    intercept_delay_seconds: float | None
    interception_ratio: float

    def __post_init__(self):
        for name in ("hit_probability", "interception_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability")
        delay = self.intercept_delay_seconds
        if delay is not None and (isinstance(delay, bool) or not isfinite(delay) or delay < 0):
            raise ValueError("intercept delay must be nonnegative, finite or None")


@dataclass(frozen=True)
class Decision:
    """A replay action and optional forecast, returned atomically before stepping."""

    action: int
    forecast: Forecast | None = None

    def __post_init__(self):
        if self.forecast is not None and not isinstance(self.forecast, Forecast):
            raise ValueError("forecast must be a Forecast or None")


@dataclass(frozen=True)
class TruthOutcome:
    elapsed_seconds: float
    truth_count: int
    eligible_count: int  # Within listening time/passband, before sensitivity.
    detectable_count: int  # Eligible and above sensitivity threshold.
    captured_count: int  # True captures before observation-buffer overflow.
    first_intercept_seconds: float | None
    # None means false-alarm modelling is unavailable, not perfect rejection.
    negative_opportunity: bool | None = None
    false_alarm: bool | None = None

    def __post_init__(self):
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive and finite")
        counts = (self.truth_count, self.eligible_count, self.detectable_count, self.captured_count)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("counts must be nonnegative integers")
        if tuple(sorted(counts, reverse=True)) != counts:
            raise ValueError("require truth >= eligible >= detectable >= captured")
        delay = self.first_intercept_seconds
        if (delay is None) != (self.captured_count == 0):
            raise ValueError("first intercept must exist exactly when a capture occurs")
        if delay is not None and (not isfinite(delay) or not 0 <= delay < self.elapsed_seconds):
            raise ValueError("first intercept must lie inside the half-open action window")
        if (self.negative_opportunity is None) != (self.false_alarm is None):
            raise ValueError("false-alarm availability fields must be supplied together")
        if any(
            v is not None and type(v) is not bool
            for v in (self.negative_opportunity, self.false_alarm)
        ):
            raise ValueError("false-alarm fields must be bool or None")


class EvaluationAccumulator:
    """Constant-memory sufficient statistics; reports never emit NaN or infinity."""

    def __init__(self):
        self.windows = self.forecasts = self.truth = self.eligible = self.detectable = 0
        self.captured = self.negatives = self.false_positives = self.fa_windows = 0
        self.tp = self.tn = self.fp = self.fn = self.ratio_windows = 0
        self.timing_pairs = self.event_windows = self.censored_windows = 0
        self.elapsed = self.reward = self.brier = self.ratio_error = 0.0
        self.timing_error = self.restricted_timing_error = 0.0

    def add(self, outcome: TruthOutcome, reward: float, forecast: Forecast | None = None):
        if not isinstance(outcome, TruthOutcome):
            raise ValueError("outcome must be a TruthOutcome")
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        if forecast is not None and not isinstance(forecast, Forecast):
            raise ValueError("forecast must be a Forecast or None")
        self.windows += 1
        self.elapsed += outcome.elapsed_seconds
        self.reward += reward
        self.truth += outcome.truth_count
        self.eligible += outcome.eligible_count
        self.detectable += outcome.detectable_count
        self.captured += outcome.captured_count
        if outcome.negative_opportunity is not None:
            self.fa_windows += 1
            self.negatives += outcome.negative_opportunity
            self.false_positives += outcome.negative_opportunity and outcome.false_alarm
        if forecast is None:
            return
        self.forecasts += 1
        hit = outcome.captured_count > 0
        predicted_hit = forecast.hit_probability >= 0.5
        self.tp += hit and predicted_hit
        self.tn += not hit and not predicted_hit
        self.fp += not hit and predicted_hit
        self.fn += hit and not predicted_hit
        self.brier += (forecast.hit_probability - hit) ** 2
        if outcome.truth_count:
            self.ratio_windows += 1
            self.ratio_error += abs(
                forecast.interception_ratio - outcome.captured_count / outcome.truth_count
            )
        actual_delay = outcome.first_intercept_seconds
        predicted_delay = forecast.intercept_delay_seconds
        self.event_windows += hit
        self.censored_windows += not hit
        if actual_delay is not None and predicted_delay is not None:
            self.timing_pairs += 1
            self.timing_error += abs(predicted_delay - actual_delay)
        # Compare min(T, horizon): no extrapolated event time for censored windows.
        horizon = outcome.elapsed_seconds
        actual_restricted = horizon if actual_delay is None else actual_delay
        predicted_restricted = horizon if predicted_delay is None else min(predicted_delay, horizon)
        self.restricted_timing_error += abs(predicted_restricted - actual_restricted)

    def report(self):
        return {
            "contract_version": CONTRACT_VERSION,
            "target": "selected_action_window_true_capture_before_overflow",
            "windows": self.windows,
            "elapsed_seconds": self.elapsed,
            "counts": {
                "truth": self.truth,
                "eligible": self.eligible,
                "detectable": self.detectable,
                "captured": self.captured,
            },
            "probability_of_detection": _ratio(self.captured, self.detectable),
            "probability_of_false_alarm": (
                _ratio(self.false_positives, self.negatives)
                if self.fa_windows == self.windows
                else None
            ),
            "false_alarm_support": {
                "modelled_windows": self.fa_windows,
                "negative_windows": self.negatives,
                "false_positive_windows": self.false_positives,
            },
            "sensitivity_loss_fraction": _ratio(self.eligible - self.detectable, self.eligible),
            "average_intercept_rate_per_second": _ratio(self.captured, self.elapsed),
            "interception_ratio": _ratio(self.captured, self.truth),
            "average_reward_per_action": _ratio(self.reward, self.windows),
            "reward_per_second": _ratio(self.reward, self.elapsed),
            "reward_sum": self.reward,
            "prediction": {
                "forecast_windows": self.forecasts,
                "coverage": _ratio(self.forecasts, self.windows),
                "classification_threshold": 0.5,
                "percentage_correct": _ratio(100 * (self.tp + self.tn), self.forecasts),
                "confusion": {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn},
                "brier_score": _ratio(self.brier, self.forecasts),
                "interception_ratio_mae": _ratio(self.ratio_error, self.ratio_windows),
                "ratio_windows": self.ratio_windows,
                "average_intercept_time_error_seconds": _ratio(
                    self.timing_error, self.timing_pairs
                ),
                "timing_pairs": self.timing_pairs,
                "event_windows": self.event_windows,
                "censored_windows": self.censored_windows,
                "timing_event_coverage": _ratio(self.timing_pairs, self.event_windows),
                "restricted_intercept_time_mae_seconds": _ratio(
                    self.restricted_timing_error, self.forecasts
                ),
            },
        }
