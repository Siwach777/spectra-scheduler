from collections import deque
from dataclasses import dataclass, field


@dataclass
class BinaryRateChangeDetector:
    """Detect a sustained change between older and recent binary observations."""

    reference_window_size: int = 4
    recent_window_size: int = 3
    minimum_rate_change: float = 0.5
    _history: dict[int, deque[int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_window_size <= 0 or self.recent_window_size <= 0:
            raise ValueError("window sizes must be positive")
        if not 0.0 < self.minimum_rate_change <= 1.0:
            raise ValueError("minimum_rate_change must be between zero and one")

    def reset(self) -> None:
        self._history.clear()

    def update(self, key: int, value: bool) -> bool:
        history_size = self.reference_window_size + self.recent_window_size
        history = self._history.setdefault(key, deque(maxlen=history_size))
        history.append(int(value))
        if len(history) < history_size:
            return False

        values = list(history)
        reference = values[: self.reference_window_size]
        recent = values[self.reference_window_size :]
        reference_rate = sum(reference) / len(reference)
        recent_rate = sum(recent) / len(recent)
        changed = abs(reference_rate - recent_rate) >= self.minimum_rate_change
        if changed:
            history.clear()
            history.extend(recent)
        return changed
