"""Shared physical-time action grid for replay and new synthetic training."""

TICK_US = 1_000.0
DWELL_US = (1_000.0, 10_000.0, 50_000.0)
DWELL_STEPS = tuple(int(dwell / TICK_US) for dwell in DWELL_US)
ACTION_CONTRACT_VERSION = 2
