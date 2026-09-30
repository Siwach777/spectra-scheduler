"""Causal coverage constraint shared by MPC collection and deployment."""

from __future__ import annotations

import numpy as np


def coverage_probe_action(
    last_visit_step,
    time_step: int,
    previous_band: int,
    retune_table,
    dwell_steps,
    remaining: int,
    revisit_limit: int,
) -> int | None:
    """Pick a legal shortest probe for unseen or overdue bands, if required.

    Age counts physical ticks since the last *listening* observation. This
    constraint allocates receiver time; it does not imply emitter discovery.
    """
    if revisit_limit <= 0:
        return None
    visits = np.asarray(last_visit_step)
    dwells = tuple(dwell_steps)
    shortest = int(np.argmin(dwells))
    delays = (
        np.zeros(len(visits), dtype=np.int64)
        if previous_band < 0
        else np.asarray(retune_table)[previous_band]
    )
    legal = delays + dwells[shortest] <= remaining
    unseen = np.flatnonzero((visits < 0) & legal)
    if len(unseen):
        # Nearby cold-start probes preserve physical listening time.
        band = int(unseen[np.argmin(delays[unseen])])
    else:
        ages = time_step - visits
        overdue = np.flatnonzero((ages >= revisit_limit) & legal)
        if not len(overdue):
            return None
        band = int(overdue[np.argmax(ages[overdue])])
    return band * len(dwells) + shortest
