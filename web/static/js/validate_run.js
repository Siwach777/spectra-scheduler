/** Reject incomplete responses before replacing a working run or its buffers. */
export function validateRun(run) {
  const {duration, num_bands: bands} = run?.scenario || {};
  if (!Number.isInteger(duration) || duration < 1 || !Number.isInteger(bands) || bands < 1) {
    throw new Error('The server returned an invalid episode shape.');
  }
  if (!Array.isArray(run.truth_grid) || run.truth_grid.length !== duration ||
      !run.truth_grid.every(signals => Array.isArray(signals) && signals.every(s =>
        Number.isInteger(s.band) && s.band >= 0 && s.band < bands && Number.isFinite(s.power_dbm)))) {
    throw new Error('The server returned an incomplete signal trace.');
  }
  for (const key of ['baseline', 'active']) {
    const p = run[key];
    if (!p?.data?.metrics || !Number.isFinite(p.data.metrics.detected_transmissions) ||
        !Array.isArray(p.data.steps) || p.data.steps.length !== duration ||
        !p.data.steps.every((s, t) => s.tick === t && Number.isInteger(s.rx_band) &&
          s.rx_band >= 0 && s.rx_band < bands && typeof s.listening === 'boolean' &&
          Number.isInteger(s.hit_count) && s.hit_count >= 0)) {
      throw new Error(`The server returned an incomplete ${key === 'baseline' ? 'reference' : 'comparison'} receiver trace.`);
    }
  }
}
