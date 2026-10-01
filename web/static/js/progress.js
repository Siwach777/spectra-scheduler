/** Prefix statistics give constant-time, reversible playback counters. */
export function buildProgress(run) {
  const n = run.scenario.duration;
  const truth = new Uint32Array(n);
  const policy = key => {
    const captured = new Uint32Array(n), listening = new Uint32Array(n), alarms = new Uint32Array(n);
    run[key].data.steps.forEach((s, t) => {
      captured[t] = (captured[t - 1] || 0) + s.hit_count;
      listening[t] = (listening[t - 1] || 0) + Number(s.listening);
      alarms[t] = (alarms[t - 1] || 0) + s.false_alarms;
    });
    return {captured, listening, alarms};
  };
  run.truth_grid.forEach((signals, t) => { truth[t] = (truth[t - 1] || 0) + signals.length; });
  return {truth, baseline: policy('baseline'), active: policy('active')};
}
