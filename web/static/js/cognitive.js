import { buildProgress } from './progress.js';
/** Display simulator metrics and observed policy state without filling missing values. */
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const ticks = value => value == null ? '—' : `${value.toFixed(1)} ticks`;

export class CognitiveDashboard {
  constructor(elements) { Object.assign(this, elements); }

  updateRunData(run) {
    this.run = run;
    this.progress = buildProgress(run);
    this.summary.innerHTML = ['Detected signals', 'Capture rate', 'Listening time', 'Retuning time'].map((label, i) => `
      <div class="summary-card panel"><div class="stat-label">${label}</div>
      <div class="stat-policy-row"><span title="${run.baseline.name}">${run.baseline.name}</span><strong data-stat="${i}-baseline">—</strong></div>
      <div class="stat-policy-row comparison"><span title="${run.active.name}">${run.active.name}</span><strong data-stat="${i}-active">—</strong></div>
      <div class="stat-detail" data-detail="${i}"></div></div>`).join('');
    this.statNodes = Array.from(this.summary.querySelectorAll('[data-stat]'));
    this.detailNodes = Array.from(this.summary.querySelectorAll('[data-detail]'));
    document.getElementById('results-reference').textContent = `Reference · ${run.baseline.name}`;
    document.getElementById('results-comparison').textContent = `Comparison · ${run.active.name}`;
    document.getElementById('details-reference').textContent = run.baseline.name;
    document.getElementById('details-comparison').textContent = run.active.name;
    this._renderResults(run);
    document.querySelector('.policy-details').open = run.active.id === 'timing-trained';
    document.querySelector('.workspace').classList.toggle('timing-demo', run.active.id === 'timing-trained');
  }

  updateProgress(tick) {
    if (!this.run) return;
    const count = this.progress.truth[tick];
    for (const node of this.statNodes) {
      const [metric, key] = node.dataset.stat.split('-');
      const p = this.progress[key];
      node.textContent = [
        `${p.captured[tick]} of ${count}`,
        count ? percent(p.captured[tick] / count) : 'No signal yet',
        `${p.listening[tick]} ticks`,
        percent((tick + 1 - p.listening[tick]) / (tick + 1)),
      ][Number(metric)];
    }
    const delta = this.progress.active.captured[tick] - this.progress.baseline.captured[tick];
    this.detailNodes[0].textContent = delta ? `${delta > 0 ? '+' : ''}${delta} detections vs reference so far` : 'Same number detected so far';
    this.detailNodes[1].textContent = 'Detected / simulated signals through this tick';
    this.detailNodes[2].textContent = `Of ${tick + 1} elapsed ticks`;
    this.detailNodes[3].textContent = 'Share of elapsed playback time';
    document.getElementById('progress-caption').textContent = `Through tick ${tick} · ${count} simulated signals`;
    const complete = tick === this.run.scenario.duration - 1;
    const b = this.run.baseline.data.metrics, a = this.run.active.data.metrics;
    const gain = b.detected_transmissions ? (a.detected_transmissions / b.detected_transmissions - 1) * 100 : null;
    const comparison = gain == null ? '' : gain > 0 ? ` ${(a.detected_transmissions / b.detected_transmissions).toFixed(2)}× as many captures in this seeded run.` : gain < 0 ? ` ${Math.abs(gain).toFixed(1)}% fewer captures in this seeded run.` : ' Both captured the same number.';
    document.getElementById('demo-outcome').textContent = complete
      ? `Run complete: ${this.run.active.name} detected ${a.detected_transmissions} signals; ${this.run.baseline.name} detected ${b.detected_transmissions}.${comparison}`
      : 'Watch the receiver decisions and detection counts evolve. Results describe this seeded example.';
  }

  updateTick(tick) {
    if (!this.run) return;
    const step = this.run.active.data.steps[tick];
    const state = step.cognitive_state || {};
    const q = state.q_values || [];
    const beliefs = state.beliefs || [];
    const forecasts = state.timing_forecasts || [];
    const decisionTick = state.decision_tick ?? step.decision_tick ?? tick;
    const forecastTicks = Math.min(state.forecast_ticks ?? 10, this.run.scenario.duration - decisionTick);
    this.beliefTitle.textContent = forecasts.length ? `Expected detections · ticks ${decisionTick}–${decisionTick + forecastTicks - 1}` : q.length ? 'Action values (Q)' : 'Estimated band hit probability';
    document.getElementById('policy-state-caption').textContent = forecasts.length
      ? `${this.run.active.name} · forecast made at tick ${decisionTick}, before receiver feedback. Selected band ${step.rx_band}; dwell ${state.dwell_ticks ?? step.dwell_ticks} listening ticks.`
      : `${this.run.active.name} · observed policy state at tick ${tick}.`;
    if (forecasts.length) {
      const maximum = Math.max(...forecasts.map(f => f.expected_detections), .001);
      this.beliefContainer.innerHTML = forecasts.map(f => this._bar(f.band, f.expected_detections / maximum * 100, f.expected_detections.toFixed(2))).join('');
    } else if (q.length) {
      const min = Math.min(...q), span = Math.max(...q) - min || 1;
      this.beliefContainer.innerHTML = q.map((value, band) => this._bar(band, 5 + (value - min) / span * 95, value.toFixed(2))).join('');
    } else if (beliefs.length) {
      this.beliefContainer.innerHTML = beliefs.map(b => this._bar(b.band, b.mean_probability * 100, percent(b.mean_probability))).join('');
    } else {
      this.beliefContainer.innerHTML = '<p class="field-help">This policy does not expose band probabilities.</p>';
    }
    const scores = state.ucb_scores || [];
    this.ucbContainer.parentElement.hidden = !scores.length;
    this.beliefContainer.parentElement.parentElement.classList.toggle('single-state', !scores.length);
    this.ucbContainer.innerHTML = scores.length ? scores.map(s => {
      const total = Math.max(s.total_score, .001);
      return `<div class="ucb-row"><span class="belief-band-tag">Band ${s.band}</span><div class="ucb-bar-track" title="Mean hit: ${s.exploitation}; exploration bonus: ${s.exploration}"><div class="ucb-exploit-fill" style="width:${s.exploitation / total * 100}%"></div><div class="ucb-explore-fill" style="width:${s.exploration / total * 100}%"></div></div><span class="belief-pct">${s.total_score.toFixed(2)}</span></div>`;
    }).join('') : '<p class="field-help">No UCB scores are available for this policy.</p>';
    this.detectorAlert.textContent = state.detected_change_count == null ? '' : `Rate shifts detected so far: ${state.detected_change_count}.`;
  }

  _bar(band, width, label) {
    return `<div class="belief-row"><span class="belief-band-tag">Band ${band}</span><div class="belief-bar-track"><div class="belief-bar-fill" style="width:${Math.max(0, Math.min(width, 100))}%"></div></div><span class="belief-pct">${label}</span></div>`;
  }

  _renderResults(run) {
    const policies = [run.baseline, run.active];
    const cell = (value, reason = '') => ({value, reason});
    const metric = (p, key) => p.data.metrics[key];
    const forecastMetric = (p, key) => p.data.evaluation?.prediction?.[key];
    const delay = (p, reacquire = false) => {
      const m = p.data.metrics;
      if (reacquire && !m.total_emitter_changes) return cell('Not applicable', 'No modeled emitter changes.');
      if (!reacquire && !m.detected_emitters) return cell('Not detected', `${m.mean_first_detection_delay.toFixed(1)} ticks is the horizon-capped bound.`);
      const censored = reacquire ? m.reacquired_changes < m.total_emitter_changes : m.detected_emitters < m.total_emitters;
      return cell(ticks(reacquire ? m.mean_reacquisition_delay : m.mean_first_detection_delay), censored ? 'Includes horizon-capped unresolved cases.' : '');
    };
    const rows = [
      ['Detected signals', 'True captures, excluding false alarms.', p => cell(`${metric(p, 'detected_transmissions')} of ${metric(p, 'total_transmissions')}`)],
      ['Interception ratio', 'Captured / all simulated signals across all bands.', p => cell(percent(metric(p, 'interception_ratio')))],
      ['Receiver sensitivity', 'Simulated detection threshold, applied equally to both receivers.', () => cell(`${run.scenario.sensitivity_dbm} dBm`)],
      ['Detection probability', 'Captured / detectable signals on the listening band.', p => {
        const m = p.data.metrics;
        return cell(m.detectable_transmissions ? percent(m.probability_of_detection) : 'Not measured', m.detectable_transmissions ? `${m.detected_transmissions} / ${m.detectable_transmissions} detectable signals` : 'No detectable signals observed.');
      }],
      ['False alarm probability', 'False alarms / listening ticks without a detectable signal.', p => {
        const negatives = p.data.steps.filter(s => s.listening && s.hit_count + s.missed_count === 0).length;
        return cell(negatives ? percent(metric(p, 'false_alarm_rate')) : 'Not measured', negatives ? `${metric(p, 'false_alarms')} false alarms / ${negatives} opportunities` : 'No negative listening opportunities.');
      }],
      ['Listening time', 'Time available to receive signals.', p => cell(`${metric(p, 'listening_steps')} of ${run.scenario.duration} ticks`)],
      ['Retuning time', 'Time spent changing the receiver band.', p => cell(percent(metric(p, 'retuning_fraction')), `${metric(p, 'retuning_steps')} ticks`)],
      ['First detection delay', 'Mean time from emitter activity to first detection.', p => delay(p)],
      ['Reacquisition delay', 'Mean time to detect again after a modeled emitter change.', p => delay(p, true)],
      ['Emitter discovery', 'Distinct emitters detected / emitters in truth.', p => cell(percent(metric(p, 'emitter_discovery_ratio')), `${metric(p, 'detected_emitters')} of ${metric(p, 'total_emitters')} emitters`)],
      ['Sensitivity loss', 'Signals below threshold / transmissions on the listening band.', p => cell(metric(p, 'eligible_transmissions') ? percent(metric(p, 'sensitivity_loss_rate')) : 'Not measured', metric(p, 'eligible_transmissions') ? '' : 'No transmissions on the listening band.')],
      ['Receiver hit rate', 'Listening ticks with a hit / listening ticks; includes false alarms.', p => cell(metric(p, 'listening_steps') ? percent(metric(p, 'hit_rate')) : 'Not measured')],
      ['Average intercept rate', 'True captured transmissions per second; one tick is 1 ms.', p => cell(p.data.evaluation?.average_intercept_rate_per_second?.toFixed(1) ?? 'Not measured')],
      ['Scan reward per tick', 'Receiver hits minus 0.05 per retuning tick, averaged across physical ticks.', p => cell(p.data.evaluation == null ? 'Not measured' : (p.data.evaluation.reward_sum / run.scenario.duration).toFixed(3))],
      ['Prediction accuracy', 'Pre-action prediction of a true capture during the selected listening dwell.', p => cell(forecastMetric(p, 'percentage_correct') == null ? 'Not reported' : `${forecastMetric(p, 'percentage_correct').toFixed(1)}%`)],
      ['Intercept-time prediction error', 'Mean absolute error when a predicted and actual intercept both occur.', p => cell(forecastMetric(p, 'average_intercept_time_error_seconds') == null ? 'Not measured' : `${(forecastMetric(p, 'average_intercept_time_error_seconds') * 1000).toFixed(2)} ms`, (forecastMetric(p, 'timing_pairs') || 0) ? `${forecastMetric(p, 'timing_pairs')} matched intercepts` : 'No matched predicted and actual intercepts.')],
      ['Forecast coverage', 'Decision windows with a forecast supplied before receiver feedback.', p => cell(forecastMetric(p, 'coverage') == null ? 'Not reported' : percent(forecastMetric(p, 'coverage')), p.data.evaluation ? `${forecastMetric(p, 'forecast_windows') || 0} of ${p.data.evaluation.windows} decisions` : '')],
      ['Longest band gap', 'Longest period without listening on a band.', p => cell(`${metric(p, 'max_band_gap')} ticks`)],
      ['Track association purity', 'Correctly grouped anonymous measurements, evaluated against truth.', p => cell(p.data.track_metrics.assigned_measurements ? percent(p.data.track_metrics.association_purity) : 'Not measured', p.data.track_metrics.assigned_measurements ? '' : 'No track measurements.')],
      ['Track pairwise F1', 'Precision and recall of measurement pair associations.', p => cell(p.data.track_metrics.assigned_measurements ? percent(p.data.track_metrics.pairwise_f1) : 'Not measured', p.data.track_metrics.assigned_measurements ? '' : 'No track measurements.')],
    ];
    if (policies.some(p => metric(p, 'average_reward') != null)) rows.push(['Policy reward per tick', 'Each policy uses its own reward definition; unlike definitions are not comparable.', p => cell(metric(p, 'average_reward') == null ? 'Not defined' : metric(p, 'average_reward').toFixed(3))]);
    const predicted = policies.filter(p => (forecastMetric(p, 'forecast_windows') || 0) > 0);
    const predictionLabels = new Set(['Prediction accuracy', 'Intercept-time prediction error', 'Forecast coverage']);
    const extraLabels = new Set(['Listening time', 'Reacquisition delay', 'Sensitivity loss', 'Receiver hit rate', 'Longest band gap', 'Track association purity', 'Track pairwise F1', 'Policy reward per tick']);
    const supported = rows.filter(([label]) => (!predictionLabels.has(label) || predicted.length) &&
      (!label.startsWith('Track ') || policies.some(p => p.data.track_metrics.assigned_measurements)));
    const renderRows = entries => entries.map(([label, description, render]) => `<tr><td>${label}<p class="metric-description">${description}</p></td>${policies.map(p => {
      const {value, reason} = render(p);
      return `<td>${value}${reason ? `<small class="metric-reason">${reason}</small>` : ''}</td>`;
    }).join('')}</tr>`).join('');
    this.fomContainer.innerHTML = renderRows(supported.filter(([label]) => !extraLabels.has(label)));
    document.getElementById('details-grid-container').innerHTML = renderRows(supported.filter(([label]) => extraLabels.has(label)));
    document.getElementById('results-capabilities').textContent = `Receiver threshold: ${run.scenario.sensitivity_dbm} dBm for both policies. ${predicted.length ? `${predicted.map(p => p.name).join(' and ')} supplies forecasts before each scan decision; prediction measurements use the actual subsequent receiver outcome.` : 'These policies do not report intercept-time forecasts; prediction accuracy and timing error are not evaluated here.'}`;
  }
}
