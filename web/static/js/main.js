import { initializeTheme } from './theme.js';
import { ApiClient } from './api_client.js';
import { EpisodeBuffer } from './buffer.js';
import { WaterfallEngine } from './waterfall.js';
import { ScrubberRibbon } from './scrubber.js';
import { PlaybackController } from './playback.js';
import { CognitiveDashboard } from './cognitive.js';
import { FeatureSpacePlot } from './feature_plot.js';

const el = id => document.getElementById(id);
const api = new ApiClient();
const buffer = new EpisodeBuffer();
let runData = null, busy = false, ready = false;
let scenarios = [], schedulers = [], presets = {};
let currentView = 'active', runNumber = 0;
initializeTheme();
const form = el('run-form');
const scenarioSelect = el('select-scenario');
const baselineSelect = el('select-baseline');
const activeSelect = el('select-active');
const presetSelect = el('select-preset');
const timeline = el('timeline');
const waterfall = new WaterfallEngine(el('waterfall-raster'), el('waterfall-hud'), el('waterfall-tooltip'));
const scrubber = new ScrubberRibbon(el('scrubber-ribbon'), el('scrubber-playhead'));
const cognitive = new CognitiveDashboard({
  summary: el('summary-grid'), beliefContainer: el('belief-bars-container'),
  beliefTitle: el('belief-title'), ucbContainer: el('ucb-stack-container'),
  detectorAlert: el('detector-alert-banner'), fomContainer: el('fom-grid-container'),
});
const featurePlot = new FeatureSpacePlot(el('feature-space-canvas'), el('track-table-body'));
const playback = new PlaybackController(updateTick, tick => {
  waterfall.render(tick);
  el('time-start').textContent = waterfall.startTick;
  el('time-end').textContent = waterfall.endTick - 1;
}, updatePlaybackState);

function setStatus(text, state = '') {
  el('run-status').textContent = text;
  el('run-status').dataset.state = state;
}

function updatePlaybackState() {
  const ended = runData && playback.currentTick === buffer.duration - 1;
  el('btn-play').textContent = playback.isPlaying ? 'Pause' : ended ? 'Replay' : 'Play';
  el('btn-play').setAttribute('aria-label', `${el('btn-play').textContent} simulation playback`);
  if (!busy && runData) setStatus(playback.isPlaying ? 'Playing' : ended ? 'Complete' : 'Paused', playback.isPlaying ? 'playing' : '');
}

function updateTick(tick) {
  if (!runData) return;
  timeline.value = tick;
  timeline.setAttribute('aria-valuetext', `Tick ${tick} of ${buffer.duration - 1}`);
  scrubber.updatePlayhead(tick);
  el('tick-counter-val').textContent = `Tick ${tick} / ${buffer.duration - 1}`;
  const a = runData.active.data.steps[tick], b = runData.baseline.data.steps[tick];
  const status = s => !s.listening ? 'retuning' : s.false_alarms ? `${s.hit_count} detections, false alarm` : s.hit_count ? `${s.hit_count} detection${s.hit_count === 1 ? '' : 's'}` : 'listening, no detection';
  el('current-observation').textContent = `${runData.active.name} · band ${a.rx_band} · ${status(a)}`;
  el('current-reference').textContent = `${runData.baseline.name} · band ${b.rx_band} · ${status(b)}`;
  const rx = currentView === 'baseline' ? b : a;
  el('waterfall-axis-y').querySelectorAll('.band-label-cell').forEach(cell => {
    cell.classList.toggle('active-listen', rx.listening && Number(cell.dataset.band) === rx.rx_band);
  });
  cognitive.updateProgress(tick);
  // Avoid updating collapsed policy details on every playback tick.
  if (document.querySelector('.policy-details').open) cognitive.updateTick(tick);
  updatePlaybackState();
}

function receiverDefaults() {
  const scenario = scenarios.find(s => s.id === scenarioSelect.value);
  if (!scenario) return;
  el('slider-sens').value = scenario.sensitivity_dbm;
  el('slider-retune').value = scenario.retune_steps;
  updateReceiverLabels();
}

function updateReceiverLabels() {
  el('val-sens').textContent = `${el('slider-sens').value} dBm`;
  const ticks = Number(el('slider-retune').value);
  el('val-retune').textContent = `${ticks} tick${ticks === 1 ? '' : 's'}`;
}

function modelNote() {
  const selected = schedulers.filter(s => [baselineSelect.value, activeSelect.value].includes(s.id) && s.artifact);
  el('model-note').hidden = selected.length === 0;
  el('model-note').textContent = selected.map(s => {
    const result = [runData?.active, runData?.baseline].find(r => r?.id === s.id);
    const model = result?.data.model || s;
    return `${s.name}: ${model.artifact}${model.training_seed == null ? '' : ` · seed ${model.training_seed}`}${model.selected_epoch == null ? '' : ` · epoch ${model.selected_epoch}`}${model.sha256 ? ` · ${model.sha256.slice(0, 12)}` : ''}`;
  }).join(' · ');
}

function markChanged() {
  el('pending-note').hidden = !runData;
  presetSelect.value = 'custom';
  el('preset-description').textContent = 'Custom comparison using the settings below.';
  modelNote();
}

function applyPreset(key) {
  const p = presets[key];
  if (!p) return;
  presetSelect.value = key;
  scenarioSelect.value = p.scenario;
  baselineSelect.value = p.baseline;
  activeSelect.value = p.active;
  el('input-seed').value = p.seed;
  el('select-variation').value = '';
  receiverDefaults();
  modelNote();
  el('preset-description').textContent = p.description;
  el('pending-note').hidden = !runData;
}

async function executeRun({autoplay = true} = {}) {
  if (busy || !ready || !form.reportValidity()) return;
  busy = true;
  const wasPlaying = playback.isPlaying;
  let succeeded = false;
  playback.pause();
  setStatus('Running');
  form.querySelectorAll('button, input, select').forEach(input => { input.disabled = true; });
  for (const id of ['btn-prev', 'btn-play', 'btn-next', 'timeline', 'btn-export']) el(id).disabled = true;
  el('btn-run-sim').textContent = 'Running…';
  el('error-banner').hidden = true;
  try {
    const data = await api.runSimulation({
      scenario: scenarioSelect.value, baseline: baselineSelect.value, active: activeSelect.value,
      seed: Number(el('input-seed').value), sensitivity_dbm: Number(el('slider-sens').value),
      retune_steps: Number(el('slider-retune').value), perturbation: el('select-variation').value || null,
    });
    // Load the buffer before making this run visible to playback callbacks.
    buffer.loadSimulationRun(data);
    playback.setDuration(data.scenario.duration, Math.max(60, 15000 / data.scenario.duration));
    runData = data;
    modelNote();
    waterfall.setBuffer(buffer, data);
    scrubber.setBuffer(buffer);
    el('waterfall-axis-y').replaceChildren(...Array.from({length: buffer.numBands}, (_, i) => {
      const cell = document.createElement('div');
      cell.className = 'band-label-cell';
      cell.dataset.band = buffer.numBands - 1 - i;
      cell.textContent = `Band ${cell.dataset.band}`;
      return cell;
    }));
    const sc = scenarios.find(s => s.id === data.scenario.id);
    el('run-title').textContent = sc.name;
    const variation = data.scenario.perturbation === 'frequency-hop' ? ' · mid-run hop' : data.scenario.perturbation ? ' · added emitter' : '';
    el('run-meta').textContent = `Run ${++runNumber} · ${buffer.numBands} bands · 1 band observed · ${buffer.duration} ticks · seed ${data.scenario.seed} · ${data.scenario.sensitivity_dbm} dBm · ${data.scenario.retune_steps} retune tick${data.scenario.retune_steps === 1 ? '' : 's'}${variation}`;
    cognitive.updateRunData(data);
    showDiagnostics(data);
    featurePlot.setData(data.active.data.active_tracks, data.active.data.archived_tracks);
    timeline.max = buffer.duration - 1;
    el('chart-empty').hidden = true;
    el('pending-note').hidden = true;
    playback.scrubTo(0);
    succeeded = true;
    if (autoplay) switchTab('tab-cockpit');
    setStatus('Paused');
  } catch (error) {
    el('error-banner').textContent = `Could not run comparison. ${error.message}${runData ? ' The previous run is still displayed.' : ''}`;
    el('error-banner').hidden = false;
    if (!runData) el('chart-empty').textContent = 'No run loaded. Adjust setup and try again.';
    setStatus('Run failed');

  } finally {
    busy = false;
    form.querySelectorAll('button, input, select').forEach(input => { input.disabled = false; });
    for (const id of ['btn-prev', 'btn-play', 'btn-next', 'timeline', 'btn-export']) el(id).disabled = !runData;
    el('btn-run-sim').textContent = 'Run comparison';
    if ((succeeded && autoplay) || (!succeeded && runData && wasPlaying)) playback.play();
    if (succeeded) updatePlaybackState();
  }
}

form.addEventListener('submit', event => { event.preventDefault(); executeRun(); });
presetSelect.addEventListener('change', () => {
  if (presetSelect.value === 'custom') { markChanged(); return; }
  applyPreset(presetSelect.value);
  executeRun();
});
scenarioSelect.addEventListener('change', () => { receiverDefaults(); markChanged(); });
for (const id of ['select-baseline', 'select-active', 'input-seed', 'select-variation']) el(id).addEventListener('input', markChanged);
for (const id of ['slider-sens', 'slider-retune']) el(id).addEventListener('input', () => { updateReceiverLabels(); markChanged(); });
el('btn-play').addEventListener('click', () => playback.toggle());
el('btn-prev').addEventListener('click', () => playback.step(-1));
el('btn-next').addEventListener('click', () => playback.step(1));
timeline.addEventListener('input', () => { playback.pause(); playback.scrubTo(Number(timeline.value)); });
el('select-speed').addEventListener('change', () => playback.setSpeed(Number(el('select-speed').value)));
el('fog-toggle').addEventListener('change', () => {
  waterfall.setFogOfWar(!el('fog-toggle').checked);
  el('truth-legend').hidden = !el('fog-toggle').checked;
});
document.querySelector('.policy-details').addEventListener('toggle', () => cognitive.updateTick(playback.currentTick));
document.querySelectorAll('.view-mode-pill').forEach(button => button.addEventListener('click', () => {
  currentView = button.dataset.view;
  document.querySelectorAll('.view-mode-pill').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', b === button); });
  waterfall.setViewMode(currentView);
  if (runData) updateTick(playback.currentTick);
}));
function switchTab(id) {
  if (id !== 'tab-cockpit') playback.pause();
  document.querySelectorAll('.nav-tab-btn').forEach(button => {
    const selected = button.dataset.tab === id;
    button.classList.toggle('active', selected);
    if (selected) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.toggle('active', tab.id === id));
  requestAnimationFrame(() => { waterfall.resize(); scrubber.resize(); featurePlot.resize(); });
}
document.querySelectorAll('.nav-tab-btn').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
document.addEventListener('themechange', () => featurePlot.render());

function showDiagnostics(data) {
  const warnings = [];
  for (const key of ['baseline', 'active']) {
    const policy = data[key], m = policy.data.metrics;
    if (m.retuning_fraction > .95) warnings.push(`${policy.name} spends ${(m.retuning_fraction * 100).toFixed(1)}% of this run retuning and listens for only ${m.listening_steps} ticks. A fixed-dwell sweep gives the receiver time to listen.`);
  }
  el('run-diagnostics').textContent = warnings.join(' ');
  el('run-diagnostics').hidden = !warnings.length;
}
el('btn-export').addEventListener('click', () => {
  if (!runData) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(runData, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = `spectra-${runData.scenario.id}-seed${runData.scenario.seed}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('keydown', event => {
  if (busy || !runData || !el('tab-cockpit').classList.contains('active') || event.target.closest('input, select, button, summary, a, textarea') || event.target.isContentEditable) return;
  if (event.code === 'Space') { event.preventDefault(); playback.toggle(); }
  else if (event.code === 'ArrowLeft') { event.preventDefault(); playback.step(-1); }
  else if (event.code === 'ArrowRight') { event.preventDefault(); playback.step(1); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) playback.pause(); });
const resize = new ResizeObserver(() => { waterfall.resize(); scrubber.resize(); featurePlot.resize(); });
resize.observe(document.querySelector('.canvas-layers-container'));
resize.observe(document.querySelector('.plot-canvas-wrapper'));

async function initialize() {
  try {
    [scenarios, schedulers, presets] = await Promise.all([api.getScenarios(), api.getSchedulers(), api.getPresets()]);
    const option = (value, text, disabled = false) => { const opt = new Option(text, value); opt.disabled = disabled; return opt; };
    scenarioSelect.replaceChildren(...scenarios.map(s => option(s.id, s.name)));
    for (const select of [baselineSelect, activeSelect]) select.replaceChildren(...schedulers.map(s => option(s.id, s.name + (s.available ? '' : ' (unavailable)'), !s.available)));
    presetSelect.replaceChildren(...Object.entries(presets).map(([key, p]) => option(key, p.name, !schedulers.find(s => s.id === p.active)?.available)), option('custom', 'Custom comparison'));
    form.querySelectorAll('input, select, button').forEach(input => { input.disabled = false; });
    ready = true;
    const params = new URLSearchParams(location.search);
    const requested = params.get('preset');
    const preferred = schedulers.find(s => s.id === 'timing-trained')?.available ? 'trained-timing' : 'video-demo';
    const preset = presets[requested] && !Array.from(presetSelect.options).find(o => o.value === requested).disabled ? requested : preferred;
    applyPreset(preset);
    const view = ['active', 'baseline', 'dual'].includes(params.get('view')) ? params.get('view') : preset === 'video-demo' ? 'dual' : 'active';
    document.querySelector(`.view-mode-pill[data-view="${view}"]`).click();
    await executeRun({autoplay: params.get('autoplay') === '1'});
  } catch (error) {
    setStatus('Unavailable');
    el('error-banner').textContent = `Could not connect to the demonstration server. ${error.message} Reload to try again.`;
    el('error-banner').hidden = false;
    el('chart-empty').textContent = 'The simulation server is unavailable.';
  }
}
initialize();
