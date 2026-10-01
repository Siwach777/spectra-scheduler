/** Focused regressions for varying episode shapes and playback boundaries. */
import assert from 'node:assert/strict';
import { EpisodeBuffer } from './static/js/buffer.js';
import { PlaybackController } from './static/js/playback.js';

const makeRun = bands => ({
  scenario: {num_bands: bands, duration: 3},
  truth_grid: [[{band: bands - 1, power_dbm: -105}], [], []],
  baseline: {data: {steps: Array.from({length: 3}, () => ({rx_band: 0, listening: true, hit_count: 0, false_alarms: 0}))}},
  active: {data: {steps: Array.from({length: 3}, () => ({rx_band: bands - 1, listening: true, hit_count: 0, false_alarms: 0}))}},
});
const buffer = new EpisodeBuffer(3, 6);
for (const bands of [6, 8, 6]) {
  buffer.loadSimulationRun(makeRun(bands));
  assert.equal(buffer.getTruthPower(0, bands - 1), -105, 'Weak truth signals must not be discarded');
  assert.equal(buffer.getTruthPower(1, bands - 1), -Infinity);
  assert.equal(buffer.truthPower.length, 3 * bands, 'Reallocate when band count changes');
}
let nextFrame = 1;
const frames = new Map();
globalThis.requestAnimationFrame = callback => {const id = nextFrame++; frames.set(id, callback); return id;};
globalThis.cancelAnimationFrame = id => frames.delete(id);
const ticks = [], states = [];
const playback = new PlaybackController(tick => ticks.push(tick), () => {}, playing => states.push(playing));
playback.play();
assert.equal(playback.isPlaying, false, 'Cannot play an empty run');
playback.scrubTo(10);
assert.equal(ticks.length, 0);
playback.setDuration(3);
playback.scrubTo(2);
playback.play();
assert.equal(playback.currentTick, 0, 'Play at the end restarts the run');
assert.equal(frames.size, 1);
function frame(now) {
  const [id, callback] = frames.entries().next().value;
  frames.delete(id);
  callback(now);
}
const start = playback.lastFrameTime;
frame(start - 1);
assert.equal(playback.currentTick, 0, 'An early animation timestamp must not move before the first tick');
frame(start + 60);
frame(start + 120);
assert.equal(playback.currentTick, 2);
assert.equal(playback.isPlaying, false);
assert.equal(frames.size, 0, 'Completion must not queue a stale animation frame');
assert.equal(states.at(-1), false);
playback.play();
playback.step(1);
assert.equal(frames.size, 0);
assert.equal(playback.isPlaying, false);
playback.setDuration(1);
playback.scrubTo(-1);
assert.equal(playback.currentTick, 0);
console.log('Frontend regressions passed: band shapes, weak signals, empty runs, completion, replay, step.');
