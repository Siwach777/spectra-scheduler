/** Simulation playback with one animation loop and explicit state changes. */
export class PlaybackController {
  constructor(onTick, onRender, onState = () => {}) {
    this.onTick = onTick;
    this.onRender = onRender;
    this.onState = onState;
    this.isPlaying = false;
    this.currentTick = 0;
    this.maxDuration = 0;
    this.speedMultiplier = 1;
    this.baseTickIntervalMs = 60;
    this.accumulatedTime = 0;
    this.lastFrameTime = 0;
    this.rafId = null;
  }

  setDuration(duration, tickIntervalMs = 60) {
    this.pause();
    this.maxDuration = duration;
    this.baseTickIntervalMs = tickIntervalMs;
    this.currentTick = 0;
    this.accumulatedTime = 0;
  }

  play() {
    if (this.isPlaying || this.maxDuration < 1) return;
    if (this.currentTick === this.maxDuration - 1) this.scrubTo(0);
    this.isPlaying = true;
    this.accumulatedTime = 0;
    this.lastFrameTime = performance.now();
    this.onState(true);
    this.rafId = requestAnimationFrame(t => this._loop(t));
  }

  pause() {
    this.isPlaying = false;
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.onState(false);
  }

  toggle() {
    if (this.isPlaying) this.pause();
    else this.play();
    return this.isPlaying;
  }

  setSpeed(multiplier) { this.speedMultiplier = Math.max(.2, multiplier); }

  scrubTo(tick) {
    if (this.maxDuration < 1) return;
    this.currentTick = Math.max(0, Math.min(Math.round(tick), this.maxDuration - 1));
    this.accumulatedTime = 0;
    this.onTick(this.currentTick);
    this.onRender(this.currentTick);
  }

  step(delta) { this.pause(); this.scrubTo(this.currentTick + delta); }

  _loop(now) {
    if (!this.isPlaying) return;
    this.rafId = null;
    // A callback can carry the frame timestamp from just before play() ran.
    // Clamp that first delta so floor() cannot move playback to tick -1.
    this.accumulatedTime += Math.max(0, Math.min(now - this.lastFrameTime, 120));
    this.lastFrameTime = now;
    const interval = this.baseTickIntervalMs / this.speedMultiplier;
    const advance = Math.floor(this.accumulatedTime / interval);
    if (advance > 0) {
      this.accumulatedTime -= advance * interval;
      this.currentTick = Math.min(this.currentTick + advance, this.maxDuration - 1);
      this.onTick(this.currentTick);
      this.onRender(this.currentTick);
      if (this.currentTick >= this.maxDuration - 1) { this.pause(); return; }
    }
    this.rafId = requestAnimationFrame(t => this._loop(t));
  }
}
