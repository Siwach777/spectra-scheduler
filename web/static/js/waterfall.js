import { COLOR_LUT } from './buffer.js';

/** Bounded, shared-coordinate raster and receiver trace display. */
export class WaterfallEngine {
  constructor(rasterCanvas, hudCanvas, tooltip) {
    Object.assign(this, {rasterCanvas, hudCanvas, tooltip});
    this.rasterCtx = rasterCanvas.getContext('2d', {alpha: false});
    this.hudCtx = hudCanvas.getContext('2d');
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    this.buffer = null;
    this.currentTick = 0;
    this.windowTicks = 120;
    this.fogOfWar = false;
    this.viewMode = 'active';
    this.width = this.height = 0;
    this.startTick = 0;
    this.endTick = 120;
    this._setupInteractions();
  }

  setBuffer(buffer, run) { this.buffer = buffer; this.run = run; this.tooltip.style.display = 'none'; this.render(0); }
  setFogOfWar(enabled) { this.fogOfWar = enabled; this.tooltip.style.display = 'none'; this.render(this.currentTick); }
  setViewMode(mode) { this.viewMode = mode; this.tooltip.style.display = 'none'; this.render(this.currentTick); }

  resize() {
    const rect = this.hudCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.width = rect.width;
    this.height = rect.height;
    const dpr = window.devicePixelRatio || 1;
    for (const canvas of [this.rasterCanvas, this.hudCanvas]) {
      canvas.width = Math.round(this.width * dpr);
      canvas.height = Math.round(this.height * dpr);
    }
    this.rasterCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.hudCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render(this.currentTick);
  }

  render(tick) {
    this.currentTick = tick;
    if (!this.buffer || !this.width || !this.height) return;
    const buf = this.buffer;
    const count = Math.min(this.windowTicks, buf.duration);
    this.startTick = Math.min(Math.max(0, tick - Math.floor(count * .75)), Math.max(0, buf.duration - count));
    this.endTick = this.startTick + count;
    const tickWidth = this.width / count, bandHeight = this.height / buf.numBands;
    if (!this.imageData || this.imageData.width !== count || this.imageData.height !== buf.numBands) {
      this.offscreenCanvas.width = count;
      this.offscreenCanvas.height = buf.numBands;
      this.imageData = this.offscreenCtx.createImageData(count, buf.numBands);
      this.pixels = new Uint32Array(this.imageData.data.buffer);
    }
    for (let band = 0; band < buf.numBands; band++) {
      for (let i = 0; i < count; i++) {
        const t = this.startTick + i;
        let power = Number.NEGATIVE_INFINITY;
        if (!this.fogOfWar) power = buf.getTruthPower(t, band);
        else if (t <= tick) {
          if (this.viewMode !== 'baseline' && buf.actRxBand[t] === band && buf.actRxListening[t]) power = buf.actMeasuredPower[t];
          if (this.viewMode !== 'active' && buf.baseRxBand[t] === band && buf.baseRxListening[t]) power = Math.max(power, buf.baseMeasuredPower[t]);
        }
        // Empty cells stay at the chart background rather than an artificial noise level.
        const norm = !Number.isFinite(power) ? 0 : Math.min(1, Math.max(.08, (power + 120) / 65));
        this.pixels[(buf.numBands - 1 - band) * count + i] = COLOR_LUT[Math.floor(norm * 255)];
      }
    }
    this.offscreenCtx.putImageData(this.imageData, 0, 0);
    this.rasterCtx.imageSmoothingEnabled = false;
    this.rasterCtx.drawImage(this.offscreenCanvas, 0, 0, this.width, this.height);
    const ctx = this.hudCtx;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#33414f';
    ctx.beginPath();
    for (let band = 1; band < buf.numBands; band++) { const y = Math.floor(band * bandHeight) + .5; ctx.moveTo(0, y); ctx.lineTo(this.width, y); }
    for (let t = Math.ceil(this.startTick / 20) * 20; t < this.endTick; t += 20) { const x = (t - this.startTick) * tickWidth; ctx.moveTo(x, 0); ctx.lineTo(x, this.height); }
    ctx.stroke();
    const trace = baseline => {
      const bands = baseline ? buf.baseRxBand : buf.actRxBand;
      const listen = baseline ? buf.baseRxListening : buf.actRxListening;
      const hits = baseline ? buf.baseRxHits : buf.actRxHits;
      const alarms = baseline ? buf.baseRxFalseAlarms : buf.actRxFalseAlarms;
      const color = baseline ? '#e2b573' : '#81c8bb';
      const offset = this.viewMode === 'dual' ? (baseline ? .24 : .62) : .45;
      for (let t = this.startTick; t <= Math.min(tick, this.endTick - 1); t++) {
        const band = bands[t];
        if (band < 0 || band >= buf.numBands) continue;
        const x = (t - this.startTick) * tickWidth, y = (buf.numBands - 1 - band) * bandHeight;
        ctx.fillStyle = baseline ? '#e2b57314' : '#81c8bb18';
        ctx.fillRect(x, y, tickWidth, bandHeight);
        if (!listen[t]) {
          ctx.strokeStyle = '#bd98617a';
          ctx.beginPath(); ctx.moveTo(x, y + bandHeight); ctx.lineTo(x + tickWidth, y); ctx.stroke();
        } else {
          ctx.fillStyle = color;
          ctx.fillRect(x, y + bandHeight * offset, Math.max(tickWidth - 1, 1), hits[t] ? Math.max(5, bandHeight * .12) : 1.5);
          if (alarms[t]) { ctx.strokeStyle = '#b5a0d4'; ctx.strokeRect(x + .5, y + 3, Math.max(1, tickWidth - 1), bandHeight - 6); }
        }
      }
      const band = bands[tick];
      if (band >= 0 && band < buf.numBands) {
        const x = (tick - this.startTick) * tickWidth, y = (buf.numBands - 1 - band) * bandHeight;
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        if (baseline && this.viewMode === 'dual') ctx.setLineDash([4, 3]);
        ctx.strokeRect(x, y + 1, Math.max(tickWidth, 4), bandHeight - 2);
        ctx.setLineDash([]); ctx.lineWidth = 1;
      }
    };
    if (this.viewMode !== 'active') trace(true);
    if (this.viewMode !== 'baseline') trace(false);
    const x = (tick - this.startTick) * tickWidth;
    ctx.strokeStyle = '#d4dce6'; ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, this.height); ctx.stroke();
  }

  _setupInteractions() {
    this.hudCanvas.addEventListener('pointermove', event => {
      if (!this.buffer || !this.width) return;
      const rect = this.hudCanvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      const tick = this.startTick + Math.floor(x / this.width * (this.endTick - this.startTick));
      const band = this.buffer.numBands - 1 - Math.floor(y / this.height * this.buffer.numBands);
      if (tick >= this.endTick || band < 0 || tick < 0) return;
      const status = (policy) => {
        if (tick > this.currentTick) return 'Not played yet';
        const s = policy.data.steps[tick];
        if (s.rx_band !== band) return 'Unobserved';
        if (!s.listening) return 'Retuning';
        return s.false_alarms ? `${s.hit_count} detections + false alarm` : s.hit_count ? `${s.hit_count} detections` : 'No detection';
      };
      const power = this.buffer.getTruthPower(tick, band);
      const truth = !this.fogOfWar ? `<div class="tooltip-row"><span class="label">Simulation truth</span><span>${!Number.isFinite(power) ? 'No signal' : `${power.toFixed(1)} dBm`}</span></div>` : '';
      this.tooltip.innerHTML = `<div class="tooltip-header">Tick ${tick} · Band ${band}</div>${truth}<div class="tooltip-row"><span class="label">Comparison</span><span>${status(this.run.active)}</span></div><div class="tooltip-row"><span class="label">Reference</span><span>${status(this.run.baseline)}</span></div>`;
      this.tooltip.style.display = 'block';
      this.tooltip.style.left = `${Math.max(4, Math.min(x + 14, this.width - this.tooltip.offsetWidth - 4))}px`;
      this.tooltip.style.top = `${Math.max(4, Math.min(y + 14, this.height - this.tooltip.offsetHeight - 4))}px`;
    });
    this.hudCanvas.addEventListener('pointerleave', () => { this.tooltip.style.display = 'none'; });
  }
}
