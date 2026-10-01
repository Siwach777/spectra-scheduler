/** Typed buffers for a bounded synthetic episode and a muted signal-power palette. */
export const COLOR_LUT = new Uint32Array(256);

for (let i = 0; i < 256; i++) {
  const f = i / 255;
  const r = Math.round(24 + f * 110), g = Math.round(33 + f * 130), b = Math.round(44 + f * 136);
  COLOR_LUT[i] = (255 << 24) | (b << 16) | (g << 8) | r;
}

export class EpisodeBuffer {
  constructor(maxDuration = 2048, numBands = 8) {
    this.maxDuration = maxDuration;
    this.numBands = numBands;
    this.duration = 0;

    // 1. Spectral Truth Grid: (maxDuration * numBands) floats for power_dbm
    this.truthPower = new Float32Array(maxDuration * numBands);
    this.truthPower.fill(Number.NEGATIVE_INFINITY);

    // 2. Baseline Receiver Arrays
    this.baseRxBand = new Int16Array(maxDuration).fill(-1);
    this.baseRxListening = new Uint8Array(maxDuration);
    this.baseRxHits = new Uint8Array(maxDuration);
    this.baseRxFalseAlarms = new Uint8Array(maxDuration);
    this.baseMeasuredPower = new Float32Array(maxDuration).fill(Number.NEGATIVE_INFINITY);

    // 3. Active (Smart) Receiver Arrays
    this.actRxBand = new Int16Array(maxDuration).fill(-1);
    this.actRxListening = new Uint8Array(maxDuration);
    this.actRxHits = new Uint8Array(maxDuration);
    this.actRxFalseAlarms = new Uint8Array(maxDuration);
    this.actMeasuredPower = new Float32Array(maxDuration).fill(Number.NEGATIVE_INFINITY);
    this.actMeasuredPw = new Float32Array(maxDuration).fill(0.0);
    this.actReason = [];
  }

  loadSimulationRun(runData) {
    const sc = runData.scenario;
    const bandsChanged = this.numBands !== sc.num_bands;
    this.numBands = sc.num_bands;
    this.duration = sc.duration;

    if (this.duration > this.maxDuration || bandsChanged) {
      this.maxDuration = Math.max(this.maxDuration, this.duration);
      this.truthPower = new Float32Array(this.maxDuration * this.numBands);
      this.baseRxBand = new Int16Array(this.maxDuration);
      this.baseRxListening = new Uint8Array(this.maxDuration);
      this.baseRxHits = new Uint8Array(this.maxDuration);
      this.baseRxFalseAlarms = new Uint8Array(this.maxDuration);
      this.baseMeasuredPower = new Float32Array(this.maxDuration);
      this.actRxBand = new Int16Array(this.maxDuration);
      this.actRxListening = new Uint8Array(this.maxDuration);
      this.actRxHits = new Uint8Array(this.maxDuration);
      this.actRxFalseAlarms = new Uint8Array(this.maxDuration);
      this.actMeasuredPower = new Float32Array(this.maxDuration);
      this.actMeasuredPw = new Float32Array(this.maxDuration);
    }

    this.truthPower.fill(Number.NEGATIVE_INFINITY);
    this.baseRxBand.fill(-1);
    this.baseRxListening.fill(0);
    this.baseRxHits.fill(0);
    this.baseRxFalseAlarms.fill(0);
    this.baseMeasuredPower.fill(Number.NEGATIVE_INFINITY);
    this.actRxBand.fill(-1);
    this.actRxListening.fill(0);
    this.actRxHits.fill(0);
    this.actRxFalseAlarms.fill(0);
    this.actMeasuredPower.fill(Number.NEGATIVE_INFINITY);
    this.actMeasuredPw.fill(0.0);
    this.actReason = [];

    // Ingest truth grid
    const truthGrid = runData.truth_grid || [];
    for (let t = 0; t < this.duration; t++) {
      const txs = truthGrid[t] || [];
      const baseOffset = t * this.numBands;
      for (const tx of txs) {
        if (tx.band >= 0 && tx.band < this.numBands) {
          this.truthPower[baseOffset + tx.band] = Math.max(
            this.truthPower[baseOffset + tx.band],
            tx.power_dbm
          );
        }
      }
    }

    // Ingest baseline steps
    const bSteps = runData.baseline?.data?.steps || [];
    for (let t = 0; t < bSteps.length; t++) {
      const s = bSteps[t];
      this.baseRxBand[t] = s.rx_band;
      this.baseRxListening[t] = s.listening ? 1 : 0;
      this.baseRxHits[t] = s.hit_count > 0 ? 1 : 0;
      this.baseRxFalseAlarms[t] = s.false_alarms > 0 ? 1 : 0;
      this.baseMeasuredPower[t] = s.measured_power ?? Number.NEGATIVE_INFINITY;
    }

    // Ingest active steps
    const aSteps = runData.active?.data?.steps || [];
    for (let t = 0; t < aSteps.length; t++) {
      const s = aSteps[t];
      this.actRxBand[t] = s.rx_band;
      this.actRxListening[t] = s.listening ? 1 : 0;
      this.actRxHits[t] = s.hit_count > 0 ? 1 : 0;
      this.actRxFalseAlarms[t] = s.false_alarms > 0 ? 1 : 0;
      this.actMeasuredPower[t] = s.measured_power ?? Number.NEGATIVE_INFINITY;
      this.actMeasuredPw[t] = s.measured_pw ?? 0.0;
      this.actReason.push(s.decision_reason || "Sweep");
    }
  }

  getTruthPower(tick, band) {
    if (tick < 0 || tick >= this.duration || band < 0 || band >= this.numBands) {
      return Number.NEGATIVE_INFINITY;
    }
    return this.truthPower[tick * this.numBands + band];
  }
}
