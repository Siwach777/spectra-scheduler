/** A full-run comparison timeline, controlled by the accessible range input. */
export class ScrubberRibbon {
  constructor(canvas, playhead) {
    this.canvas = canvas;
    this.playhead = playhead;
    this.ctx = canvas.getContext('2d');
    this.buffer = null;
  }
  setBuffer(buffer) { this.buffer = buffer; this.drawBarcode(); }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawBarcode();
  }
  drawBarcode() {
    if (!this.buffer) return;
    const {width, height} = this.canvas.getBoundingClientRect();
    this.ctx.clearRect(0, 0, width, height);
    const dx = width / this.buffer.duration;
    for (let t = 0; t < this.buffer.duration; t++) {
      const hit = this.buffer.actRxHits[t], listening = this.buffer.actRxListening[t];
      this.ctx.fillStyle = hit ? '#5d9e92' : listening ? '#bfcad4' : '#d2b183';
      const y = hit ? 3 : listening ? height * .7 : height * .45;
      this.ctx.fillRect(t * dx, y, Math.max(dx, 1), height - y);
    }
  }
  updatePlayhead(tick) { this.playhead.style.left = `${tick / Math.max(1, this.buffer.duration - 1) * 100}%`; }
}
