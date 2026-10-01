/** Final anonymous track snapshot, using bounds derived from the measurements. */
export class FeatureSpacePlot {
  constructor(canvas, tableBody) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tableBody = tableBody;
    this.activeTracks = [];
    this.archivedTracks = [];
    this.minPw = 0;
    this.maxPw = 10;
    this.minPwr = -100;
    this.maxPwr = -50;
  }

  setData(active = [], archived = []) {
    this.activeTracks = active;
    this.archivedTracks = archived;
    this.minPw = 0; this.maxPw = 10; this.minPwr = -100; this.maxPwr = -50;
    const tracks = [...active, ...archived];
    if (tracks.length) {
      const pws = tracks.map(t => t.mean_pw_us), powers = tracks.map(t => t.mean_power_dbm);
      const padPw = Math.max(.5, (Math.max(...pws) - Math.min(...pws)) * .15);
      this.minPw = Math.max(0, Math.min(...pws) - padPw);
      this.maxPw = Math.max(...pws) + padPw;
      this.minPwr = Math.floor((Math.min(...powers) - 5) / 5) * 5;
      this.maxPwr = Math.ceil((Math.max(...powers) + 5) / 5) * 5;
    }
    this.render();
    this.renderTable();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  render() {
    const {width: w, height: h} = this.canvas.getBoundingClientRect();
    if (!w || !h) return;
    const ctx = this.ctx;
    const style = getComputedStyle(document.documentElement);
    const muted = style.getPropertyValue('--muted').trim();
    const ink = style.getPropertyValue('--ink').trim();
    const line = style.getPropertyValue('--line').trim();
    ctx.clearRect(0, 0, w, h);
    const m = {left: 65, right: 40, top: 26, bottom: 50};
    const pw = w - m.left - m.right, ph = h - m.top - m.bottom;
    if (pw <= 0 || ph <= 0) return;
    const xOf = value => m.left + pw * (value - this.minPw) / (this.maxPw - this.minPw);
    const yOf = value => m.top + ph * (1 - (value - this.minPwr) / (this.maxPwr - this.minPwr));
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.fillStyle = muted;
    ctx.font = '11px sans-serif';
    for (let i = 0; i <= 4; i++) {
      const power = this.minPwr + (this.maxPwr - this.minPwr) * i / 4;
      const y = yOf(power);
      ctx.beginPath(); ctx.moveTo(m.left, y); ctx.lineTo(w - m.right, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(power.toFixed(0), m.left - 9, y + 4);
      const pulse = this.minPw + (this.maxPw - this.minPw) * i / 4;
      const x = xOf(pulse);
      ctx.beginPath(); ctx.moveTo(x, m.top); ctx.lineTo(x, h - m.bottom); ctx.stroke();
      ctx.textAlign = 'center'; ctx.fillText(pulse.toFixed(1), x, h - m.bottom + 18);
    }
    ctx.fillText('Measured pulse width (µs)', m.left + pw / 2, h - 12);
    ctx.save(); ctx.translate(17, m.top + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('Measured power (dBm)', 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(m.left - 8, m.top - 8, pw + 16, ph + 16); ctx.clip();
    for (const track of this.archivedTracks) {
      ctx.strokeStyle = '#aeb9c4'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(xOf(track.mean_pw_us), yOf(track.mean_power_dbm), 5, 0, Math.PI * 2); ctx.stroke();
    }
    for (const track of this.activeTracks) {
      const x = xOf(track.mean_pw_us), y = yOf(track.mean_power_dbm);
      const r = Math.min(5 + Math.sqrt(track.observation_count), 12);
      ctx.fillStyle = track.confirmed ? '#e1f0eb' : '#f7eddc';
      ctx.strokeStyle = track.confirmed ? '#3b8677' : '#b48641'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = ink; ctx.textAlign = x > w - 110 ? 'right' : 'left';
      ctx.fillText(`#${track.track_id}`, x > w - 110 ? x - r - 5 : x + r + 5, y + 4);
    }
    ctx.restore();
    if (!this.activeTracks.length && !this.archivedTracks.length) {
      ctx.fillStyle = muted; ctx.textAlign = 'center';
      ctx.fillText('No measured tracks in this run.', m.left + pw / 2, m.top + ph / 2);
    }
  }

  renderTable() {
    if (!this.activeTracks.length) {
      this.tableBody.innerHTML = '<tr><td colspan="6" class="muted">No active tracks at the end of this run.</td></tr>';
      return;
    }
    this.tableBody.innerHTML = this.activeTracks.map(t => `<tr><td>#${t.track_id}</td><td>${t.last_band}</td><td>${t.mean_power_dbm.toFixed(1)}</td><td>${t.mean_pw_us.toFixed(2)}</td><td>${t.observation_count}</td><td><span class="track-state ${t.confirmed ? '' : 'tentative'}">${t.confirmed ? 'Confirmed' : 'Tentative'}</span></td></tr>`).join('');
  }
}
