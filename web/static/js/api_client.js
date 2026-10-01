/**
 * Zero-dependency Fetch Client for Spectra Scheduler Backend API
 */

export class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  async getScenarios() {
    const res = await fetch(`${this.baseUrl}/api/scenarios`);
    if (!res.ok) throw new Error(`Failed to fetch scenarios: ${res.statusText}`);
    return res.json();
  }

  async getSchedulers() {
    const res = await fetch(`${this.baseUrl}/api/schedulers`);
    if (!res.ok) throw new Error(`Failed to fetch schedulers: ${res.statusText}`);
    return res.json();
  }

  async getPresets() {
    const res = await fetch(`${this.baseUrl}/api/presets`);
    if (!res.ok) throw new Error(`Failed to fetch presets: ${res.statusText}`);
    return res.json();
  }

  async runSimulation(payload) {
    let res;
    try {
      res = await fetch(`${this.baseUrl}/api/run`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload), signal: AbortSignal.timeout(20000),
      });
    } catch (error) {
      if (error.name === 'TimeoutError') throw new Error('The server took too long. Please retry the run.');
      throw new Error('The demonstration server is unreachable. Check that web/server.py is running.');
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `Simulation run failed (${res.status})`);
    }
    return res.json();
  }
}
