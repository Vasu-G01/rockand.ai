/**
 * Client for the AstroLand AI backend (Flask, http://127.0.0.1:5000).
 *
 * The AI pilot asks the backend twice:
 *   - /api/plan   once per flight (and every few seconds) for the predicted path
 *   - /api/predict every 100 ms for the next control decision
 *
 * If the backend is unreachable the local fallback controller below takes over,
 * so the autopilot keeps working without a server.
 */

const BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:5000").replace(/\/+$/, "");

/** seconds between control decisions — must match DECISION_DT in the backend */
export const AI_INTERVAL = 0.1;
/** seconds between re-plans of the predicted path */
export const PLAN_EVERY = 4;
/** how far ahead the plan is rolled out (10 Hz steps) */
export const PLAN_STEPS = 160;

async function request(path, { method = "GET", body, timeout = 3000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`${path} responded ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export const health = () => request("/api/health");
export const modelInfo = () => request("/api/model");
export const trainStatus = () => request("/api/train/status");
export const startTraining = (timesteps) => request("/api/train", { method: "POST", body: { timesteps } });
export const stopTraining = () => request("/api/train/stop", { method: "POST", body: {} });
export const reloadModel = () => request("/api/model/reload", { method: "POST", body: {} });
export const predict = (state, padX) =>
  request("/api/predict", { method: "POST", body: { state, pad_x: padX }, timeout: 2000 });
export const plan = (state, padX, steps = PLAN_STEPS) =>
  request("/api/plan", { method: "POST", body: { state, pad_x: padX, steps }, timeout: 4000 });
export const saveFlight = (flight) =>
  request("/api/flights", { method: "POST", body: flight, timeout: 4000 });
export const listFlights = (limit = 50) => request(`/api/flights?limit=${limit}`);

/** The canvas rocket state, shaped exactly like the backend's State dataclass. */
export function telemetryState(r) {
  return {
    x: r.x,
    y: r.y,
    vx: r.vx,
    vy: r.vy,
    a: r.a,
    w: r.w,
    fuel: r.fuel,
    t: r.t,
    throttle: r.throttle,
    status: r.status,
  };
}

/**
 * Local guidance controller — a port of heuristic_action() in backend/ml/predict.py.
 * Used when the backend is offline. Keep the constants in sync with App.jsx.
 */
export function fallbackControl(s, padX) {
  const GROUND_Y = 480;
  const PAD_HALF = 65;

  const alt = Math.max(0, GROUND_Y - s.y);
  const dx = padX - s.x;

  let desVx;
  if (Math.abs(dx) <= PAD_HALF * 0.5) desVx = Math.max(-10, Math.min(10, -s.vx * 0.6));
  else desVx = Math.max(-30, Math.min(30, dx * 0.2));
  const errVx = desVx - s.vx;

  let desVy;
  if (alt > 200) desVy = 40;
  else if (alt > 80) desVy = 22;
  else if (alt > 30) desVy = 10;
  else desVy = 3;

  const tilt = alt > 120 ? 0.35 : 0.18;
  const aCmd = Math.max(-tilt, Math.min(tilt, errVx * 0.008));

  // PD on (angle, angular velocity)
  const signal = 3.0 * (aCmd - s.a) - 1.2 * s.w;

  return {
    thrust: s.fuel > 0 && s.vy > desVy + 1.0,
    left: signal < -0.03,
    right: signal > 0.03,
  };
}
