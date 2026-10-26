import { useEffect, useRef, useState } from "react";

import * as api from "./api";
import "./index.css";

const W = 1080;
const H = 560;
const GRAVITY = 9.8 * 5;
const DRY_MASS = 1000;
const FUEL_START = 300;
const MAX_THRUST = 125000;
const BURN_RATE = 20;
const TORQUE = 2.6;

const GROUND_Y = H - 80;
const PAD_HALF = 65; // landing pad is 130px wide, centred
const LIMIT_VY = 45;
const LIMIT_VX = 25;
const LIMIT_RAD = 0.21; // ~12 degrees
const LIMIT_DEG = (LIMIT_RAD * 180) / Math.PI;

const rand = (min, max) => min + Math.random() * (max - min);

// Both the rocket and the landing pad spawn at random positions inside the
// screen bounds on every new game (mirror PAD_X_MIN/MAX + random_start in
// backend/simulation/physics.py so the trained policy sees the same world).
const PAD_X_MIN = PAD_HALF + 50;
const PAD_X_MAX = W - PAD_HALF - 50;
const randomPadX = () => rand(PAD_X_MIN, PAD_X_MAX);

// The rocket launches from the upper middle of the screen with a random
// sideways drift, so no two runs start the same way.
const randomStart = () => ({
  x: rand(W * 0.32, W * 0.68),
  y: rand(70, 300),
  vx: (Math.random() < 0.5 ? -1 : 1) * rand(25, 55),
});

const newRocket = () => {
  const s = randomStart();
  return {
    x: s.x, y: s.y, vx: s.vx, vy: 0, a: 0, w: 0,
    fuel: FUEL_START, throttle: 0, t: 0,
    status: "FLYING", msg: "Mission in progress",
  };
};

const CONTROL_KEYS = ["ArrowUp", "ArrowLeft", "ArrowRight", "Space", "KeyW", "KeyA", "KeyD"];

const initialTelemetry = {
  x: W / 2, vx: 0, vy: 0, angle: 0, w: 0, speed: 0,
  fuel: FUEL_START, alt: 0,
  status: "FLYING", throttle: 0, time: 0, msg: "",
};

/** Samples kept for the live speed / angular-velocity graphs (~8 s at 20 Hz). */
const MOTION_SAMPLES = 160;

/** Star field behind the rocket — generated once, outside render. */
const STARS = Array.from({ length: 140 }, () => ({
  x: Math.random() * W,
  y: Math.random() * (H - 200),
  s: Math.random() * 1.6 + 0.5,
  p: Math.random() * 6.28,
}));

export default function App() {
  const canvasRef = useRef(null);
  const keys = useRef({});
  const rocket = useRef(newRocket());
  // Landing pad for this game: the state feeds the UI, the ref feeds the
  // render loop (which must not depend on React state).
  const [padX, setPadX] = useState(() => randomPadX());
  const padRef = useRef(padX);
  const particles = useRef([]);
  const trail = useRef([]);
  const attempt = useRef(0);
  const pausedRef = useRef(false);
  const stars = useRef(STARS);

  const [telemetry, setTelemetry] = useState(initialTelemetry);
  const [history, setHistory] = useState([]);
  const [motion, setMotion] = useState([]); // {sp, w} samples for the graphs
  const [log, setLog] = useState([]);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  /* ---------------- AI pilot (backend) ---------------- */
  const [aiOn, setAiOn] = useState(false);
  const [backend, setBackend] = useState("checking"); // online | offline | checking
  const [model, setModel] = useState(null);           // /api/model info
  const [training, setTraining] = useState(null);     // /api/train/status
  const [pilotSource, setPilotSource] = useState("model");
  const [trainTimesteps, setTrainTimesteps] = useState(500000);
  const [planLabel, setPlanLabel] = useState(null);
  const [manualHeld, setManualHeld] = useState(false);

  // the render loop lives in a mount-only effect, so it talks to these refs
  const aiRef = useRef(false);
  const controlsRef = useRef({ thrust: false, left: false, right: false });
  const planRef = useRef([]); // predicted path [{x, y}, ...]
  const aiClock = useRef(0);
  const planClock = useRef(0);
  const needsPlan = useRef(false);
  const controlBusy = useRef(false);
  const planBusy = useRef(false);
  const sourceRef = useRef("model");

  useEffect(() => {
    aiRef.current = aiOn;
  }, [aiOn]);

  const toggleAI = () => {
    const next = !aiOn;
    if (next) {
      needsPlan.current = true; // ask the backend for a fresh best path
      aiClock.current = 0;
      planClock.current = 0;
    } else {
      planRef.current = [];
      setPlanLabel(null);
    }
    setAiOn(next);
  };

  /** Ask the backend for the next control (falls back to local guidance). */
  const askControl = (r, pad) => {
    if (controlBusy.current) return;
    controlBusy.current = true;
    const state = api.telemetryState(r);
    api
      .predict(state, pad)
      .then((res) => {
        controlsRef.current = res.controls;
        sourceRef.current = res.source;
        setPilotSource(res.source);
        setBackend("online");
      })
      .catch(() => {
        controlsRef.current = api.fallbackControl(state, pad);
        sourceRef.current = "local";
        setPilotSource("local");
        setBackend("offline");
      })
      .finally(() => {
        controlBusy.current = false;
      });
  };

  /** Ask the backend to roll the policy forward = the predicted "best path". */
  const askPlan = (r, pad) => {
    if (planBusy.current) return;
    planBusy.current = true;
    api
      .plan(api.telemetryState(r), pad)
      .then((res) => {
        planRef.current = (res.path || []).map(([x, y]) => ({ x, y }));
        const landing = res.predicted_landing;
        setPlanLabel(
          landing
            ? landing.ok
              ? "Prediction: soft landing on the pad"
              : `Prediction: ${landing.message}`
            : "Prediction: descent in progress"
        );
        setBackend("online");
      })
      .catch(() => {
        planRef.current = [];
        setPlanLabel(null);
        setBackend("offline");
      })
      .finally(() => {
        planBusy.current = false;
      });
  };

  // keep the model / training panels in sync with the backend
  const trainingRunning = !!(training && training.running);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        setBackend("online");
        setModel(h.model);
        setTraining(h.training);
      } catch {
        if (alive) setBackend("offline");
      }
    };
    poll();
    const id = setInterval(poll, trainingRunning ? 1000 : 6000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [trainingRunning]);

  const startTraining = () => {
    api
      .startTraining(trainTimesteps)
      .then(() => setBackend("online"))
      .catch(() => setBackend("offline"));
  };
  const stopTraining = () => {
    api.stopTraining().catch(() => {});
  };

  const reset = () => {
    const fresh = newRocket();
    rocket.current = fresh;
    const nextPad = randomPadX(); // new game = new landing spot
    padRef.current = nextPad;
    setPadX(nextPad);
    particles.current = [];
    trail.current = [];
    controlsRef.current = { thrust: false, left: false, right: false };
    planRef.current = [];
    setPlanLabel(null);
    aiClock.current = 0;
    planClock.current = 0;
    needsPlan.current = aiRef.current;
    setHistory([]);
    setMotion([]);
    setPaused(false);
    // show the freshly spawned rocket, not the neutral placeholder values
    setTelemetry({
      ...initialTelemetry,
      x: fresh.x,
      vx: fresh.vx,
      speed: Math.abs(fresh.vx),
      alt: Math.max(0, GROUND_Y - fresh.y),
    });
  };

  useEffect(() => {
    const down = (e) => {
      keys.current[e.code] = true;
      if (["ArrowUp", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) e.preventDefault();
      if (CONTROL_KEYS.includes(e.code)) setManualHeld(true); // keys beat the autopilot
      if (e.code === "KeyR") reset();
      if (e.code === "KeyP") setPaused((v) => !v);
    };
    const up = (e) => {
      keys.current[e.code] = false;
      if (CONTROL_KEYS.includes(e.code)) setManualHeld(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);

    let frame;
    let last = performance.now();
    let clock = 0, sample = 0, tele = 0, trailT = 0;
    let lastStatus = "FLYING";

    const spawn = (p) => particles.current.push(p);

    const explode = (x, y) => {
      for (let i = 0; i < 70; i++) {
        const ang = Math.random() * Math.PI * 2;
        const sp = 40 + Math.random() * 260;
        spawn({
          kind: "fire", x, y,
          vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp - 60,
          life: 0.6 + Math.random() * 0.8, max: 1.4, size: 3 + Math.random() * 6,
        });
      }
      for (let i = 0; i < 30; i++) {
        spawn({
          kind: "smoke", x: x + (Math.random() - 0.5) * 30, y,
          vx: (Math.random() - 0.5) * 60, vy: -20 - Math.random() * 60,
          life: 1.5 + Math.random(), max: 2.5, size: 8 + Math.random() * 10,
        });
      }
    };

    const loop = (now) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        frame = requestAnimationFrame(loop);
        return;
      }
      const ctx = canvas.getContext("2d");
      const r = rocket.current;
      const landingPadX = padRef.current; // where the pad sits for this game
      const dt = Math.min((now - last) / 1000, 0.04);
      last = now;
      const isPaused = pausedRef.current;
      if (!isPaused) clock += dt;

      /* ---------- physics (unchanged from your version) ---------- */
      if (!isPaused && r.status === "FLYING") {
        const k = keys.current;
        const manualKeys =
          k.Space || k.ArrowUp || k.KeyW || k.ArrowLeft || k.KeyA || k.ArrowRight || k.KeyD;
        const aiActive = aiRef.current && !manualKeys; // you can always grab the sticks

        let thrusting;
        let left;
        let right;
        if (aiActive) {
          const c = controlsRef.current;
          thrusting = !!c.thrust;
          left = !!c.left;
          right = !!c.right;
        } else {
          thrusting = k.Space || k.ArrowUp || k.KeyW;
          left = k.ArrowLeft || k.KeyA;
          right = k.ArrowRight || k.KeyD;
        }

        // 1) next control decision every 100 ms, 2) re-plan the path now and then
        if (aiRef.current) {
          aiClock.current += dt;
          planClock.current += dt;
          if (aiActive && aiClock.current >= api.AI_INTERVAL) {
            aiClock.current = 0;
            askControl(r, padRef.current);
          }
          if (needsPlan.current || planClock.current >= api.PLAN_EVERY) {
            planClock.current = 0;
            needsPlan.current = false;
            askPlan(r, padRef.current);
          }
        }

        r.throttle = thrusting && r.fuel > 0 ? 1 : 0;
        if (left) r.w -= TORQUE * dt;
        if (right) r.w += TORQUE * dt;
        r.w *= Math.max(0, 1 - 0.35 * dt);
        r.a += r.w * dt;
        r.t += dt;

        if (r.throttle) {
          const burn = Math.min(r.fuel, BURN_RATE * dt);
          r.fuel -= burn;
          const mass = DRY_MASS + r.fuel;
          const thrust = MAX_THRUST * (burn / Math.max(BURN_RATE * dt, 0.001));
          r.vx += ((Math.sin(r.a) * thrust) / mass) * dt;
          r.vy -= ((Math.cos(r.a) * thrust) / mass) * dt;

          // exhaust particles leave the nozzle, opposite to thrust
          const nx = r.x - Math.sin(r.a) * 24;
          const ny = r.y + Math.cos(r.a) * 24;
          for (let i = 0; i < 3; i++) {
            const sp = 170 + Math.random() * 120;
            spawn({
              kind: "fire", x: nx, y: ny,
              vx: -Math.sin(r.a) * sp + (Math.random() - 0.5) * 50 + r.vx * 0.3,
              vy: Math.cos(r.a) * sp + (Math.random() - 0.5) * 30 + r.vy * 0.3,
              life: 0.35 + Math.random() * 0.25, max: 0.6, size: 3 + Math.random() * 4,
            });
          }
          if (Math.random() < 0.5) {
            spawn({
              kind: "smoke", x: nx, y: ny,
              vx: -Math.sin(r.a) * 60 + (Math.random() - 0.5) * 40,
              vy: Math.cos(r.a) * 60,
              life: 1 + Math.random() * 0.6, max: 1.6, size: 4 + Math.random() * 4,
            });
          }
        }

        r.vy += GRAVITY * dt;
        r.x += r.vx * dt;
        r.y += r.vy * dt;
        r.x = Math.max(20, Math.min(W - 20, r.x));

        trailT += dt;
        if (trailT > 0.04) {
          trailT = 0;
          trail.current.push({ x: r.x, y: r.y });
          if (trail.current.length > 90) trail.current.shift();
        }

        if (r.y >= GROUND_Y) {
          r.y = GROUND_Y;
          const onPad = Math.abs(r.x - landingPadX) <= PAD_HALF;
          const soft =
            Math.abs(r.vy) <= LIMIT_VY && Math.abs(r.vx) <= LIMIT_VX && Math.abs(r.a) <= LIMIT_RAD;
          const ok = soft && onPad;
          const score = ok
            ? Math.round(200 + r.fuel - Math.abs(r.vy) * 2)
            : -Math.round(Math.abs(r.vy) + Math.abs(r.vx) * 2 + Math.abs(r.a) * 50);
          const impact = { vx: r.vx, vy: r.vy, angle: r.a };

          r.status = ok ? "LANDED" : "CRASHED";
          r.msg = ok
            ? "Landing successful!"
            : soft
            ? "Missed the landing pad"
            : "Rocket destroyed. Try again";
          if (!ok) explode(r.x, r.y);
          r.vx = 0;
          r.vy = 0;

          attempt.current += 1;
          const mode = aiRef.current ? "ai" : "manual";
          const entry = { id: attempt.current, status: r.status, score, mode };
          setLog((old) => [entry, ...old].slice(0, 50));
          // keep the backend history in sync (best effort — the box may be offline)
          api
            .saveFlight({
              mode,
              pilot: aiRef.current ? sourceRef.current : null,
              status: r.status,
              score,
              time: r.t,
              fuel: r.fuel,
              vx: impact.vx,
              vy: impact.vy,
              angle: impact.angle,
              message: r.msg,
            })
            .catch(() => {});
        }

        sample += dt;
        if (sample >= 0.15) {
          sample = 0;
          const p = { x: r.x, y: r.y };
          setHistory((old) => [...old.slice(-79), p]);
        }
      }

      /* ---------- particles ---------- */
      if (!isPaused) {
        const list = particles.current;
        for (let i = list.length - 1; i >= 0; i--) {
          const p = list[i];
          p.life -= dt;
          if (p.life <= 0) {
            list.splice(i, 1);
            continue;
          }
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          if (p.kind === "fire") p.vy += 80 * dt;
          else p.vy -= 10 * dt;
        }
      }

      /* ---------- draw: sky ---------- */
      const sky = ctx.createLinearGradient(0, 0, 0, H);
      sky.addColorStop(0, "#04081a");
      sky.addColorStop(0.55, "#111c47");
      sky.addColorStop(1, "#1c4452");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);

      const mg = ctx.createRadialGradient(930, 95, 4, 930, 95, 130);
      mg.addColorStop(0, "rgba(190,225,255,0.5)");
      mg.addColorStop(1, "rgba(190,225,255,0)");
      ctx.fillStyle = mg;
      ctx.fillRect(780, 0, 300, 260);
      ctx.fillStyle = "#e8f3ff";
      ctx.beginPath();
      ctx.arc(930, 95, 24, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#fff";
      stars.current.forEach((s) => {
        ctx.globalAlpha = 0.35 + 0.65 * Math.abs(Math.sin(clock * 0.8 + s.p));
        ctx.fillRect(s.x, s.y, s.s, s.s);
      });
      ctx.globalAlpha = 1;

      const ridge = (base, amp, ph, col) => {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.moveTo(0, H);
        for (let x = 0; x <= W; x += 10) {
          ctx.lineTo(x, base + Math.sin(x * 0.006 + ph) * amp + Math.sin(x * 0.017 + ph * 2) * amp * 0.35);
        }
        ctx.lineTo(W, H);
        ctx.closePath();
        ctx.fill();
      };
      ridge(H - 200, 40, 1, "#13264d");
      ridge(H - 150, 30, 3, "#0f3142");
      ridge(H - 105, 20, 5, "#0f3a30");

      /* ground + glowing pad */
      const gg = ctx.createLinearGradient(0, H - 65, 0, H);
      gg.addColorStop(0, "#17503a");
      gg.addColorStop(1, "#08201a");
      ctx.fillStyle = gg;
      ctx.fillRect(0, H - 65, W, 65);
      ctx.fillStyle = "rgba(70,230,177,0.5)";
      ctx.fillRect(0, H - 65, W, 2);

      const beam = ctx.createLinearGradient(0, H - 330, 0, H - 70);
      beam.addColorStop(0, "rgba(70,230,177,0)");
      beam.addColorStop(1, "rgba(70,230,177,0.14)");
      ctx.fillStyle = beam;
      ctx.fillRect(landingPadX - PAD_HALF, H - 330, PAD_HALF * 2, 260);

      ctx.save();
      ctx.shadowColor = "#46e6b1";
      ctx.shadowBlur = 24;
      ctx.fillStyle = "#46e6b1";
      ctx.fillRect(landingPadX - PAD_HALF, H - 70, PAD_HALF * 2, 6);
      ctx.restore();
      const pulse = 0.5 + 0.5 * Math.sin(clock * 4);
      [-PAD_HALF, PAD_HALF].forEach((dx) => {
        ctx.fillStyle = `rgba(255,214,102,${0.5 + pulse * 0.5})`;
        ctx.beginPath();
        ctx.arc(landingPadX + dx, H - 74, 4, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = "rgba(200,225,235,0.75)";
      ctx.font = "12px 'Share Tech Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("LANDING ZONE", landingPadX, H - 82);
      ctx.textAlign = "start";

      /* trail */
      const tr = trail.current;
      for (let i = 1; i < tr.length; i++) {
        const f = i / tr.length;
        ctx.strokeStyle = `rgba(94,234,255,${f * 0.65})`;
        ctx.lineWidth = 1 + f * 2.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(tr[i - 1].x, tr[i - 1].y);
        ctx.lineTo(tr[i].x, tr[i].y);
        ctx.stroke();
      }

      /* predicted path from the trained policy (AI pilot) */
      const planPts = planRef.current;
      if (aiRef.current && planPts.length > 1) {
        ctx.save();
        ctx.setLineDash([10, 8]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255,214,102,0.92)";
        ctx.beginPath();
        ctx.moveTo(planPts[0].x, planPts[0].y);
        for (let i = 1; i < planPts.length; i++) ctx.lineTo(planPts[i].x, planPts[i].y);
        ctx.stroke();
        ctx.setLineDash([]);
        // breadcrumbs every second of predicted flight
        ctx.fillStyle = "rgba(255,214,102,0.7)";
        for (let i = 0; i < planPts.length; i += 10) {
          ctx.beginPath();
          ctx.arc(planPts[i].x, planPts[i].y, 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
        // predicted touchdown point
        const end = planPts[planPts.length - 1];
        ctx.fillStyle = "#ffd666";
        ctx.beginPath();
        ctx.moveTo(end.x, end.y - 8);
        ctx.lineTo(end.x + 8, end.y);
        ctx.lineTo(end.x, end.y + 8);
        ctx.lineTo(end.x - 8, end.y);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      /* smoke first, then rocket, then fire on top */
      particles.current.forEach((p) => {
        if (p.kind !== "smoke") return;
        const t = p.life / p.max;
        ctx.fillStyle = `rgba(165,180,195,${t * 0.22})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (1.8 - t), 0, Math.PI * 2);
        ctx.fill();
      });

      if (r.status !== "CRASHED") {
        ctx.save();
        ctx.translate(r.x, r.y);
        ctx.rotate(r.a);

        if (r.throttle && r.fuel > 0 && r.status === "FLYING") {
          const fl = 28 + Math.random() * 16;
          const fg = ctx.createLinearGradient(0, 18, 0, 18 + fl);
          fg.addColorStop(0, "#ffffff");
          fg.addColorStop(0.3, "#ffd27a");
          fg.addColorStop(1, "rgba(255,90,30,0)");
          ctx.fillStyle = fg;
          ctx.shadowColor = "#ff9a3c";
          ctx.shadowBlur = 30;
          ctx.beginPath();
          ctx.moveTo(-6, 18);
          ctx.quadraticCurveTo(0, 18 + fl * 1.3, 6, 18);
          ctx.closePath();
          ctx.fill();
          ctx.shadowBlur = 0;
        }

        // legs
        ctx.strokeStyle = "#8fa3bd";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(-8, 12); ctx.lineTo(-17, 24);
        ctx.moveTo(8, 12); ctx.lineTo(17, 24);
        ctx.stroke();

        // fins
        ctx.fillStyle = "#2b3a55";
        ctx.beginPath();
        ctx.moveTo(-9, 2); ctx.lineTo(-21, 22); ctx.lineTo(-9, 16);
        ctx.closePath(); ctx.fill();
        ctx.beginPath();
        ctx.moveTo(9, 2); ctx.lineTo(21, 22); ctx.lineTo(9, 16);
        ctx.closePath(); ctx.fill();

        // body
        const bg = ctx.createLinearGradient(-11, 0, 11, 0);
        bg.addColorStop(0, "#9fb0c6");
        bg.addColorStop(0.35, "#ffffff");
        bg.addColorStop(1, "#8ea0b8");
        ctx.fillStyle = bg;
        ctx.beginPath();
        ctx.moveTo(0, -34);
        ctx.bezierCurveTo(14, -18, 12, 4, 11, 18);
        ctx.lineTo(-11, 18);
        ctx.bezierCurveTo(-12, 4, -14, -18, 0, -34);
        ctx.fill();

        // nose cone
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.moveTo(0, -34);
        ctx.quadraticCurveTo(8, -26, 9, -18);
        ctx.lineTo(-9, -18);
        ctx.quadraticCurveTo(-8, -26, 0, -34);
        ctx.fill();

        // window
        ctx.fillStyle = "#0b1d36";
        ctx.strokeStyle = "#5eeaff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(0, -6, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // nozzle
        ctx.fillStyle = "#3a4660";
        ctx.fillRect(-6, 18, 12, 4);
        ctx.restore();
      }

      ctx.globalCompositeOperation = "lighter";
      particles.current.forEach((p) => {
        if (p.kind !== "fire") return;
        const t = p.life / p.max;
        ctx.fillStyle = `hsla(${15 + 40 * t},100%,${45 + 20 * t}%,${t * 0.85})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * t + 1, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalCompositeOperation = "source-over";

      /* ---------- telemetry (throttled) ---------- */
      tele += dt;
      if (tele > 0.05 || r.status !== lastStatus) {
        tele = 0;
        lastStatus = r.status;
        const speed = Math.hypot(r.vx, r.vy);
        setTelemetry({
          x: r.x,
          vx: r.vx,
          vy: r.vy,
          angle: (r.a * 180) / Math.PI,
          w: r.w,
          speed,
          fuel: r.fuel,
          alt: Math.max(0, GROUND_Y - r.y),
          status: r.status,
          throttle: r.throttle,
          time: r.t,
          msg: r.msg,
        });
        if (r.status === "FLYING") {
          const sample = { sp: speed, w: r.w };
          setMotion((old) => (old.length >= MOTION_SAMPLES ? [...old.slice(1), sample] : [...old, sample]));
        }
      }

      frame = requestAnimationFrame(loop);
    };

    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const t = telemetry;
  const fuelPercent = Math.max(0, (t.fuel / FUEL_START) * 100);
  const mm = String(Math.floor(t.time / 60)).padStart(2, "0");
  const ss = String(Math.floor(t.time % 60)).padStart(2, "0");

  // live landing read-outs: green while the value is inside the limits,
  // red the moment it drifts out — updated with every telemetry tick.
  const okVy = Math.abs(t.vy) <= LIMIT_VY;
  const okVx = Math.abs(t.vx) <= LIMIT_VX;
  const okAngle = Math.abs(t.angle) <= LIMIT_DEG;
  const dx = Math.round(t.x - padX);
  const okPad = Math.abs(dx) <= PAD_HALF;

  const conditions = [
    { label: "Vertical speed", limit: `≤ ${LIMIT_VY} m/s`, value: `${t.vy.toFixed(1)} m/s`, ok: okVy },
    { label: "Horizontal speed", limit: `≤ ${LIMIT_VX} m/s`, value: `${t.vx.toFixed(1)} m/s`, ok: okVx },
    { label: "Angle", limit: `≤ ${Math.round(LIMIT_DEG)}°`, value: `${t.angle.toFixed(1)}°`, ok: okAngle },
    { label: "Over the pad", limit: `± ${PAD_HALF} px`, value: `${dx > 0 ? "+" : ""}${dx} px`, ok: okPad },
  ];

  // scales for the live speed / angular-velocity graphs
  const speedMax = Math.max(60, Math.ceil(motion.reduce((m, s) => Math.max(m, s.sp), 0) / 10) * 10);
  const wMax = Math.max(1, Math.ceil(motion.reduce((m, s) => Math.max(m, Math.abs(s.w)), 0) * 10) / 10);

  const attempts = log.length;
  const landings = log.filter((l) => l.status === "LANDED").length;
  const best = attempts ? Math.max(...log.map((l) => l.score)) : "–";

  const pilotLabel = !aiOn
    ? "Manual flight"
    : pilotSource === "model"
      ? "PPO policy"
      : pilotSource === "heuristic"
        ? "Guidance law"
        : pilotSource === "local"
          ? "Local fallback"
          : "Standing by";
  const backendLabel =
    backend === "online" ? "Connected" : backend === "offline" ? "Offline" : "Checking…";

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <div className="grass-block" aria-hidden="true" />
          <div>
            <h1>ROCKET LANDING SIMULATOR</h1>
            <p>Fly &nbsp;•&nbsp; Burn &nbsp;•&nbsp; Land</p>
          </div>
        </div>
        <p className="quote">“Small steps, big landings.”</p>
        <div className="pilot">
          <div className="creeper" aria-hidden="true" />
          <div>
            <strong>Mission 001</strong>
            <span className={`tag ${paused ? "warn" : ""}`}>{paused ? "Paused" : "Live"}</span>
          </div>
        </div>
      </header>

      <div className="shell">
        <section className="center">
          <div className="panel stage-panel">
            <div className="stage">
              <canvas ref={canvasRef} width={W} height={H} className="simulation-canvas" />

              <div className="hud hud-left">
                <HudRow icon="🚀" label="Altitude" value={`${t.alt.toFixed(0)} m`} />
                <HudRow icon="⬇️" label="Vertical speed" value={`${t.vy.toFixed(1)} m/s`} tone={okVy ? "ok" : "bad"} />
                <HudRow icon="↔️" label="Horizontal speed" value={`${t.vx.toFixed(1)} m/s`} tone={okVx ? "ok" : "bad"} />
                <HudRow icon="⛽" label="Fuel" value={`${fuelPercent.toFixed(0)}%`} />
              </div>

              <div className="hud hud-right">
                <HudRow icon="⏱️" label="Time" value={`${mm}:${ss}`} />
                <HudRow icon="📐" label="Angle" value={`${t.angle.toFixed(1)}°`} tone={okAngle ? "ok" : "bad"} />
                <HudRow
                  icon={<i className={`dot ${t.status.toLowerCase()}`} />}
                  label="Status"
                  value={t.status === "FLYING" ? "In flight" : t.status === "LANDED" ? "Landed" : "Crashed"}
                />
              </div>

              {t.status !== "FLYING" && (
                <div className={`result ${t.status.toLowerCase()}`}>
                  <h2>{t.status === "LANDED" ? "LANDED!" : "CRASHED"}</h2>
                  <p>{t.msg}</p>
                  <button className="mc-btn" onClick={reset}>Try again (R)</button>
                </div>
              )}
              {paused && t.status === "FLYING" && (
                <div className="result paused">
                  <h2>PAUSED</h2>
                  <button className="mc-btn" onClick={() => setPaused(false)}>Resume (P)</button>
                </div>
              )}
            </div>
          </div>

          <div className="bottom-row">
            <Panel icon="🧱" title="Flight stats">
              <div className="stat-grid">
                <div><span>Attempts</span><strong>{attempts}</strong></div>
                <div><span>Landings</span><strong>{landings}</strong></div>
                <div><span>Best score</span><strong>{best}</strong></div>
                <div><span>Engine</span>
                  <strong className={t.throttle ? "ok" : ""}>
                    {t.fuel <= 0 ? "No fuel" : t.throttle ? "Burning" : "Standby"}
                  </strong>
                </div>
              </div>
              <div className="last5">
                <h4>Last 5 results</h4>
                {log.length === 0 && <p className="empty">No landings yet. Take off!</p>}
                {log.slice(0, 5).map((l) => (
                  <div key={l.id} className={`result-row ${l.status.toLowerCase()}`}>
                    <span>#{l.id}</span>
                    <span>
                      {l.mode === "ai" ? "🤖 " : "🎮 "}
                      {l.status === "LANDED" ? "Landed" : "Crash"}
                    </span>
                    <b>{l.score}</b>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel icon="📈" title="Trajectory monitor">
              <svg className="traj-svg" viewBox="0 0 320 190" role="img" aria-label="Live rocket trajectory">
                {[30, 65, 100, 135, 170].map((y) => (
                  <line key={y} x1="12" y1={y} x2="308" y2={y} className="chart-grid" />
                ))}
                {[40, 100, 160, 220, 280].map((x) => (
                  <line key={x} x1={x} y1="15" x2={x} y2="175" className="chart-grid" />
                ))}
                {history.length > 1 && (
                  <polyline
                    points={history
                      .map((p) => `${12 + (p.x / W) * 296},${15 + (p.y / H) * 155}`)
                      .join(" ")}
                    className="trajectory-line"
                  />
                )}
                <circle
                  cx={12 + (padX / W) * 296}
                  cy={15 + (GROUND_Y / H) * 155}
                  r="5"
                  className="target-dot"
                />
              </svg>
            </Panel>

            <Panel icon="📊" title="Speed & angular velocity">
              <div className="charts">
                <div className="chart-head">
                  <span><i className="swatch speed" />Speed</span>
                  <strong>{t.speed.toFixed(1)} m/s</strong>
                </div>
                <LineChart points={motion.map((s) => s.sp)} color="#5eeaff" min={0} max={speedMax} />

                <div className="chart-head">
                  <span><i className="swatch spin" />Angular velocity</span>
                  <strong>{t.w.toFixed(2)} rad/s</strong>
                </div>
                <LineChart points={motion.map((s) => s.w)} color="#ffd666" min={-wMax} max={wMax} zero />
              </div>
            </Panel>
          </div>
        </section>

        <aside className="right">
          <Panel icon="🚀" title="Rocket controls">
            <ul className="controls">
              <li><b className="key">W</b> Thrust <em>(↑ / Space)</em></li>
              <li><b className="key">A</b> Rotate left <em>(←)</em></li>
              <li><b className="key">D</b> Rotate right <em>(→)</em></li>
              <li><b className="key">R</b> Restart mission</li>
            </ul>
          </Panel>

          <Panel icon="🎯" title="Landing conditions">
            <ul className="conditions">
              {conditions.map((c) => (
                <li key={c.label} className={c.ok ? "ok" : "bad"}>
                  <span className="mark">{c.ok ? "✔" : "✖"}</span>
                  <span className="cond-label">
                    {c.label}
                    <em>{c.limit}</em>
                  </span>
                  <b className={`cond-value ${c.ok ? "ok" : "bad"}`}>{c.value}</b>
                </li>
              ))}
            </ul>
          </Panel>

          <Panel icon="🧠" title="ML autopilot">
            <div className="ai-grid">
              <div><span>Backend</span><strong className={backend === "online" ? "ok" : backend === "offline" ? "bad" : ""}>{backendLabel}</strong></div>
              <div><span>Policy</span><strong>{model && model.loaded ? "Trained PPO" : "Guidance law"}</strong></div>
              <div><span>Pilot</span><strong>{pilotLabel}</strong></div>
              <div>
                <span>Eval landings</span>
                <strong>
                  {model && model.success_rate != null
                    ? `${Math.round(model.success_rate * 100)}%`
                    : "–"}
                </strong>
              </div>
            </div>

            {aiOn && manualHeld && (
              <p className="ai-note warn">Manual override — you have the controls</p>
            )}
            {planLabel && <p className="ai-note">{planLabel}</p>}

            {trainingRunning ? (
              <div className="training">
                <div className="train-bar">
                  <i style={{ width: `${Math.min(100, training.percent || 0)}%` }} />
                </div>
                <div className="train-meta">
                  <span>{Number(training.step || 0).toLocaleString()} / {Number(training.total || 0).toLocaleString()} steps</span>
                  <span>{training.fps || 0} fps</span>
                  <span>
                    {training.success_rate != null
                      ? `${Math.round(training.success_rate * 100)}% landing`
                      : "learning…"}
                  </span>
                </div>
                <button className="mc-btn stone" onClick={stopTraining}>⏹ Stop training</button>
              </div>
            ) : (
              <div className="training">
                <label className="train-pick">
                  <span>Training budget</span>
                  <select
                    value={trainTimesteps}
                    onChange={(e) => setTrainTimesteps(Number(e.target.value))}
                  >
                    <option value={150000}>Quick · 150k steps</option>
                    <option value={500000}>Standard · 500k steps</option>
                    <option value={2000000}>Thorough · 2M steps</option>
                  </select>
                </label>
                <button className="mc-btn" onClick={startTraining}>🎓 Train model</button>
                {training && training.eval && (
                  <p className="ai-note">
                    Last run: {Math.round(training.eval.success_rate * 100)}% landings over{" "}
                    {training.eval.episodes} attempts
                  </p>
                )}
                {training && training.error && <p className="ai-note bad">{training.error}</p>}
              </div>
            )}
          </Panel>

          <Panel icon="🧰" title="Quick actions">
            <div className="actions">
              <button
                className={`mc-btn ai-toggle ${aiOn ? "on" : ""}`}
                onClick={toggleAI}
              >
                {aiOn ? "🤖 AI pilot: ON" : "🤖 AI pilot: OFF"}
              </button>
              <button className="mc-btn stone" onClick={() => setPaused((v) => !v)}>
                {paused ? "▶ Resume" : "Ⅱ Pause"}
              </button>
              <button className="mc-btn stone" onClick={reset}>↻ Reset simulation</button>
              <button className="mc-btn stone" onClick={() => { setLog([]); attempt.current = 0; }}>
                🗑 Clear results
              </button>
            </div>
          </Panel>
        </aside>
      </div>

      <footer className="footer">
        <span>Rocket Landing Simulator</span>
        <span>React + Canvas</span>
        <span>Reach for the stars!</span>
      </footer>
    </main>
  );
}

function Panel({ icon, title, children }) {
  return (
    <section className="panel">
      <h3 className="panel-title"><span>{icon}</span>{title}</h3>
      {children}
    </section>
  );
}

function HudRow({ icon, label, value, tone }) {
  return (
    <div className="hud-row">
      <span className="hud-icon">{icon}</span>
      <div>
        <small>{label}</small>
        <strong className={tone ? `tone ${tone}` : ""}>{value}</strong>
      </div>
    </div>
  );
}

/** Tiny live line chart used for the speed / angular-velocity panels. */
function LineChart({ points, color, min, max, zero = false }) {
  const CW = 320;
  const CH = 96;
  const padL = 8;
  const padR = 8;
  const padT = 10;
  const padB = 12;
  const span = Math.max(max - min, 1e-6);
  const innerW = CW - padL - padR;
  const innerH = CH - padT - padB;
  const X = (i) => padL + (i / Math.max(points.length - 1, 1)) * innerW;
  const Y = (v) => padT + (1 - (Math.min(max, Math.max(min, v)) - min) / span) * innerH;

  const path = points
    .map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p).toFixed(1)}`)
    .join(" ");
  const last = points.length ? points[points.length - 1] : min;

  return (
    <svg
      className="line-chart"
      viewBox={`0 0 ${CW} ${CH}`}
      role="img"
      preserveAspectRatio="none"
    >
      {[0.25, 0.5, 0.75].map((f) => (
        <line
          key={f}
          x1={padL}
          x2={CW - padR}
          y1={padT + f * innerH}
          y2={padT + f * innerH}
          className="chart-grid"
        />
      ))}
      <line x1={padL} x2={CW - padR} y1={padT + innerH} y2={padT + innerH} className="chart-axis" />
      {zero && (
        <line x1={padL} x2={CW - padR} y1={Y(0)} y2={Y(0)} className="chart-zero" />
      )}
      {points.length > 1 && <path d={path} className="chart-line" style={{ stroke: color }} />}
      <circle cx={X(Math.max(points.length - 1, 0))} cy={Y(last)} r="3.5" fill={color} />
    </svg>
  );
}