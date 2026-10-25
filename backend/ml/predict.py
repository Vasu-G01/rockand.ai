"""The pilot: turns rocket telemetry into a control decision.

Loads the trained PPO policy from ml/models/ (see train_model.py) and exposes

    act()   -> one control decision, used by /api/predict at 10 Hz
    plan()  -> a predicted flight path, drawn by the frontend as the "best path"

and falls back to a classical guidance controller when no model has been
trained yet, so the AI pilot button works out of the box.
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np

from config import META_PATH, MODEL_PATH, PLAN_MAX_STEPS
from ml.env import observe
from simulation import landing as L
from simulation import physics as P


def heuristic_action(s: P.State, pad_x: float) -> int:
    """PD-style guidance used before a model is trained (and as a safety net).

    Returns an index into P.ACTIONS, mirroring the JS fallback in the frontend.
    """
    dx = pad_x - s.x
    alt = s.alt

    # horizontal: head for the pad, damp the drift once we are over it
    if abs(dx) <= P.PAD_HALF * 0.5:
        des_vx = max(-10.0, min(10.0, -s.vx * 0.6))
    else:
        des_vx = max(-30.0, min(30.0, dx * 0.2))
    err_vx = des_vx - s.vx

    # vertical: gentle descent as the ground comes up
    if alt > 200:
        des_vy = 40.0
    elif alt > 80:
        des_vy = 22.0
    elif alt > 30:
        des_vy = 10.0
    else:
        des_vy = 3.0

    # attitude: tilt so horizontal thrust pushes us toward the pad.
    # Allow a wider nose-over while high up; stay near upright for touchdown.
    tilt = 0.35 if alt > 120 else 0.18
    a_cmd = max(-tilt, min(tilt, err_vx * 0.008))

    # PD on (angle, angular velocity) — damping matters because releasing the
    # "key" does not stop the rotation (weak aerodynamic damping in the sim).
    signal = 3.0 * (a_cmd - s.a) - 1.2 * s.w
    right = signal > 0.03
    left = signal < -0.03

    # thrust whenever we fall faster than the target descent rate
    thrust = s.fuel > 0 and s.vy > des_vy + 1.0

    if thrust and right:
        action = 5  # thrust + rotate right
    elif thrust and left:
        action = 4  # thrust + rotate left
    elif thrust:
        action = 1
    elif right:
        action = 3
    elif left:
        action = 2
    else:
        action = 0
    return action


class Pilot:
    """Singleton that owns the loaded policy and answers act()/plan() calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._meta: dict = {}
        self.reload()

    # ------------------------------------------------------------- lifecycle
    def reload(self) -> dict:
        """(Re)load the model from disk; called at startup and after training."""
        with self._lock:
            self._model = None
            self._meta = {}
            if MODEL_PATH.exists():
                try:
                    from stable_baselines3 import PPO

                    self._model = PPO.load(str(MODEL_PATH), device="cpu")
                    if META_PATH.exists():
                        self._meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                    return self.info
                except Exception as error:  # corrupt / incompatible file
                    self._meta = {"error": str(error)}
            return self.info

    @property
    def info(self) -> dict:
        eval_result = (self._meta.get("eval") or {}) if self._meta else {}
        return {
            "loaded": self._model is not None,
            "source": "model" if self._model is not None else "heuristic",
            "path": str(MODEL_PATH),
            "timesteps": self._meta.get("timesteps"),
            "trained_at": self._meta.get("trained_at"),
            "success_rate": eval_result.get("success_rate"),
            "landings": eval_result.get("landings"),
            "eval_episodes": eval_result.get("episodes"),
            "error": self._meta.get("error"),
        }

    # -------------------------------------------------------------- decisions
    def act(self, state: P.State, pad_x: float) -> dict:
        """One control decision for the given telemetry."""
        obs = observe(state, pad_x)
        with self._lock:
            model = self._model
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            source = "model"
        else:
            action = heuristic_action(state, pad_x)
            source = "heuristic"

        thrust, left, right = P.ACTIONS[action]
        return {
            "action": action,
            "name": P.ACTION_NAMES[action],
            "source": source,
            "controls": {"thrust": thrust, "left": left, "right": right},
            "observation": [round(float(v), 5) for v in obs],
        }

    def plan(self, state: P.State, pad_x: float, steps: int = PLAN_MAX_STEPS) -> dict:
        """Roll the policy forward and return the predicted path + controls."""
        steps = max(1, min(int(steps), PLAN_MAX_STEPS))
        s = P.State.from_dict(state.to_dict())  # work on a copy
        path: list = []
        actions: list = []
        controls: list = []
        landing = None
        source = "model" if self._model is not None else "heuristic"

        for _ in range(steps):
            decision = self.act(s, pad_x)
            action = decision["action"]
            P.hold(s, P.ACTIONS[action])
            contact = P.has_contact(s)
            if contact:
                # the rollout crosses the ground inside one decision step; pin
                # the last point to the surface so the drawn path stops there
                s.y = P.GROUND_Y
            path.append([round(s.x, 1), round(s.y, 1)])
            actions.append(action)
            controls.append(decision["controls"])
            if contact:
                result = L.resolve_contact(s, pad_x)
                landing = {
                    "ok": result.ok,
                    "score": result.score,
                    "message": result.message,
                    "on_pad": result.on_pad,
                    "soft": result.soft,
                    "vx": 0.0,
                    "vy": 0.0,
                }
                break

        return {
            "source": source,
            "path": path,
            "actions": actions,
            "controls": controls,
            "dt": P.DECISION_DT,
            "predicted_landing": landing,
            "steps": len(actions),
        }


# module-level singleton used by the Flask app
pilot = Pilot()
