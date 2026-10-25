"""AstroLand AI — Flask API connecting the React simulator to the ML models.

Endpoints
---------
GET  /api/health          server + model + training overview
GET  /api/model           info about the loaded policy
POST /api/model/reload    re-read ml/models/ppo_rocket_land.zip (after training)
POST /api/predict         one control decision for a telemetry state  (10 Hz)
POST /api/plan            predicted flight path + control sequence   ("best path")
POST /api/train           start PPO training in a background process
GET  /api/train/status    live training progress
POST /api/train/stop      request a graceful stop
POST /api/flights         store a finished flight in SQLite
GET  /api/flights         flight history + summary
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    BASE_DIR,
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    HOST,
    ML_DIR,
    PLAN_MAX_STEPS,
    PORT,
    STATUS_PATH,
    STOP_FLAG,
)
from database import database as db
from ml.predict import pilot

app = Flask(__name__)
CORS(app)

db.init_db()

TRAIN_PROC: dict = {"proc": None}
_train_lock = threading.Lock()
_reload_guard = {"was_running": False}


# --------------------------------------------------------------------- helpers
def read_status() -> dict:
    """Training status, preferring the file written by the trainer process."""
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}
    else:
        status = {}

    proc = TRAIN_PROC["proc"]
    if proc is not None:
        code = proc.poll()
        if code is not None and status.get("running"):
            # process died without writing its final status
            status = {
                "running": False,
                "crashed": True,
                "returncode": code,
                "message": f"Training process exited with code {code}",
            }
            STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        elif code is None:
            status["running"] = True

    # first time we see training finish -> swap in the freshly trained policy
    running = bool(status.get("running"))
    if _reload_guard["was_running"] and not running:
        pilot.reload()
    _reload_guard["was_running"] = running

    return status or {"running": False, "message": "No training run yet"}


def parse_flight_request(data) -> tuple:
    """Validate {"state": {...}, "pad_x": number} payloads."""
    if not isinstance(data, dict):
        raise ValueError("JSON body is required")
    state = data.get("state")
    if not isinstance(state, dict):
        raise ValueError("`state` object is required")
    for key in ("x", "y", "vx", "vy"):
        if key not in state:
            raise ValueError(f"state.{key} is required")
        float(state[key])
    if "pad_x" not in data:
        raise ValueError("`pad_x` is required")
    pad_x = float(data["pad_x"])
    return state, pad_x


def to_state(payload: dict):
    from simulation.physics import State

    return State.from_dict(payload)


# ------------------------------------------------------------------- health
@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model": pilot.info,
            "training": read_status(),
            "database": db.summary(),
        }
    )


@app.get("/api/model")
def model_info():
    return jsonify(pilot.info)


@app.post("/api/model/reload")
def reload_model():
    return jsonify(pilot.reload())


# ---------------------------------------------------------------- prediction
@app.post("/api/predict")
def predict():
    try:
        state_json, pad_x = parse_flight_request(request.get_json(silent=True))
        state = to_state(state_json)
        return jsonify(pilot.act(state, pad_x))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # noqa: BLE001
        app.logger.exception("prediction failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/plan")
def plan():
    try:
        data = request.get_json(silent=True) or {}
        state_json, pad_x = parse_flight_request(data)
        steps = int(data.get("steps", PLAN_MAX_STEPS))
        state = to_state(state_json)
        return jsonify(pilot.plan(state, pad_x, steps))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:  # noqa: BLE001
        app.logger.exception("planning failed")
        return jsonify({"error": str(error)}), 500


# ------------------------------------------------------------------ training
@app.post("/api/train")
def start_training():
    status = read_status()
    if status.get("running"):
        return jsonify({"error": "Training is already running", "status": status}), 409

    data = request.get_json(silent=True) or {}
    timesteps = int(data.get("timesteps", DEFAULT_TIMESTEPS))
    timesteps = max(1_000, min(timesteps, 5_000_000))
    seed = int(data.get("seed", DEFAULT_SEED))

    STOP_FLAG.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(ML_DIR / "train_model.py"),
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
    ]
    with _train_lock:
        TRAIN_PROC["proc"] = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        _reload_guard["was_running"] = True

    return jsonify({"started": True, "timesteps": timesteps, "seed": seed})


@app.get("/api/train/status")
def training_status():
    return jsonify(read_status())


@app.post("/api/train/stop")
def stop_training():
    STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.write_text("stop", encoding="utf-8")
    return jsonify({"stopping": True, **read_status()})


# ------------------------------------------------------------------- flights
@app.post("/api/flights")
def save_flight():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "status" not in data:
            raise ValueError("`status` is required")
        flight_id = db.save_flight(data)
        return jsonify({"id": flight_id, "summary": db.summary()})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/flights")
def flights():
    limit = int(request.args.get("limit", 50))
    return jsonify({"flights": db.list_flights(limit), "summary": db.summary()})


if __name__ == "__main__":
    print(f"AstroLand AI API on http://{HOST}:{PORT}  (model: {pilot.info['source']})")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
