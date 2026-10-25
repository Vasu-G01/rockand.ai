"""Train the PPO autopilot that lands the rocket.

Run from the backend folder:

    .venv\\Scripts\\python.exe ml/train_model.py --timesteps 500000

The script writes
    ml/models/ppo_rocket_land.zip     trained policy used by /api/predict
    ml/models/training_meta.json      timesteps + evaluation success rate
    ml/models/training_status.json    live progress, polled by the Flask API

and honours ml/models/training_stop.flag as a graceful stop request
(the "Stop training" button in the UI writes it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on sys.path

from config import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    STATUS_PATH,
    STOP_FLAG,
)
from ml.env import RocketLandEnv, all_envs, combined_stats  # noqa: E402
from simulation import physics as P  # noqa: E402

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

# one thread for the tiny policy net: on 4 cores the default thread pool is
# slower than a single thread for 10-input MLPs (measured 820 vs 690 pred/s)
torch.set_num_threads(1)

N_ENVS = 4
# spawn curriculum: [0, EASY_STEPS) close to the pad, then wider, then the
# exact random spawn the browser uses.
CURRICULUM = [(60_000, 0), (150_000, 1)]


# --------------------------------------------------------------------- status
def _json_default(obj):
    """Serialise numpy scalars (np.float64, np.bool_) as native Python."""
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_status(payload: dict) -> bool:
    """Write training_status.json so the API never reads a half-written file.

    Atomic replace is retried because Windows refuses to rename a file that
    another process (Flask polling /api/train/status) currently has open.
    If the reader stays in the way we simply skip this update: a stale but
    valid file beats a corrupt one, and status reporting must never take a
    training run down with it. Never write in place - that tears reads.
    """
    text = json.dumps(payload, indent=2, default=_json_default)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        tmp = STATUS_PATH.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, STATUS_PATH)
            return True
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            if attempt == 0:
                print(f"status write retrying: {exc}", file=sys.stderr)
            time.sleep(0.02)
    print("status write skipped (file busy)", file=sys.stderr)
    return False


class StatusCallback(BaseCallback):
    """Streams progress into training_status.json and honours the stop flag."""

    def __init__(self, total: int, started_at: float, seed: int):
        super().__init__()
        self.total = total
        self.started_at = started_at
        self.seed = seed
        self.last_write = 0.0
        self.curriculum = -1

    def _on_training_start(self) -> None:
        self._write("Training started")

    def _on_step(self) -> bool:
        # widen the spawn distribution as the policy improves
        step = int(self.num_timesteps)
        level = 2
        for threshold, lvl in CURRICULUM:
            if step < threshold:
                level = lvl
                break
        if level != self.curriculum:
            self.curriculum = level
            for env in all_envs():
                env.curriculum = level

        if STOP_FLAG.exists():
            self._write("Stopped by user", running=True, stopping=True)
            return False
        now = time.time()
        if now - self.last_write >= 2.0:
            self.last_write = now
            self._write("Training")
        return True

    def _write(self, message: str, running: bool = True, stopping: bool = False) -> None:
        stats = combined_stats()
        step = int(self.num_timesteps)
        elapsed = time.time() - self.started_at
        fps = step / elapsed if elapsed > 0 else 0.0
        write_status(
            {
                "running": running,
                "stopping": stopping,
                "message": message,
                "step": step,
                "total": self.total,
                "percent": round(100.0 * step / self.total, 1),
                "elapsed": round(elapsed, 1),
                "eta": round((self.total - step) / fps, 1) if fps > 0 else None,
                "fps": round(fps),
                "episodes": stats.get("episodes", 0),
                "success_rate": stats.get("success_rate"),
                "return_mean": stats.get("return_mean"),
                "curriculum": self.curriculum,
                "pid": os.getpid(),
                "seed": self.seed,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )


# ---------------------------------------------------------------- evaluation
def evaluate(model, episodes: int = 40, seed: int = 1000) -> dict:
    """Fly `episodes` deterministic landings from random spawns."""
    env = RocketLandEnv(seed=seed)
    env.curriculum = 2  # always score on the full spawn distribution
    successes = 0
    returns = []
    touchdowns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        landing = info.get("landing")
        if landing and landing["ok"]:
            successes += 1
            touchdowns.append(landing)
    env.close()
    return {
        "episodes": episodes,
        "landings": successes,
        "success_rate": round(successes / episodes, 3),
        "return_mean": round(sum(returns) / len(returns), 2),
        "sample_landing": touchdowns[-1] if touchdowns else None,
    }


def build_model(env, seed: int, resume: bool) -> PPO:
    if resume and MODEL_PATH.exists():
        print(f"Resuming from {MODEL_PATH}")
        return PPO.load(str(MODEL_PATH), env=env, device="cpu")
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        device="cpu",
        verbose=1,
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the rocket-landing PPO policy")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true", help="continue from the saved model")
    parser.add_argument("--eval-only", action="store_true", help="evaluate the saved model")
    parser.add_argument("--eval-episodes", type=int, default=40)
    args = parser.parse_args(argv)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if not MODEL_PATH.exists():
            print(f"No trained model at {MODEL_PATH} — train first.")
            return 1
        model = PPO.load(str(MODEL_PATH), device="cpu")
        result = evaluate(model, args.eval_episodes, seed=args.seed + 5000)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0

    STOP_FLAG.unlink(missing_ok=True)
    started_at = time.time()
    write_status(
        {
            "running": True,
            "message": "Starting up",
            "step": 0,
            "total": args.timesteps,
            "percent": 0.0,
            "elapsed": 0.0,
            "eta": None,
            "fps": 0,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )

    vec_env = DummyVecEnv(
        [
            # Monitor supplies the episode stats SB3 logs (ep_rew_mean, ep_len_mean)
            lambda i=i: Monitor(RocketLandEnv(seed=args.seed + i))
            for i in range(N_ENVS)
        ]
    )
    model = build_model(vec_env, args.seed, args.resume)
    callback = StatusCallback(args.timesteps, started_at, args.seed)

    # learn(total_timesteps=N, reset_num_timesteps=False) means "N *more* steps",
    # so turn the absolute target into the remaining budget when resuming.
    already = int(getattr(model, "num_timesteps", 0) or 0) if args.resume else 0
    remaining = max(args.timesteps - already, 0)
    if args.resume:
        print(f"Resuming at {already} steps, {remaining} to go (target {args.timesteps})")

    error = None
    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining, callback=callback, reset_num_timesteps=False
            )
        else:
            print("Model is already at the target — evaluating only.")
    except KeyboardInterrupt:
        error = "Interrupted by user"
        print("Interrupted — saving partial model.")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"Training failed: {error}")
        traceback.print_exc()
    finally:
        # Nothing in here may skip the final status write: Flask keys its
        # hot-reload off the running -> finished transition in that file.
        steps_done = int(getattr(callback, "num_timesteps", 0) or 0)
        eval_result = None
        stopping = STOP_FLAG.exists()
        STOP_FLAG.unlink(missing_ok=True)
        try:
            model.save(str(MODEL_PATH))
            print(f"Model saved to {MODEL_PATH}")

            vec_env.close()
            eval_result = evaluate(model, args.eval_episodes, seed=args.seed + 5000)
            meta = {
                "model_path": str(MODEL_PATH),
                "timesteps": steps_done,
                "timesteps_target": args.timesteps,
                "seed": args.seed,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_seconds": round(time.time() - started_at, 1),
                "stopped_early": bool(stopping),
                "error": error,
                "eval": eval_result,
            }
            META_PATH.write_text(
                json.dumps(meta, indent=2, default=_json_default), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            error = error or f"{type(exc).__name__}: {exc}"
            print("Post-training step failed:")
            traceback.print_exc()

        if error:
            message = f"Training failed: {error}"
        elif stopping:
            message = f"Stopped early at {steps_done} steps"
        else:
            message = "Training finished"
        write_status(
            {
                "running": False,
                "stopped": bool(stopping),
                "error": error,
                "message": message,
                "step": steps_done,
                "total": args.timesteps,
                "percent": round(100.0 * steps_done / max(args.timesteps, 1), 1),
                "eval": eval_result,
                "model_path": str(MODEL_PATH),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        if eval_result is not None:
            print(f"Eval: {json.dumps(eval_result, default=_json_default)}")

    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
