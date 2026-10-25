"""End-to-end test: fly a whole landing through the HTTP API.

This is the browser's loop in Python — ask /api/predict for one control,
hold it for 100 ms, repeat — so it measures what the UI actually experiences.

    .venv\\Scripts\\python.exe tests\\test_api_flight.py            # report
    .venv\\Scripts\\python.exe -m unittest discover -s tests         # as a test

Skipped automatically when the Flask server is not running.
"""

from __future__ import annotations

import json
import random
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import landing as L  # noqa: E402
from simulation import physics as P  # noqa: E402

BASE = "http://127.0.0.1:5000"


def _post(path: str, payload: dict, timeout: float = 5.0):
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def api_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=2) as response:
            return json.load(response).get("ok") is True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def model_info() -> dict:
    try:
        with urllib.request.urlopen(f"{BASE}/api/model", timeout=2) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def fly_episode(seed: int) -> dict | None:
    """One AI flight: random spawn, 10 Hz decisions straight from the API."""
    rng = random.Random(seed)
    pad_x = rng.uniform(P.PAD_X_MIN, P.PAD_X_MAX)
    state = P.State(
        x=rng.uniform(P.W * 0.32, P.W * 0.68),
        y=rng.uniform(70, 300),
        vx=rng.choice([-1, 1]) * rng.uniform(25, 55),
        vy=0.0,
    )
    for _ in range(600):  # 60 s cap
        decision = _post("/api/predict", {"state": state.to_dict(), "pad_x": pad_x})
        controls = decision["controls"]
        P.hold(state, (controls["thrust"], controls["left"], controls["right"]))
        if P.has_contact(state):
            result = L.resolve_contact(state, pad_x)
            return {"ok": result.ok, "score": result.score, "source": decision["source"]}
    return None


def report(episodes: int = 20, seed: int = 1234) -> dict:
    outcomes = [fly_episode(seed + i) for i in range(episodes)]
    done = [o for o in outcomes if o]
    landings = sum(1 for o in done if o["ok"])
    return {
        "episodes": episodes,
        "finished": len(done),
        "landings": landings,
        "success_rate": round(landings / max(len(done), 1), 3),
        "mean_score": round(sum(o["score"] for o in done) / max(len(done), 1), 1),
        "source": done[0]["source"] if done else None,
    }


@unittest.skipUnless(api_up(), "Flask API is not running on :5000")
class ApiFlightTest(unittest.TestCase):
    def test_flights_reach_the_ground(self):
        outcome = fly_episode(99)
        self.assertIsNotNone(outcome, "flight did not terminate within 60 s")
        self.assertIn(outcome["source"], ("model", "heuristic"))

    def test_trained_model_lands_most_flights(self):
        info = model_info()
        if not info.get("loaded") or (info.get("timesteps") or 0) < 150_000:
            self.skipTest("no sufficiently trained model loaded yet")
        result = report(episodes=20)
        print(f"\nAPI flight report: {result}")
        self.assertGreaterEqual(
            result["success_rate"], 0.5,
            f"trained model only landed {result['success_rate']:.0%} of flights",
        )


if __name__ == "__main__":
    if not api_up():
        print(f"API not reachable at {BASE} — start app.py first.")
        raise SystemExit(2)
    print(json.dumps(report(), indent=2))
