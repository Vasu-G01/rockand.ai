"""ML tests: features, the fallback guidance law and the Pilot endpoints.

Run with:  .venv\\Scripts\\python.exe -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ml.env import OBS_SIZE, RocketLandEnv, observe  # noqa: E402
from ml.predict import Pilot, heuristic_action  # noqa: E402
from simulation import physics as P  # noqa: E402


def _state(**kw):
    base = dict(x=540.0, y=200.0, vx=40.0, vy=10.0)
    base.update(kw)
    return P.State(**base)


class FeatureTest(unittest.TestCase):
    def test_observation_shape_and_dtype(self):
        obs = observe(_state(), pad_x=700)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertEqual(obs.dtype, np.float32)

    def test_observation_is_bounded(self):
        obs = observe(_state(x=1000, y=60, vx=400, vy=-400, a=3.0, w=50), pad_x=115)
        self.assertTrue(np.all(np.abs(obs) <= 3.0), obs)

    def test_pad_indicator(self):
        self.assertEqual(observe(_state(x=700), 700)[8], 1.0)
        self.assertEqual(observe(_state(x=300), 700)[8], 0.0)


class GuidanceTest(unittest.TestCase):
    def test_returns_a_valid_action(self):
        for x in (100, 540, 1000):
            action = heuristic_action(_state(x=x), pad_x=700)
            self.assertTrue(0 <= action < len(P.ACTIONS))

    def test_thrusts_when_falling_too_fast(self):
        action = heuristic_action(_state(vy=60, a=0.0), pad_x=540)
        self.assertTrue(P.ACTIONS[action][0], "expected thrust while dropping fast")

    def test_steers_toward_a_pad_on_the_right(self):
        # pad far to the right -> nose should be commanded over
        action = heuristic_action(_state(x=200, vx=0), pad_x=900)
        self.assertTrue(P.ACTIONS[action][2], "expected right rotation")


class EnvironmentTest(unittest.TestCase):
    def test_spaces(self):
        env = RocketLandEnv(seed=1)
        self.assertEqual(env.action_space.n, len(P.ACTIONS))
        self.assertEqual(env.observation_space.shape, (OBS_SIZE,))
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        env.close()

    def test_episode_reaches_an_outcome(self):
        env = RocketLandEnv(seed=2)
        env.curriculum = 0
        env.reset(seed=3)
        done = False
        steps = 0
        while not done and steps < 500:
            _, _, terminated, truncated, _ = env.step(0)  # idle: let it fall
            done = terminated or truncated
            steps += 1
            if done:
                env.reset()
        self.assertTrue(done)
        env.close()

    def test_contact_ends_episode(self):
        env = RocketLandEnv(seed=3)
        env.curriculum = 0
        env.reset(seed=4)
        env.s.y = P.GROUND_Y - 1
        env.s.vy = 20
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertIn("landing", info)
        env.close()


class PilotTest(unittest.TestCase):
    def test_act_returns_controls(self):
        pilot = Pilot()
        decision = pilot.act(_state(), pad_x=700)
        self.assertIn(decision["action"], range(len(P.ACTIONS)))
        self.assertIn(decision["source"], ("model", "heuristic"))
        thrust, left, right = P.ACTIONS[decision["action"]]
        self.assertEqual(decision["controls"], {"thrust": thrust, "left": left, "right": right})

    def test_plan_returns_a_path_and_stops_at_the_ground(self):
        pilot = Pilot()
        result = pilot.plan(
            _state(y=P.GROUND_Y - 6, vy=25, vx=0), pad_x=540, steps=200
        )
        self.assertGreater(len(result["path"]), 1)
        for x, y in result["path"]:
            self.assertTrue(20 <= x <= P.W - 20)
            self.assertTrue(y <= P.GROUND_Y + 1)
        # a drop from just above the pad must terminate with a prediction
        self.assertIsNotNone(result["predicted_landing"])
        self.assertEqual(len(result["actions"]), len(result["path"]))


if __name__ == "__main__":
    unittest.main()
