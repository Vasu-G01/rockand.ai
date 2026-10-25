"""Physics parity tests: backend integration must match frontend/src/App.jsx.

Run with:  .venv\\Scripts\\python.exe -m unittest discover tests
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import landing as L  # noqa: E402
from simulation import physics as P  # noqa: E402


class FakeRng:
    """Deterministic stand-in for numpy's Generator: always the range midpoint."""

    def uniform(self, lo, hi):
        return (lo + hi) / 2

    def choice(self, options):
        return options[0]


class PhysicsTest(unittest.TestCase):
    def test_free_fall_gains_downward_speed(self):
        s = P.State(x=500, y=200, vx=0, vy=0)
        P.hold(s, P.IDLE, 0.1)
        self.assertGreater(s.vy, 0)  # y grows downwards
        self.assertGreater(s.y, 200)
        self.assertEqual(s.vx, 0)

    def test_upright_thrust_slows_descent(self):
        s = P.State(x=500, y=200, vx=0, vy=40)
        P.hold(s, P.THRUST, 0.1)
        self.assertLess(s.vy, 40)
        self.assertLess(s.fuel, P.FUEL_START)

    def test_thrust_consumes_fuel_and_stops_at_zero(self):
        s = P.State(x=500, y=200, vx=0, vy=0, fuel=0.01)
        P.hold(s, P.THRUST, 0.1)
        self.assertEqual(s.fuel, 0.0)
        self.assertEqual(s.throttle, 0)

    def test_rotation_damps_without_input(self):
        s = P.State(x=500, y=200, vx=0, vy=0, w=2.0)
        P.hold(s, P.IDLE, 0.1)
        self.assertLess(abs(s.w), 2.0)
        self.assertNotEqual(s.a, 0.0)  # angle still integrated

    def test_horizontal_bounds(self):
        s = P.State(x=P.W - 10, y=200, vx=500, vy=0)
        P.hold(s, P.IDLE, 0.1)
        self.assertLessEqual(s.x, P.W - 20)
        self.assertGreaterEqual(s.x, 20)

    def test_cannot_fly_off_the_top(self):
        s = P.State(x=500, y=31, vx=0, vy=-40)
        P.hold(s, P.IDLE, 0.1)
        self.assertGreaterEqual(s.y, 30.0)
        self.assertGreaterEqual(s.vy, 0.0)

    def test_action_set_has_six_controls(self):
        self.assertEqual(len(P.ACTIONS), 6)
        self.assertEqual(len(set(P.ACTIONS)), 6)
        self.assertEqual(P.ACTION_NAMES[5], "thrust_right")


class LandingTest(unittest.TestCase):
    def _state(self, **kw):
        base = dict(x=540.0, y=P.GROUND_Y, vx=0.0, vy=10.0, a=0.0)
        base.update(kw)
        return P.State(**base)

    def test_soft_landing_on_pad(self):
        result = L.evaluate(self._state(x=700, vy=-5, vx=3), pad_x=700)
        self.assertTrue(result.ok)
        self.assertGreater(result.score, 0)
        self.assertEqual(result.message, "Landing successful!")

    def test_hard_impact_crashes(self):
        result = L.evaluate(self._state(vy=90), pad_x=700)
        self.assertFalse(result.ok)
        self.assertFalse(result.soft)
        self.assertLess(result.score, 0)

    def test_off_pad_crashes_even_if_slow(self):
        result = L.evaluate(self._state(x=200, vy=5), pad_x=700)
        self.assertFalse(result.ok)
        self.assertTrue(result.soft)
        self.assertTrue(result.on_pad is False)
        self.assertEqual(result.message, "Missed the landing pad")

    def test_tilted_touchdown_crashes(self):
        result = L.evaluate(self._state(a=0.5, vy=5), pad_x=540)
        self.assertFalse(result.ok)
        self.assertFalse(result.soft)

    def test_resolve_contact_zeroes_velocity(self):
        s = self._state(vy=30)
        result = L.resolve_contact(s, pad_x=700)
        self.assertEqual(s.status, "LANDED" if result.ok else "CRASHED")
        self.assertEqual(s.vy, 0)
        self.assertEqual(s.vx, 0)


class SpawnTest(unittest.TestCase):
    def test_pad_spawn_range_matches_frontend(self):
        pad = P.random_pad_x(FakeRng())
        self.assertGreaterEqual(pad, P.PAD_HALF + 50)
        self.assertLessEqual(pad, P.W - P.PAD_HALF - 50)

    def test_start_matches_frontend(self):
        s = P.random_start(FakeRng())
        self.assertTrue(P.W * 0.32 <= s.x <= P.W * 0.68)
        self.assertTrue(70 <= s.y <= 300)
        self.assertGreaterEqual(abs(s.vx), 25)
        self.assertLessEqual(abs(s.vx), 55)
        self.assertEqual(s.vy, 0)
        self.assertEqual(s.fuel, P.FUEL_START)

    def test_decision_timing(self):
        self.assertAlmostEqual(P.SUBSTEPS * P.SUB_DT, P.DECISION_DT)
        self.assertAlmostEqual(P.DECISION_DT, 0.1, places=6)
        self.assertAlmostEqual(math.sin(0), 0)


if __name__ == "__main__":
    unittest.main()
