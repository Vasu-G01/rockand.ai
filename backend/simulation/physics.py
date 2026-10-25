"""2D rocket physics — a 1:1 port of the integration used by the React canvas
simulation (frontend/src/App.jsx).

The browser flies with these equations and the autopilot plans with the same
ones, so the predicted path drawn in the UI matches what the rocket actually
does. If you change a constant here, change it in App.jsx too.
"""

import math
from dataclasses import dataclass

# --- world / rocket constants (mirror App.jsx) ---
W = 1080
H = 560
GRAVITY = 60.0
DRY_MASS = 1000.0
FUEL_START = 300.0
MAX_THRUST = 125000.0
BURN_RATE = 20.0
TORQUE = 2.6

GROUND_Y = H - 80
PAD_HALF = 65
LIMIT_VY = 45.0
LIMIT_VX = 25.0
LIMIT_RAD = 0.21

# --- autopilot timing ---
# The pilot is queried every DECISION_DT seconds and its control is held while
# the physics keeps integrating, exactly like the browser does between two
# predictions. SUBSTEPS x SUB_DT == DECISION_DT.
DECISION_DT = 0.1
SUBSTEPS = 5
SUB_DT = DECISION_DT / SUBSTEPS

# spawn ranges (mirror randomStart / randomPadX in App.jsx)
PAD_X_MIN = PAD_HALF + 50
PAD_X_MAX = W - PAD_HALF - 50


def clamp(lo, hi, v):
    return lo if v < lo else hi if v > hi else v


@dataclass
class State:
    """Rocket state in canvas coordinates (y grows downwards)."""

    x: float
    y: float
    vx: float
    vy: float
    a: float = 0.0  # angle, 0 = upright
    w: float = 0.0  # angular velocity (rad/s)
    fuel: float = FUEL_START
    throttle: int = 0
    t: float = 0.0
    status: str = "FLYING"

    @property
    def alt(self) -> float:
        return max(0.0, GROUND_Y - self.y)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "a": self.a, "w": self.w, "fuel": self.fuel,
            "throttle": self.throttle, "t": self.t, "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            x=float(d["x"]), y=float(d["y"]), vx=float(d["vx"]), vy=float(d["vy"]),
            a=float(d.get("a", 0.0)), w=float(d.get("w", 0.0)),
            fuel=float(d.get("fuel", FUEL_START)),
            throttle=int(d.get("throttle", 0)),
            t=float(d.get("t", 0.0)),
            status=str(d.get("status", "FLYING")),
        )


# control triple: (thrust, steer left, steer right)
Controls = tuple  # (thrust: bool, left: bool, right: bool)

IDLE: Controls = (False, False, False)
THRUST: Controls = (True, False, False)
LEFT: Controls = (False, True, False)
RIGHT: Controls = (False, False, True)
THRUST_LEFT: Controls = (True, True, False)
THRUST_RIGHT: Controls = (True, False, True)

ACTION_NAMES = [
    "idle", "thrust", "rotate_left", "rotate_right",
    "thrust_left", "thrust_right",
]
ACTIONS: list = [IDLE, THRUST, LEFT, RIGHT, THRUST_LEFT, THRUST_RIGHT]

# mirror App.jsx: thrusting = Space / ArrowUp / KeyW, left = ArrowLeft / KeyA,
# right = ArrowRight / KeyD


def integrate(s: State, ctrl: Controls, dt: float) -> None:
    """Advance the physics by dt seconds in place (identical to App.jsx)."""
    thrust, left, right = ctrl

    s.throttle = 1 if (thrust and s.fuel > 0) else 0
    if left:
        s.w -= TORQUE * dt
    if right:
        s.w += TORQUE * dt
    s.w *= max(0.0, 1 - 0.35 * dt)
    s.a += s.w * dt
    s.t += dt

    if s.throttle:
        burn = max(0.0, min(s.fuel, BURN_RATE * dt))
        s.fuel = max(0.0, s.fuel - burn)
        mass = DRY_MASS + s.fuel
        thrust_n = MAX_THRUST * (burn / max(BURN_RATE * dt, 0.001))
        s.vx += (math.sin(s.a) * thrust_n / mass) * dt
        s.vy -= (math.cos(s.a) * thrust_n / mass) * dt

    s.vy += GRAVITY * dt
    s.x += s.vx * dt
    s.y += s.vy * dt
    s.x = clamp(20.0, W - 20.0, s.x)
    if s.y < 30.0:  # never fly off the top of the screen
        s.y = 30.0
        s.vy = max(s.vy, 0.0)


def hold(s: State, ctrl: Controls, total_dt: float = DECISION_DT) -> None:
    """Apply one control decision for total_dt seconds in SUBSTEPS chunks."""
    for _ in range(SUBSTEPS):
        integrate(s, ctrl, SUB_DT)


def has_contact(s: State) -> bool:
    return s.y >= GROUND_Y


def random_pad_x(rng) -> float:
    # float() on purpose: numpy's Generator yields np.float64, which turns the
    # landing comparisons into np.bool_ and those are not JSON serialisable.
    return float(rng.uniform(PAD_X_MIN, PAD_X_MAX))


def random_start(rng) -> State:
    """Random spawn identical to the frontend's newRocket()."""
    return State(
        x=float(rng.uniform(W * 0.32, W * 0.68)),
        y=float(rng.uniform(70, 300)),
        vx=float(rng.choice([-1.0, 1.0]) * rng.uniform(25, 55)),
        vy=0.0,
        a=0.0,
        w=0.0,
        fuel=FUEL_START,
    )


def state_from_request(payload: dict) -> State:
    """Build a State from the JSON the frontend posts (same shape as to_dict)."""
    return State.from_dict(payload)
