"""Touchdown evaluation: did we land, and what is the score?

Port of the contact logic at the bottom of the physics block in App.jsx so the
backend, the training reward and the browser all agree on LANDED vs CRASHED.
"""

from dataclasses import dataclass

from .physics import (
    GROUND_Y,
    LIMIT_RAD,
    LIMIT_VX,
    LIMIT_VY,
    PAD_HALF,
    State,
)


@dataclass
class LandingResult:
    ok: bool          # landed safely on the pad
    on_pad: bool
    soft: bool        # inside velocity / angle limits
    score: int
    message: str


def is_soft(s: State) -> bool:
    return abs(s.vy) <= LIMIT_VY and abs(s.vx) <= LIMIT_VX and abs(s.a) <= LIMIT_RAD


def is_on_pad(s: State, pad_x: float) -> bool:
    return abs(s.x - pad_x) <= PAD_HALF


def score_for(ok: bool, s: State) -> int:
    if ok:
        return round(200 + s.fuel - abs(s.vy) * 2)
    return -round(abs(s.vy) + abs(s.vx) * 2 + abs(s.a) * 50)


def evaluate(s: State, pad_x: float) -> LandingResult:
    on_pad = is_on_pad(s, pad_x)
    soft = is_soft(s)
    ok = soft and on_pad
    if ok:
        message = "Landing successful!"
    elif soft:
        message = "Missed the landing pad"
    else:
        message = "Rocket destroyed. Try again"
    return LandingResult(
        ok=ok, on_pad=on_pad, soft=soft,
        score=score_for(ok, s), message=message,
    )


def resolve_contact(s: State, pad_x: float) -> LandingResult:
    """Mark the state as finished, zero the velocity and score it (as App.jsx does)."""
    result = evaluate(s, pad_x)
    s.y = GROUND_Y
    s.status = "LANDED" if result.ok else "CRASHED"
    s.vx = 0.0
    s.vy = 0.0
    s.throttle = 0
    return result
