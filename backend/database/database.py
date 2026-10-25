"""SQLite storage for finished flights (landing attempts)."""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading

from config import DB_PATH

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT    NOT NULL,
    mode    TEXT    NOT NULL,          -- 'manual' | 'ai'
    pilot   TEXT,                      -- 'model' | 'heuristic' | NULL
    status  TEXT    NOT NULL,          -- 'LANDED' | 'CRASHED'
    score   INTEGER NOT NULL,
    time    REAL    NOT NULL,
    fuel    REAL    NOT NULL,
    vx      REAL    NOT NULL,
    vy      REAL    NOT NULL,
    angle   REAL    NOT NULL,
    message TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)


def save_flight(flight: dict) -> int:
    """Persist one finished flight and return its id."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO flights (ts, mode, pilot, status, score, time, fuel,
                                 vx, vy, angle, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                flight.get("ts") or datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                str(flight.get("mode", "manual")),
                flight.get("pilot"),
                str(flight.get("status", "FLYING")),
                int(flight.get("score", 0)),
                float(flight.get("time", 0.0)),
                float(flight.get("fuel", 0.0)),
                float(flight.get("vx", 0.0)),
                float(flight.get("vy", 0.0)),
                float(flight.get("angle", 0.0)),
                flight.get("message"),
            ),
        )
        return int(cur.lastrowid)


def list_flights(limit: int = 50) -> list[dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM flights ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]


def summary() -> dict:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS attempts,"
            "       SUM(CASE WHEN status='LANDED' THEN 1 ELSE 0 END) AS landings,"
            "       MAX(score) AS best"
            " FROM flights"
        ).fetchone()
    attempts = int(row["attempts"] or 0)
    landings = int(row["landings"] or 0)
    return {
        "attempts": attempts,
        "landings": landings,
        "best": row["best"],
        "rate": round(landings / attempts, 3) if attempts else None,
    }


if __name__ == "__main__":
    init_db()
    print(json.dumps(summary(), indent=2))
