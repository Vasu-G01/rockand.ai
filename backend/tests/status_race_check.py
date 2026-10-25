"""Not a unit test: hammer write_status while another handle reads the file.

Reproduces the Windows PermissionError that killed a training run (Flask
holding training_status.json open during os.replace). Checks the two
invariants that matter: a reader never sees a torn file, and training is
never blocked by status reporting. Run manually:

    .venv\\Scripts\\python.exe tests/status_race_check.py
"""

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STATUS_PATH  # noqa: E402
from ml.train_model import write_status  # noqa: E402

stop = threading.Event()
reads = {"n": 0, "torn": 0, "locked": 0}


def reader():
    # same access pattern as app.read_status(): open, read, parse, short gap
    while not stop.is_set():
        try:
            text = STATUS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:  # transient sharing violation: readers just skip
            reads["locked"] += 1
            continue
        try:
            json.loads(text)
            reads["n"] += 1
        except json.JSONDecodeError:  # a partial file: never acceptable
            reads["torn"] += 1
            reads["snippet"] = text[:120]
        time.sleep(0.005)


t = threading.Thread(target=reader, daemon=True)
t.start()

written = 0
t0 = time.time()
for i in range(400):
    written += bool(write_status({"running": True, "step": i, "note": "race check"}))
elapsed = time.time() - t0

stop.set()
t.join(timeout=2)
assert write_status({"running": False, "step": 399, "note": "final"}), "final write failed"
final = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

print(f"400 write attempts in {elapsed:.2f}s | written={written} "
      f"| reads={reads['n']} torn={reads['torn']} locked={reads['locked']}")
if reads["torn"]:
    print(f"first torn read: {reads['snippet']!r}")
assert reads["torn"] == 0, "a reader saw a partially written file"
assert written >= 380, f"only {written}/400 status updates landed"
assert final["step"] == 399, "final status was not written"
print("OK")
