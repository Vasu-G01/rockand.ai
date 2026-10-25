"""Central configuration for the AstroLand AI backend."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"
MODEL_DIR = ML_DIR / "models"
MODEL_PATH = MODEL_DIR / "ppo_rocket_land.zip"
META_PATH = MODEL_DIR / "training_meta.json"
STATUS_PATH = MODEL_DIR / "training_status.json"
STOP_FLAG = MODEL_DIR / "training_stop.flag"
DB_PATH = BASE_DIR / "database" / "astro.db"

# server
HOST = "127.0.0.1"
PORT = 5000

# training
DEFAULT_TIMESTEPS = 500_000
DEFAULT_SEED = 7
MAX_EPISODE_STEPS = 400  # 40 s of flight at 10 Hz decisions

# serving
PLAN_MAX_STEPS = 200  # 20 s of predicted flight per plan
