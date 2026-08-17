"""Application settings. Override any of these with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NEST_DATA_DIR", BASE_DIR / "data" / "jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = float(os.getenv("NEST_MAX_UPLOAD_MB", 300))
MAX_FACES = int(os.getenv("NEST_MAX_FACES", 400_000))
JOB_WORKERS = int(os.getenv("NEST_JOB_WORKERS", 1))     # engine is single-core
JOB_TTL_HOURS = float(os.getenv("NEST_JOB_TTL_HOURS", 24))

DEFAULT_PROFILE = os.getenv("NEST_DEFAULT_PROFILE", "quick")
#: "sampled" (no extra dependency) or "bvh" (needs python-fcl, far faster)
DEFAULT_DISTANCE_BACKEND = os.getenv("NEST_DISTANCE_BACKEND", "sampled")
DEFAULT_CLEARANCE = float(os.getenv("NEST_DEFAULT_CLEARANCE", 5.0))
DEFAULT_TOP_N = int(os.getenv("NEST_DEFAULT_TOP_N", 10))

RENDER_DPI = int(os.getenv("NEST_RENDER_DPI", 110))
