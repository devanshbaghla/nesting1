"""Application settings. Override any of these with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NEST_DATA_DIR", BASE_DIR / "data" / "jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = float(os.getenv("NEST_MAX_UPLOAD_MB", 300))
#: Triangle ceiling for an upload. 0 means no limit — cost scales with face
#: count across sampling, rasterisation and BVH construction, so a very large
#: mesh is slow rather than rejected.
MAX_FACES = int(os.getenv("NEST_MAX_FACES", 0))
JOB_WORKERS = int(os.getenv("NEST_JOB_WORKERS", 1))     # engine is single-core
JOB_TTL_HOURS = float(os.getenv("NEST_JOB_TTL_HOURS", 24))

#: The UI offers no tuning controls, so this is what every run uses.
#: 'fast' trades accuracy for speed on purpose -- it skips the continuous
#: refinement, so a delivered gap is whatever the lattice left rather than
#: the clearance requested. Measured against 'quick': 8x faster on
#: electric_drill for a 2% larger box, but 5x faster on diamonds for a box
#: three times the size and a 26 mm gap where 5 mm was asked for.
#: Set NEST_DEFAULT_PROFILE=quick to get the accurate engine back.
DEFAULT_PROFILE = os.getenv("NEST_DEFAULT_PROFILE", "fast")
#: "sampled" (no extra dependency) or "bvh" (needs python-fcl, far faster)
DEFAULT_DISTANCE_BACKEND = os.getenv("NEST_DISTANCE_BACKEND", "sampled")
DEFAULT_CLEARANCE = float(os.getenv("NEST_DEFAULT_CLEARANCE", 5.0))
#: Stage 7 (refine, export, verify) scales linearly with this, so it is the
#: cheapest lever left once the profile is fixed.
DEFAULT_TOP_N = int(os.getenv("NEST_DEFAULT_TOP_N", 5))
