"""
utils.py — Shared constants, paths, and helpers for the DPSR pipeline.

All modules import from here so parameters live in exactly one place.
"""


import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Project layout ─────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).parent.parent   # ISRO_Hackathon/
DATA_DIR          = PROJECT_ROOT / "data"
OUTPUT_DIR        = PROJECT_ROOT / "results"

# Input rasters (produced by the earlier hillshade / PSR-rasterise pipeline)
DEM_PATH          = DATA_DIR  / "ldem_85s_20m_float.lbl"   # PDS label → .img
ILLUMINATION_PATH = OUTPUT_DIR / "illumination.tif"
PSR_MASK_PATH     = OUTPUT_DIR / "PSR_mask.tif"

# Output
DPSR_PATH         = OUTPUT_DIR / "DPSR.tif"

# ── Sun geometry (geographic convention: azimuth 0=North, 90=East) ────────────
# Maximum solar elevation at the lunar south pole (~89.5°S) is ~1.54°.
# Azimuth cycles through 360° over one lunar month (27.3 days).
# For a single-epoch illumination map, use the approximate peak elevation.
# For annual illumination, call compute_solar_illumination() at multiple
# azimuths and take the union (see pipeline/step_illumination.py).
SUN_ELEVATION = 1.54        # degrees — peak solar elevation at 89.5°S latitude
SUN_AZIMUTH   = 0.0         # degrees — starting azimuth; set per epoch or sweep

# ── Algorithm parameters ───────────────────────────────────────────────────────
N_ANGLES      = 72          # rays per pixel  (360° / 5° = 72)

# Search radius justification:
#   South-polar craters span a wide range of sizes:
#     Shackleton  ~21 km diam  → PSR interior to rim ≤ 10.5 km
#     Haworth     ~51 km diam  → PSR interior to rim ≤ 25.5 km
#     Amundsen    ~103 km diam → PSR interior to rim ≤ 51.5 km
#   A 10 km radius (500 px) misses the illuminated rim of Haworth and larger
#   craters.  50 km (2500 px) covers the full south-polar crater population.
#   Early exit means computation cost scales with actual blocking distance,
#   not MAX_DISTANCE, so the wall-clock impact is moderate (~2–3× vs 500).
MAX_DISTANCE  = 2500        # max ray length in pixels  (2500 × 20 m = 50 km)
CELLSIZE      = 20.0        # metres per pixel

# Multiprocessing chunk size (pixels handed to each worker at a time).
# Tune to balance process-spawn overhead vs granularity.
# Larger  → fewer process launches, lower overhead
# Smaller → better load balancing when rays terminate early
CHUNK_SIZE    = 200_000

# Sentinel written into padding columns of the Bresenham ray table
# so the Numba kernel knows when a ray has ended.
RAY_SENTINEL  = np.int32(-32767)

# ── Logging ────────────────────────────────────────────────────────────────────
def get_logger(name: str = "dpsr") -> logging.Logger:
    """
    Return a configured logger that timestamps every message.

    Usage
    -----
    log = get_logger(__name__)
    log.info("Loading DEM …")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt     = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Simple timer context manager ───────────────────────────────────────────────
class Timer:
    """
    Usage
    -----
    with Timer("Loading DEM"):
        dem = load_dem()
    # prints: "Loading DEM  done in 2.3 s"
    """
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        print(f"  {self.label:40s}  {elapsed:7.2f} s")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0
