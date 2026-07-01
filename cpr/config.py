"""
config.py
=========
Single source of truth for all pipeline parameters and file paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scene selection  ("faustini" | "south_pole_oct25")
# ---------------------------------------------------------------------------
SCENE = "faustini"

# ---------------------------------------------------------------------------
# Directory layout  (all relative to THIS file, i.e. cpr/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

if SCENE == "faustini":
    # Faustini crater scene: 2021-05-06, lat -85.7 to -89.25
    _DATE    = "20210506"
    _STEM    = "ch2_sar_ncxl_20210506t022608652"
    DATA_DIR = BASE_DIR / "faustini" / "data" / "calibrated" / _DATE
    OUT_DIR  = BASE_DIR / "faustini" / "outputs"
else:
    # South-pole Oct-2025 scene
    _DATE    = "20251025"
    _STEM    = "ch2_sar_ncxl_20251025t211236510"
    DATA_DIR = BASE_DIR / "data" / "data" / "calibrated" / _DATE
    OUT_DIR  = BASE_DIR / "outputs"

PREV_DIR = OUT_DIR / "previews"
CPR_DIR  = OUT_DIR / "cpr"
LOG_DIR  = OUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Chandrayaan-2 DFSAR SLI full-polarimetric GeoTIFFs
# ---------------------------------------------------------------------------
_SLI = f"{_STEM}_d_sli_xx_fp"
SLI_PATHS = {
    "HH": DATA_DIR / f"{_SLI}_hh_d18.tif",
    "HV": DATA_DIR / f"{_SLI}_hv_d18.tif",
    "VH": DATA_DIR / f"{_SLI}_vh_d18.tif",
    "VV": DATA_DIR / f"{_SLI}_vv_d18.tif",
}

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
CPR_OUTPUT_NAME = "Calculated_CPR.tif"

# ---------------------------------------------------------------------------
# Calibration & masking
# ---------------------------------------------------------------------------
# From XML: calibration_constant = 70.308868.
# Cancels in the CPR ratio, but kept here for reference.
CALIBRATION_K = 70.308868
NODATA        = -9999.0
EPSILON       = 1e-10        # guard against division-by-zero

# Output range for --research mode's published CPR(mu_c) formula. That
# formula only stays under 2 when HH/VV differ by >15dB (rare for
# natural terrain), so the raw ratio is log10-rescaled onto this range
# rather than hard-clipped (see cpr.compute_cpr_research docstring).
CPR_RESEARCH_VALID_RANGE = (0.0, 2.0)

# ---------------------------------------------------------------------------
# Multilook / speckle reduction
# ---------------------------------------------------------------------------
# Applied to |SC|^2 and |OC|^2 power images BEFORE computing CPR.
# SLI pixel spacing: ~9.4 m azimuth x ~25 m range.
#
# Recommended windows (azimuth x range):
#   (19, 1)  = 19 looks  -- preserves range resolution, matches GRI azimuth looks
#   (19, 3)  = 57 looks  -- ~178 m az x 75 m rg, good balance for this swath
#   (25, 3)  = 75 looks  -- higher ENL, smoother CPR
#
# For polarimetric CPR, >= 25 effective looks is recommended for stable
# ratio estimation.  Increase range samples with care: scene is only 244 wide.
MULTILOOK_WINDOW = (19, 3)   # (azimuth_lines, range_samples)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
# Downsample factor for preview images (azimuth direction only).
# At 272 631 lines, factor=100 gives ~2726-line previews.
PREVIEW_DOWNSAMPLE = 100
PREVIEW_PERCENTILE = (2, 98)   # stretch range

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
# Chunk size (lines) used when writing the CPR GeoTIFF.
WRITE_BLOCK = 2000
