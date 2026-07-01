"""
config.py
=========
Single source of truth for all DOP pipeline parameters and file paths.

Reuses the same Chandrayaan-2 DFSAR full-pol SLI GeoTIFFs already staged
for the CPR pipeline (cpr/) -- no data is duplicated on disk.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scene selection  ("faustini" | "south_pole_oct25")
# ---------------------------------------------------------------------------
SCENE = "faustini"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent          # dop/
PROJECT_ROOT = BASE_DIR.parent
CPR_DIR     = PROJECT_ROOT / "cpr"

if SCENE == "faustini":
    # Faustini crater scene: 2021-05-06, lat -85.7 to -89.25
    _DATE    = "20210506"
    _STEM    = "ch2_sar_ncxl_20210506t022608652"
    DATA_DIR = CPR_DIR / "faustini" / "data" / "calibrated" / _DATE
else:
    # South-pole Oct-2025 scene
    _DATE    = "20251025"
    _STEM    = "ch2_sar_ncxl_20251025t211236510"
    DATA_DIR = CPR_DIR / "data" / "data" / "calibrated" / _DATE

OUT_DIR  = BASE_DIR / "outputs"
PREV_DIR = OUT_DIR / "previews"
DOP_DIR  = OUT_DIR / "dop"
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
DOP_OUTPUT_NAME = "Calculated_DOP.tif"

# ---------------------------------------------------------------------------
# Calibration & masking
# ---------------------------------------------------------------------------
NODATA  = -9999.0
EPSILON = 1e-10        # guard against division-by-zero in DOP = num / S0

# ---------------------------------------------------------------------------
# Multilook / speckle reduction
# ---------------------------------------------------------------------------
# IMPORTANT: this MUST match the CPR pipeline's window (cpr/config.py) so
# that DOP and CPR are computed at the same effective spatial resolution
# and number of looks, which is required for a physically consistent
# CPR/DOP fusion product downstream.
#
# A single-look coherency/covariance matrix is always rank-1 (a fully
# polarized / "pure" scattering state), which makes DOP == 1 everywhere
# by construction -- multilooking is what allows the matrix to become
# rank-deficient and DOP to drop below 1 for depolarizing targets.
MULTILOOK_WINDOW = (19, 3)   # (azimuth_lines, range_samples) -- 57 looks

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
PREVIEW_DOWNSAMPLE = 100
PREVIEW_PERCENTILE = (2, 98)   # stretch range

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
WRITE_BLOCK = 2000   # chunk size (lines) used when writing the DOP GeoTIFF
