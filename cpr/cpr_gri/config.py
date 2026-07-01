"""
config.py
=========
Single source of truth for all Ground Range Image (GRI) CPR pipeline
parameters and file paths.

Reuses the same Chandrayaan-2 DFSAR calibrated GRI GeoTIFFs already
staged under cpr/ (no data is duplicated on disk), and compares the
result against the official DFSAR CPR mosaic staged under DFSAR/.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scene selection  ("faustini" | "south_pole_oct25")
# ---------------------------------------------------------------------------
SCENE = "faustini"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent           # cpr_gri/
PROJECT_ROOT = BASE_DIR.parent
CPR_DIR      = PROJECT_ROOT / "cpr"

if SCENE == "faustini":
    # Faustini crater scene: 2021-05-06, lat -85.7 to -89.25
    _DATE    = "20210506"
    _STEM    = "ch2_sar_ncxl_20210506t022608652"
    DATA_DIR = CPR_DIR / "faustini" / "data" / "calibrated" / _DATE
    GEOM_DIR = CPR_DIR / "faustini" / "geometry" / "calibrated" / _DATE
else:
    # South-pole Oct-2025 scene
    _DATE    = "20251025"
    _STEM    = "ch2_sar_ncxl_20251025t211236510"
    DATA_DIR = CPR_DIR / "data" / "data" / "calibrated" / _DATE
    GEOM_DIR = CPR_DIR / "data" / "geometry" / "calibrated" / _DATE

OUT_DIR    = BASE_DIR / "outputs"
PREV_DIR   = OUT_DIR / "previews"
VALID_DIR  = OUT_DIR / "validation"
LOG_DIR    = OUT_DIR / "logs"
CPR_GRI_DIR = OUT_DIR / "cpr_gri"

# ---------------------------------------------------------------------------
# Chandrayaan-2 DFSAR GRI full-polarimetric GeoTIFFs + incidence angle
# ---------------------------------------------------------------------------
_GRI = f"{_STEM}_d_gri_xx_fp"
GRI_PATHS = {
    "HH": DATA_DIR / f"{_GRI}_hh_d18.tif",
    "HV": DATA_DIR / f"{_GRI}_hv_d18.tif",
    "VH": DATA_DIR / f"{_GRI}_vh_d18.tif",
    "VV": DATA_DIR / f"{_GRI}_vv_d18.tif",
}
INCIDENCE_PATH = DATA_DIR / f"{_STEM}_d_gri_in_fp_xx_d18.tif"

# PDS4 XML label carrying the (single, scene-wide) radiometric calibration
# constant -- see reader.parse_calibration_constant().
GRI_LABEL_XML = DATA_DIR / f"{_STEM}_d_gri_xx_fp_xx_d18.xml"

# Geometry tie-point table (Latitude, Longitude, Range, Incidence_Angle)
# for the GRI ground-range grid -- used only for STEP 9 georeferencing.
GEOM_CSV = GEOM_DIR / f"{_STEM}_g_gri_xx_fp_xx_d18.csv"

# ---------------------------------------------------------------------------
# Official DFSAR CPR mosaic (for STEP 9 comparison)
# ---------------------------------------------------------------------------
# NOTE: this is a south-polar MOSAIC product (2025-06-30) in Moon 2000 South
# Polar Stereographic projection, not a per-scene product for the exact
# 2021-05-06 Faustini pass. It is the only real "official" CPR raster
# available in this dataset, so it is used as an independent spatial/
# order-of-magnitude consistency check rather than a pixel-exact ground
# truth (see validation.py docstring for the full caveat).
OFFICIAL_CPR_MOSAIC = (
    PROJECT_ROOT / "DFSAR" / "ch2_sar_ndxl_20250630mpcpspwest_d_fp_xxx"
    / "data" / "derived" / "20250630"
    / "ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif"
)

# IAU mean radius of the Moon (sphere), matches the mosaic's declared
# GEOGCS SPHEROID["Moon_2000_IAU_IAG", 1737400, 0] (flattening 0).
MOON_RADIUS_M = 1737400.0

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
CPR_GRI_OUTPUT_NAME = "Calculated_CPR_GRI.tif"

# ---------------------------------------------------------------------------
# Calibration & masking
# ---------------------------------------------------------------------------
# Fallback used only if GRI_LABEL_XML is missing or unparsable; the real
# value is read from the PDS4 label at runtime (see reader.py). Both
# scenes in this dataset carry the same constant (70.308868).
CALIBRATION_CONSTANT_FALLBACK = 70.308868

NODATA  = -9999.0
EPSILON = 1e-10

# Physically expected CPR range (bare regolith ~0.05-0.5, ice candidates
# >1.0, official mosaic max ~4.9 -- see cpr/faustini/outputs). The
# published co-pol-only CPR(mu_c) formula (--research mode) is exact but
# numerically ill-conditioned: its denominator (sqrt(sigma_HH)-
# sqrt(sigma_VV))^2 approaches zero whenever the two co-pol channels have
# similar power, which is common for natural terrain, so raw output can
# spike to enormous values that are numerical artifacts, not real CPR.
# Pixels outside this range are masked invalid rather than clipped, so
# statistics/previews reflect only physically credible values instead of
# being distorted by a boundary pile-up.
CPR_RESEARCH_VALID_RANGE = (0.0, 2.0)

# ---------------------------------------------------------------------------
# Multilook / speckle reduction
# ---------------------------------------------------------------------------
# MUST match the SLI CPR pipeline's window (cpr/config.py MULTILOOK_WINDOW)
# per the task spec, so the GRI-derived CPR is directly comparable.
# Note GRI is already an 18-azimuth-look L1B product (see PDS4 label
# isda:azimuth_looks=18), so this multilook is applied ON TOP of that,
# giving a heavily speckle-reduced result -- this is what was requested,
# not a bug.
MULTILOOK_WINDOW = (19, 3)   # (azimuth_lines, range_samples)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
PREVIEW_DOWNSAMPLE = 4          # GRI is already short (thousands of lines,
                                 # not hundreds of thousands like SLI)
PREVIEW_PERCENTILE = (2, 98)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
WRITE_BLOCK = 2000   # chunk size (lines) used when writing the CPR GeoTIFF
