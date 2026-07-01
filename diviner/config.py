"""
config.py
=========
Single source of truth for all Diviner integration pipeline parameters
and file paths.

IMPORTANT — this module only READS existing pipeline outputs; it never
modifies, renames, or overwrites any file outside its own outputs/ subtree.
"""

from pathlib import Path

from rasterio.enums import Resampling

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent   # diviner/
PROJECT_ROOT = BASE_DIR.parent                    # ISRO_Hackathon/

# ---------------------------------------------------------------------------
# Input: Diviner thermal datasets (raw downloads — read-only)
# ---------------------------------------------------------------------------
TEMP_DATA_DIR = PROJECT_ROOT / "temp" / "data"
TMEAN_GRD     = TEMP_DATA_DIR / "polar_south_80_Tmean.grd"
ZIT_TIF       = TEMP_DATA_DIR / "polar_south_80_zit_float32.tif"
PUMP_GRD      = TEMP_DATA_DIR / "polar_south_80_pump.grd"

# ---------------------------------------------------------------------------
# Input: existing pipeline products (read-only)
# ---------------------------------------------------------------------------
DEM_PATH  = PROJECT_ROOT / "data"    / "ldem_85s_20m_float.lbl"
PSR_PATH  = PROJECT_ROOT / "results" / "PSR_mask.tif"
DPSR_PATH = PROJECT_ROOT / "results" / "DPSR.tif"

# CPR already aligned to reference polar-stereo grid (DFSAR data pipeline output)
CPR_PATH = (
    PROJECT_ROOT / "DFSAR" / "data_pipeline" / "outputs" / "aligned" /
    "CPR.tif"
)

# DOP computed from the DFSAR full-pol SLI
DOP_PATH = PROJECT_ROOT / "dop" / "outputs" / "dop" / "Calculated_DOP.tif"

# ---------------------------------------------------------------------------
# Reference grid — every input is reprojected/resampled to match this raster
# ---------------------------------------------------------------------------
# Default: DPSR.tif (derived from LOLA 20 m DEM; defines the master
# polar-stereographic grid to which PSR and DPSR are already aligned).
# To target the DFSAR grid instead, switch to CPR_PATH.
REFERENCE_GRID = DPSR_PATH

# ---------------------------------------------------------------------------
# Output directories (never touch anything outside these paths)
# ---------------------------------------------------------------------------
OUT_DIR     = PROJECT_ROOT / "outputs" / "diviner"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "previews"
ALIGNED_DIR = OUT_DIR / "aligned"
REPORT_DIR  = OUT_DIR / "reports"
LOG_DIR     = OUT_DIR / "logs"

# GeoTIFF names for GRD conversions (Step 2)
TMEAN_TIF_NAME = "Tmean_converted.tif"
PUMP_TIF_NAME  = "Pump_converted.tif"

# Aligned output names (Steps 3–4)
TMEAN_ALIGNED_NAME = "Tmean_aligned.tif"
ZIT_ALIGNED_NAME   = "ZIT_aligned.tif"
PUMP_ALIGNED_NAME  = "Pump_aligned.tif"
DEM_ALIGNED_NAME   = "DEM_aligned.tif"
SLOPE_ALIGNED_NAME = "Slope_aligned.tif"
PSR_ALIGNED_NAME   = "PSR_aligned.tif"
DPSR_ALIGNED_NAME  = "DPSR_aligned.tif"
CPR_ALIGNED_NAME   = "CPR_aligned.tif"
DOP_ALIGNED_NAME   = "DOP_aligned.tif"

# Feature stack and ice confidence (Steps 7, 9)
FEATURE_STACK_NAME = "Feature_Stack.tif"
ICE_CONF_TIF_NAME  = "Ice_Confidence_Map.tif"
ICE_CONF_PNG_NAME  = "Ice_Confidence_Map.png"

# Report files (Steps 11–12)
STATS_CSV_NAME  = "statistics_report.csv"
PDF_REPORT_NAME = "diviner_pipeline_report.pdf"

# ---------------------------------------------------------------------------
# Calibration & masking
# ---------------------------------------------------------------------------
NODATA  = -9999.0
EPSILON = 1e-10

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
PREVIEW_PERCENTILE_LO = 2
PREVIEW_PERCENTILE_HI = 98
FIGURE_DPI            = 150

# Colour maps for each band
CMAPS: dict = {
    "DEM":           "terrain",
    "Slope":         "hot_r",
    "PSR":           "Blues",
    "DPSR":          "Purples",
    "CPR":           "viridis",
    "DOP":           "magma",
    "Tmean":         "RdYlBu_r",   # cool = blue (ice-friendly), hot = red
    "ZIT":           "plasma",
    "Pump":          "YlOrRd",
    "IceConfidence": "YlGnBu",
}

# ---------------------------------------------------------------------------
# Resampling methods
# ---------------------------------------------------------------------------
RESAMPLE_CONTINUOUS = Resampling.bilinear   # DEM, CPR, DOP, Diviner
RESAMPLE_MASK       = Resampling.nearest    # PSR, DPSR binary masks

# ---------------------------------------------------------------------------
# Physics-based Ice Confidence weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
# Physical justification documented in ice_score.py.
ICE_SCORE_WEIGHTS: dict = {
    "CPR":   0.20,   # high CPR → volume/double-bounce → ice grains likely
    "DOP":   0.12,   # low DOP  → depolarising target → volumetric scatter
    "Tmean": 0.20,   # low T    → thermally stable environment → ice preserved
    "ZIT":   0.15,   # low ZIT  → cold trap at zero incidence
    "Pump":  0.13,   # high pump → efficient volatile trapping
    "PSR":   0.10,   # inside PSR → never illuminated → ice thermally stable
    "DPSR":  0.05,   # inside DPSR → doubly shadowed → extra thermal stability
    "Slope": 0.05,   # low slope → flat terrain → ice accumulation favoured
}
# Sum = 1.00

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
REPORT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
HIST_BINS          = 256

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
WRITE_BLOCK = 2000   # rows per rasterio streaming write chunk

# Moon mean radius (metres) — used to convert geographic pixel sizes to metres
# when computing slope from a DEM stored in lat/lon degrees.
MOON_RADIUS_M = 1_737_400.0
