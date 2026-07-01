"""
config.py
=========
Central configuration for the Lunar South Pole Ice Detection Pipeline.

To point the pipeline at your actual data:
  - Place files in the datasets/<DEM|PSR|DPSR|DFSAR>/ sub-directories, OR
  - Override the *_DIR variables below with absolute paths to your data.

All other parameters are tuned for Chandrayaan-2 DFSAR + LOLA DEM.
"""
from __future__ import annotations
from pathlib import Path
from rasterio.enums import Resampling

# ── Root layout ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Dataset directories (override to point at real data) ─────────────────────
DEM_DIR   = DATASETS_DIR / "DEM"
PSR_DIR   = DATASETS_DIR / "PSR"
DPSR_DIR  = DATASETS_DIR / "DPSR"
DFSAR_DIR = DATASETS_DIR / "DFSAR"

# ── Output directories ────────────────────────────────────────────────────────
ALIGNED_DIR  = OUTPUTS_DIR / "aligned"
PREVIEWS_DIR = OUTPUTS_DIR / "previews"
LOG_FILE     = OUTPUTS_DIR / "processing.log"

# ── Supported file extensions ─────────────────────────────────────────────────
RASTER_EXTENSIONS: frozenset[str] = frozenset({
    ".tif", ".tiff", ".img", ".hgt", ".vrt", ".nc", ".bil",
})
VECTOR_EXTENSIONS: frozenset[str] = frozenset({
    ".shp", ".geojson", ".gpkg", ".gdb",
})

# ── DFSAR product keyword → canonical band label ─────────────────────────────
# Matched case-insensitively against the file stem.
# Add new product codes here without touching any other module.
DFSAR_PRODUCT_KEYWORDS: dict[str, str] = {
    "cpr": "CPR",   # Circular Polarisation Ratio
    "srd": "SRD",   # Stokes-derived Radar Parameter
    "odd": "ODD",   # Odd-bounce Scattering
    "vol": "VOL",   # Volume Scattering
    "hlx": "HLX",   # Helix Scattering
    "evn": "EVN",   # Eigenvalue Parameter
}

# ── Feature stack — band order in FeatureStack.tif / .npy ────────────────────
BAND_NAMES: list[str] = [
    "DEM",        # 01 — elevation
    "Slope",      # 02 — slope in degrees  (derived from DEM)
    "Hillshade",  # 03 — hillshade         (derived from DEM)
    "PSR",        # 04 — permanently shadowed region mask
    "DPSR",       # 05 — doubly permanently shadowed region mask
    "CPR",        # 06 — circular polarisation ratio
    "SRD",        # 07 — stokes radar derivative
    "VOL",        # 08 — volume scattering
    "ODD",        # 09 — odd-bounce scattering
    "HLX",        # 10 — helix scattering
    "EVN",        # 11 — eigenvalue parameter
]

# ── Normalisation ─────────────────────────────────────────────────────────────
NORM_P_LOW:  int = 2
NORM_P_HIGH: int = 98

# ── Hillshade illumination parameters ────────────────────────────────────────
HILLSHADE_AZIMUTH:  float = 315.0   # geographic azimuth (0=N, CW), degrees
HILLSHADE_ALTITUDE: float = 45.0    # sun angle above horizon, degrees
HILLSHADE_Z_FACTOR: float = 1.0     # vertical exaggeration

# ── Resampling methods ────────────────────────────────────────────────────────
RESAMPLE_CONTINUOUS: Resampling = Resampling.bilinear  # DEMs, SAR products
RESAMPLE_MASK:       Resampling = Resampling.nearest   # PSR / DPSR binary masks

# ── Visualisation ─────────────────────────────────────────────────────────────
FIGURE_DPI:  int        = 150
FIGURE_SIZE: tuple      = (10, 8)

BAND_CMAPS: dict[str, str] = {
    "DEM":       "terrain",
    "Slope":     "hot_r",
    "Hillshade": "gray",
    "PSR":       "Blues",
    "DPSR":      "Purples",
    "CPR":       "viridis",
    "SRD":       "plasma",
    "VOL":       "inferno",
    "ODD":       "magma",
    "HLX":       "cividis",
    "EVN":       "YlOrRd",
}
DEFAULT_CMAP: str = "gray"

# ── Memory guard ──────────────────────────────────────────────────────────────
# Log a warning when an array exceeds this size in MB.
LARGE_ARRAY_WARN_MB: float = 512.0
