"""
config.py
=========
Central configuration for the Chandrayaan-2 DFSAR Lunar South Pole
Ice Detection and Landing Site Selection pipeline.

Directory layout (auto-configured from the file's location):
  DFSAR/
    data_pipeline/           <- this project
      datasets/
        DEM/                 <- place LOLA DEM here
        PSR/                 <- place PSR raster or shapefile here
        DPSR/                <- place DPSR raster or shapefile here
      outputs/
        aligned/             <- reprojected + resampled layers
        previews/            <- PNG visualisations
    ch2_sar_ndxl_*/          <- DFSAR product folders (auto-scanned)
    data/                    <- raw GRI / SLI / SRI products (auto-scanned)

Override any *_DIR path below to redirect at custom data locations.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from rasterio.enums import Resampling

# -- Root layout ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent          # …/DFSAR/data_pipeline/
DFSAR_ROOT   = PROJECT_ROOT.parent                      # …/DFSAR/
ISRO_ROOT    = PROJECT_ROOT.parent.parent               # …/ISRO_Hackathon/
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# -- Dataset source directories ------------------------------------------------
# DEM, PSR, DPSR -> place files in these local folders.
DEM_DIR  = DATASETS_DIR / "DEM"
PSR_DIR  = DATASETS_DIR / "PSR"
DPSR_DIR = DATASETS_DIR / "DPSR"

# DFSAR derived products -> scanned recursively from the DFSAR parent folder.
# All .tif files under DFSAR/ (including both pass subdirectories) are found.
DFSAR_DIR = DFSAR_ROOT

# -- Explicit file overrides ---------------------------------------------------
# When set (not None), these take priority over the directory scan above.
# Useful when your datasets live in a shared location rather than datasets/.
#
# After running setup_dem.py, DEM_FILE is auto-set to the converted GeoTIFF.
# PSR_FILE and DPSR_FILE point directly at your pre-existing data.
DEM_FILE:  Optional["Path"] = DATASETS_DIR / "DEM" / "LOLA_DEM_20m.tif"
PSR_FILE:  Optional["Path"] = ISRO_ROOT / "data" / "LOLA_PSR_75S_120M_82S_060M_5KM2_FINAL.shp"
DPSR_FILE: Optional["Path"] = ISRO_ROOT / "results" / "DPSR.tif"

# -- Output directories --------------------------------------------------------
ALIGNED_DIR  = OUTPUTS_DIR / "aligned"
PREVIEWS_DIR = OUTPUTS_DIR / "previews"
LOG_FILE     = OUTPUTS_DIR / "processing.log"

# -- File type discovery -------------------------------------------------------
RASTER_EXTENSIONS: frozenset[str] = frozenset({
    ".tif", ".tiff", ".img", ".hgt", ".vrt", ".bil",
})
VECTOR_EXTENSIONS: frozenset[str] = frozenset({
    ".shp", ".geojson", ".gpkg",
})

# Folders inside DFSAR_DIR that are NOT DFSAR product directories.
# These are excluded from the DFSAR product scan.
DFSAR_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "data_pipeline",   # this project folder
    "DFSAR_Images",    # visualisations from previous scripts
    "outputs",
    "__pycache__",
})

# -- DFSAR product keyword -> canonical band label -----------------------------
# Matched case-insensitively against the file stem (token at index 5 in the
# underscore-split CH-2 filename convention, e.g. "cpr" in "_d_cpr_xx_").
# Add new product codes here — no other module needs changing.
DFSAR_PRODUCT_KEYWORDS: dict[str, str] = {
    "cpr": "CPR",   # Circular Polarisation Ratio
    "srd": "SRD",   # Stokes Radar Decomposition
    "odd": "ODD",   # Odd-bounce Scattering
    "vol": "VOL",   # Volume Scattering
    "hlx": "HLX",   # Helix Scattering
    "evn": "EVN",   # Eigenvalue Parameter
    "trt": "TRT",   # Total Power / Trace
    # Raw Level-2 products (labelled but not included in the feature stack)
    "gri": "GRI",   # Geocoded RCS Image
    "sli": "SLI",   # Single Look Image
    "sri": "SRI",   # Sigma-nought RCS Image
}

# -- Feature stack — ordered band definitions ----------------------------------
# This order defines the band indices in FeatureStack.tif and FeatureStack.npy.
# Bands not found in the data are filled with NaN (a warning is logged).
BAND_NAMES: list[str] = [
    "DEM",        # 01 — elevation (m)
    "Slope",      # 02 — slope (degrees)        derived from DEM
    "Hillshade",  # 03 — hillshade (0–255)      derived from DEM
    "PSR",        # 04 — PSR binary mask        (1 = shadowed)
    "DPSR",       # 05 — DPSR binary mask       (1 = doubly shadowed)
    "CPR",        # 06 — circular polarisation ratio
    "SRD",        # 07 — stokes radar decomposition
    "VOL",        # 08 — volume scattering
    "ODD",        # 09 — odd-bounce scattering
    "HLX",        # 10 — helix scattering
    "EVN",        # 11 — eigenvalue parameter
]

# Binary layers — use nearest resampling and threshold normalisation.
MASK_LAYER_NAMES: frozenset[str] = frozenset({"PSR", "DPSR"})

# -- Normalisation -------------------------------------------------------------
NORM_P_LOW:  int = 2
NORM_P_HIGH: int = 98

# -- Hillshade illumination ----------------------------------------------------
HILLSHADE_AZIMUTH:  float = 315.0   # degrees from N, clockwise
HILLSHADE_ALTITUDE: float = 45.0    # sun elevation above horizon, degrees
HILLSHADE_Z_FACTOR: float = 1.0     # vertical exaggeration

# -- Resampling methods --------------------------------------------------------
RESAMPLE_CONTINUOUS: Resampling = Resampling.bilinear
RESAMPLE_MASK:       Resampling = Resampling.nearest

# -- Visualisation -------------------------------------------------------------
FIGURE_DPI:  int   = 150
FIGURE_SIZE: tuple = (10, 8)

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
    "TRT":       "Greens",
    "GRI":       "gray",
    "SLI":       "gray",
    "SRI":       "gray",
}
DEFAULT_CMAP: str = "gray"

# -- Memory --------------------------------------------------------------------
LARGE_ARRAY_WARN_MB: float = 512.0
