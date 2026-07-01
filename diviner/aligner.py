"""
aligner.py
==========
STEP 3 + STEP 4 of the Diviner integration pipeline, plus terrain
(slope) derivation used in the feature stack.

STEP 3 — Reproject every dataset into the same CRS as the reference grid.
STEP 4 — Resample every dataset to the *exact* pixel grid of the
          reference raster: identical width, height, transform, CRS.

Both steps are performed in a single rasterio.warp.reproject() call.  If
the source CRS already matches the target the warp degenerates to a pure
affine resampling.  All outputs go to outputs/diviner/aligned/ and are
never overwritten.

Slope derivation
----------------
Slope (°) is computed from the aligned DEM using the standard finite-
difference formula.  Geographic CRS pixel sizes (degrees) are converted
to metres using the Moon's mean radius (1 737 400 m) before differencing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

log = logging.getLogger("diviner_pipeline.aligner")

MOON_RADIUS_M = 1_737_400.0   # metres


# ---------------------------------------------------------------------------
# Reference grid loader
# ---------------------------------------------------------------------------

def load_reference_grid(ref_path: Path) -> dict:
    """
    Open the reference raster and return its spatial metadata.

    Returns
    -------
    dict with: crs, transform, width, height, nodata, profile.
    """
    with rasterio.open(ref_path) as src:
        return {
            "crs":       src.crs,
            "transform": src.transform,
            "width":     src.width,
            "height":    src.height,
            "nodata":    src.nodata,
            "profile":   src.profile.copy(),
        }


# ---------------------------------------------------------------------------
# Core aligner
# ---------------------------------------------------------------------------

def align_raster(
    src_path:     Path,
    out_dir:      Path,
    out_name:     str,
    ref:          dict,
    resampling:   Resampling = Resampling.bilinear,
    nodata:       float = -9999.0,
    force:        bool  = False,
    scale_factor: float = 1.0,
) -> Path:
    """
    Reproject and resample *src_path* so it exactly matches *ref*.

    Parameters
    ----------
    src_path     : input raster (any CRS / resolution).
    out_dir      : directory for the aligned GeoTIFF.
    out_name     : output filename.
    ref          : reference grid dict from :func:`load_reference_grid`.
    resampling   : Resampling.bilinear for continuous data;
                   Resampling.nearest for binary masks (PSR, DPSR).
    nodata       : output nodata sentinel (float32 -9999.0).
    force        : if True, re-warp even when the output already exists.
    scale_factor : multiply valid pixel values by this factor after reading
                   (use 1000.0 to convert LOLA DEM from km to metres).

    Returns
    -------
    Path to the aligned GeoTIFF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists() and not force:
        log.info(f"  [skip] {out_name} already aligned — skipping.")
        return out_path

    log.info(f"  Aligning   {src_path.name}  →  {out_name}")
    log.info(f"    Target CRS  : {ref['crs']}")
    log.info(f"    Target size : {ref['width']} x {ref['height']}")
    log.info(f"    Resampling  : {resampling.name}")

    with rasterio.open(src_path) as src:
        src_crs       = src.crs
        src_transform = src.transform
        src_nodata    = src.nodata if src.nodata is not None else nodata

        src_data = src.read(1).astype(np.float32)

    # Replace any pre-existing nodata / non-finite values with our sentinel
    bad = (~np.isfinite(src_data)) | (src_data == np.float32(src_nodata))
    src_data[bad] = np.float32(nodata)

    # Apply optional unit conversion (e.g. LOLA DEM km → m)
    if scale_factor != 1.0:
        valid_px = src_data != np.float32(nodata)
        src_data[valid_px] = src_data[valid_px] * np.float32(scale_factor)
        log.info(f"    Scale factor {scale_factor} applied to valid pixels")

    # Allocate destination buffer
    dst_data = np.full(
        (ref["height"], ref["width"]),
        fill_value=nodata,
        dtype=np.float32,
    )

    # Handle missing CRS by assigning Moon geographic
    if src_crs is None:
        from rasterio.crs import CRS
        src_crs = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
        log.warning(
            f"  {src_path.name}: CRS missing — assigned Moon geographic "
            f"CRS for warp.  Verify spatial metadata before publishing."
        )

    reproject(
        source         = src_data,
        destination    = dst_data,
        src_transform  = src_transform,
        src_crs        = src_crs,
        dst_transform  = ref["transform"],
        dst_crs        = ref["crs"],
        resampling     = resampling,
        src_nodata     = nodata,
        dst_nodata     = nodata,
    )

    out_profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     ref["width"],
        "height":    ref["height"],
        "count":     1,
        "crs":       ref["crs"],
        "transform": ref["transform"],
        "nodata":    nodata,
        "compress":  "lzw",
    }

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(dst_data, 1)
        dst.update_tags(
            SOURCE     = src_path.name,
            ALIGNED_TO = str(ref["crs"]),
            RESAMPLING = resampling.name,
            NODATA     = str(nodata),
        )

    log.info(f"    Saved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Slope derivation from an aligned DEM
# ---------------------------------------------------------------------------

def compute_slope(
    dem_arr:   np.ndarray,
    transform,
    crs,
    nodata:    float = -9999.0,
) -> np.ndarray:
    """
    Compute slope in degrees from a DEM array already on the target grid.

    Uses the standard 4-neighbour finite-difference formula:

        dz/dx = ∂z/∂x (range)
        dz/dy = ∂z/∂y (azimuth)
        slope = arctan( sqrt( (dz/dx)² + (dz/dy)² ) )

    Geographic CRS (lat/lon in degrees) — pixel sizes are converted to
    metres using the Moon's radius at the scene centre latitude so that
    the gradient magnitude has meaningful units.

    Projected CRS — pixel sizes are taken directly from the transform
    (already in metres or the relevant linear unit).

    Parameters
    ----------
    dem_arr   : 2-D float32 DEM array aligned to the reference grid.
    transform : affine transform of the DEM (same as reference grid).
    crs       : CRS of the DEM; used to distinguish geographic / projected.
    nodata    : nodata sentinel in *dem_arr*.

    Returns
    -------
    slope_deg : 2-D float32 array of slope values in degrees; nodata where
                the input DEM was nodata.
    """
    valid_mask = np.isfinite(dem_arr) & (dem_arr != nodata)
    work = dem_arr.astype(np.float64)
    work[~valid_mask] = np.nan

    # Pixel sizes in transform units (always positive)
    px = abs(float(transform.a))
    py = abs(float(transform.e))

    if crs is not None and crs.is_geographic:
        # Convert degree-sized pixels to metres at scene centre latitude
        h, w = dem_arr.shape
        cx, cy = transform * (w / 2.0, h / 2.0)   # (lon, lat) of centre
        lat_rad = np.radians(cy)
        m_per_deg_lat = (np.pi / 180.0) * MOON_RADIUS_M
        m_per_deg_lon = (np.pi / 180.0) * MOON_RADIUS_M * abs(np.cos(lat_rad))
        px_m = px * m_per_deg_lon
        py_m = py * m_per_deg_lat
        log.debug(
            f"  Geographic CRS: px={px_m:.1f} m/col  py={py_m:.1f} m/row "
            f"(centre lat {cy:.2f}°)"
        )
    else:
        px_m, py_m = px, py

    dz_dx = np.gradient(work, px_m, axis=1)   # ∂z/∂x  (column direction)
    dz_dy = np.gradient(work, py_m, axis=0)   # ∂z/∂y  (row direction)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[~valid_mask] = np.float32(nodata)

    return slope_deg


def save_slope(
    slope_arr: np.ndarray,
    out_dir:   Path,
    out_name:  str,
    ref:       dict,
    nodata:    float = -9999.0,
) -> Path:
    """Save a computed slope array as a float32 GeoTIFF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists():
        log.info(f"  [skip] {out_name} already exists — slope save skipped.")
        return out_path

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     ref["width"],
        "height":    ref["height"],
        "count":     1,
        "crs":       ref["crs"],
        "transform": ref["transform"],
        "nodata":    nodata,
        "compress":  "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(slope_arr, 1)
        dst.update_tags(
            PIPELINE = "Slope derived from LOLA DEM (diviner/aligner.py)",
            FORMULA  = "arctan(sqrt((dz/dx)^2 + (dz/dy)^2))  [degrees]",
            NODATA   = str(nodata),
        )

    log.info(f"    Saved slope: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Load an aligned band from disk
# ---------------------------------------------------------------------------

def load_aligned(path: Path, nodata: float = -9999.0) -> Optional[np.ndarray]:
    """
    Read band 1 of an aligned GeoTIFF into a float32 numpy array.
    Returns None if the file does not exist (optional bands: CPR, DOP).
    """
    if not path.exists():
        log.warning(f"  Optional band not found — skipping: {path.name}")
        return None

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata if src.nodata is not None else nodata

    arr[(arr == np.float32(nd)) | ~np.isfinite(arr)] = np.float32(nodata)
    return arr
