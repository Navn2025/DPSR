"""
preprocessor.py
===============
Terrain derivatives and vector rasterisation.

  compute_slope()     — slope in degrees from DEM elevation array
  compute_hillshade() — shaded relief using solar illumination formula
  rasterize_vector()  — burn shapefile / GeoPackage polygons to a raster
  derive_terrain()    — convenience wrapper: slope + hillshade in one call
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

from config import (
    HILLSHADE_AZIMUTH, HILLSHADE_ALTITUDE, HILLSHADE_Z_FACTOR,
    ALIGNED_DIR,
)
from utils import get_logger

log = get_logger("preprocessor")


# -- Terrain derivatives -------------------------------------------------------

def compute_slope(
    dem:   np.ndarray,
    res_x: float,
    res_y: float,
) -> np.ndarray:
    """
    Compute slope in degrees.

    Uses numpy.gradient which returns the central-difference approximation.
    For projected CRS (units metres) res_x/res_y are the pixel sizes in metres.

    Note: for geographic CRS (degrees) the result is technically in m/deg and
    should be converted using the local metre-per-degree factor. For projected
    (polar stereographic) LOLA DEM data this function is numerically correct.
    """
    nan_mask = ~np.isfinite(dem)
    filled   = np.where(nan_mask, 0.0, dem)

    # np.gradient(f, dy, dx) -> (∂f/∂row, ∂f/∂col)
    dz_drow, dz_dcol = np.gradient(filled, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dz_dcol**2 + dz_drow**2)))
    slope[nan_mask] = np.nan
    return slope.astype("float32")


def compute_hillshade(
    dem:      np.ndarray,
    res_x:    float,
    res_y:    float,
    azimuth:  float = HILLSHADE_AZIMUTH,
    altitude: float = HILLSHADE_ALTITUDE,
    z_factor: float = HILLSHADE_Z_FACTOR,
) -> np.ndarray:
    """
    Compute hillshade (0–255) using the standard solar illumination model.

    azimuth  : geographic azimuth of light source (0=N, 90=E, 315=NW)
    altitude : sun angle above horizon in degrees
    z_factor : vertical exaggeration (>1 amplifies relief)
    """
    nan_mask = ~np.isfinite(dem)
    filled   = np.where(nan_mask, 0.0, dem * z_factor)

    dz_drow, dz_dcol = np.gradient(filled, res_y, res_x)
    slope_rad  = np.arctan(np.sqrt(dz_dcol**2 + dz_drow**2))

    # Aspect: mathematical angle (0=East, CCW)
    aspect_rad = np.arctan2(-dz_drow, dz_dcol)

    # Convert geographic azimuth -> mathematical angle
    az_math_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad     = np.radians(altitude)

    hs = (
        np.cos(alt_rad) * np.cos(slope_rad)
        + np.sin(alt_rad) * np.sin(slope_rad) * np.cos(az_math_rad - aspect_rad)
    )
    hs = np.clip(hs * 255.0, 0.0, 255.0)
    hs[nan_mask] = np.nan
    return hs.astype("float32")


# -- Vector -> raster -----------------------------------------------------------

def rasterize_vector(
    vector_path: Path,
    ref_profile: dict,
    dst_path:    Path,
    burn_value:  float = 1.0,
) -> Optional[np.ndarray]:
    """
    Burn vector polygon geometries into a raster matching *ref_profile*.

    Returns a float32 binary mask (1.0 inside polygons, 0.0 outside).
    Saves the result to *dst_path*.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf     = gpd.read_file(vector_path)
        dst_crs = ref_profile["crs"]

        if gdf.crs and str(gdf.crs) != str(dst_crs):
            log.info(f"Reprojecting vector {vector_path.name} -> {dst_crs}")
            gdf = gdf.to_crs(dst_crs)

        if gdf.empty:
            log.warning(f"{vector_path.name} has no features; mask will be all zeros.")

        shapes = (
            (geom.__geo_interface__, burn_value)
            for geom in gdf.geometry
            if geom is not None and geom.is_valid
        )

        result = rasterize(
            shapes,
            out_shape=(ref_profile["height"], ref_profile["width"]),
            transform=ref_profile["transform"],
            fill=0.0,
            dtype="float32",
        )

        p = ref_profile.copy()
        p.update(count=1, dtype="float32", nodata=None,
                 compress="lzw", tiled=True)
        with rasterio.open(dst_path, "w", **p) as dst:
            dst.write(result, 1)

        log.info(f"Rasterized -> {dst_path.name}")
        return result

    except Exception as exc:
        log.error(f"Rasterization failed [{vector_path.name}]: {exc}")
        return None


# -- Convenience wrapper -------------------------------------------------------

def derive_terrain(
    dem_array:   np.ndarray,
    ref_profile: dict,
    out_dir:     Path = ALIGNED_DIR,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute and save Slope and Hillshade from *dem_array*.

    Uses GPU (CuPy/PyTorch) if available; falls back to numpy.
    Returns (slope, hillshade) float32 arrays.
    """
    res_x = abs(float(ref_profile["transform"].a))
    res_y = abs(float(ref_profile["transform"].e))

    # Try GPU terrain computation
    try:
        from gpu_accel import gpu_slope_hillshade
        slope, hillshade = gpu_slope_hillshade(
            dem_array, res_x, res_y,
            azimuth=HILLSHADE_AZIMUTH,
            altitude=HILLSHADE_ALTITUDE,
            z_factor=HILLSHADE_Z_FACTOR,
        )
        log.info("Terrain computed on GPU")
    except Exception as gpu_exc:
        log.warning(f"GPU terrain failed ({gpu_exc}); using numpy")
        slope     = compute_slope(dem_array, res_x, res_y)
        hillshade = compute_hillshade(dem_array, res_x, res_y)

    _save(slope,     out_dir / "Slope.tif",     ref_profile)
    _save(hillshade, out_dir / "Hillshade.tif", ref_profile)

    log.info(f"Slope range: {np.nanmin(slope):.2f} - {np.nanmax(slope):.2f} deg")
    log.info(f"Hillshade range: {np.nanmin(hillshade):.1f} - {np.nanmax(hillshade):.1f}")
    return slope, hillshade


def _save(array: np.ndarray, path: Path, ref_profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = ref_profile.copy()
    p.update(count=1, dtype="float32", nodata=float("nan"),
             compress="lzw", tiled=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype("float32"), 1)
    log.info(f"Saved -> {path.name}")
