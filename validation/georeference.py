"""
georeference.py
===============
Build ground control points (GCPs) from the DFSAR geometry CSV and prepare
the SLI-derived CPR for reprojection to Moon South Pole Stereographic.

Tie-point structure (SLI CSV)
------------------------------
The CSV contains N_AZ_TIES × N_RNG_TIES rows in row-major order:
    rows 0..N_RNG-1        → azimuth tie 0, range ties 0..N_RNG-1
    rows N_RNG..2*N_RNG-1  → azimuth tie 1, range ties 0..N_RNG-1
    ...

Pixel positions of tie points (0-indexed):
    az_px[i]  = i × (H - 1) / (N_AZ - 1)   where H = 272631 (scene height)
    rng_px[j] = j × (W - 1) / (N_RNG - 1)  where W = 244   (scene width)

GCP convention (rasterio)
--------------------------
    GroundControlPoint(row=az_px, col=rng_px, x=lon, y=lat, z=0)
    src_crs = Moon geographic (sphere R=1737400m)
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

log = logging.getLogger("validation.georeference")


# ---------------------------------------------------------------------------
# Tie-point grid reconstruction
# ---------------------------------------------------------------------------

def build_tie_grids(
    df: pd.DataFrame,
    N_AZ: int,
    N_RNG: int,
    scene_height: int,
    scene_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape the flat CSV into 2-D tie-point grids.

    Returns
    -------
    az_px   : (N_AZ,)   pixel row of each azimuth tie line
    rng_px  : (N_RNG,)  pixel col of each range tie column
    lat_grid : (N_AZ, N_RNG)  latitude  at each tie point
    lon_grid : (N_AZ, N_RNG)  longitude at each tie point
    """
    # Pixel positions of tie points (uniform spacing)
    az_px  = np.linspace(0, scene_height - 1, N_AZ)
    rng_px = np.linspace(0, scene_width  - 1, N_RNG)

    lat_arr = df["lat"].values[:N_AZ * N_RNG]
    lon_arr = df["lon"].values[:N_AZ * N_RNG]

    lat_grid = lat_arr.reshape(N_AZ, N_RNG)
    lon_grid = lon_arr.reshape(N_AZ, N_RNG)

    log.info(
        f"  Tie-point grid: {N_AZ} az x {N_RNG} rng"
        f"  lat [{lat_grid.min():.4f}, {lat_grid.max():.4f}]"
        f"  lon [{lon_grid.min():.4f}, {lon_grid.max():.4f}]"
    )
    return az_px, rng_px, lat_grid, lon_grid


# ---------------------------------------------------------------------------
# GCP creation
# ---------------------------------------------------------------------------

def create_gcps(
    az_px:    np.ndarray,
    rng_px:   np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    az_stride: int = 10,
) -> List[GroundControlPoint]:
    """
    Build rasterio GroundControlPoint objects from the tie-point grid.

    Parameters
    ----------
    az_px, rng_px   : tie-point pixel positions (1-D)
    lat_grid, lon_grid : (N_AZ, N_RNG) coordinate arrays
    az_stride       : use every az_stride-th azimuth tie row

    Returns
    -------
    List of GroundControlPoint objects (lon/lat, Moon geographic CRS)
    """
    gcps = []
    selected_az = range(0, len(az_px), az_stride)

    for i in selected_az:
        for j in range(len(rng_px)):
            lat = float(lat_grid[i, j])
            lon = float(lon_grid[i, j])
            row = float(az_px[i])
            col = float(rng_px[j])

            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue

            gcps.append(
                GroundControlPoint(
                    row=row, col=col,
                    x=lon, y=lat, z=0.0,
                )
            )

    n_az_used = len(selected_az)
    log.info(
        f"  Created {len(gcps)} GCPs  ({n_az_used} az tie rows x {len(rng_px)} range cols)"
        f"  [stride={az_stride}]"
    )
    return gcps


# ---------------------------------------------------------------------------
# Tie-point → full-scene lat/lon interpolation (for diagnostic use)
# ---------------------------------------------------------------------------

def interpolate_full_latlon(
    az_px:    np.ndarray,
    rng_px:   np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    scene_height: int,
    scene_width:  int,
    block_size:   int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinearly interpolate the (N_AZ, N_RNG) tie-point grids to the full
    (scene_height, scene_width) pixel grid.

    Memory-efficient: processes in horizontal blocks of `block_size` rows.

    Returns
    -------
    lat_full : (scene_height, scene_width) float32
    lon_full : (scene_height, scene_width) float32
    """
    from scipy.interpolate import RegularGridInterpolator

    lat_interp = RegularGridInterpolator(
        (az_px, rng_px), lat_grid, method="linear", bounds_error=False,
        fill_value=None,   # extrapolate
    )
    lon_interp = RegularGridInterpolator(
        (az_px, rng_px), lon_grid, method="linear", bounds_error=False,
        fill_value=None,
    )

    rng_full  = np.arange(scene_width, dtype=np.float32)
    lat_full  = np.empty((scene_height, scene_width), dtype=np.float32)
    lon_full  = np.empty((scene_height, scene_width), dtype=np.float32)

    for row_start in range(0, scene_height, block_size):
        row_end = min(row_start + block_size, scene_height)
        az_block = np.arange(row_start, row_end, dtype=np.float32)
        Az, Rng  = np.meshgrid(az_block, rng_full, indexing="ij")
        pts = np.column_stack([Az.ravel(), Rng.ravel()])

        lat_full[row_start:row_end, :] = lat_interp(pts).reshape(
            row_end - row_start, scene_width
        )
        lon_full[row_start:row_end, :] = lon_interp(pts).reshape(
            row_end - row_start, scene_width
        )

        if row_start % 50000 == 0:
            log.debug(f"    Interpolating lat/lon: row {row_start}/{scene_height}")

    log.info(
        f"  Full lat/lon grid: lat [{lat_full.min():.4f}, {lat_full.max():.4f}]"
        f"  lon [{lon_full.min():.4f}, {lon_full.max():.4f}]"
    )
    return lat_full, lon_full
