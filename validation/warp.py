"""
warp.py
=======
Reproject the SLI-derived CPR (no CRS, identity transform) to the
Moon South Pole Stereographic grid of the official DFSAR CPR product.

Strategy
--------
1. GCP reprojection (primary):
   Use rasterio.warp.reproject with the full set of tie-point GCPs.
   rasterio passes these to GDAL's warper, which fits a 3rd-order polynomial
   from pixel (col, row) to geographic (lon, lat), then applies the
   geographic → stereographic CRS transformation.

2. Scipy inverse-mapping fallback:
   If the GCP warp produces noisy results (detected from the percentage of
   filled pixels), fall back to an explicit lat/lon grid interpolation:
     - Interpolate the full (H, W) lat/lon grid from tie points.
     - Convert to Moon stereo (x, y) using pyproj.
     - Build a KD-tree on a downsampled source grid.
     - For each target pixel, find nearest source pixels and bilinearly
       interpolate CPR values.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator

import config as cfg

log = logging.getLogger("validation.warp")


# ---------------------------------------------------------------------------
# Primary: GCP-based warp
# ---------------------------------------------------------------------------

def warp_with_gcps(
    src_path:   Path,
    gcps:       List[GroundControlPoint],
    src_crs:    CRS,
    dst_crs:    CRS,
    dst_transform,
    dst_shape:  Tuple[int, int],
    output_path: Path,
    src_nodata: float = cfg.NODATA_CALC,
    dst_nodata: float = cfg.NODATA_GEOREF,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reproject CPR using GCPs via rasterio.warp.reproject.

    Parameters
    ----------
    src_path     : path to Calculated_CPR.tif (no CRS)
    gcps         : list of GroundControlPoint in Moon geographic CRS
    src_crs      : Moon geographic CRS
    dst_crs      : Moon South Pole Stereographic CRS (matches official CPR)
    dst_transform: affine transform for the target grid
    dst_shape    : (height, width) of the output raster
    output_path  : where to write Calculated_CPR_Georeferenced.tif
    """
    dst_h, dst_w = dst_shape
    dst_arr = np.full((dst_h, dst_w), dst_nodata, dtype=np.float32)

    log.info(f"  Source CPR       : {src_path.name}")
    log.info(f"  GCPs used        : {len(gcps)}")
    log.info(f"  Src CRS          : {src_crs.to_string()[:60]}...")
    log.info(f"  Dst CRS          : {dst_crs.to_string()[:60]}...")
    log.info(f"  Target shape     : {dst_h} x {dst_w}")
    log.info(f"  Target resolution: {dst_transform.a:.2f} m")
    log.info(f"  Target bounds    : {cfg.TARGET_BOUNDS}")

    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_data[src_data == src_nodata] = np.nan

    log.info("  Running GCP reprojection (rasterio.warp.reproject)...")

    reproject(
        source           = src_data,
        destination      = dst_arr,
        gcps             = gcps,
        src_crs          = src_crs,
        dst_crs          = dst_crs,
        dst_transform    = dst_transform,
        resampling       = resampling,
        src_nodata       = np.nan,
        dst_nodata       = dst_nodata,
        num_threads      = 4,
    )

    n_filled = int(np.isfinite(dst_arr).sum())
    fill_pct = 100 * n_filled / dst_arr.size
    log.info(f"  Filled pixels: {n_filled:,}  ({fill_pct:.2f}% of target grid)")

    if fill_pct < 0.5:
        log.warning(
            f"  GCP warp filled < 0.5% of target. "
            f"Switching to scipy inverse-mapping fallback."
        )
        return None   # signal fallback

    _write_georeferenced(dst_arr, dst_crs, dst_transform, output_path, dst_nodata)
    return dst_arr


# ---------------------------------------------------------------------------
# Fallback: scipy inverse-mapping
# ---------------------------------------------------------------------------

def warp_scipy_inverse(
    src_path:     Path,
    az_px:        np.ndarray,
    rng_px:       np.ndarray,
    lat_grid:     np.ndarray,
    lon_grid:     np.ndarray,
    scene_height: int,
    scene_width:  int,
    dst_crs:      CRS,
    dst_transform,
    dst_shape:    Tuple[int, int],
    output_path:  Path,
    src_nodata:   float = cfg.NODATA_CALC,
    dst_nodata:   float = cfg.NODATA_GEOREF,
) -> np.ndarray:
    """
    Fallback georeferencing using full lat/lon interpolation + scipy inverse mapping.

    Steps:
    1.  Interpolate lat/lon at every SLI pixel (272631 x 244).
    2.  Convert to Moon stereographic (x, y) using pyproj.
    3.  Build a 2-D interpolator: (x_stereo, y_stereo) → (src_row, src_col).
    4.  For each target pixel (x_t, y_t), find source (row, col) and
        sample CPR with map_coordinates.
    """
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    log.info("  Scipy fallback: interpolating full lat/lon grid ...")

    # Step 1: Full lat/lon grid
    lat_interp = RegularGridInterpolator(
        (az_px, rng_px), lat_grid, method="linear",
        bounds_error=False, fill_value=None,
    )
    lon_interp = RegularGridInterpolator(
        (az_px, rng_px), lon_grid, method="linear",
        bounds_error=False, fill_value=None,
    )

    # Evaluate at the tie-point rows only (8521 × 244) to keep memory down
    # For the inverse mapping we only need the bounding polygon, not every pixel.
    # We'll use 1024 evenly-spaced rows × all 244 cols.
    N_ROWS_SAMPLE = 1024
    az_sample  = np.linspace(0, scene_height - 1, N_ROWS_SAMPLE)
    rng_sample = np.arange(scene_width, dtype=float)
    Az, Rng = np.meshgrid(az_sample, rng_sample, indexing="ij")
    pts = np.column_stack([Az.ravel(), Rng.ravel()])

    lat_s = lat_interp(pts).reshape(N_ROWS_SAMPLE, scene_width)
    lon_s = lon_interp(pts).reshape(N_ROWS_SAMPLE, scene_width)

    # Step 2: Convert to Moon stereographic
    log.info("  Converting sampled lat/lon to Moon stereographic ...")
    transformer = Transformer.from_crs(cfg.MOON_GEO_WKT, dst_crs.to_wkt(), always_xy=True)
    x_s, y_s = transformer.transform(lon_s.ravel(), lat_s.ravel())
    x_s = x_s.reshape(N_ROWS_SAMPLE, scene_width)
    y_s = y_s.reshape(N_ROWS_SAMPLE, scene_width)

    # Step 3: Build inverse mapping interpolators
    # Along azimuth: y_stereo is roughly monotonic with source row
    # Along range:   x_stereo is roughly monotonic with source col
    y_mean_per_row = y_s.mean(axis=1)     # (N_ROWS_SAMPLE,)
    x_mean_per_col = x_s.mean(axis=0)    # (scene_width,)

    # Monotonicity check
    if y_mean_per_row[0] < y_mean_per_row[-1]:
        az_order = 1
    else:
        az_order = -1
    if x_mean_per_col[0] < x_mean_per_col[-1]:
        rng_order = 1
    else:
        rng_order = -1

    # Sorted versions for interpolation
    az_sorted   = az_sample[::az_order]
    y_sorted    = y_mean_per_row[::az_order]
    rng_sorted  = np.arange(scene_width, dtype=float)[::rng_order]
    x_sorted    = x_mean_per_col[::rng_order]

    inv_y = RegularGridInterpolator(
        (y_sorted,), az_sorted.reshape(-1, 1),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    inv_x = RegularGridInterpolator(
        (x_sorted,), rng_sorted.reshape(-1, 1),
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # Step 4: Build target grid and do inverse mapping
    log.info("  Building target grid and sampling CPR ...")
    with rasterio.open(src_path) as src:
        cpr_src = src.read(1).astype(np.float32)
    cpr_src[cpr_src == src_nodata] = np.nan

    dst_h, dst_w = dst_shape
    dst_arr = np.full((dst_h, dst_w), dst_nodata, dtype=np.float32)

    # Scene bounding box in stereo space
    x_min, y_min = float(x_s.min()), float(y_s.min())
    x_max, y_max = float(x_s.max()), float(y_s.max())
    log.info(f"  Scene extent (stereo): x=[{x_min:.0f}, {x_max:.0f}]  y=[{y_min:.0f}, {y_max:.0f}]")

    # Destination affine: dst_transform
    t = dst_transform
    for ti in range(dst_h):
        y_t = t.f + ti * t.e           # t.e is negative (top-down)
        if y_t < y_min or y_t > y_max:
            continue

        x_row = t.c + np.arange(dst_w) * t.a   # x coords for this row
        in_scene = (x_row >= x_min) & (x_row <= x_max)
        if not in_scene.any():
            continue

        x_sub   = x_row[in_scene]
        col_sub = np.where(in_scene)[0]

        # Inverse map y_t → source row
        src_row = float(inv_y(np.array([[y_t]]))[0, 0])
        if not np.isfinite(src_row):
            continue
        src_row = np.clip(src_row, 0, scene_height - 1)

        # Inverse map x → source col
        src_cols = inv_x(x_sub.reshape(-1, 1))[:, 0]
        valid = np.isfinite(src_cols)
        if not valid.any():
            continue

        src_cols = np.clip(src_cols[valid], 0, scene_width - 1)
        c_tgt    = col_sub[valid]

        # Sample with map_coordinates
        coords = np.array([
            np.full(len(src_cols), src_row),
            src_cols,
        ])
        vals = map_coordinates(cpr_src, coords, order=1, mode="nearest")
        dst_arr[ti, c_tgt] = vals.astype(np.float32)

        if ti % 2000 == 0:
            log.debug(f"    Target row {ti}/{dst_h}")

    n_filled = int(np.isfinite(dst_arr).sum())
    fill_pct = 100 * n_filled / dst_arr.size
    log.info(f"  Filled pixels: {n_filled:,}  ({fill_pct:.2f}%)")

    _write_georeferenced(dst_arr, dst_crs, dst_transform, output_path, dst_nodata)
    return dst_arr


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_georeferenced(
    arr:         np.ndarray,
    crs:         CRS,
    transform,
    output_path: Path,
    nodata:      float,
) -> None:
    """Write the georeferenced CPR array to a LZW-compressed float32 GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     arr.shape[1],
        "height":    arr.shape[0],
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "nodata":    nodata if np.isfinite(nodata) else float("nan"),
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            PRODUCT    = "Calculated_CPR_Georeferenced",
            CRS        = "Moon South Pole Stereographic",
            METHOD     = "GCP reproject (rasterio.warp.reproject)",
            REFERENCE  = "Putrevu et al. (2023) JGR Planets 10.1029/2023JE007745",
        )

    size_mb = output_path.stat().st_size / 1024**2
    log.info(f"  Written: {output_path.name}  ({size_mb:.1f} MB)")
