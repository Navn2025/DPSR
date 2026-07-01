"""
reproject.py
============
Reproject and resample any raster to match the DEM reference grid.

All layers are aligned to:
  - The DEM CRS
  - The DEM affine transform (origin + pixel size)
  - The DEM width and height

GPU path  (CuPy / PyTorch):
  When source and destination share the same CRS, uses gpu_accel.resample_to_grid()
  — a custom bilinear sampler that runs entirely on the GPU (~2 sec/layer vs
  ~2 min/layer on CPU).  Falls back to rasterio.warp automatically on failure.

CPU path:
  rasterio.warp.reproject  (GDAL, single-threaded).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as warp_reproject

from config import ALIGNED_DIR, RESAMPLE_CONTINUOUS
from utils import get_logger, warn_large
from config import LARGE_ARRAY_WARN_MB

log = get_logger("reproject")

# Import GPU module once; fall back gracefully if unavailable
try:
    from gpu_accel import resample_to_grid as _gpu_resample, backend_info as _gpu_backend
    _GPU_RESAMPLE_AVAILABLE = True
    log.info(f"GPU resampling backend: {_gpu_backend()}")
except Exception as _gpu_err:
    _GPU_RESAMPLE_AVAILABLE = False
    log.info(f"GPU resampling unavailable ({_gpu_err}); using rasterio.warp")


# -- Reference profile ---------------------------------------------------------

def get_reference_profile(dem_path: Path) -> dict:
    """
    Read the DEM and return a rasterio profile that describes the target grid.
    Every other layer will be reprojected / resampled to match this profile.
    """
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    # Force float32 output regardless of source dtype
    profile.update(
        count=1,
        dtype="float32",
        nodata=float("nan"),
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        driver="GTiff",
    )

    log.info(f"Reference CRS       : {profile.get('crs')}")
    log.info(f"Reference transform : {profile.get('transform')}")
    log.info(f"Reference size      : {profile['width']} x {profile['height']}")
    return profile


# -- Single-raster reprojection ------------------------------------------------

def reproject_raster(
    src_path:    Path,
    dst_path:    Path,
    ref_profile: dict,
    resampling:  Resampling = RESAMPLE_CONTINUOUS,
    band_idx:    int = 1,
) -> Optional[np.ndarray]:
    """
    Reproject Band-*band_idx* of *src_path* to the *ref_profile* grid.

    Saves the result to *dst_path* and returns the float32 array so callers
    can use it in-memory without a second read.  Returns None on failure.

    Parameters
    ----------
    src_path    : source raster (any CRS / resolution)
    dst_path    : output path inside outputs/aligned/
    ref_profile : reference profile obtained from get_reference_profile()
    resampling  : Resampling.bilinear for continuous, .nearest for masks
    band_idx    : 1-indexed band to read from the source
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    dst_crs   = ref_profile["crs"]
    dst_trans = ref_profile["transform"]
    dst_w     = ref_profile["width"]
    dst_h     = ref_profile["height"]

    try:
        with rasterio.open(src_path) as src:
            # Read and cast
            src_data  = src.read(band_idx).astype("float32")
            src_nd    = src.nodata
            src_crs   = src.crs
            src_trans = src.transform

            # Replace source NoData with NaN before warping
            if src_nd is not None:
                try:
                    bad = np.isnan(float(src_nd))
                except (ValueError, TypeError):
                    bad = False
                if bad:
                    src_data[~np.isfinite(src_data)] = np.nan
                else:
                    src_data[src_data == float(src_nd)] = np.nan

            # If source has no CRS, assume it matches the target
            if src_crs is None:
                log.warning(
                    f"{src_path.name} has no CRS — assuming it already matches "
                    f"target CRS ({dst_crs}). Verify this manually."
                )
                src_crs = dst_crs

        # ── GPU path: same CRS → pure resampling (no coordinate transform) ──
        same_crs = _crs_match(src_crs, dst_crs)
        used_gpu = False

        if _GPU_RESAMPLE_AVAILABLE and same_crs:
            try:
                is_mask = (resampling == Resampling.nearest)
                dst_data = _gpu_resample(
                    src_data, src_trans, dst_trans,
                    (dst_h, dst_w), is_mask=is_mask,
                )
                used_gpu = True
            except Exception as gpu_exc:
                log.warning(
                    f"GPU resampling failed [{src_path.name}]: {gpu_exc} "
                    f"-- falling back to rasterio.warp"
                )

        # ── CPU path (rasterio.warp) ──
        if not used_gpu:
            dst_data = np.full((dst_h, dst_w), np.nan, dtype="float32")
            with rasterio.open(src_path) as src:
                warp_reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_trans,
                    src_crs=src_crs,
                    dst_transform=dst_trans,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )

        # Save aligned raster
        out_profile = ref_profile.copy()
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(dst_data, 1)

        tag = "GPU" if used_gpu else "CPU"
        warn_large(dst_path.stem, dst_data.nbytes, LARGE_ARRAY_WARN_MB)
        log.info(f"Aligned [{tag}] -> {dst_path.name}")
        return dst_data

    except Exception as exc:
        log.error(f"Reprojection failed [{src_path.name}]: {exc}")
        return None


def _crs_match(crs_a, crs_b) -> bool:
    """Return True if both CRS refer to the same projection (loose check)."""
    if crs_a is None or crs_b is None:
        return False
    try:
        return crs_a.to_epsg() == crs_b.to_epsg() and crs_a.to_epsg() is not None
    except Exception:
        pass
    try:
        return crs_a.to_wkt() == crs_b.to_wkt()
    except Exception:
        return str(crs_a) == str(crs_b)


# -- Read already-aligned raster -----------------------------------------------

def read_aligned(path: Path) -> Optional[np.ndarray]:
    """Read Band-1 of a previously aligned float32 raster."""
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            nd   = src.nodata
            if nd is not None:
                data[data == float(nd)] = np.nan
        return data
    except Exception as exc:
        log.error(f"Cannot read aligned raster {path.name}: {exc}")
        return None


# -- Save a bare array with reference georeferencing ---------------------------

def save_aligned(
    array:       np.ndarray,
    name:        str,
    ref_profile: dict,
    out_dir:     Path = ALIGNED_DIR,
) -> Path:
    """Write *array* as a float32 GeoTIFF in *out_dir* with *ref_profile* CRS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tif"
    p = ref_profile.copy()
    p.update(count=1, dtype="float32", nodata=float("nan"),
             compress="lzw", tiled=True)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(array.astype("float32"), 1)
    log.info(f"Saved aligned layer -> {out_path.name}")
    return out_path
