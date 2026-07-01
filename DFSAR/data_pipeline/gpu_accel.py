"""
gpu_accel.py
============
GPU-accelerated raster processing for the CH-2 DFSAR pipeline.

Backend priority (auto-detected at import time):
  1. CuPy    -- pip install cupy-cuda12x   (for CUDA 12)
                pip install cupy-cuda11x   (for CUDA 11)
  2. PyTorch -- pip install torch torchvision --index-url ^
                  https://download.pytorch.org/whl/cu121
  3. Parallel CPU -- concurrent.futures.ProcessPoolExecutor  (stdlib, ~4-8x)
  4. Sequential CPU -- always available

Public API
  resample_to_grid()        -- bilinear resampling (same CRS, any resolution)
  gpu_normalize()           -- percentile stretch to [0, 1]
  gpu_slope_hillshade()     -- slope (deg) + hillshade (0-255) from DEM
  backend_info()            -- human-readable backend string
  vram_free_gb()            -- free VRAM in GB (0 if no GPU)
"""
from __future__ import annotations

import gc
import math
import os
import concurrent.futures
from typing import Optional, Tuple

import numpy as np
from rasterio.transform import Affine

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

_BACKEND = "cpu"          # "cupy" | "torch" | "cpu_parallel" | "cpu"
_GPU_NAME = ""
_N_CPU_WORKERS = max(1, (os.cpu_count() or 4) - 1)   # leave one core free

# --- Try CuPy ---------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as _cp_map_coords

    _cp_dev = cp.cuda.Device(0)
    _cp_dev.use()
    # Warmup check
    _tmp = cp.zeros(1)
    del _tmp
    _BACKEND = "cupy"

    mem  = cp.cuda.runtime.memGetInfo()
    _GPU_NAME = f"CUDA GPU  free={mem[0]/1e9:.1f}GB / total={mem[1]/1e9:.1f}GB"

except Exception as _e:
    cp = None                                # type: ignore[assignment]
    _cp_map_coords = None

# --- Try PyTorch (if no CuPy) -----------------------------------------------
if _BACKEND == "cpu":
    try:
        import torch                         # type: ignore[import]
        import torch.nn.functional as _TF   # type: ignore[import]

        if torch.cuda.is_available():
            _BACKEND = "torch"
            _GPU_NAME = torch.cuda.get_device_name(0)
        else:
            torch = None                     # type: ignore[assignment]
    except ImportError:
        torch = None                         # type: ignore[assignment]

# --- Parallel CPU fallback --------------------------------------------------
if _BACKEND == "cpu" and _N_CPU_WORKERS > 1:
    _BACKEND = "cpu_parallel"


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def backend_info() -> str:
    """Human-readable description of the active backend."""
    if _BACKEND == "cupy":
        return f"CuPy GPU  ({_GPU_NAME})"
    if _BACKEND == "torch":
        return f"PyTorch GPU  ({_GPU_NAME})"
    if _BACKEND == "cpu_parallel":
        return f"Parallel CPU  ({_N_CPU_WORKERS} workers)"
    return "Sequential CPU  (install cupy or torch for GPU)"


def vram_free_gb() -> float:
    """Free GPU VRAM in GB; 0 if no GPU."""
    if _BACKEND == "cupy" and cp is not None:
        mem = cp.cuda.runtime.memGetInfo()
        return mem[0] / 1e9
    if _BACKEND == "torch" and torch is not None:   # type: ignore[union-attr]
        return torch.cuda.mem_get_info()[0] / 1e9
    return 0.0


# --------------------------------------------------------------------------- #
# GPU / parallel CPU bilinear resampling
# --------------------------------------------------------------------------- #

def resample_to_grid(
    src:           np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_shape:     Tuple[int, int],
    is_mask:       bool = False,
    chunk_rows:    int  = 1024,
) -> np.ndarray:
    """
    Resample *src* from its affine grid to the destination grid.

    Assumes the same CRS (no coordinate reprojection — only geometric
    resampling).  Bilinear for continuous data; nearest for masks.

    Parameters
    ----------
    src           : (H_src, W_src) float32 array, may contain NaN
    src_transform : affine transform of the source raster
    dst_transform : affine transform of the destination grid (DEM reference)
    dst_shape     : (H_dst, W_dst)
    is_mask       : use nearest-neighbour if True, else bilinear
    chunk_rows    : rows per GPU chunk (reduce if VRAM is tight)
    """
    order = 0 if is_mask else 1

    if _BACKEND == "cupy":
        return _cupy_resample(src, src_transform, dst_transform,
                              dst_shape, order, chunk_rows)
    if _BACKEND == "torch":
        return _torch_resample(src, src_transform, dst_transform,
                               dst_shape, is_mask)
    # CPU path
    return _cpu_resample(src, src_transform, dst_transform,
                         dst_shape, order)


# --------------------------------------------------------------------------- #
# CuPy implementation (chunked to keep VRAM usage low)
# --------------------------------------------------------------------------- #

_SENTINEL = np.float32(-1e30)   # sentinel for NaN in GPU arrays


def _cupy_resample(
    src: np.ndarray,
    src_t: Affine,
    dst_t: Affine,
    dst_shape: Tuple[int, int],
    order: int,
    chunk_rows: int,
) -> np.ndarray:
    """
    CuPy bilinear resampling.

    VRAM strategy for RTX 3050 6GB:
      - NaN pixels are replaced with a sentinel value (-1e30) so only ONE
        copy of the source array needs to live in VRAM.
      - Coordinates use float32 (sufficient for pixel indices up to ~30 000).
      - chunk_rows controls the working-set size per iteration.

    Peak VRAM: source_bytes + ~200 MB chunk overhead.
    For 24181x24794 DFSAR source (float32): ~2.4 GB + 0.2 GB = 2.6 GB.
    """
    H, W = dst_shape

    # Replace NaN with sentinel; upload once
    src_f32   = src.astype(np.float32)
    src_sent  = np.where(np.isfinite(src_f32), src_f32, _SENTINEL)
    src_gpu   = cp.asarray(src_sent)
    del src_sent, src_f32

    result = np.empty((H, W), dtype=np.float32)

    # Destination column coords -> source column coords (computed once, on GPU)
    # float32 is fine: pixel coords max ~30 000, well within float32 range
    cols_dst = cp.arange(W, dtype=cp.float32)
    x_dst    = np.float32(dst_t.c) + (cols_dst + 0.5) * np.float32(dst_t.a)
    col_src  = (x_dst - np.float32(src_t.c)) / np.float32(src_t.a) - 0.5   # (W,)

    for r0 in range(0, H, chunk_rows):
        r1    = min(r0 + chunk_rows, H)
        nrows = r1 - r0

        rows_dst = cp.arange(r0, r1, dtype=cp.float32)
        y_dst    = np.float32(dst_t.f) + (rows_dst + 0.5) * np.float32(dst_t.e)
        row_src  = (y_dst - np.float32(src_t.f)) / np.float32(src_t.e) - 0.5  # (nrows,)

        r_coords = cp.repeat(row_src, W)     # (nrows*W,)
        c_coords = cp.tile(col_src, nrows)   # (nrows*W,)
        coords   = cp.vstack([r_coords, c_coords])

        out_flat = _cp_map_coords(
            src_gpu, coords, order=order,
            mode="constant", cval=_SENTINEL,
        )
        chunk = out_flat.reshape(nrows, W)

        # Pixels that landed on sentinel (NaN or out-of-bounds) -> NaN
        chunk = cp.where(chunk < _SENTINEL * 0.5, cp.float32("nan"), chunk)
        result[r0:r1] = chunk.get()

        del r_coords, c_coords, coords, out_flat, chunk, row_src, y_dst, rows_dst
        cp.get_default_memory_pool().free_all_blocks()

    del src_gpu, col_src, x_dst, cols_dst
    cp.get_default_memory_pool().free_all_blocks()
    return result


# --------------------------------------------------------------------------- #
# PyTorch implementation (uses grid_sample, very fast on GPU)
# --------------------------------------------------------------------------- #

def _torch_resample(
    src: np.ndarray,
    src_t: Affine,
    dst_t: Affine,
    dst_shape: Tuple[int, int],
    is_mask: bool,
) -> np.ndarray:
    H, W       = dst_shape
    src_H, src_W = src.shape
    mode = "nearest" if is_mask else "bilinear"

    src_f32  = src.astype(np.float32)
    nan_mask = ~np.isfinite(src_f32)
    src_clean = np.where(nan_mask, np.float32(0), src_f32)

    # Destination pixel centers in map coords
    cols = torch.arange(W, dtype=torch.float64)
    rows = torch.arange(H, dtype=torch.float64)
    x_dst = dst_t.c + (cols + 0.5) * dst_t.a          # (W,)
    y_dst = dst_t.f + (rows + 0.5) * dst_t.e          # (H,)

    col_src = (x_dst - src_t.c) / src_t.a - 0.5       # (W,)
    row_src = (y_dst - src_t.f) / src_t.e - 0.5       # (H,)

    # Normalize to [-1, 1] for grid_sample
    col_norm = (2.0 * col_src / (src_W - 1) - 1.0).float()
    row_norm = (2.0 * row_src / (src_H - 1) - 1.0).float()

    # grid: (1, H, W, 2)  last dim = (x=col, y=row) normalized
    col_grid = col_norm.unsqueeze(0).expand(H, W)   # (H, W)
    row_grid = row_norm.unsqueeze(1).expand(H, W)   # (H, W)
    grid = torch.stack([col_grid, row_grid], dim=-1).unsqueeze(0)

    src_t_tensor  = torch.from_numpy(src_clean).unsqueeze(0).unsqueeze(0)
    nan_t_tensor  = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    if torch.cuda.is_available():
        src_t_tensor = src_t_tensor.cuda()
        nan_t_tensor = nan_t_tensor.cuda()
        grid = grid.cuda()

    with torch.no_grad():
        out = torch.nn.functional.grid_sample(
            src_t_tensor, grid, mode=mode,
            padding_mode="zeros", align_corners=False,
        )
        nan_out = torch.nn.functional.grid_sample(
            nan_t_tensor, grid, mode="bilinear",
            padding_mode="ones", align_corners=False,
        )

    result   = out.squeeze().cpu().numpy().astype(np.float32)
    nan_grid = nan_out.squeeze().cpu().numpy() > 0.5
    result[nan_grid] = np.nan
    return result


# --------------------------------------------------------------------------- #
# CPU scipy bilinear resampling (fallback, single-threaded)
# --------------------------------------------------------------------------- #

def _cpu_resample(
    src: np.ndarray,
    src_t: Affine,
    dst_t: Affine,
    dst_shape: Tuple[int, int],
    order: int,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    H, W = dst_shape
    src_f32   = src.astype(np.float32)
    nan_mask  = ~np.isfinite(src_f32)
    src_clean = np.where(nan_mask, np.float32(0), src_f32)

    cols_dst = np.arange(W, dtype=np.float64)
    rows_dst = np.arange(H, dtype=np.float64)

    x_dst = dst_t.c + (cols_dst + 0.5) * dst_t.a
    y_dst = dst_t.f + (rows_dst + 0.5) * dst_t.e

    col_src = (x_dst - src_t.c) / src_t.a - 0.5   # (W,)
    row_src = (y_dst - src_t.f) / src_t.e - 0.5   # (H,)

    r_coords = np.repeat(row_src, W)
    c_coords = np.tile(col_src,  H)
    coords   = np.vstack([r_coords, c_coords])

    out_flat = map_coordinates(src_clean, coords, order=order,
                               mode="constant", cval=0.0)
    nan_flat = map_coordinates(nan_mask.astype(np.float32), coords,
                               order=1, mode="constant", cval=1.0)

    result = out_flat.reshape(H, W).astype(np.float32)
    result[nan_flat.reshape(H, W) > 0.5] = np.nan
    return result


# --------------------------------------------------------------------------- #
# GPU normalization
# --------------------------------------------------------------------------- #

def gpu_normalize(
    array:  np.ndarray,
    p_low:  int = 2,
    p_high: int = 98,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Percentile-stretch normalisation on GPU (or CPU if no GPU).

    is_mask=True  ->  binary 0/1 output.
    is_mask=False ->  float [0, 1] output.
    """
    if is_mask:
        return _normalize_mask(array)

    if _BACKEND == "cupy":
        return _cupy_normalize(array, p_low, p_high)
    if _BACKEND == "torch":
        return _torch_normalize(array, p_low, p_high)

    # CPU fallback
    from normalizer import normalize_continuous
    return normalize_continuous(array, p_low, p_high)


def _normalize_mask(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    out = np.where(finite & (array > 0), np.float32(1), np.float32(0))
    out[~finite] = np.nan
    return out.astype(np.float32)


def _cupy_normalize(array: np.ndarray, p_low: int, p_high: int) -> np.ndarray:
    a           = cp.asarray(array, dtype=cp.float32)
    finite_mask = cp.isfinite(a)
    valid       = a[finite_mask]
    if valid.size == 0:
        del a, finite_mask, valid
        cp.get_default_memory_pool().free_all_blocks()
        return np.full_like(array, np.nan, dtype=np.float32)
    lo = float(cp.percentile(valid, p_low))
    hi = float(cp.percentile(valid, p_high))
    del valid
    if hi <= lo:
        del a, finite_mask
        cp.get_default_memory_pool().free_all_blocks()
        return np.zeros_like(array, dtype=np.float32)
    clipped          = cp.clip(a, np.float32(lo), np.float32(hi))
    normed           = (clipped - np.float32(lo)) / np.float32(hi - lo)
    normed[~finite_mask] = cp.float32("nan")
    result           = normed.get().astype(np.float32)
    del a, finite_mask, clipped, normed
    cp.get_default_memory_pool().free_all_blocks()
    return result


def _torch_normalize(array: np.ndarray, p_low: int, p_high: int) -> np.ndarray:
    a      = torch.from_numpy(array.astype(np.float32))
    finite = torch.isfinite(a)
    valid  = a[finite]
    if valid.numel() == 0:
        return np.full_like(array, np.nan, dtype=np.float32)
    lo = float(torch.quantile(valid, p_low  / 100.0))
    hi = float(torch.quantile(valid, p_high / 100.0))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    clipped = torch.clamp(a, lo, hi)
    normed  = (clipped - lo) / (hi - lo)
    normed[~finite] = float("nan")
    return normed.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# GPU slope + hillshade
# --------------------------------------------------------------------------- #

def gpu_slope_hillshade(
    dem:      np.ndarray,
    res_x:    float,
    res_y:    float,
    azimuth:  float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and hillshade (0-255) from DEM on GPU.
    Falls back to CPU (numpy) if no GPU is available.
    """
    if _BACKEND == "cupy":
        return _cupy_terrain(dem, res_x, res_y, azimuth, altitude, z_factor)
    if _BACKEND == "torch":
        return _torch_terrain(dem, res_x, res_y, azimuth, altitude, z_factor)
    # CPU numpy fallback
    from preprocessor import compute_slope, compute_hillshade
    return (
        compute_slope(dem, res_x, res_y),
        compute_hillshade(dem, res_x, res_y, azimuth, altitude, z_factor),
    )


def _terrain_arrays(xp, dem, res_x, res_y, azimuth, altitude, z_factor):
    """Shared terrain computation on whichever array namespace xp is."""
    nan_mask = ~xp.isfinite(dem)
    filled   = xp.where(nan_mask, xp.float32(0), dem * z_factor)

    # np.gradient convention: gradient(f, dy, dx) -> (df/dy, df/dx)
    if xp is np:
        dz_drow, dz_dcol = xp.gradient(filled, res_y, res_x)
    else:
        dz_drow, dz_dcol = xp.gradient(filled, res_y, res_x)

    slope_rad  = xp.arctan(xp.sqrt(dz_dcol**2 + dz_drow**2))
    aspect_rad = xp.arctan2(-dz_drow, dz_dcol)

    az_rad  = xp.float32(math.radians(360.0 - azimuth + 90.0))
    alt_rad = xp.float32(math.radians(altitude))

    hs = (
        xp.cos(alt_rad) * xp.cos(slope_rad)
        + xp.sin(alt_rad) * xp.sin(slope_rad) * xp.cos(az_rad - aspect_rad)
    )
    hs = xp.clip(hs * 255.0, 0.0, 255.0)

    slope_deg = xp.degrees(slope_rad)
    slope_deg[nan_mask] = xp.float32("nan")
    hs[nan_mask]        = xp.float32("nan")

    return slope_deg.astype(xp.float32), hs.astype(xp.float32)


def _cupy_terrain(dem, res_x, res_y, azimuth, altitude, z_factor):
    dem_gpu  = cp.asarray(dem, dtype=cp.float32)
    slope_g, hs_g = _terrain_arrays(cp, dem_gpu, res_x, res_y,
                                     azimuth, altitude, z_factor)
    slope = slope_g.get()
    hs    = hs_g.get()
    del dem_gpu, slope_g, hs_g
    cp.get_default_memory_pool().free_all_blocks()
    return slope, hs


def _torch_terrain(dem, res_x, res_y, azimuth, altitude, z_factor):
    # PyTorch doesn't have gradient(); fall back to numpy for terrain
    from preprocessor import compute_slope, compute_hillshade
    return (
        compute_slope(dem, res_x, res_y),
        compute_hillshade(dem, res_x, res_y, azimuth, altitude, z_factor),
    )


# --------------------------------------------------------------------------- #
# Parallel CPU reprojection helper (used when no GPU is found)
# --------------------------------------------------------------------------- #

def parallel_reproject_batch(
    tasks: list[dict],
    n_workers: int = _N_CPU_WORKERS,
) -> dict[str, Optional[np.ndarray]]:
    """
    Reproject a batch of rasters in parallel using ProcessPoolExecutor.

    Each task is a dict:
      { "label": str, "src_path": Path, "dst_path": Path,
        "ref_profile": dict, "resampling": Resampling }

    Returns { label -> ndarray or None }
    """
    from reproject import reproject_raster

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                reproject_raster,
                t["src_path"], t["dst_path"], t["ref_profile"], t["resampling"]
            ): t["label"]
            for t in tasks
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            try:
                results[label] = fut.result()
            except Exception as exc:
                from utils import get_logger
                get_logger("gpu_accel").error(
                    f"Parallel reproject failed [{label}]: {exc}"
                )
                results[label] = None

    return results
