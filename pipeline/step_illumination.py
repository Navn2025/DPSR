"""
step_illumination.py — Physically based solar illumination / shadow map.

Replaces the hillshade-threshold proxy with a proper shadow-casting model.

Algorithm
---------
For each pixel P, cast a ray in the direction of the sun along the DEM.
P is illuminated iff no terrain along that ray exceeds the sun's elevation
angle:

    max { (h_d - h_P) / dist_d  :  d along ray toward sun } < tan(E)

where E is the sun elevation angle and dist_d is horizontal distance to step d.

Backends
--------
  CPU  :  Numba @njit(parallel=True)  — used automatically when no GPU
  GPU  :  Numba CUDA                  — used automatically when CUDA available

GPU strategy for annual illumination
-------------------------------------
The elevation array (920 MB for 15k×15k DEM) is transferred to the GPU ONCE
and kept there for all 72 azimuth passes.  Only the tiny per-azimuth ray
vectors (2500 × 3 arrays = ~30 KB) are sent each iteration.

The CUDA kernel writes the union (OR) of illumination masks in-place into a
single device array.  Pixels already marked lit are skipped via early return,
so each subsequent azimuth requires less work than the previous one.

Speedup vs CPU: ~100–300× (5 min/az CPU → 2–10 s/az GPU).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import math
import time

import numpy as np
from numba import njit, prange

from pipeline.utils import CELLSIZE, MAX_DISTANCE, SUN_ELEVATION, SUN_AZIMUTH, get_logger

log = get_logger("illumination")

# ── CUDA availability check ───────────────────────────────────────────────────
_CUDA_AVAILABLE = False
try:
    from numba import cuda as _cuda
    _CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    pass

_TPB = 16   # CUDA threads per block per dimension  (16×16 = 256 threads/block)


# ── Ray precomputation (shared by CPU and GPU paths) ─────────────────────────

def _sun_ray(sun_az_deg: float, max_dist: int, cellsize: float):
    """
    Integer (row, col) offsets and distances for a ray toward the sun.

    Geographic convention: azimuth 0=North, 90=East.
    Row increases southward → dr = -cos(az), dc = +sin(az).

    Returns
    -------
    dr   : int32   (max_dist,)   row offsets (toward sun)
    dc   : int32   (max_dist,)   col offsets (toward sun)
    dist : float32 (max_dist,)   Euclidean horizontal distance in metres
    """
    az = math.radians(sun_az_deg)
    dr_unit = -math.cos(az)
    dc_unit =  math.sin(az)

    steps = np.arange(1, max_dist + 1, dtype=np.float64)
    dr    = np.round(dr_unit * steps).astype(np.int32)
    dc    = np.round(dc_unit * steps).astype(np.int32)
    dist  = np.sqrt(dr.astype(np.float64)**2 +
                    dc.astype(np.float64)**2) * cellsize
    dist  = np.where(dist < 1.0, 1.0, dist).astype(np.float32)
    return dr, dc, dist


# ── CPU shadow kernel (Numba parallel) ───────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _shadow_kernel_cpu(elevation, sun_dr, sun_dc, sun_dist, sun_tan):
    """
    CPU shadow casting.  Each pixel cast one ray toward the sun.
    Returns uint8 (H,W): 1=illuminated, 0=shadow.
    """
    H, W     = elevation.shape
    max_dist = sun_dr.shape[0]
    result   = np.ones((H, W), dtype=np.uint8)

    for row in prange(H):
        for col in range(W):
            cur_h = elevation[row, col]
            for d in range(max_dist):
                r = row + sun_dr[d]
                c = col + sun_dc[d]
                if r < 0 or r >= H or c < 0 or c >= W:
                    break
                if (elevation[r, c] - cur_h) / sun_dist[d] >= sun_tan:
                    result[row, col] = 0
                    break
    return result


@njit(parallel=True, cache=True, fastmath=True)
def _shadow_union_cpu(elevation, sun_dr, sun_dc, sun_dist, sun_tan, combined):
    """
    CPU annual update: OR current illumination into 'combined' in place.
    Pixels already lit (combined=1) skip the ray walk.
    """
    H, W     = elevation.shape
    max_dist = sun_dr.shape[0]

    for row in prange(H):
        for col in range(W):
            if combined[row, col]:
                continue
            cur_h = elevation[row, col]
            lit   = True
            for d in range(max_dist):
                r = row + sun_dr[d]
                c = col + sun_dc[d]
                if r < 0 or r >= H or c < 0 or c >= W:
                    break
                if (elevation[r, c] - cur_h) / sun_dist[d] >= sun_tan:
                    lit = False
                    break
            if lit:
                combined[row, col] = np.uint8(1)


# ── CUDA shadow kernels ───────────────────────────────────────────────────────

if _CUDA_AVAILABLE:
    from numba import cuda as _cuda

    @_cuda.jit(cache=True, fastmath=True)
    def _shadow_kernel_cuda(elevation, dr, dc, dist, n_steps, sun_tan, out):
        """
        Single-epoch shadow map.
        Each CUDA thread handles one (row, col) pixel.
        out[row,col] = 1 if illuminated, 0 if in shadow.
        """
        row, col = _cuda.grid(2)
        H = elevation.shape[0]
        W = elevation.shape[1]
        if row >= H or col >= W:
            return

        cur_h = elevation[row, col]
        for d in range(n_steps):
            r = row + dr[d]
            c = col + dc[d]
            if r < 0 or r >= H or c < 0 or c >= W:
                break
            if (elevation[r, c] - cur_h) / dist[d] >= sun_tan:
                out[row, col] = 0
                return          # shadow — stop
        out[row, col] = 1       # illuminated

    @_cuda.jit(cache=True, fastmath=True)
    def _shadow_union_cuda(elevation, dr, dc, dist, n_steps, sun_tan, combined):
        """
        Annual illumination update kernel.
        Marks pixel as lit (combined=1) if sunlight from this azimuth reaches it.
        Pixels already lit are skipped → work shrinks with each azimuth pass.

        Memory layout: elevation is (H,W) float32, row-major.
        Threads in a warp share the same dr[d]/dc[d] offset, so they access
        consecutive cols in the same row → coalesced global memory reads.
        """
        row, col = _cuda.grid(2)
        H = elevation.shape[0]
        W = elevation.shape[1]
        if row >= H or col >= W:
            return
        if combined[row, col]:  # already lit → skip entirely
            return

        cur_h = elevation[row, col]
        for d in range(n_steps):
            r = row + dr[d]
            c = col + dc[d]
            if r < 0 or r >= H or c < 0 or c >= W:
                break
            if (elevation[r, c] - cur_h) / dist[d] >= sun_tan:
                return          # blocked → leave combined as 0
        combined[row, col] = 1  # sunlight reaches pixel → mark lit


def _cuda_grid(H, W):
    """Return (blocks_2d, threads_2d) for a 2D CUDA launch over (H, W)."""
    import math
    bx = math.ceil(W / _TPB)
    by = math.ceil(H / _TPB)
    return (by, bx), (_TPB, _TPB)


# ── Public API — single epoch ─────────────────────────────────────────────────

def compute_solar_illumination(
    elevation:  np.ndarray,
    sun_az_deg: float = SUN_AZIMUTH,
    sun_el_deg: float = SUN_ELEVATION,
    cellsize:   float = CELLSIZE,
    max_dist:   int   = MAX_DISTANCE,
    use_gpu:    bool  = False,
) -> np.ndarray:
    """
    Binary illumination map for a single sun position.

    Returns uint8 (H, W): 1=illuminated, 0=shadow.
    """
    sun_tan = math.tan(math.radians(sun_el_deg))
    dr, dc, dist = _sun_ray(sun_az_deg, max_dist, cellsize)

    log.info("Shadow cast: az=%.1f°  el=%.2f°  tan=%.5f  ray=%d px (%.0f km)  backend=%s",
             sun_az_deg, sun_el_deg, sun_tan, max_dist, max_dist * cellsize / 1000,
             "CUDA" if (use_gpu and _CUDA_AVAILABLE) else "CPU")

    t0 = time.perf_counter()

    if use_gpu and _CUDA_AVAILABLE:
        H, W      = elevation.shape
        blocks, tpb = _cuda_grid(H, W)
        d_elev    = _cuda.to_device(elevation)
        d_dr      = _cuda.to_device(dr)
        d_dc      = _cuda.to_device(dc)
        d_dist    = _cuda.to_device(dist)
        d_out     = _cuda.device_array((H, W), dtype=np.uint8)
        _shadow_kernel_cuda[blocks, tpb](d_elev, d_dr, d_dc, d_dist,
                                         len(dr), sun_tan, d_out)
        _cuda.synchronize()
        illum = d_out.copy_to_host()
    else:
        illum = _shadow_kernel_cpu(elevation, dr, dc, dist, sun_tan)

    log.info("  done %.1f s  lit=%.2f%%", time.perf_counter() - t0,
             100.0 * illum.mean())
    return illum


# ── Public API — annual illumination ─────────────────────────────────────────

def compute_annual_illumination(
    elevation:  np.ndarray,
    n_azimuths: int   = 72,
    sun_el_deg: float = SUN_ELEVATION,
    cellsize:   float = CELLSIZE,
    max_dist:   int   = MAX_DISTANCE,
    use_gpu:    bool  = None,   # None = auto-detect
) -> np.ndarray:
    """
    Annual illumination: union over all sun azimuths at peak solar elevation.

    Returns uint8 (H, W): 1 = pixel receives sunlight from ≥1 azimuth,
                           0 = permanent shadow from all directions.

    GPU path (use_gpu=True or auto-detected):
      Elevation transferred once.  Each azimuth launches one CUDA kernel.
      Expected speedup: 100–300× over CPU.
    """
    if use_gpu is None:
        use_gpu = _CUDA_AVAILABLE

    backend = "CUDA" if (use_gpu and _CUDA_AVAILABLE) else "CPU"
    log.info("Annual illumination: %d azimuths  el=%.2f°  ray=%d px (%.0f km)  backend=%s",
             n_azimuths, sun_el_deg, max_dist, max_dist * cellsize / 1000, backend)

    azimuths = np.linspace(0.0, 360.0, n_azimuths, endpoint=False)
    sun_tan  = math.tan(math.radians(sun_el_deg))
    H, W     = elevation.shape
    t_total  = time.perf_counter()

    if use_gpu and _CUDA_AVAILABLE:
        # ── GPU path ────────────────────────────────────────────────────────
        blocks, tpb = _cuda_grid(H, W)

        # Transfer elevation once — stays on GPU for all azimuth passes
        log.info("  Transferring elevation to GPU (%.0f MB) …",
                 elevation.nbytes / 1e6)
        d_elev    = _cuda.to_device(elevation)
        d_combined = _cuda.to_device(np.zeros((H, W), dtype=np.uint8))

        for i, az in enumerate(azimuths):
            dr, dc, dist = _sun_ray(float(az), max_dist, cellsize)
            d_dr   = _cuda.to_device(dr)
            d_dc   = _cuda.to_device(dc)
            d_dist = _cuda.to_device(dist)

            _shadow_union_cuda[blocks, tpb](
                d_elev, d_dr, d_dc, d_dist, len(dr), sun_tan, d_combined
            )
            _cuda.synchronize()

            elapsed = time.perf_counter() - t_total
            eta     = elapsed / (i + 1) * (n_azimuths - i - 1)
            log.info("  az %5.1f°  [%2d/%2d]  %.1f s  ETA %.0f s",
                     az, i + 1, n_azimuths, elapsed, eta)

        combined = d_combined.copy_to_host()

    else:
        # ── CPU path ────────────────────────────────────────────────────────
        combined = np.zeros((H, W), dtype=np.uint8)

        for i, az in enumerate(azimuths):
            dr, dc, dist = _sun_ray(float(az), max_dist, cellsize)
            _shadow_union_cpu(elevation, dr, dc, dist, sun_tan, combined)

            elapsed = time.perf_counter() - t_total
            eta     = elapsed / (i + 1) * (n_azimuths - i - 1)
            log.info("  az %5.1f°  [%2d/%2d]  lit=%.1f%%  %.1f s  ETA %.0f s",
                     az, i + 1, n_azimuths, 100.0 * combined.mean(), elapsed, eta)

    total_t = time.perf_counter() - t_total
    log.info("Annual illumination done  %.0f s (%.1f min)  lit=%.2f%%  shadow=%.2f%%",
             total_t, total_t / 60,
             100.0 * combined.mean(),
             100.0 * (combined == 0).mean())
    return combined


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, rasterio
    from pipeline.utils import DEM_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Compute solar shadow map")
    parser.add_argument("--az",     type=float, default=SUN_AZIMUTH)
    parser.add_argument("--el",     type=float, default=SUN_ELEVATION)
    parser.add_argument("--annual", action="store_true")
    parser.add_argument("--n-az",   type=int,   default=72)
    parser.add_argument("--gpu",    action="store_true")
    parser.add_argument("--cpu",    action="store_true")
    args = parser.parse_args()

    use_gpu = None          # auto-detect
    if args.gpu: use_gpu = True
    if args.cpu: use_gpu = False

    log.info("Loading DEM ...")
    with rasterio.open(DEM_PATH) as ds:
        elev = ds.read(1).astype(np.float32) * 1000.0
        meta = ds.meta.copy()
        meta.update(dtype="uint8", count=1, compress="lzw", nodata=None)

    if args.annual:
        illum = compute_annual_illumination(elev, n_azimuths=args.n_az,
                                            sun_el_deg=args.el, use_gpu=use_gpu)
        out = OUTPUT_DIR / "illumination_annual.tif"
    else:
        illum = compute_solar_illumination(elev, sun_az_deg=args.az,
                                           sun_el_deg=args.el, use_gpu=use_gpu)
        out = OUTPUT_DIR / "illumination.tif"

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(illum, 1)
    log.info("Saved: %s  lit=%.2f%%", out, 100.0 * illum.mean())
