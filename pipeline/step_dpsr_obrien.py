"""
pipeline/step_dpsr_obrien.py
============================
DPSR algorithm from O'Brien & Byrne (2022),
"Double Shadows at the Lunar Poles", PSJ 3:258.

Algorithm (Sections 2.2-2.3 + Appendix)
-----------------------------------------
Single shadowing (PSR):
  For each pixel, cast 720 rays at 0.5 deg spacing to d_max ~150 km.
  Compute curvature-corrected horizon angle (Eq. A4).
  Compare against max solar elevation angle (Eq. 2).
  If horizon exceeds sun in ALL directions -> permanently shadowed.

Double shadowing (DPSR):
  For every PSR pixel, cast the same 720 rays.
  A pixel is visible if its curvature-corrected elevation angle >=
  the maximum angle from all terrain closer to the observer.
  If ANY visible pixel has psr_mask == 0 (non-PSR) -> NOT doubly shadowed.
  If ALL visible pixels are PSR -> pixel IS doubly shadowed (DPSR).

Post-processing:
  Remove connected components < 5 pixels (8-connected).

Curvature-corrected elevation angle (Appendix, Eq. A4):
  tan(mu) = R1 * (R2 - sqrt(d^2 + R1^2)) / (d * R2)
  where:
    R1 = Moon radius + elevation at observer
    R2 = Moon radius + elevation at target
    d  = horizontal distance in metres (gnomonic / stereographic)

  At d = 50 km, the correction lowers apparent elevation by ~720 m,
  which is crucial for correctly classifying deep polar craters.

Inputs:  DEM (float32, metres)  +  binary PSR mask (uint8)
         No illumination raster needed.
Outputs: binary DPSR mask (uint8)

Computational notes
-------------------
Paper uses 720 angles x 7500 px (150 km) = ~108 T operations for 20 M PSR px.
Default here: 360 angles x 2500 px (50 km) with early-exit optimisation.
Reducing to 1 deg spacing introduces at most a few percent error (paper p.5).
GPU path: elevation + PSR transferred once; PSR pixel list split into 256-thread
blocks; early-exit causes warp divergence only for the rare true-DPSR pixels.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy.ndimage import label as _nd_label

MOON_R = 1_737_400.0   # metres  (lunar reference radius, Smith et al. 2010)

# ── CUDA availability ─────────────────────────────────────────────────────────

_CUDA_AVAILABLE = False
try:
    from numba import cuda as _cuda
    _CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    pass

_THREADS = 256   # CUDA threads per block (1-D launch over PSR pixels)


# ── Bresenham ray precomputation ──────────────────────────────────────────────

def precompute_rays(
    n_angles: int   = 360,
    max_dist: int   = 2500,
    cellsize: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute Bresenham integer ray offsets for n_angles azimuth directions.

    Returns
    -------
    ray_dr   : int32   (n_angles, max_dist)   row offsets
    ray_dc   : int32   (n_angles, max_dist)   col offsets
    ray_dist : float32 (n_angles, max_dist)   horizontal distance in metres
    ray_len  : int32   (n_angles,)            valid steps per ray
    """
    _SENTINEL = 999_999
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)

    ray_dr   = np.full((n_angles, max_dist), _SENTINEL, dtype=np.int32)
    ray_dc   = np.full((n_angles, max_dist), _SENTINEL, dtype=np.int32)
    ray_dist = np.zeros((n_angles, max_dist), dtype=np.float32)
    ray_len  = np.zeros(n_angles, dtype=np.int32)

    for a, angle in enumerate(angles):
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        end_r = int(round(max_dist * sin_a))
        end_c = int(round(max_dist * cos_a))
        abs_dr = abs(end_r)
        abs_dc = abs(end_c)
        sr = 1 if end_r >= 0 else -1
        sc = 1 if end_c >= 0 else -1

        dr_list: list[int] = []
        dc_list: list[int] = []
        r, c = 0, 0

        if abs_dc >= abs_dr:
            err = 2 * abs_dr - abs_dc
            for _ in range(min(abs_dc, max_dist)):
                c += sc
                if err >= 0:
                    r += sr
                    err -= 2 * abs_dc
                err += 2 * abs_dr
                dr_list.append(r)
                dc_list.append(c)
        else:
            err = 2 * abs_dc - abs_dr
            for _ in range(min(abs_dr, max_dist)):
                r += sr
                if err >= 0:
                    c += sc
                    err -= 2 * abs_dr
                err += 2 * abs_dc
                dr_list.append(r)
                dc_list.append(c)

        n = min(len(dr_list), max_dist)
        ray_len[a] = n
        dr_arr = np.array(dr_list[:n], dtype=np.int32)
        dc_arr = np.array(dc_list[:n], dtype=np.int32)
        ray_dr[a, :n] = dr_arr
        ray_dc[a, :n] = dc_arr
        dist_arr = np.sqrt(dr_arr.astype(np.float64)**2 +
                           dc_arr.astype(np.float64)**2) * cellsize
        ray_dist[a, :n] = np.maximum(dist_arr, 1.0).astype(np.float32)

    return ray_dr, ray_dc, ray_dist, ray_len


# ── CPU kernel (Numba parallel) ───────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _dpsr_cpu(elevation, psr, psr_rows, psr_cols,
              ray_dr, ray_dc, ray_dist, ray_len, moon_r):
    """
    Classify each PSR pixel as DPSR (1) or not (0).

    Implements O'Brien & Byrne Sec 2.3:
      visibility target = psr == 0   (NOT illumination raster)
      elevation angle   = curvature-corrected (Eq. A4)
    """
    n_psr    = psr_rows.shape[0]
    n_angles = ray_dr.shape[0]
    n_rows   = elevation.shape[0]
    n_cols   = elevation.shape[1]
    result   = np.zeros(n_psr, dtype=np.uint8)

    for i in prange(n_psr):
        row   = psr_rows[i]
        col   = psr_cols[i]
        cur_h = elevation[row, col]
        R1    = moon_r + cur_h
        R1_sq = R1 * R1          # precomputed once per PSR pixel
        is_dpsr = True

        for a in range(n_angles):
            highest_tan = -1.0e18
            n_steps     = ray_len[a]

            for d in range(n_steps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist = float(ray_dist[a, d])
                R2   = moon_r + elevation[r, c]

                # Curvature-corrected elevation angle (O'Brien & Byrne Eq. A4):
                #   tan(mu) = R1 * (R2 - sqrt(d^2 + R1^2)) / (d * R2)
                tan_mu = R1 * (R2 - math.sqrt(dist * dist + R1_sq)) / (dist * R2)

                # Visibility: check BEFORE updating highest_tan
                # (checking after would trivially pass for any new maximum)
                if psr[r, c] == 0 and tan_mu >= highest_tan:
                    is_dpsr = False    # visible non-PSR surface -> not DPSR
                    break

                if tan_mu > highest_tan:
                    highest_tan = tan_mu

            if not is_dpsr:
                break    # early exit: skip remaining azimuths

        if is_dpsr:
            result[i] = 1

    return result


# ── CUDA kernel ───────────────────────────────────────────────────────────────

if _CUDA_AVAILABLE:
    from numba import cuda as _cuda

    @_cuda.jit(cache=True, fastmath=True)
    def _dpsr_cuda(elevation, psr, psr_rows, psr_cols,
                   ray_dr, ray_dc, ray_dist, ray_len, moon_r, result):
        """
        GPU DPSR kernel.  One CUDA thread per PSR pixel.

        Memory layout:
          elevation, psr  : (H, W) row-major; row access per step is
                            scattered, but early exit limits total access.
          ray_dr/dc/dist  : (n_angles, max_dist) accessed sequentially per
                            angle step -- L2 cache friendly for the thread block.
          psr_rows/cols   : linear list of PSR pixel coordinates.
        """
        i = _cuda.grid(1)
        if i >= psr_rows.shape[0]:
            return

        n_rows   = elevation.shape[0]
        n_cols   = elevation.shape[1]
        n_angles = ray_dr.shape[0]

        row   = psr_rows[i]
        col   = psr_cols[i]
        cur_h = elevation[row, col]
        R1    = moon_r + cur_h
        R1_sq = R1 * R1
        is_dpsr = True

        for a in range(n_angles):
            highest_tan = -1.0e18
            n_steps     = ray_len[a]

            for d in range(n_steps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist   = ray_dist[a, d]
                R2     = moon_r + elevation[r, c]
                tan_mu = R1 * (R2 - math.sqrt(dist * dist + R1_sq)) / (dist * R2)

                if psr[r, c] == 0 and tan_mu >= highest_tan:
                    is_dpsr = False
                    break

                if tan_mu > highest_tan:
                    highest_tan = tan_mu

            if not is_dpsr:
                break

        result[i] = 1 if is_dpsr else 0


# ── Connected-component filter ────────────────────────────────────────────────

def remove_small_components(dpsr_raster: np.ndarray, min_size: int = 5):
    """
    Remove DPSR connected components with fewer than min_size pixels.

    Uses 8-connectivity (paper: "five contiguous pixels", Fig. 3 caption).
    Returns filtered raster and count of removed pixels.
    """
    struct = np.ones((3, 3), dtype=np.int32)          # 8-connectivity
    labeled, n_comp = _nd_label(dpsr_raster, structure=struct)

    if n_comp == 0:
        return dpsr_raster.copy(), 0

    # np.bincount on ravel() view is O(N), no extra allocation for the ravel
    sizes      = np.bincount(labeled.ravel())
    sizes[0]   = 0                                     # exclude background
    keep_mask  = sizes >= min_size
    filtered   = keep_mask[labeled].astype(np.uint8)

    removed = int(dpsr_raster.sum()) - int(filtered.sum())
    return filtered, removed


# ── Public API ────────────────────────────────────────────────────────────────

def compute_dpsr(
    elevation:     np.ndarray,
    psr_mask:      np.ndarray,
    n_angles:      int   = 360,
    max_dist:      int   = 2500,
    cellsize:      float = 20.0,
    moon_r:        float = MOON_R,
    use_gpu:       bool  = None,
    min_component: int   = 5,
) -> np.ndarray:
    """
    Compute DPSR map following O'Brien & Byrne (2022).

    Parameters
    ----------
    elevation      : DEM in metres (float32, H x W)
    psr_mask       : binary PSR map (uint8, H x W); 1=PSR 0=non-PSR
    n_angles       : azimuth directions (paper: 720; practical default: 360)
    max_dist       : max ray pixels   (paper: 7500=150km; default: 2500=50km)
    cellsize       : pixel size in metres
    moon_r         : lunar reference radius in metres
    use_gpu        : True / False / None (auto-detect)
    min_component  : remove DPSR components smaller than this (paper: 5)

    Returns
    -------
    dpsr_raster : uint8 (H, W)   1 = DPSR,  0 = not DPSR
    """
    if use_gpu is None:
        use_gpu = _CUDA_AVAILABLE

    backend = "CUDA" if (use_gpu and _CUDA_AVAILABLE) else "CPU Numba"
    max_km  = max_dist * cellsize / 1000.0
    print(f"  O'Brien & Byrne (2022) DPSR")
    print(f"    {n_angles} azimuth directions x {max_dist} px ({max_km:.0f} km)  [{backend}]")
    print(f"    Curvature correction: ON (Eq. A4, R_Moon={moon_r/1000:.1f} km)")
    print(f"    Visibility target: psr_mask==0  (NOT illumination raster)")
    print(f"    Min component size: {min_component} px (8-connected)")

    # PSR pixel list
    psr_rows, psr_cols = np.where(psr_mask == 1)
    psr_rows = psr_rows.astype(np.int32)
    psr_cols = psr_cols.astype(np.int32)
    n_psr = len(psr_rows)
    print(f"  PSR pixels to process : {n_psr:,}")

    # Precompute Bresenham ray offsets
    t0 = time.perf_counter()
    ray_dr, ray_dc, ray_dist, ray_len = precompute_rays(n_angles, max_dist, cellsize)
    print(f"  Ray precomputation : {time.perf_counter()-t0:.1f} s")

    # --- Classify ---
    t1 = time.perf_counter()

    if use_gpu and _CUDA_AVAILABLE:
        n_blocks = math.ceil(n_psr / _THREADS)
        print(f"  GPU: {n_blocks} blocks x {_THREADS} threads")
        print(f"  Transferring arrays to GPU ...")
        d_elev = _cuda.to_device(elevation)
        d_psr  = _cuda.to_device(psr_mask)
        d_rows = _cuda.to_device(psr_rows)
        d_cols = _cuda.to_device(psr_cols)
        d_dr   = _cuda.to_device(ray_dr)
        d_dc   = _cuda.to_device(ray_dc)
        d_dist = _cuda.to_device(ray_dist)
        d_rlen = _cuda.to_device(ray_len)
        d_out  = _cuda.device_array(n_psr, dtype=np.uint8)

        print(f"  Launching CUDA kernel ...")
        _dpsr_cuda[n_blocks, _THREADS](
            d_elev, d_psr, d_rows, d_cols,
            d_dr, d_dc, d_dist, d_rlen,
            float(moon_r), d_out,
        )
        _cuda.synchronize()
        flags = d_out.copy_to_host()

    else:
        print(f"  Running CPU parallel kernel ...")
        flags = _dpsr_cpu(
            elevation, psr_mask, psr_rows, psr_cols,
            ray_dr, ray_dc, ray_dist, ray_len, float(moon_r),
        )

    elapsed = time.perf_counter() - t1
    print(f"  Kernel time : {elapsed:.1f} s  ({elapsed/60:.1f} min)  "
          f"speed={n_psr/max(elapsed,0.001):,.0f} px/s")

    # Scatter results back to raster grid
    dpsr_raster = np.zeros(elevation.shape, dtype=np.uint8)
    dpsr_raster[psr_rows[flags == 1], psr_cols[flags == 1]] = 1
    raw_count = int(dpsr_raster.sum())
    print(f"  Raw DPSR pixels : {raw_count:,}  ({100*raw_count/max(n_psr,1):.3f}% of PSR)")

    # Remove small connected components (paper: <5 pixels excluded)
    if min_component > 1 and raw_count > 0:
        dpsr_raster, removed = remove_small_components(dpsr_raster, min_component)
        final_count = int(dpsr_raster.sum())
        print(f"  After removing <{min_component}-px components: "
              f"removed {removed:,}  ->  {final_count:,} DPSR pixels")

    final = int(dpsr_raster.sum())
    print(f"  Final DPSR : {final:,}  ({100*final/max(n_psr,1):.4f}% of PSR)  "
          f"({100*final/elevation.size:.4f}% of DEM)")

    return dpsr_raster
