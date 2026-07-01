"""
step03_precompute_rays.py  —  Precompute Bresenham ray pixel offsets.

Purpose
-------
Generate integer (row, col) offsets and horizontal distances for N_ANGLES
azimuth directions out to MAX_DIST pixels, using Bresenham's Line Algorithm.
These tables are passed to the Numba visibility kernel (step04) so that the
inner ray-walking loop requires only integer additions and array lookups —
no trigonometry, no floating-point coordinate arithmetic.

Scientific role
---------------
O'Brien & Byrne (2022), Section 2.3, describe a ray-casting approach:
"For each PSR pixel we trace rays outward in 720 azimuth directions
(0.5° spacing) to a distance of 150 km."

We reduce to 360 directions (1.0° spacing) and 50 km — see utils.py for
the full justification.  The angular spacing affects only which pixels are
visited along each ray; the visibility formula (Eq. A4) is unchanged.

Why Bresenham (not DDA / floating-point stepping)?
--------------------------------------------------
Direct Digital Analyser (DDA) approach — what a naive implementation does:
    r = int(observer_row + d * sin(az))   # inside the inner loop
    c = int(observer_col + d * cos(az))   # → repeated sin/cos/int casts

Problems with DDA:
  1. sin() and cos() are called once per step per pixel per angle:
       20M PSR px × 360 angles × 2500 steps = 18 TRILLION trig calls.
  2. int() truncation on a shallow-angle ray produces runs of identical
     (r, c) pairs — wasted iterations visiting the same pixel twice.
  3. Float → int casting at every step adds overhead.

Bresenham approach (this module):
  • Trig functions are called ONCE per angle (360 calls total at startup).
  • Each grid pixel along the line is visited EXACTLY once — no duplicates.
  • Results are stored in int32 arrays; the kernel loop does only:
      r = observer_row + ray_dr[a, d]   (integer add + lookup)
      c = observer_col + ray_dc[a, d]   (integer add + lookup)
  • ray_len[a] gives the true pixel count per ray (typically ~0.5–1.0 ×
    max_dist depending on angle), so the kernel avoids processing padding.

Inputs
------
n_angles : int   — number of azimuth directions  (default: N_ANGLES = 360)
max_dist : int   — maximum ray length in pixels   (default: MAX_DIST = 2500)
cellsize : float — pixel size in metres            (default: CELLSIZE = 20.0)

Outputs
-------
ray_dr   : int32   ndarray (n_angles, max_dist)
           Row offsets from observer pixel.  Columns beyond ray_len[a] are
           filled with RAY_SENTINEL (999 999) — never reached by the kernel.

ray_dc   : int32   ndarray (n_angles, max_dist)
           Column offsets from observer pixel.

ray_dist : float32 ndarray (n_angles, max_dist)
           Euclidean horizontal distance in metres from observer to each
           step along the ray.  Used in the curvature-correction formula.
           Minimum value is 1.0 m (guard against division by zero at origin).

ray_len  : int32   ndarray (n_angles,)
           Number of valid (non-padding) pixels per ray.

Mathematical basis
------------------
Azimuth a_k = 2πk / N_ANGLES,  k = 0 … N_ANGLES-1.

Convention: azimuth 0 points North (+row direction in DEM arrays with
North-up convention), rotating clockwise through East, South, West.
Actually in array indexing: row increases downward, so:
    Δrow = max_dist × sin(azimuth)   (positive = down = south in South Polar)
    Δcol = max_dist × cos(azimuth)   (positive = right = east)

The Bresenham algorithm then fills integer steps (dr, dc) from (0,0) to
(end_r, end_c) with no skipped or duplicate pixels.

Horizontal distance to the k-th step:
    dist[k] = sqrt(dr[k]^2 + dc[k]^2) × cellsize   (metres)

This is the 2-D Euclidean distance on the DEM plane (flat-plane approximation
for short distances).  The curvature correction in step04_visibility.py uses
this as d in Eq. A4, which already accounts for the spherical geometry.

Time complexity  : O(N_ANGLES × MAX_DIST)  —  72 × 2500 ≈ 180 k ops at startup
Memory complexity: O(N_ANGLES × MAX_DIST)
                  ray_dr + ray_dc : 2 × 360 × 2500 × 4 = 7.2 MB  (int32)
                  ray_dist        :     360 × 2500 × 4 = 3.6 MB  (float32)
                  ray_len         :     360 × 4        = 1.4 kB  (int32)
                  Total           : ≈ 11 MB

Optimisation strategy
---------------------
• This function runs ONCE at pipeline startup; runtime is negligible (< 1 s).
• The resulting arrays are passed directly to the Numba kernel, which
  accesses them via sequential indexing (ray_dr[a, d]) — cache-friendly.
• int32 (not int64) halves the ray table memory, fitting it comfortably in L3.

Reference
---------
O'Brien & Byrne (2022), PSJ 3:258, Section 2.3.
Bresenham, J.E. (1965). Algorithm for computer control of a digital plotter.
  IBM Systems Journal, 4(1), 25–30.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dpsr.utils import N_ANGLES, MAX_DIST, CELLSIZE, RAY_SENTINEL, Timer, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Bresenham line for one angle
# ---------------------------------------------------------------------------

def _bresenham_offsets(
    sin_a:    float,
    cos_a:    float,
    max_dist: int,
) -> tuple[list[int], list[int]]:
    """
    Compute Bresenham integer offsets (dr, dc) for one azimuth direction.

    Parameters
    ----------
    sin_a    : sin(azimuth)  — row component of the direction unit vector
    cos_a    : cos(azimuth)  — col component
    max_dist : maximum ray length in pixels

    Returns
    -------
    dr_list, dc_list : lists of row and col offsets from the origin pixel.

    Notes
    -----
    The algorithm walks from pixel (0,0) toward the computed endpoint
    (end_r, end_c), stepping one grid unit at a time in the dominant axis
    direction and conditionally stepping in the minor axis using a running
    error accumulator.  Each output pixel is visited exactly once.

    Two cases:
      abs_dc >= abs_dr  (shallow ray — dominant axis is column)
      abs_dr >  abs_dc  (steep  ray — dominant axis is row)
    """
    end_r = int(round(max_dist * sin_a))
    end_c = int(round(max_dist * cos_a))

    abs_dr = abs(end_r)
    abs_dc = abs(end_c)
    sr     = 1 if end_r >= 0 else -1
    sc     = 1 if end_c >= 0 else -1

    dr_list: list[int] = []
    dc_list: list[int] = []

    r, c = 0, 0

    if abs_dc >= abs_dr:
        err = 2 * abs_dr - abs_dc
        for _ in range(min(abs_dc, max_dist)):
            c  += sc
            if err >= 0:
                r    += sr
                err  -= 2 * abs_dc
            err += 2 * abs_dr
            dr_list.append(r)
            dc_list.append(c)
    else:
        err = 2 * abs_dc - abs_dr
        for _ in range(min(abs_dr, max_dist)):
            r  += sr
            if err >= 0:
                c    += sc
                err  -= 2 * abs_dr
            err += 2 * abs_dc
            dr_list.append(r)
            dc_list.append(c)

    return dr_list, dc_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def precompute_rays(
    n_angles: int   = N_ANGLES,
    max_dist: int   = MAX_DIST,
    cellsize: float = CELLSIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute Bresenham ray offsets for all azimuth directions.

    Parameters
    ----------
    n_angles : number of equally-spaced azimuth directions
               Paper: 720 (0.5°).  Default: 360 (1.0°).
    max_dist : maximum ray length in pixels
               Paper: 7500 (150 km).  Default: 2500 (50 km).
    cellsize : pixel size in metres (must match DEM; default 20 m).

    Returns
    -------
    ray_dr   : int32   (n_angles, max_dist)  row offsets; RAY_SENTINEL padding
    ray_dc   : int32   (n_angles, max_dist)  col offsets; RAY_SENTINEL padding
    ray_dist : float32 (n_angles, max_dist)  horizontal distance in metres
    ray_len  : int32   (n_angles,)           valid pixel count per ray

    Time complexity  : O(n_angles × max_dist)  ≈ 180 k ops — runs in < 1 s
    Memory           : ≈ 11 MB for default parameters

    Example
    -------
    >>> dr, dc, dist, rlen = precompute_rays()
    >>> dr.shape
    (360, 2500)
    >>> rlen.min(), rlen.max()
    (1767, 2500)
    """
    with Timer("Ray precomputation"):
        angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)

        ray_dr   = np.full((n_angles, max_dist), RAY_SENTINEL, dtype=np.int32)
        ray_dc   = np.full((n_angles, max_dist), RAY_SENTINEL, dtype=np.int32)
        ray_dist = np.zeros((n_angles, max_dist), dtype=np.float32)
        ray_len  = np.zeros(n_angles,             dtype=np.int32)

        for a, angle in enumerate(angles):
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            dr_list, dc_list = _bresenham_offsets(sin_a, cos_a, max_dist)

            n = min(len(dr_list), max_dist)
            ray_len[a] = n

            dr_arr = np.array(dr_list[:n], dtype=np.int32)
            dc_arr = np.array(dc_list[:n], dtype=np.int32)

            ray_dr[a, :n] = dr_arr
            ray_dc[a, :n] = dc_arr

            dist_m = (
                np.sqrt(dr_arr.astype(np.float64)**2 + dc_arr.astype(np.float64)**2)
                * cellsize
            )
            ray_dist[a, :n] = np.maximum(dist_m, 1.0).astype(np.float32)

    dda_len    = max_dist
    avg_bres   = float(ray_len.mean())
    saving_pct = (1.0 - avg_bres / dda_len) * 100.0

    log.info(
        "Rays ready  n_angles=%d  max_dist=%d (%.0f km)  "
        "avg_len=%.0f px  Bresenham saving=%.1f%%  RAM=%.1f MB",
        n_angles, max_dist, max_dist * cellsize / 1000.0,
        avg_bres, saving_pct,
        (ray_dr.nbytes + ray_dc.nbytes + ray_dist.nbytes) / 1e6,
    )

    return ray_dr, ray_dc, ray_dist, ray_len


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dr, dc, dist, rlen = precompute_rays()
    print(f"\n  ray_dr   shape : {dr.shape}   dtype : {dr.dtype}")
    print(f"  ray_dc   shape : {dc.shape}   dtype : {dc.dtype}")
    print(f"  ray_dist shape : {dist.shape}   dtype : {dist.dtype}")
    print(f"  ray_len  shape : {rlen.shape}   dtype : {rlen.dtype}")
    print(f"  ray_len  min / max : {rlen.min()} / {rlen.max()} px")
    print(f"  ray_dist max       : {dist.max():.0f} m")
