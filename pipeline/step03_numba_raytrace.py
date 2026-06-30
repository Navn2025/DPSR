"""
step03_numba_raytrace.py — Numba-compiled ray-casting kernel (CPU).

This module contains the computational heart of the DPSR pipeline.
It uses Numba's @njit(parallel=True) to:
  • Compile the kernel to native machine code (≈ 50–100× faster than CPython)
  • Parallelise the outer PSR-pixel loop across all CPU cores via OpenMP

Scientific algorithm (UNCHANGED from original)
-----------------------------------------------
For each PSR pixel P at position (row, col):
  For each of 72 azimuth directions:
    Walk outward along the ray.
    Track the terrain horizon as tan(elevation_angle) = Δh / distance.
    For every pixel Q along the ray:
      1. If Q is illuminated AND its terrain angle ≥ current horizon:
         → P can see sunlit terrain → P is NOT DPSR.  Stop all rays.
      2. Update horizon if Q is higher than all previous steps.
  If no illuminated terrain is visible from ANY direction → P is DPSR.

Key implementation choices
--------------------------
  tan(angle) comparison instead of arctan / degrees
    Saves arctan() + np.degrees() inside 720 billion iterations.
    Monotonically equivalent: tan is order-preserving for angles in (−90°,+90°).

  Visibility check BEFORE horizon update
    Ensures a pixel shadowed by closer terrain is correctly blocked.
    (Bug in naive implementations: updating horizon first makes the
    condition trivially true for any new maximum.)

  Early exit at pixel level
    Once visible illuminated terrain is found, the remaining 71
    directions are skipped entirely.  For non-DPSR pixels (majority)
    this can skip 95%+ of ray work.

  Bresenham integer offsets (from step02)
    ray_dr / ray_dc are int32 look-up tables; no float arithmetic per step.
    ray_len gives the exact number of valid pixels per ray (avoids
    processing RAY_SENTINEL padding).

Complexity
----------
  Worst case  : O(P × A × D)  ≈ 720 billion iterations
  With early exit and Bresenham:
    P = 20 016 516  PSR pixels
    A = 72          angles   (avg ~2 checked for non-DPSR pixels)
    D = 432         avg ray length (Bresenham, vs 500 DDA)
  Effective:  ~20M × 2 × 432  (non-DPSR)  +  ~small_fraction × 72 × 432 (DPSR)
    ≈ 17 billion  compiled ops  on 12 cores  ≈ 1–5 minutes

Runtime comparison
------------------
  Implementation              Throughput        Estimate (20 M PSR pixels)
  ─────────────────────────── ─────────────     ──────────────────────────
  Pure Python (baseline)       ~133 k px/s       40+ hours
  + Precomputed offsets         ~200 k px/s       28 hours
  + Numba @njit (1 thread)    ~4–8 M px/s        40–80 minutes
  + prange (12 cores)         ~40–80 M px/s       2–5 minutes   ← this module
  + early exit                ~80–200 M px/s      1–3 minutes
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
from numba import njit, prange

from pipeline.utils import RAY_SENTINEL


# ── Core Numba kernel ──────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def classify_psr_pixels(
    elevation:    np.ndarray,   # float32 (H, W)  elevation in metres
    illumination: np.ndarray,   # uint8   (H, W)  1=illuminated
    psr_rows:     np.ndarray,   # int32   (P,)
    psr_cols:     np.ndarray,   # int32   (P,)
    ray_dr:       np.ndarray,   # int32   (A, D)  Bresenham row offsets
    ray_dc:       np.ndarray,   # int32   (A, D)  Bresenham col offsets
    ray_dist:     np.ndarray,   # float32 (A, D)  Euclidean distances metres
    ray_len:      np.ndarray,   # int32   (A,)    valid steps per ray
) -> np.ndarray:                # uint8   (P,)    1=DPSR  0=not DPSR
    """
    Classify each PSR pixel as DPSR (1) or non-DPSR (0).

    Parameters
    ----------
    elevation    : DEM in metres, float32, C-contiguous
    illumination : binary illumination map, uint8
    psr_rows     : row indices of all PSR pixels, int32
    psr_cols     : col indices of all PSR pixels, int32
    ray_dr       : Bresenham row-offset table  (N_ANGLES × MAX_DISTANCE)
    ray_dc       : Bresenham col-offset table  (N_ANGLES × MAX_DISTANCE)
    ray_dist     : real-world distances        (N_ANGLES × MAX_DISTANCE), metres
    ray_len      : number of valid pixels per Bresenham ray (N_ANGLES,)

    Returns
    -------
    result : uint8 array of length P  (1 = DPSR,  0 = not DPSR)

    Notes
    -----
    • prange  → OpenMP thread pool; each thread handles a contiguous block
      of PSR pixels with no shared writes (race-free).
    • fastmath=True allows LLVM to use SIMD / FMA instructions.
    • cache=True   writes compiled bitcode to disk; subsequent runs skip JIT.
    • nogil=True   releases Python's GIL (relevant when used via threading).
    """
    n_psr    = psr_rows.shape[0]
    n_angles = ray_dr.shape[0]
    n_rows   = elevation.shape[0]
    n_cols   = elevation.shape[1]

    result = np.zeros(n_psr, dtype=np.uint8)

    for i in prange(n_psr):          # ← parallel loop  (12 threads on 12-core CPU)
        row   = psr_rows[i]
        col   = psr_cols[i]
        cur_h = elevation[row, col]

        is_dpsr = True

        for a in range(n_angles):
            # Terrain horizon stored as tangent (rise / run).
            # Using tan avoids arctan + degrees inside the inner loop.
            highest_tan = -1.0e18

            n_steps = ray_len[a]

            for d in range(n_steps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]

                # Inline bounds check  (cheaper than function call overhead)
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist        = ray_dist[a, d]
                terrain_tan = (elevation[r, c] - cur_h) / dist

                # ── Visibility check BEFORE horizon update ─────────────────
                # An illuminated pixel is visible iff it lies on or above
                # the terrain horizon formed by all closer pixels.
                # We must check illumination before updating highest_tan;
                # otherwise the comparison is trivially true for any new
                # horizon maximum.
                if illumination[r, c] == 1 and terrain_tan >= highest_tan:
                    is_dpsr = False
                    break            # found illuminated visible terrain

                if terrain_tan > highest_tan:
                    highest_tan = terrain_tan

            if not is_dpsr:
                break                # early exit: skip remaining angles

        if is_dpsr:
            result[i] = 1

    return result


# ── Non-parallel version (used by multiprocessing workers in step04) ───────────

@njit(cache=True, fastmath=True)
def classify_chunk(
    elevation:    np.ndarray,
    illumination: np.ndarray,
    psr_rows:     np.ndarray,
    psr_cols:     np.ndarray,
    ray_dr:       np.ndarray,
    ray_dc:       np.ndarray,
    ray_dist:     np.ndarray,
    ray_len:      np.ndarray,
) -> np.ndarray:
    """
    Same algorithm as classify_psr_pixels but WITHOUT prange.

    Used by individual worker processes in step04_parallel_processing.py
    so that each process runs a single-threaded kernel on its chunk.
    (Using prange inside worker processes would over-subscribe the CPU.)

    Parameters / Returns: identical to classify_psr_pixels.
    """
    n_psr    = psr_rows.shape[0]
    n_angles = ray_dr.shape[0]
    n_rows   = elevation.shape[0]
    n_cols   = elevation.shape[1]

    result = np.zeros(n_psr, dtype=np.uint8)

    for i in range(n_psr):           # ← sequential (parallelism is at process level)
        row   = psr_rows[i]
        col   = psr_cols[i]
        cur_h = elevation[row, col]

        is_dpsr = True

        for a in range(n_angles):
            highest_tan = -1.0e18
            n_steps     = ray_len[a]

            for d in range(n_steps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]

                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist        = ray_dist[a, d]
                terrain_tan = (elevation[r, c] - cur_h) / dist

                if illumination[r, c] == 1 and terrain_tan >= highest_tan:
                    is_dpsr = False
                    break

                if terrain_tan > highest_tan:
                    highest_tan = terrain_tan

            if not is_dpsr:
                break

        if is_dpsr:
            result[i] = 1

    return result


# ── JIT warm-up ────────────────────────────────────────────────────────────────

def warmup(ray_dr: np.ndarray, ray_dc: np.ndarray,
           ray_dist: np.ndarray, ray_len: np.ndarray) -> None:
    """
    Trigger Numba JIT compilation on a tiny dummy array.

    The first call to an @njit function compiles it; subsequent calls
    reuse the compiled binary (or load from cache if cache=True).
    Call warmup() once at startup so the compilation cost is not charged
    to the timed DPSR computation.

    Parameters
    ----------
    ray_dr, ray_dc, ray_dist, ray_len : precomputed ray tables (from step02)
    """
    _e   = np.zeros((8, 8), dtype=np.float32)
    _ill = np.zeros((8, 8), dtype=np.uint8)
    _r   = np.array([3], dtype=np.int32)
    _c   = np.array([3], dtype=np.int32)
    _dr2 = ray_dr[:, :2].copy()
    _dc2 = ray_dc[:, :2].copy()
    _dd2 = ray_dist[:, :2].copy()
    _rl  = np.minimum(ray_len, 2).copy()

    classify_psr_pixels(_e, _ill, _r, _c, _dr2, _dc2, _dd2, _rl)
    classify_chunk(     _e, _ill, _r, _c, _dr2, _dc2, _dd2, _rl)
