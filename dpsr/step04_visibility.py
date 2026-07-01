"""
step04_visibility.py  —  Curvature-corrected DEM visibility kernel (CPU + GPU).

Purpose
-------
Implement the DEM-based horizon analysis described in O'Brien & Byrne (2022),
Section 2.3 and Appendix (Eq. A4).  For every PSR pixel, cast rays in
N_ANGLES azimuth directions.  Along each ray, maintain the maximum
curvature-corrected terrain elevation angle seen so far.  If any terrain pixel
that lies ABOVE this running horizon (i.e., is visible) belongs to the non-PSR
class (psr_mask == 0), the centre pixel is immediately classified as NOT DPSR.
If no visible non-PSR terrain is found in any direction, the pixel IS DPSR.

Scientific definition (O'Brien & Byrne 2022, Sec. 2.3)
-------------------------------------------------------
    DPSR pixel P  ⟺  psr_mask[P] == 1
                      AND  ∀ azimuth a:
                          ∄ pixel Q visible from P along ray a
                          with psr_mask[Q] == 0

A pixel Q at range d from P is VISIBLE from P if and only if:

    tan(μ_Q) ≥  max_{Q' before Q along ray}  tan(μ_Q')

where μ_Q is the curvature-corrected elevation angle from P to Q.

Curvature correction — O'Brien & Byrne (2022) Appendix, Equation A4
--------------------------------------------------------------------
On a spherical body, distant terrain is depressed relative to the flat-terrain
prediction because the surface curves away from the horizontal.  The corrected
elevation angle μ at horizontal distance d is:

    tan(μ) = R₁ · (R₂ − √(d² + R₁²))  /  (d · R₂)        [Eq. A4]

where:
    R₁ = R_Moon + h_observer   (radial distance of observer from Moon centre)
    R₂ = R_Moon + h_target     (radial distance of target from Moon centre)
    d  = horizontal distance in metres (flat-plane, from ray_dist table)

Physical interpretation
~~~~~~~~~~~~~~~~~~~~~~~
Without curvature (flat Moon):  tan(μ) = (h₂ − h₁) / d
With curvature:                  tan(μ) = R₁(R₂ − √(d² + R₁²)) / (d R₂)

The difference at d = 50 km:
    Correction ≈ d / (2 R_Moon) = 50 000 / (2 × 1 737 400) ≈ 0.0144 rad
    Height offset ≈ d × correction = 50 000 × 0.0144 ≈ 720 m

Without curvature correction, PSR pixels deep in Shackleton or Haworth would
incorrectly "see" distant non-PSR terrain that is actually below their horizon
due to lunar curvature, causing them to be misclassified as non-DPSR.

Visibility logic (in implementation order, per step along each ray)
-------------------------------------------------------------------
1. Compute curvature-corrected tan(μ) for target pixel Q  [Eq. A4]
2. CHECK: if psr[Q] == 0 AND tan(μ_Q) >= highest_tan_so_far
          → visible non-PSR terrain found → NOT DPSR → early exit
3. UPDATE horizon: if tan(μ_Q) > highest_tan_so_far → update

CRITICAL: step 2 BEFORE step 3.
If we updated the horizon first, the comparison at step 2 would be trivially
true for any new maximum — incorrectly marking every new peak as visible.
Checking before updating means a pixel is visible only if it clears the
horizon formed by ALL closer terrain.

Inputs
------
elevation : float32 ndarray (H, W)   — DEM in metres (from step01)
psr_mask  : uint8   ndarray (H, W)   — 1=PSR  0=non-PSR (from step02)
psr_rows  : int32   ndarray (P,)     — PSR pixel row indices (from step02)
psr_cols  : int32   ndarray (P,)     — PSR pixel col indices (from step02)
ray_dr    : int32   ndarray (A, D)   — Bresenham row offsets (from step03)
ray_dc    : int32   ndarray (A, D)   — Bresenham col offsets (from step03)
ray_dist  : float32 ndarray (A, D)   — horizontal distances in metres (step03)
ray_len   : int32   ndarray (A,)     — valid steps per ray (from step03)
moon_r    : float                    — lunar reference radius in metres

Outputs
-------
flags : uint8 ndarray (P,)   — 1 = DPSR,  0 = not DPSR  (per PSR pixel)

Time complexity
---------------
Worst case   : O(P × A × D) — P PSR pixels × A angles × D steps
               ≈ 20 M × 360 × 2500 = 18 × 10¹² operations

With early exit (non-DPSR pixels stop after finding ONE visible non-PSR pixel):
  Non-DPSR pixels (>99.9% of PSR): average ~2 angles × ~50 steps ← early exit
  DPSR pixels (< 0.1% of PSR):     all 360 angles × up to 2500 steps

Effective: ≈ 20 M × 2 × 50 + small_number × 360 × 2500 ≈ 2 billion ops
Compiled Numba on 12 cores: ≈ 2–8 minutes wall time.

Memory complexity
-----------------
O(1) extra per pixel (only scalars: cur_h, R1, R1sq, highest_tan, is_dpsr).
The ray tables and DEM are read-only shared across all parallel threads.

Optimisation strategy
---------------------
• @njit(parallel=True): compiles to native machine code and parallelises
  the outer PSR-pixel loop over all CPU cores via OpenMP.
• prange(n_psr): OpenMP parallel for — each thread handles its own contiguous
  block of PSR pixels.  No shared writes (result[i] is written by only one
  thread).  Race-free by construction.
• Precomputed R1_sq = R1 × R1: avoids squaring inside the inner loop.
• dist² = dist × dist: avoids math.pow() (slower than multiply).
• fastmath=True: allows LLVM to use fused multiply-add (FMA) instructions.
• cache=True: compiled bitcode persists to disk; subsequent runs skip JIT.
• nogil=True: releases Python GIL (relevant for threading, not prange).
• Early-exit break at pixel level AND angle level: for non-DPSR pixels,
  the kernel typically terminates after < 5% of the theoretical work.

CUDA GPU path
-------------
One CUDA thread per PSR pixel.  The DEM + PSR mask are transferred to GPU
VRAM once; the ray tables (11 MB) fit in L2 and are accessed sequentially
per thread → coalesced reads for nearby threads in the same warp.

Reference
---------
O'Brien & Byrne (2022), PSJ 3:258, Section 2.3 and Appendix Eq. A4.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dpsr.utils import MOON_R, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# CUDA availability probe
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = False
try:
    from numba import cuda as _cuda
    _CUDA_AVAILABLE = _cuda.is_available()
except BaseException:
    # BaseException catches KeyboardInterrupt during slow CUDA init on CPU-only machines
    pass

_CUDA_THREADS = 256   # threads per CUDA block (one block per 256 PSR pixels)


# ---------------------------------------------------------------------------
# CPU kernel — Numba parallel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _classify_dpsr_cpu(
    elevation: np.ndarray,   # float32 (H, W)
    psr:       np.ndarray,   # uint8   (H, W)
    psr_rows:  np.ndarray,   # int32   (P,)
    psr_cols:  np.ndarray,   # int32   (P,)
    ray_dr:    np.ndarray,   # int32   (A, D)
    ray_dc:    np.ndarray,   # int32   (A, D)
    ray_dist:  np.ndarray,   # float32 (A, D)
    ray_len:   np.ndarray,   # int32   (A,)
    moon_r:    float,
) -> np.ndarray:             # uint8   (P,)
    """
    Classify PSR pixels as DPSR or non-DPSR using curvature-corrected
    horizon analysis.  Implements O'Brien & Byrne (2022), Sec. 2.3, Eq. A4.

    Each call to prange(n_psr) spawns n_psr independent tasks, each
    processed by one OpenMP thread.  No locks, no shared mutable state.

    Returns
    -------
    result : uint8 array (P,)  —  1 = DPSR,  0 = not DPSR
    """
    n_psr  = psr_rows.shape[0]
    n_a    = ray_dr.shape[0]
    n_rows = elevation.shape[0]
    n_cols = elevation.shape[1]
    result = np.zeros(n_psr, dtype=np.uint8)

    for i in prange(n_psr):           # OpenMP parallel — each i independent
        row   = psr_rows[i]
        col   = psr_cols[i]
        h_obs = elevation[row, col]

        # Pre-compute observer terms once per PSR pixel  (Eq. A4)
        R1    = moon_r + h_obs
        R1sq  = R1 * R1

        is_dpsr = True

        for a in range(n_a):
            # Horizon: maximum curvature-corrected tan(elevation_angle) seen so far
            # Initialise to a large negative value — flat horizon at -90°
            highest_tan = -1.0e18

            nsteps = ray_len[a]

            for d in range(nsteps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]

                # Bounds check — ray left DEM extent
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist = float(ray_dist[a, d])   # metres, guaranteed > 0

                # Curvature-corrected elevation angle  [O'Brien & Byrne Eq. A4]
                #   tan(μ) = R₁ · (R₂ − √(d² + R₁²)) / (d · R₂)
                R2     = moon_r + elevation[r, c]
                tan_mu = R1 * (R2 - math.sqrt(dist * dist + R1sq)) / (dist * R2)

                # ── Visibility check BEFORE horizon update ────────────────────
                # Q is visible from P iff tan(μ_Q) >= highest_tan (above horizon)
                # We check BEFORE updating highest_tan so that only pixels
                # genuinely above the terrain horizon count as visible.
                if psr[r, c] == 0 and tan_mu >= highest_tan:
                    # Visible non-PSR pixel found → P cannot be DPSR
                    is_dpsr = False
                    break   # stop this ray

                # Update running horizon
                if tan_mu > highest_tan:
                    highest_tan = tan_mu

            # Early exit: once one direction finds visible non-PSR, skip rest
            if not is_dpsr:
                break

        if is_dpsr:
            result[i] = 1

    return result


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

if _CUDA_AVAILABLE:
    from numba import cuda as _cuda

    @_cuda.jit(cache=True, fastmath=True)
    def _classify_dpsr_cuda(
        elevation, psr, psr_rows, psr_cols,
        ray_dr, ray_dc, ray_dist, ray_len,
        moon_r, result,
    ):
        """
        GPU kernel: one CUDA thread per PSR pixel.

        Memory access pattern
        ---------------------
        elevation / psr : (H, W) — scattered row/col access per ray step;
                          L2 cache (~6–12 MB) absorbs repeated access to the
                          same DEM tiles by nearby threads.
        ray_dr/dc/dist  : (A, D) — sequential column access per step →
                          coalesced reads for adjacent threads in the same warp
                          when they are on the same angle / step index.
        psr_rows/cols   : (P,)   — one read per thread → fully coalesced.
        result          : (P,)   — one write per thread → coalesced.

        Warp divergence note
        --------------------
        The is_dpsr=False early-exit causes warp divergence only for the
        rare pixels that actually ARE DPSR (< 0.05% of PSR pixels).  For
        the vast majority, all threads in a warp exit early and re-converge
        quickly.  Performance impact is negligible.
        """
        i = _cuda.grid(1)
        if i >= psr_rows.shape[0]:
            return

        n_rows = elevation.shape[0]
        n_cols = elevation.shape[1]
        n_a    = ray_dr.shape[0]

        row   = psr_rows[i]
        col   = psr_cols[i]
        h_obs = elevation[row, col]
        R1    = moon_r + h_obs
        R1sq  = R1 * R1
        is_dpsr = True

        for a in range(n_a):
            highest_tan = -1.0e18
            nsteps      = ray_len[a]

            for d in range(nsteps):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                dist   = ray_dist[a, d]
                R2     = moon_r + elevation[r, c]
                tan_mu = R1 * (R2 - math.sqrt(dist * dist + R1sq)) / (dist * R2)

                if psr[r, c] == 0 and tan_mu >= highest_tan:
                    is_dpsr = False
                    break

                if tan_mu > highest_tan:
                    highest_tan = tan_mu

            if not is_dpsr:
                break

        result[i] = 1 if is_dpsr else 0


# ---------------------------------------------------------------------------
# JIT warm-up
# ---------------------------------------------------------------------------

def warmup_jit(
    ray_dr:   np.ndarray,
    ray_dc:   np.ndarray,
    ray_dist: np.ndarray,
    ray_len:  np.ndarray,
    moon_r:   float = MOON_R,
) -> None:
    """
    Trigger Numba JIT compilation on a tiny dummy array.

    Purpose
    -------
    Numba compiles @njit functions on first invocation.  Calling warmup_jit()
    once during initialisation (with a 4×4 dummy DEM) separates JIT time from
    the reported DPSR computation time.

    Inputs  : precomputed ray tables from step03
    Outputs : none (compilation artefact written to __pycache__)
    Time    : ~15–30 s on first run;  < 0.1 s on subsequent runs (cache hit)
    """
    log.info("JIT warm-up — compiling Numba kernel (first run only) …")
    _e  = np.zeros((8, 8), dtype=np.float32)
    _p  = np.zeros((8, 8), dtype=np.uint8)
    _r  = np.array([3], dtype=np.int32)
    _c  = np.array([3], dtype=np.int32)
    _dr = np.ascontiguousarray(ray_dr[:, :4])
    _dc = np.ascontiguousarray(ray_dc[:, :4])
    _dd = np.ascontiguousarray(ray_dist[:, :4])
    _rl = np.minimum(ray_len, 4).copy()

    _classify_dpsr_cpu(_e, _p, _r, _c, _dr, _dc, _dd, _rl, float(moon_r))
    log.info("JIT warm-up complete.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_dpsr_pixels(
    elevation: np.ndarray,
    psr_mask:  np.ndarray,
    psr_rows:  np.ndarray,
    psr_cols:  np.ndarray,
    ray_dr:    np.ndarray,
    ray_dc:    np.ndarray,
    ray_dist:  np.ndarray,
    ray_len:   np.ndarray,
    moon_r:    float = MOON_R,
    use_gpu:   bool  = False,
) -> np.ndarray:
    """
    Classify all PSR pixels as DPSR or non-DPSR.

    Parameters
    ----------
    elevation : float32 (H, W) — DEM in metres
    psr_mask  : uint8   (H, W) — 1=PSR  0=non-PSR
    psr_rows  : int32   (P,)   — row indices of PSR pixels
    psr_cols  : int32   (P,)   — col indices of PSR pixels
    ray_dr    : int32   (A, D) — Bresenham row offsets
    ray_dc    : int32   (A, D) — Bresenham col offsets
    ray_dist  : float32 (A, D) — horizontal distances (metres)
    ray_len   : int32   (A,)   — valid steps per ray
    moon_r    : float          — lunar radius (metres)
    use_gpu   : bool           — True to use CUDA kernel (requires CUDA GPU)

    Returns
    -------
    flags : uint8 (P,)  — 1 = DPSR,  0 = not DPSR

    Notes
    -----
    The CPU kernel (use_gpu=False) uses Numba prange for OpenMP parallelism.
    The GPU kernel (use_gpu=True) launches one CUDA thread per PSR pixel.
    Both implement the identical scientific algorithm (Eq. A4).
    """
    n_psr = len(psr_rows)
    backend = "CUDA GPU" if (use_gpu and _CUDA_AVAILABLE) else "CPU (Numba parallel)"
    log.info(
        "Visibility kernel starting  n_psr=%s  n_angles=%d  max_dist=%d  backend=%s",
        f"{n_psr:,}", ray_dr.shape[0], ray_dr.shape[1], backend,
    )

    if use_gpu and _CUDA_AVAILABLE:
        return _run_gpu(elevation, psr_mask, psr_rows, psr_cols,
                        ray_dr, ray_dc, ray_dist, ray_len, moon_r)
    else:
        return _classify_dpsr_cpu(
            elevation, psr_mask, psr_rows, psr_cols,
            ray_dr, ray_dc, ray_dist, ray_len, float(moon_r),
        )


def _run_gpu(
    elevation, psr_mask, psr_rows, psr_cols,
    ray_dr, ray_dc, ray_dist, ray_len, moon_r,
) -> np.ndarray:
    """Transfer arrays to GPU, launch CUDA kernel, copy results back."""
    from numba import cuda

    n_psr   = len(psr_rows)
    n_blocks = math.ceil(n_psr / _CUDA_THREADS)

    log.info("  GPU: %d blocks × %d threads = %d threads",
             n_blocks, _CUDA_THREADS, n_blocks * _CUDA_THREADS)
    log.info("  Transferring arrays to GPU VRAM …")

    d_elev = cuda.to_device(np.ascontiguousarray(elevation))
    d_psr  = cuda.to_device(np.ascontiguousarray(psr_mask))
    d_rows = cuda.to_device(psr_rows)
    d_cols = cuda.to_device(psr_cols)
    d_dr   = cuda.to_device(np.ascontiguousarray(ray_dr))
    d_dc   = cuda.to_device(np.ascontiguousarray(ray_dc))
    d_dd   = cuda.to_device(np.ascontiguousarray(ray_dist))
    d_rl   = cuda.to_device(ray_len)
    d_out  = cuda.device_array(n_psr, dtype=np.uint8)

    log.info("  Launching CUDA kernel …")
    _classify_dpsr_cuda[n_blocks, _CUDA_THREADS](
        d_elev, d_psr, d_rows, d_cols,
        d_dr, d_dc, d_dd, d_rl,
        float(moon_r), d_out,
    )
    cuda.synchronize()
    return d_out.copy_to_host()
