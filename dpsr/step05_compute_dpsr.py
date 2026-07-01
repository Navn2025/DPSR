"""
step05_compute_dpsr.py  —  Orchestrate the complete DPSR classification pipeline.

Purpose
-------
Load all inputs, precompute ray tables, run the visibility kernel, scatter
results into a raster, and save the raw DPSR map before small-region
post-processing (step06).

This module is the main entry point for the DPSR computation.  It chains
steps 01–04 in order, prints a parameter summary at startup, and reports
timing and pixel statistics at each stage.

Scientific output
-----------------
DPSR_RAW_PATH : GeoTIFF uint8  (H, W)
    1 = pixel is DPSR (raw, before connected-component filtering)
    0 = pixel is not DPSR

The final filtered output is produced by step06_remove_small_regions.py,
which removes connected components smaller than 5 pixels (8-connected),
as specified by O'Brien & Byrne (2022), Fig. 3 caption.

Inputs  (all from earlier steps)
---------------------------------
DEM_PATH     → step01_load_dem
PSR_MASK_TIF → step02_load_psr
ray tables   → step03_precompute_rays
kernel       → step04_visibility

Outputs
-------
DPSR_RAW_PATH : raw DPSR raster (before small-region removal)
Returns       : dpsr_raster (uint8 ndarray H×W), meta dict

Runtime estimate (20 M PSR pixels, 360 angles, 50 km, 12-core CPU)
--------------------------------------------------------------------
  JIT compile  :  ~25 s (first run only; cached thereafter)
  DEM load     :  ~5 s
  Ray precomp  :  < 1 s
  Kernel       :  2–8 min  (varies with PSR density and early-exit rate)
  Save         :  < 5 s
  Total        :  ~3–10 min (CPU)  /  ~1–3 min (GPU)

Reference
---------
O'Brien & Byrne (2022), PSJ 3:258.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpsr.utils import (
    N_ANGLES, MAX_DIST, CELLSIZE, MOON_R, MIN_COMPONENT,
    DPSR_RAW_PATH, PARAMETER_TABLE,
    ELEV_PNG, PSR_PNG, DPSR_RAW_PNG,
    Timer, get_logger, save_preview,
)
from dpsr.step01_load_dem  import load_dem
from dpsr.step02_load_psr  import load_psr_mask, extract_psr_indices
from dpsr.step03_precompute_rays import precompute_rays
from dpsr.step04_visibility import (
    classify_dpsr_pixels, warmup_jit,
    _CUDA_AVAILABLE,
)

log = get_logger("dpsr.pipeline")


# ---------------------------------------------------------------------------
# Helper: scatter flags → raster
# ---------------------------------------------------------------------------

def flags_to_raster(
    flags:    np.ndarray,   # uint8 (P,)
    psr_rows: np.ndarray,   # int32 (P,)
    psr_cols: np.ndarray,   # int32 (P,)
    shape:    tuple,        # (H, W)
) -> np.ndarray:            # uint8 (H, W)
    """
    Scatter per-pixel DPSR flags back into a 2-D raster.

    Purpose
    -------
    The visibility kernel processes a flat list of PSR pixels (faster for
    parallelism).  This function maps results back to the (row, col) grid.

    Inputs
    ------
    flags    : uint8 (P,)  — 1=DPSR  0=non-DPSR  (output of step04)
    psr_rows : int32 (P,)  — row index of each PSR pixel
    psr_cols : int32 (P,)  — col index of each PSR pixel
    shape    : (H, W)      — output raster dimensions

    Outputs
    -------
    dpsr_raster : uint8 (H, W)  — 1=DPSR  0=everywhere else

    Complexity : O(P)  — vectorised index assignment, no Python loop.
    Memory     : O(H × W) for the output raster (~230 MB for 15 168² at uint8).
    """
    raster            = np.zeros(shape, dtype=np.uint8)
    dpsr_mask         = flags == 1
    raster[psr_rows[dpsr_mask], psr_cols[dpsr_mask]] = 1
    return raster


def save_raster(raster: np.ndarray, path: Path, meta: dict) -> None:
    """
    Write a uint8 raster to a LZW-compressed GeoTIFF.

    Inputs
    ------
    raster : uint8 ndarray (H, W)
    path   : output file path
    meta   : rasterio metadata dict (CRS, transform, …) from step01
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_meta = meta.copy()
    write_meta.update(dtype="uint8", count=1, compress="lzw",
                      driver="GTiff", nodata=None)
    with rasterio.open(path, "w", **write_meta) as dst:
        dst.write(raster, 1)
    log.info("Saved  %s  (%.1f MB)", path.name, path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_dpsr(
    use_gpu:   bool  = False,
    n_angles:  int   = N_ANGLES,
    max_dist:  int   = MAX_DIST,
    cellsize:  float = CELLSIZE,
    moon_r:    float = MOON_R,
    out_path:  Path  = DPSR_RAW_PATH,
    force:     bool  = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run the complete DPSR classification pipeline (steps 01–04).

    Parameters
    ----------
    use_gpu   : use CUDA GPU kernel if available
    n_angles  : azimuth directions  (default: 360 at 1° spacing)
    max_dist  : max ray length in pixels  (default: 2500 = 50 km)
    cellsize  : pixel size in metres  (default: 20.0)
    moon_r    : lunar radius in metres  (default: 1 737 400)
    out_path  : where to save the raw DPSR GeoTIFF
    force     : if True, recompute even if out_path already exists

    Returns
    -------
    dpsr_raster : uint8 (H, W)
    elevation   : float32 (H, W)
    psr_mask    : uint8 (H, W)
    meta        : rasterio metadata dict

    Notes
    -----
    If out_path already exists and force=False, the saved raster is loaded
    and returned directly (skips expensive computation).
    """
    t_total = time.perf_counter()

    # ── Print parameter table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DPSR Pipeline  —  O'Brien & Byrne (2022), PSJ 3:258")
    print("=" * 70)
    print(PARAMETER_TABLE)
    backend = "CUDA GPU" if (use_gpu and _CUDA_AVAILABLE) else "CPU (Numba parallel)"
    max_km  = max_dist * cellsize / 1000.0
    print(f"  Backend : {backend}")
    print(f"  Angles  : {n_angles}  ({360.0/n_angles:.1f}° spacing)")
    print(f"  Rays    : {max_dist} px  ({max_km:.0f} km)")
    print(f"  Moon R  : {moon_r:.0f} m")
    print("=" * 70 + "\n")

    if out_path.exists() and not force:
        log.info("Raw DPSR already exists — loading from %s", out_path.name)
        with rasterio.open(out_path) as ds:
            dpsr_raster = ds.read(1)
        with Timer("Step 1  Load DEM"):
            elevation, meta = load_dem()
        save_preview(elevation, ELEV_PNG, cmap="terrain",
                     label="LOLA DEM — Elevation (m)", pct_clip=1.0)
        with Timer("Step 2  Load PSR"):
            psr_mask = load_psr_mask(dem_meta=meta, dem_shape=elevation.shape)
        save_preview(psr_mask, PSR_PNG, cmap="Blues",
                     label="PSR mask  (1 = PSR)", vmin=0, vmax=1)
        save_preview(dpsr_raster, DPSR_RAW_PNG, cmap="hot",
                     label=f"DPSR raw  ({int(dpsr_raster.sum()):,} px)", vmin=0, vmax=1)
        return dpsr_raster, elevation, psr_mask, meta

    # ── Step 1: Load DEM ─────────────────────────────────────────────────────
    print("[1/5]  Loading DEM …")
    with Timer("Step 1  Load DEM"):
        elevation, meta = load_dem()
    save_preview(elevation, ELEV_PNG, cmap="terrain",
                 label="LOLA DEM — Elevation (m)", pct_clip=1.0)

    # ── Step 2: Load PSR mask ────────────────────────────────────────────────
    print("\n[2/5]  Loading PSR mask …")
    with Timer("Step 2  Load PSR"):
        psr_mask             = load_psr_mask(dem_meta=meta, dem_shape=elevation.shape)
        psr_rows, psr_cols   = extract_psr_indices(psr_mask)
        n_psr                = len(psr_rows)
    save_preview(psr_mask, PSR_PNG, cmap="Blues",
                 label="PSR mask  (1 = PSR)", vmin=0, vmax=1)

    print(f"         PSR pixels : {n_psr:,}  ({100.0*n_psr/psr_mask.size:.2f}% of DEM)")

    # ── Step 3: Precompute rays ───────────────────────────────────────────────
    print("\n[3/5]  Precomputing ray tables …")
    ray_dr, ray_dc, ray_dist, ray_len = precompute_rays(
        n_angles=n_angles, max_dist=max_dist, cellsize=cellsize,
    )

    # ── Step 4: JIT warm-up + visibility kernel ───────────────────────────────
    print("\n[4/5]  Running visibility kernel …")
    if not (use_gpu and _CUDA_AVAILABLE):
        warmup_jit(ray_dr, ray_dc, ray_dist, ray_len, moon_r)

    t_kernel = time.perf_counter()
    flags    = classify_dpsr_pixels(
        elevation, psr_mask, psr_rows, psr_cols,
        ray_dr, ray_dc, ray_dist, ray_len,
        moon_r  = moon_r,
        use_gpu = (use_gpu and _CUDA_AVAILABLE),
    )
    kernel_s = time.perf_counter() - t_kernel
    rate     = n_psr / max(kernel_s, 1e-6)

    n_dpsr_raw = int(flags.sum())
    log.info(
        "Kernel done  %.1f s (%.1f min)  throughput=%.0f px/s  raw_DPSR=%s",
        kernel_s, kernel_s / 60.0, rate, f"{n_dpsr_raw:,}",
    )
    print(f"         Raw DPSR   : {n_dpsr_raw:,}  ({100.0*n_dpsr_raw/n_psr:.4f}% of PSR)")

    # ── Step 5: Scatter → raster + save ─────────────────────────────────────
    print("\n[5/5]  Saving raw DPSR raster …")
    with Timer("Step 5  Save"):
        dpsr_raster = flags_to_raster(flags, psr_rows, psr_cols, elevation.shape)
        save_raster(dpsr_raster, out_path, meta)
    save_preview(dpsr_raster, DPSR_RAW_PNG, cmap="hot",
                 label=f"DPSR raw  ({int(dpsr_raster.sum()):,} px)", vmin=0, vmax=1)

    # Sanity check: every DPSR pixel must be within PSR
    n_outside_psr = int(((dpsr_raster == 1) & (psr_mask == 0)).sum())
    if n_outside_psr > 0:
        log.error(
            "SANITY FAIL: %d DPSR pixels are outside PSR! "
            "Check DEM/PSR alignment.", n_outside_psr,
        )
    else:
        log.info("Sanity check PASSED: all DPSR pixels are within PSR.")

    total_s = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Step 5 complete  —  total wall time {total_s:.1f} s ({total_s/60:.1f} min)")
    print(f"  Raw DPSR saved → {out_path}")
    print(f"{'=' * 70}\n")
    print("  Next step: python -m dpsr.step06_remove_small_regions")

    return dpsr_raster, elevation, psr_mask, meta


# ---------------------------------------------------------------------------
# CLI  —  python -m dpsr.step05_compute_dpsr [--gpu] [--redo]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPSR classification — O'Brien & Byrne (2022)"
    )
    parser.add_argument("--gpu",  action="store_true", help="Use CUDA GPU")
    parser.add_argument("--redo", action="store_true", help="Force recompute")
    parser.add_argument("--angles",   type=int,   default=N_ANGLES,
                        help=f"Azimuth directions (default {N_ANGLES})")
    parser.add_argument("--max-dist", type=int,   default=MAX_DIST,
                        help=f"Max ray length pixels (default {MAX_DIST})")
    args = parser.parse_args()

    run_dpsr(
        use_gpu  = args.gpu,
        n_angles = args.angles,
        max_dist = args.max_dist,
        force    = args.redo,
    )
