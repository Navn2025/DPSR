"""
step05_generate_dpsr.py — Main CPU pipeline entry point.

Orchestrates the full DPSR extraction:
  step01  →  Load DEM, illumination, PSR mask
  step02  →  Precompute Bresenham ray offsets
  step03  →  Numba @njit(parallel=True) kernel  [RECOMMENDED]
  step04  →  Multiprocessing + SharedMemory     [ALTERNATIVE]

Run
---
  python step05_generate_dpsr.py              # uses Numba parallel (default)
  python step05_generate_dpsr.py --mp         # uses multiprocessing fallback

Computational complexity summary
---------------------------------
                              Before            After (Numba parallel)
  ─────────────────────────── ───────────────── ─────────────────────────
  Inner loop language         CPython           Compiled machine code
  Parallelism                 None              12 CPU cores (prange)
  Trig per step               arctan + degrees  None (precomputed offsets)
  Redundant pixels per ray    ~14 % (DDA)       0 % (Bresenham)
  Early exit                  Yes               Yes
  ─────────────────────────── ───────────────── ─────────────────────────
  Throughput (20 M PSR px)    ~133 k px/s       ~40–200 M px/s
  Wall-clock estimate         40+ hours         1–5 minutes
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from pipeline.utils import DPSR_PATH, OUTPUT_DIR, Timer, get_logger
from pipeline.step01_load import load_all
from pipeline.step02_precompute_rays import precompute_rays
from pipeline.step03_numba_raytrace import classify_psr_pixels, warmup

log = get_logger("dpsr")


# ── Result → raster ───────────────────────────────────────────────────────────

def flags_to_raster(
    dpsr_flags: np.ndarray,   # uint8 (P,)
    psr_rows:   np.ndarray,   # int32 (P,)
    psr_cols:   np.ndarray,   # int32 (P,)
    shape:      tuple,        # (H, W)
) -> np.ndarray:              # uint8 (H, W)
    """
    Scatter the per-pixel classification flags back into a 2-D raster.

    Input
    -----
    dpsr_flags : 1=DPSR  0=non-DPSR  for each PSR pixel
    psr_rows   : row index of each PSR pixel
    psr_cols   : col index of each PSR pixel
    shape      : (H, W) of the output raster

    Output
    ------
    dpsr_raster : uint8 (H, W)  1=DPSR  0=everywhere else

    Complexity : O(P)  — vectorised index assignment, no Python loop.
    """
    raster = np.zeros(shape, dtype=np.uint8)
    mask   = dpsr_flags == 1
    raster[psr_rows[mask], psr_cols[mask]] = 1
    return raster


def save_dpsr(dpsr_raster: np.ndarray, meta: dict) -> None:
    """
    Write the DPSR raster to a LZW-compressed GeoTIFF.

    Input
    -----
    dpsr_raster : uint8 (H, W)
    meta        : rasterio metadata dict (CRS, transform, …)
    """
    out = DPSR_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    write_meta = meta.copy()
    write_meta.update(dtype="uint8", count=1, compress="lzw", driver="GTiff")

    with rasterio.open(out, "w", **write_meta) as dst:
        dst.write(dpsr_raster, 1)

    size_mb = out.stat().st_size / 1e6
    log.info("DPSR raster saved → %s  (%.1f MB)", out, size_mb)


def plot_results(
    psr_mask:    np.ndarray,
    dpsr_raster: np.ndarray,
    elapsed_s:   float,
) -> None:
    """
    Side-by-side comparison plot of PSR mask and extracted DPSR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)

    axes[0].imshow(psr_mask,    cmap="gray", interpolation="none")
    axes[0].set_title("PSR mask  (input)")
    axes[0].axis("off")

    axes[1].imshow(dpsr_raster, cmap="hot",  interpolation="none")
    axes[1].set_title(
        f"DPSR  (output)\n"
        f"pixels={dpsr_raster.sum():,}  time={elapsed_s:.0f} s"
    )
    axes[1].axis("off")

    plt.tight_layout()
    fig_path = OUTPUT_DIR.parent / "images" / "DPSR_result.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info("Result plot saved → %s", fig_path)
    plt.show()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(use_multiprocessing: bool = False) -> None:
    """
    Execute the full DPSR extraction pipeline.

    Parameters
    ----------
    use_multiprocessing : if True, use step04 (SharedMemory + Pool)
                          instead of step03 (Numba prange)
    """
    log.info("=" * 60)
    log.info("DPSR Extraction  —  Optimised CPU Pipeline")
    log.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("[1/5] Loading rasters …")
    with Timer("Load"):
        data = load_all()

    elevation    = data["elevation"]
    illumination = data["illumination"]
    psr_mask     = data["psr_mask"]
    psr_rows     = data["psr_rows"]
    psr_cols     = data["psr_cols"]
    meta         = data["meta"]
    n_psr        = len(psr_rows)

    # ── 2. Precompute ray offsets ─────────────────────────────────────────────
    log.info("[2/5] Precomputing Bresenham ray offsets …")
    with Timer("Rays"):
        ray_dr, ray_dc, ray_dist, ray_len = precompute_rays()

    # ── 3. JIT warm-up ────────────────────────────────────────────────────────
    log.info("[3/5] Compiling Numba kernels …")
    with Timer("JIT compile"):
        warmup(ray_dr, ray_dc, ray_dist, ray_len)

    # ── 4. Ray-cast ───────────────────────────────────────────────────────────
    log.info("[4/5] Classifying %s PSR pixels …", f"{n_psr:,}")

    if use_multiprocessing:
        log.info("      Mode: multiprocessing + SharedMemory")
        from pipeline.step04_parallel_processing import run_parallel
        t0         = time.perf_counter()
        dpsr_flags = run_parallel(
            elevation, illumination, psr_rows, psr_cols,
            ray_dr, ray_dc, ray_dist, ray_len,
        )
    else:
        log.info("      Mode: Numba @njit(parallel=True)  [recommended]")
        t0 = time.perf_counter()
        dpsr_flags = classify_psr_pixels(
            elevation, illumination,
            psr_rows, psr_cols,
            ray_dr, ray_dc, ray_dist, ray_len,
        )

    elapsed   = time.perf_counter() - t0
    rate      = n_psr / max(elapsed, 1e-6)
    log.info(
        "      Done in %.1f s (%.1f min)  throughput=%.0f px/s  DPSR=%s",
        elapsed, elapsed / 60, rate, f"{int(dpsr_flags.sum()):,}",
    )

    # ── 5. Save ───────────────────────────────────────────────────────────────
    log.info("[5/5] Saving output …")
    with Timer("Save"):
        dpsr_raster = flags_to_raster(dpsr_flags, psr_rows, psr_cols,
                                       elevation.shape)
        save_dpsr(dpsr_raster, meta)

    plot_results(psr_mask, dpsr_raster, elapsed)

    log.info("=" * 60)
    log.info("Pipeline complete.  Output → %s", DPSR_PATH)
    log.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPSR extraction — optimised CPU pipeline"
    )
    parser.add_argument(
        "--mp", action="store_true",
        help="Use multiprocessing + SharedMemory instead of Numba prange",
    )
    args = parser.parse_args()
    run(use_multiprocessing=args.mp)
