"""
run_pipeline.py  —  Single command to run the complete DPSR pipeline.

Usage
-----
    python -m dpsr.run_pipeline              # CPU, default params
    python -m dpsr.run_pipeline --gpu        # CUDA GPU
    python -m dpsr.run_pipeline --redo       # force recompute all steps
    python -m dpsr.run_pipeline --angles 720 --max-dist 7500  # paper params

Steps
-----
    [1] Load DEM           (step01_load_dem)
    [2] Load PSR mask      (step02_load_psr)
    [3] Precompute rays    (step03_precompute_rays)
    [4] Visibility kernel  (step04_visibility)
    [5] Save raw DPSR      (step05_compute_dpsr)
    [6] Remove small regions (step06_remove_small_regions)
    [7] Validate           (step07_validation)

Reference
---------
O'Brien & Byrne (2022), PSJ 3:258.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpsr.utils import (
    N_ANGLES, MAX_DIST, DPSR_FINAL_PATH, SUMMARY_PNG, get_logger, save_summary,
)
from dpsr.step05_compute_dpsr         import run_dpsr
from dpsr.step06_remove_small_regions import run_filter
from dpsr.step07_validation           import run_validation

log = get_logger("dpsr.run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full DPSR pipeline — O'Brien & Byrne (2022)"
    )
    parser.add_argument("--gpu",      action="store_true",
                        help="Use CUDA GPU for visibility kernel")
    parser.add_argument("--redo",     action="store_true",
                        help="Force recompute all steps (delete cached outputs)")
    parser.add_argument("--angles",   type=int, default=N_ANGLES,
                        help=f"Number of azimuth directions (paper: 720; default: {N_ANGLES})")
    parser.add_argument("--max-dist", type=int, default=MAX_DIST,
                        help=f"Max ray length in pixels (paper: 7500; default: {MAX_DIST})")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation step (faster)")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # ── Steps 1–5: classify ──────────────────────────────────────────────────
    dpsr_raw, elevation, psr_mask, meta = run_dpsr(
        use_gpu  = args.gpu,
        n_angles = args.angles,
        max_dist = args.max_dist,
        force    = args.redo,
    )

    # ── Step 6: small-region removal ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6  —  Remove small DPSR connected components")
    print("=" * 70)
    dpsr_final = run_filter(meta=meta, force=args.redo)

    # ── Step 7: validation ───────────────────────────────────────────────────
    if not args.no_validate and DPSR_FINAL_PATH.exists():
        print("\n" + "=" * 70)
        print("  STEP 7  —  Scientific validation")
        print("=" * 70)
        run_validation()

    # ── Summary composite image ───────────────────────────────────────────────
    save_summary(
        arrays=[
            (elevation, "terrain",  "Elevation (m)"),
            (psr_mask,  "Blues",    "PSR mask"),
            (dpsr_raw,  "Oranges",  "DPSR raw"),
            (dpsr_final,"hot",      f"DPSR final  ({int(dpsr_final.sum()):,} px)"),
        ],
        png_path=SUMMARY_PNG,
        suptitle="DPSR Pipeline Summary — O'Brien & Byrne (2022)",
    )

    total = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete  —  {total:.1f} s ({total/60:.1f} min)")
    print(f"  Outputs saved to results/ and images/:")
    print(f"    results/PSR_mask.tif")
    print(f"    results/DPSR_raw.tif")
    print(f"    results/DPSR.tif")
    print(f"    images/elevation.png")
    print(f"    images/PSR_mask.png")
    print(f"    images/DPSR_raw.png")
    print(f"    images/DPSR.png")
    print(f"    images/DPSR_summary.png")
    print(f"    images/dpsr_validation.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
