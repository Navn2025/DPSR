"""
step06_remove_small_regions.py  —  Remove small DPSR connected components.

Purpose
-------
Apply the post-processing filter described in O'Brien & Byrne (2022), Fig. 3
caption: remove connected components of DPSR pixels that are smaller than
5 pixels.  This eliminates single-pixel noise arising from DEM artefacts,
interpolation errors, or the discrete angular sampling of the ray-casting.

Scientific basis
----------------
O'Brien & Byrne (2022) state (Fig. 3 caption):
  "DPSR regions comprising fewer than five contiguous pixels are excluded."

Connectivity : 8-connected (diagonal neighbours count as connected)
Threshold    : 5 pixels minimum

At the LOLA 20 m/px resolution:
  1 pixel  =  400 m²  =  0.0004 km²
  5 pixels = 2000 m²  =  0.002  km²

This threshold is conservative — it retains scientifically meaningful DPSR
clusters while removing isolated pixels that are almost certainly noise.

Why 8-connectivity?
-------------------
8-connectivity (all 8 neighbours including diagonals) is the standard choice
for binary morphology on rectangular grids.  It correctly connects diagonal
chains of DPSR pixels that would be split by 4-connectivity, preventing
legitimate small DPSRs from being over-fragmented and under-counted.
The paper specifies this in Figure 3 and Table 1 (connected cluster sizes).

Algorithm
---------
1. Label connected components:  scipy.ndimage.label (8-connected structure)
2. Count component sizes:        np.bincount on flattened label array — O(H×W)
3. Keep only large components:   boolean mask → filtered uint8 output

This is equivalent to morphological opening with a 5-pixel structuring element
but is faster (O(H×W)) and correctly handles irregular shapes.

Inputs
------
dpsr_raw  : uint8 ndarray (H, W) — raw DPSR raster (from step05)
min_size  : int — minimum connected-component size (default 5, paper value)

Outputs
-------
dpsr_final  : uint8 ndarray (H, W) — filtered DPSR raster
n_removed   : int — number of pixels removed
n_comps_in  : int — number of connected components before filtering
n_comps_out : int — number of connected components after filtering

Also saves DPSR_FINAL_PATH (GeoTIFF co-registered with DEM).

Time complexity  : O(H × W)  —  two passes: label + bincount
Memory complexity: O(H × W)  —  labeled array (int32) ≈ 920 MB, then discarded

Optimisation strategy
---------------------
• np.bincount on labeled.ravel() is O(N) and allocates a single vector of
  length n_components + 1 — no per-component loop.
• Boolean index assignment (keep_mask[labeled]) is a vectorised look-up —
  no Python loop.
• labeled array is computed in-place by scipy.ndimage and immediately
  converted to the keep mask — the int32 array is freed after the mask is
  built.

Reference
---------
O'Brien & Byrne (2022), PSJ 3:258, Figure 3 caption and Section 2.3.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from scipy.ndimage import label as _scipy_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dpsr.utils import (
    MIN_COMPONENT, CONNECTIVITY, DPSR_RAW_PATH, DPSR_FINAL_PATH,
    DPSR_PNG, Timer, get_logger, save_preview,
)

log = get_logger(__name__)

# 8-connected structure: all 8 neighbours are considered adjacent
_STRUCT_8 = np.ones((3, 3), dtype=np.int32)


def remove_small_components(
    dpsr_raw:  np.ndarray,
    min_size:  int = MIN_COMPONENT,
) -> tuple[np.ndarray, int, int, int]:
    """
    Remove DPSR connected components smaller than min_size pixels.

    Parameters
    ----------
    dpsr_raw : uint8 ndarray (H, W) — raw DPSR raster  (1=DPSR  0=non-DPSR)
    min_size : minimum cluster size in pixels  (paper: 5)

    Returns
    -------
    dpsr_filtered : uint8 ndarray (H, W) — DPSR after small-component removal
    n_removed     : number of pixels removed
    n_comps_in    : number of components before filtering
    n_comps_out   : number of components after filtering

    Notes
    -----
    Background (value 0) is labelled 0 by scipy.ndimage.label and is excluded
    from size counting by setting sizes[0] = 0.

    Time  : O(H × W)
    Memory: O(H × W) for the int32 labeled array (freed before return)
    """
    if int(dpsr_raw.sum()) == 0:
        log.warning("Raw DPSR raster is empty — no pixels to filter.")
        return dpsr_raw.copy(), 0, 0, 0

    with Timer("Connected-component labelling"):
        labeled, n_comps_in = _scipy_label(dpsr_raw, structure=_STRUCT_8)

    log.info("Connected components before filter: %d", n_comps_in)

    # Component sizes via bincount (O(N), no Python loop)
    sizes       = np.bincount(labeled.ravel())   # sizes[label_id] = pixel count
    sizes[0]    = 0                               # exclude background (label=0)
    keep_mask   = sizes >= min_size               # True for components to keep

    # Map each pixel's label to keep/discard — fully vectorised
    dpsr_filtered = keep_mask[labeled].astype(np.uint8)
    del labeled   # free int32 array (~920 MB)

    n_comps_out = int(keep_mask[1:].sum())        # exclude background (index 0)
    n_removed   = int(dpsr_raw.sum()) - int(dpsr_filtered.sum())

    log.info(
        "Filter: kept %d / %d components  (>= %d px)  "
        "removed %d px  final DPSR = %d px",
        n_comps_out, n_comps_in, min_size,
        n_removed, int(dpsr_filtered.sum()),
    )
    return dpsr_filtered, n_removed, n_comps_in, n_comps_out


def run_filter(
    dpsr_raw_path:   Path = DPSR_RAW_PATH,
    dpsr_final_path: Path = DPSR_FINAL_PATH,
    meta:            Optional[dict] = None,
    min_size:        int  = MIN_COMPONENT,
    force:           bool = False,
) -> np.ndarray:
    """
    Load raw DPSR raster, apply small-region filter, save filtered result.

    Parameters
    ----------
    dpsr_raw_path   : path to raw DPSR GeoTIFF (from step05)
    dpsr_final_path : output path for filtered DPSR GeoTIFF
    meta            : rasterio metadata dict; if None, read from dpsr_raw_path
    min_size        : minimum component size (paper: 5)
    force           : recompute even if output already exists

    Returns
    -------
    dpsr_final : uint8 ndarray (H, W)
    """
    if dpsr_final_path.exists() and not force:
        log.info("Filtered DPSR already exists — loading %s", dpsr_final_path.name)
        with rasterio.open(dpsr_final_path) as ds:
            dpsr_cached = ds.read(1)
        save_preview(dpsr_cached, DPSR_PNG, cmap="hot",
                     label=f"DPSR final  ({int(dpsr_cached.sum()):,} px | "
                           f"{int(dpsr_cached.sum()) * 20.0**2 / 1e6:.4f} km²)",
                     vmin=0, vmax=1)
        return dpsr_cached

    if not dpsr_raw_path.exists():
        raise FileNotFoundError(
            f"Raw DPSR not found: {dpsr_raw_path}\n"
            "Run step05_compute_dpsr.py first."
        )

    with rasterio.open(dpsr_raw_path) as ds:
        dpsr_raw = ds.read(1)
        if meta is None:
            meta = ds.meta.copy()

    log.info(
        "Raw DPSR pixels: %s  (%.4f%% of raster)",
        f"{int(dpsr_raw.sum()):,}",
        100.0 * dpsr_raw.mean(),
    )

    dpsr_final, n_removed, n_in, n_out = remove_small_components(
        dpsr_raw, min_size=min_size,
    )

    # Print component-size distribution
    _print_size_stats(dpsr_raw, min_size)

    # Save
    dpsr_final_path.parent.mkdir(parents=True, exist_ok=True)
    write_meta = meta.copy()
    write_meta.update(dtype="uint8", count=1, compress="lzw",
                      driver="GTiff", nodata=None)
    with rasterio.open(dpsr_final_path, "w", **write_meta) as dst:
        dst.write(dpsr_final, 1)
    log.info("Saved  %s  (%.1f MB)", dpsr_final_path.name,
             dpsr_final_path.stat().st_size / 1e6)

    n_final = int(dpsr_final.sum())
    save_preview(dpsr_final, DPSR_PNG, cmap="hot",
                 label=f"DPSR final  ({n_final:,} px | {n_final * 20.0**2 / 1e6:.4f} km²)",
                 vmin=0, vmax=1)

    n_psr   = dpsr_raw.size     # approximate; use PSR mask if available
    print(f"\n  Post-processing summary")
    print(f"  ========================")
    print(f"  Components before filter : {n_in:,}")
    print(f"  Components after  filter : {n_out:,}  (>= {min_size} px)")
    print(f"  Pixels removed           : {n_removed:,}")
    print(f"  Final DPSR pixels        : {n_final:,}")
    print(f"  Final DPSR area          : {n_final * 20.0**2 / 1e6:.3f} km²")
    print(f"  Connectivity             : 8-connected (paper specification)")
    print(f"  Min cluster size         : {min_size} px = {min_size * 20.0**2:.0f} m²")

    return dpsr_final


def _print_size_stats(dpsr_raw: np.ndarray, min_size: int) -> None:
    """Print component-size histogram for diagnostics."""
    labeled, n_comps = _scipy_label(dpsr_raw, structure=_STRUCT_8)
    if n_comps == 0:
        return
    sizes   = np.bincount(labeled.ravel())[1:]   # skip background (label=0)
    bins    = [1, 2, 3, 4, 5, 10, 25, 50, 100, 500, 1000, np.inf]
    print("\n  Component size distribution (before filter):")
    print(f"  {'Size range':>16}   {'Count':>8}   {'Pixels':>10}")
    print(f"  {'-'*40}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        hi_str  = str(int(hi)) if hi != np.inf else "∞"
        count   = int(((sizes >= lo) & (sizes < hi)).sum())
        px      = int(sizes[(sizes >= lo) & (sizes < hi)].sum())
        flag    = "  ← removed" if hi <= min_size else ""
        if count > 0:
            print(f"  [{lo:>6} – {hi_str:>6}]   {count:>8,}   {px:>10,}{flag}")


# ---------------------------------------------------------------------------
# CLI  —  python -m dpsr.step06_remove_small_regions [--min-size N] [--redo]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove small DPSR connected components"
    )
    parser.add_argument("--min-size", type=int, default=MIN_COMPONENT,
                        help=f"Min pixels per component (paper: {MIN_COMPONENT})")
    parser.add_argument("--redo", action="store_true",
                        help="Recompute even if output exists")
    args = parser.parse_args()

    dpsr = run_filter(min_size=args.min_size, force=args.redo)
    print(f"\n  Final DPSR pixels : {int(dpsr.sum()):,}")
    print(f"  Final DPSR area   : {int(dpsr.sum()) * 20.0**2 / 1e6:.4f} km²")
    print(f"  Output            : {DPSR_FINAL_PATH}")
