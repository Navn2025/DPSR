"""
feature_stack.py
================
Assemble, save, and analyse the multi-band feature cube.

Memory strategy
  The cube (15168 x 15168 x 11 @ float32 = 9.4 GB) is too large to hold
  in RAM.  It is stored on disk as a memory-mapped file (FeatureStack.dat).
  Numpy accesses it one band-slice at a time (~900 MB), so peak RAM usage
  stays manageable.

Output files
  outputs/FeatureStack.dat  — raw float32 binary (H x W x C), C-order
  outputs/FeatureStack.tif  — multi-band GeoTIFF with band descriptions

Loading the .dat file later
  import numpy as np
  cube = np.memmap("FeatureStack.dat", dtype="float32",
                   mode="r", shape=(H, W, C))
  dem_band = cube[:, :, 0]   # shape (H, W)

Extension points
  The feature cube is the primary input for:
    - Ice Probability Mapping  (ML classifiers)
    - Landing Site Selection   (multi-criteria scoring)
    - Rover Path Planning      (cost-surface generation)
    - Ice Volume Estimation    (3-D integration)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from config import BAND_NAMES, OUTPUTS_DIR
from utils import get_logger, section

log = get_logger("feature_stack")

STACK_DAT  = OUTPUTS_DIR / "FeatureStack.dat"
STACK_META = OUTPUTS_DIR / "FeatureStack_meta.json"
STACK_TIF  = OUTPUTS_DIR / "FeatureStack.tif"


# -- Initialise memmap cube on disk -------------------------------------------

def create_cube_memmap(ref_profile: dict) -> np.memmap:
    """
    Create (or overwrite) the on-disk feature cube as a memory-mapped array.

    The file is initialised to zeros.  Missing bands are filled with NaN
    band-by-band in write_band_to_cube().

    Returns the writable np.memmap object.
    """
    H = ref_profile["height"]
    W = ref_profile["width"]
    C = len(BAND_NAMES)

    STACK_DAT.parent.mkdir(parents=True, exist_ok=True)
    cube = np.memmap(STACK_DAT, dtype="float32", mode="w+", shape=(H, W, C))
    log.info(
        f"Memmap cube created: {H} x {W} x {C} = "
        f"{H * W * C * 4 / 1_073_741_824:.2f} GB  ->  {STACK_DAT}"
    )
    return cube


def write_band_to_cube(
    cube: np.memmap,
    band_name: str,
    array: Optional[np.ndarray],
    ref_profile: dict,
) -> None:
    """
    Write one normalised band into the memmap cube, then flush to disk.
    Call this inside the main processing loop right after normalisation;
    the caller can then `del array` to free RAM.
    """
    H = ref_profile["height"]
    W = ref_profile["width"]

    if band_name not in BAND_NAMES:
        return                          # extra DFSAR product, not in the stack

    i = BAND_NAMES.index(band_name)

    if array is None:
        cube[:, :, i] = np.nan
        log.warning(f"  Band {i+1:02d} [{band_name}] -- NOT FOUND, NaN filled.")
        return

    if array.shape != (H, W):
        cube[:, :, i] = np.nan
        log.warning(
            f"  Band {i+1:02d} [{band_name}] -- shape {array.shape} != "
            f"({H},{W}). NaN filled."
        )
        return

    cube[:, :, i] = array
    cube.flush()

    finite = np.isfinite(array)
    log.info(
        f"  Band {i+1:02d} [{band_name}] -- "
        f"valid: {np.sum(finite):,}  NaN: {np.sum(~finite):,}"
    )


# -- Legacy batch-assembly API (memory-safe via memmap) -----------------------

def assemble_stack(
    bands:       dict[str, Optional[np.ndarray]],
    ref_profile: dict,
) -> np.memmap:
    """
    Build the feature cube from a pre-populated *bands* dict.

    Uses np.memmap so no 9+ GB RAM block is allocated.
    Process each band sequentially; caller should `del` arrays after calling.
    """
    cube = create_cube_memmap(ref_profile)
    H = ref_profile["height"]
    W = ref_profile["width"]

    for band_name in BAND_NAMES:
        write_band_to_cube(cube, band_name, bands.get(band_name), ref_profile)

    return cube


# -- Save metadata JSON --------------------------------------------------------

def save_stack_meta(cube: np.memmap) -> None:
    """
    Save a JSON sidecar describing the .dat memmap so users can reload it.
    """
    H, W, C = cube.shape
    meta = {
        "shape":    [H, W, C],
        "dtype":    "float32",
        "order":    "C",
        "dat_file": str(STACK_DAT),
        "bands":    {str(i + 1): name for i, name in enumerate(BAND_NAMES)},
        "load_example": (
            "import numpy as np\n"
            f"cube = np.memmap(r'{STACK_DAT}', dtype='float32', "
            f"mode='r', shape=({H}, {W}, {C}))\n"
            "dem = cube[:, :, 0]  # Band 1 = DEM"
        ),
    }
    STACK_META.parent.mkdir(parents=True, exist_ok=True)
    with open(STACK_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info(f"Metadata saved -> {STACK_META.name}")


# -- Save GeoTIFF (band by band from memmap) ----------------------------------

def save_stack_tif(
    cube:        np.memmap,
    ref_profile: dict,
    path:        Path = STACK_TIF,
) -> None:
    """
    Write the feature cube as a multi-band GeoTIFF with band descriptions.
    Reads one slice (~900 MB) at a time from the memmap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W, C = cube.shape

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=C,
        nodata=float("nan"),
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    log.info(f"Writing FeatureStack.tif ({C} bands)...")
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(C):
            band_slice = np.array(cube[:, :, i])   # loads one slice into RAM
            dst.write(band_slice, i + 1)
            dst.set_band_description(i + 1, BAND_NAMES[i])
            del band_slice

    log.info(f"Saved FeatureStack.tif -> {path}")


# -- Per-band statistics (one slice at a time) --------------------------------

def compute_statistics(cube: np.memmap) -> list[dict]:
    """
    Compute per-band statistics without loading the whole cube into RAM.
    Reads one band slice at a time from the memmap.
    """
    H, W, _ = cube.shape
    total = H * W
    stats: list[dict] = []

    for i, name in enumerate(BAND_NAMES):
        layer  = np.array(cube[:, :, i])        # read one band from disk
        finite = layer[np.isfinite(layer)]
        miss   = 100.0 * (total - finite.size) / max(total, 1)
        del layer

        if finite.size == 0:
            row = dict(band=i+1, name=name,
                       min=np.nan, max=np.nan, mean=np.nan,
                       median=np.nan, std=np.nan, missing_pct=100.0)
        else:
            row = dict(
                band=i+1, name=name,
                min=float(np.min(finite)),
                max=float(np.max(finite)),
                mean=float(np.mean(finite)),
                median=float(np.median(finite)),
                std=float(np.std(finite)),
                missing_pct=miss,
            )
        stats.append(row)
        del finite

    return stats


def print_statistics(stats: list[dict]) -> None:
    section("STEP 11 -- PER-BAND STATISTICS (normalised cube)")
    hdr = (
        f"  {'Band':>4}  {'Name':<12}  "
        f"{'Min':>10}  {'Max':>10}  {'Mean':>10}  "
        f"{'Median':>10}  {'Std':>10}  {'Missing%':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in stats:
        def f(v): return f"{v:10.4f}" if np.isfinite(v) else f"{'N/A':>10}"
        print(
            f"  {r['band']:>4}  {r['name']:<12}  "
            f"{f(r['min'])}  {f(r['max'])}  {f(r['mean'])}  "
            f"{f(r['median'])}  {f(r['std'])}  "
            f"{r['missing_pct']:>8.2f}%"
        )
