"""
main.py
=======
Chandrayaan-2 DFSAR  --  Lunar South Pole Ice Detection Pipeline
================================================================
Memory-efficient streaming pipeline: each layer is normalised,
written to the on-disk memmap feature cube, and freed before the
next layer is processed.  Peak RAM per band: ~1.8 GB.

Steps
  01  Dataset discovery
  02  Metadata report
  03  Validation
  04  Reference grid (DEM CRS + transform)
  05+06  Reproject + resample every layer to the DEM grid (saved to disk)
  07  Terrain derivatives (Slope, Hillshade) from DEM
  08  Streaming: normalise -> write to memmap cube -> save PNG -> free
  09  Feature stack metadata + GeoTIFF save
  10  Per-band statistics
  11  Final summary + log

Usage
  cd DFSAR/data_pipeline
  set PYTHONUTF8=1   (Windows -- prevents Unicode console errors)
  python main.py

Prerequisites
  Run setup_dem.py once to convert LOLA PDS3 DEM to GeoTIFF.
  PSR and DPSR are auto-resolved from ISRO_Hackathon/data/ and results/.
  DFSAR products are auto-scanned from the DFSAR/ parent folder.
"""
from __future__ import annotations

import sys
import time
import gc
from pathlib import Path
from typing import Optional

import numpy as np

# -- GPU acceleration (optional) -----------------------------------------------
try:
    from gpu_accel import backend_info as _gpu_backend_info, gpu_normalize as _gpu_normalize
    _GPU_NORMALIZE = True
except Exception:
    _GPU_NORMALIZE = False

# -- Project modules -----------------------------------------------------------
from config import (
    OUTPUTS_DIR, ALIGNED_DIR, PREVIEWS_DIR, LOG_FILE,
    BAND_NAMES, MASK_LAYER_NAMES,
    RESAMPLE_CONTINUOUS, RESAMPLE_MASK,
)
from utils import setup_logging, Timer, memory_str, section, subsection, progress
from loader import DatasetCatalog, RasterInfo, VectorInfo
from validator import validate_catalog, print_validation_summary
from reproject import get_reference_profile, reproject_raster, read_aligned, save_aligned
from preprocessor import derive_terrain, rasterize_vector
from normalizer import normalize_layer
from feature_stack import (
    create_cube_memmap, write_band_to_cube,
    save_stack_meta, save_stack_tif,
    compute_statistics, print_statistics,
)
from visualizer import save_band_preview, save_composite_overview


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline() -> None:

    t_wall = time.perf_counter()

    # -- 0. Bootstrap ----------------------------------------------------------
    for d in (OUTPUTS_DIR, ALIGNED_DIR, PREVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logging(LOG_FILE)
    log.info("=" * 68)
    log.info("  CH-2 DFSAR Lunar South Pole Ice Detection Pipeline -- START")
    log.info("=" * 68)
    if _GPU_NORMALIZE:
        log.info(f"  Compute backend: {_gpu_backend_info()}")
    else:
        log.info("  Compute backend: Sequential CPU (install cupy or torch for GPU)")

    counters = dict(discovered=0, processed=0, skipped=0, corrupted=0)

    # -- 1. Discover -----------------------------------------------------------
    catalog = DatasetCatalog()
    with Timer("Discovery", log):
        catalog.discover()
    catalog.print_catalog()

    counters["discovered"] = len(catalog.all_rasters) + len(catalog.all_vectors)

    if catalog.dem is None:
        log.error(
            "No DEM found -- cannot determine the reference grid.\n"
            "  Run:  python setup_dem.py\n"
            "  This converts the LOLA PDS3 DEM to GeoTIFF (one-time step)."
        )
        sys.exit(1)

    # -- 3. Validate -----------------------------------------------------------
    section("STEP 3 -- VALIDATION")
    with Timer("Validation", log):
        val = validate_catalog(catalog.all_rasters, catalog.all_vectors)
    print_validation_summary(val)

    # -- 4. Reference grid -----------------------------------------------------
    section("STEP 4 -- REFERENCE GRID (from DEM)")
    with Timer("Reference profile", log):
        ref = get_reference_profile(catalog.dem.path)
    H, W = ref["height"], ref["width"]

    # =========================================================================
    # STEP 5+6: Align (reproject + resample) all layers to disk.
    #
    # We do NOT keep the reprojected arrays in memory.
    # Each layer is saved to outputs/aligned/<Label>.tif and then freed.
    # =========================================================================
    section("STEP 5+6 -- REPROJECT & RESAMPLE (saving to disk)")

    # Tracks which aligned TIF files exist for the streaming step below.
    aligned_paths: dict[str, Optional[Path]] = {}

    def _align(info: RasterInfo, label: str, rs) -> Optional[Path]:
        """Reproject *info* to the DEM grid; return path of the saved TIF."""
        if not info.valid:
            log.warning(f"  Skipping invalid layer: {label}")
            counters["corrupted"] += 1
            return None
        dst = ALIGNED_DIR / f"{label}.tif"
        if dst.exists():
            log.info(f"  Already aligned (cached): {dst.name}")
            counters["processed"] += 1
            return dst
        arr = reproject_raster(info.path, dst, ref, rs)
        if arr is not None:
            counters["processed"] += 1
            del arr           # free immediately -- we only needed the save
            gc.collect()
            return dst
        counters["skipped"] += 1
        return None

    # DEM
    subsection("DEM")
    with Timer("DEM align", log):
        aligned_paths["DEM"] = _align(catalog.dem, "DEM", RESAMPLE_CONTINUOUS)

    # PSR
    subsection("PSR")
    if catalog.psr is None:
        log.warning("PSR not available.")
        aligned_paths["PSR"] = None
    elif isinstance(catalog.psr, VectorInfo):
        dst = ALIGNED_DIR / "PSR.tif"
        if dst.exists():
            log.info("  PSR already rasterized (cached).")
            aligned_paths["PSR"] = dst
            counters["processed"] += 1
        else:
            arr = rasterize_vector(catalog.psr.path, ref, dst)
            if arr is not None:
                counters["processed"] += 1
                del arr
                gc.collect()
            aligned_paths["PSR"] = dst if dst.exists() else None
    else:
        aligned_paths["PSR"] = _align(catalog.psr, "PSR", RESAMPLE_MASK)

    # DPSR
    subsection("DPSR")
    if catalog.dpsr is None:
        log.warning("DPSR not available.")
        aligned_paths["DPSR"] = None
    elif isinstance(catalog.dpsr, VectorInfo):
        dst = ALIGNED_DIR / "DPSR.tif"
        if dst.exists():
            log.info("  DPSR already rasterized (cached).")
            aligned_paths["DPSR"] = dst
            counters["processed"] += 1
        else:
            arr = rasterize_vector(catalog.dpsr.path, ref, dst)
            if arr is not None:
                counters["processed"] += 1
                del arr
                gc.collect()
            aligned_paths["DPSR"] = dst if dst.exists() else None
    else:
        aligned_paths["DPSR"] = _align(catalog.dpsr, "DPSR", RESAMPLE_MASK)

    # DFSAR products
    subsection("DFSAR Products")
    dfsar_labels = list(catalog.dfsar.keys())
    for idx, label in enumerate(dfsar_labels, 1):
        progress(idx, len(dfsar_labels), label)
        info = catalog.dfsar[label]
        rs   = RESAMPLE_MASK if label in MASK_LAYER_NAMES else RESAMPLE_CONTINUOUS
        aligned_paths[label] = _align(info, label, rs)

    # =========================================================================
    # STEP 7: Terrain derivatives -- Slope + Hillshade
    # Computed in-memory from the aligned DEM, then saved and freed.
    # =========================================================================
    section("STEP 7 -- TERRAIN DERIVATIVES (Slope + Hillshade)")

    dem_aligned_path = aligned_paths.get("DEM")
    slope_path       = ALIGNED_DIR / "Slope.tif"
    hillshade_path   = ALIGNED_DIR / "Hillshade.tif"

    if dem_aligned_path and dem_aligned_path.exists():
        if slope_path.exists() and hillshade_path.exists():
            log.info("  Slope + Hillshade already computed (cached).")
            aligned_paths["Slope"]     = slope_path
            aligned_paths["Hillshade"] = hillshade_path
        else:
            dem_arr = read_aligned(dem_aligned_path)
            if dem_arr is not None:
                with Timer("Slope + Hillshade", log):
                    slope, hillshade = derive_terrain(dem_arr, ref, ALIGNED_DIR)
                del dem_arr
                gc.collect()
                aligned_paths["Slope"]     = slope_path
                aligned_paths["Hillshade"] = hillshade_path
                del slope, hillshade
                gc.collect()
            else:
                log.error("Could not read aligned DEM for terrain derivation.")
                aligned_paths["Slope"]     = None
                aligned_paths["Hillshade"] = None
    else:
        log.error("Aligned DEM not found -- cannot compute Slope/Hillshade.")
        aligned_paths["Slope"]     = None
        aligned_paths["Hillshade"] = None

    # =========================================================================
    # STEP 8+9: Streaming normalise -> memmap cube -> PNG preview
    #
    # For each layer:
    #   1. Read the aligned TIF from disk  (~900 MB into RAM)
    #   2. Normalise                        (~900 MB)
    #   3. Write normalised band to memmap  (disk write, RAM freed after)
    #   4. Save PNG preview
    #   5. del arrays + gc.collect()       (free 900 MB before next layer)
    #
    # Peak RAM per iteration: ~1.8 GB (raw + normed).
    # The cube itself lives on disk as FeatureStack.dat.
    # =========================================================================
    section("STEP 8+9 -- NORMALISE + FEATURE STACK (streaming, one band at a time)")

    cube = create_cube_memmap(ref)

    # All labels we want to process for previews (primary + extra DFSAR)
    all_labels = list(aligned_paths.keys())

    # Pre-build a small normalization report table
    norm_report_rows: list[tuple] = []

    total_bands = len(all_labels)
    for idx, label in enumerate(all_labels, 1):
        progress(idx, total_bands, label)

        tif_path = aligned_paths.get(label)
        is_mask  = label in MASK_LAYER_NAMES

        # -- Read from disk
        if tif_path and tif_path.exists():
            raw = read_aligned(tif_path)
        else:
            raw = None

        # -- Collect raw stats for report
        if raw is not None:
            valid_raw = raw[np.isfinite(raw)]
            r_min = float(valid_raw.min()) if valid_raw.size else float("nan")
            r_max = float(valid_raw.max()) if valid_raw.size else float("nan")
        else:
            r_min = r_max = float("nan")

        # -- Normalise (GPU if available)
        if raw is not None:
            if _GPU_NORMALIZE:
                try:
                    normed = _gpu_normalize(raw, is_mask=is_mask)
                except Exception as gpu_exc:
                    log.warning(f"GPU normalize failed [{label}]: {gpu_exc} -- CPU fallback")
                    normed = normalize_layer(raw, label, is_mask=is_mask)
            else:
                normed = normalize_layer(raw, label, is_mask=is_mask)
            del raw          # free ~900 MB
            gc.collect()
        else:
            normed = None

        # -- Collect normalised stats
        if normed is not None:
            valid_n = normed[np.isfinite(normed)]
            n_min = float(valid_n.min()) if valid_n.size else float("nan")
            n_max = float(valid_n.max()) if valid_n.size else float("nan")
        else:
            n_min = n_max = float("nan")

        norm_report_rows.append((label, r_min, r_max, n_min, n_max))

        # -- Write to memmap cube (only primary BAND_NAMES bands)
        write_band_to_cube(cube, label, normed, ref)

        # -- Save PNG preview
        try:
            save_band_preview(normed, label, PREVIEWS_DIR)
        except Exception as exc:
            log.error(f"Preview failed [{label}]: {exc}")

        del normed       # free ~900 MB
        gc.collect()

    # -- Normalization report table
    section("STEP 8 -- NORMALISATION REPORT")
    hdr = f"  {'Layer':<18}  {'Raw Min':>12}  {'Raw Max':>12}  {'Norm Min':>10}  {'Norm Max':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in norm_report_rows:
        label, rm, rM, nm, nM = row
        def _f(v): return f"{v:12.4g}" if np.isfinite(v) else f"{'N/A':>12}"
        print(f"  {label:<18}  {_f(rm)}  {_f(rM)}  {_f(nm):>10}  {_f(nM):>10}")

    # -- Composite overview PNG (reads from PREVIEWS_DIR files already saved)
    try:
        save_composite_overview(
            {lbl: None for lbl in all_labels},   # placeholder -- overview reads files
            PREVIEWS_DIR,
        )
    except Exception as exc:
        log.warning(f"Composite overview skipped: {exc}")

    # =========================================================================
    # STEP 10: Save FeatureStack.tif (reads from memmap, one band at a time)
    # =========================================================================
    section("STEP 10 -- SAVE FEATURE STACK")
    save_stack_meta(cube)
    with Timer("Save FeatureStack.tif", log):
        save_stack_tif(cube, ref)

    # =========================================================================
    # STEP 11: Statistics (reads from memmap, one band at a time)
    # =========================================================================
    section("STEP 11 -- PER-BAND STATISTICS")
    with Timer("Statistics", log):
        stats = compute_statistics(cube)
    print_statistics(stats)

    # =========================================================================
    # STEP 13: Final summary
    # =========================================================================
    elapsed = time.perf_counter() - t_wall
    H_c, W_c, C_c = cube.shape

    section("STEP 13 -- FINAL SUMMARY")
    print(f"  Datasets discovered    : {counters['discovered']}")
    print(f"  Successfully aligned   : {counters['processed']}")
    print(f"  Skipped / failed       : {counters['skipped']}")
    print(f"  Corrupted              : {counters['corrupted']}")
    print(f"  Feature stack shape    : {H_c} x {W_c} x {C_c}")
    print(f"  Stack on disk          : {H_c * W_c * C_c * 4 / 1_073_741_824:.2f} GB")
    print(f"  Current process memory : {memory_str()}")
    print(f"  Total execution time   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Outputs                : {OUTPUTS_DIR}")
    print(f"  Log                    : {LOG_FILE}")

    log.info(f"Pipeline complete in {elapsed:.1f}s")


# =============================================================================
# Extension stubs -- implement in future sprints
# =============================================================================

def ice_probability_map(
    cube:      "np.memmap",
    cpr_band:  int = 5,    # 0-indexed band index for CPR
    psr_band:  int = 3,    # PSR
    dpsr_band: int = 4,    # DPSR
) -> "np.ndarray":
    """
    Per-pixel ice probability from the feature cube.

    Simple heuristic:
      P = CPR_norm x (1 - VOL_norm) x PSR_mask x DPSR_mask

    For supervised ML:
      features = cube.reshape(-1, C)
      proba    = trained_model.predict_proba(features)[:, 1].reshape(H, W)
    """
    raise NotImplementedError("Ice probability mapping -- TODO Sprint 2")


def select_landing_sites(
    ice_prob:      "np.ndarray",
    slope:         "np.ndarray",
    *,
    min_ice_prob:  float = 0.6,
    max_slope_deg: float = 15.0,
) -> "np.ndarray":
    """Boolean mask: ice_prob >= threshold AND slope <= max."""
    raise NotImplementedError("Landing site selection -- TODO Sprint 3")


def plan_rover_path(
    cost_surface: "np.ndarray",
    start:        "tuple[int, int]",
    goal:         "tuple[int, int]",
) -> "list[tuple[int, int]]":
    """A* over a slope-derived cost surface."""
    raise NotImplementedError("Rover path planning -- TODO Sprint 4")


def estimate_ice_volume(
    ice_prob:      "np.ndarray",
    pixel_area_m2: float,
    depth_m:       float = 1.0,
    threshold:     float = 0.5,
) -> float:
    """V = sum(area x depth) for pixels where P >= threshold."""
    raise NotImplementedError("Ice volume estimation -- TODO Sprint 5")


# =============================================================================

if __name__ == "__main__":
    run_pipeline()
