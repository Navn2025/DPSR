"""
main.py  --  Validation pipeline orchestrator.

Run from the project root:
    python validation/main.py

Steps
-----
1.  Inspect all metadata and geometry files.
2.  Parse SLI geometry CSV → build tie-point grids.
3.  Create GCPs in Moon geographic CRS.
4.  Reproject Calculated_CPR.tif → Calculated_CPR_Georeferenced.tif
    (Moon South Pole Stereographic, 20m, matches official CPR grid exactly).
5.  Load both georeferenced CPR arrays.
6.  Extract overlap pixels (valid in both products).
7.  Compute all metrics (Pearson r, Spearman r, RMSE, MAE, Bias, R², SSIM,
    Histogram Intersection, Mutual Information).
8.  Generate all figures.
9.  Print per-raster statistics.
10. Write validation_report.txt.
"""

import sys
from pathlib import Path

# Make sibling modules importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

import config as cfg
from utils import setup_logger, Timer, array_stats, log_stats, section, memory_mb
from reader import (
    read_tiff_metadata, read_tiff_data,
    discover_geometry_files, read_geometry_csv,
)
from georeference import build_tie_grids, create_gcps
from warp import warp_with_gcps, warp_scipy_inverse
from metrics import extract_overlap, compute_all_metrics
from plots import generate_all_plots
from report import generate_report


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
for d in (cfg.FIG_DIR, cfg.LOG_DIR, cfg.OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = setup_logger(cfg.LOG_DIR, name="validation")
T_total = Timer()


# ===========================================================================
# STEP 1  --  Inspect metadata and geometry files
# ===========================================================================
section("STEP 1  |  Metadata Inspection", log)

log.info("Calculated CPR:")
calc_meta = read_tiff_metadata(cfg.CALC_CPR_PATH)
log.info(f"  Path      : {cfg.CALC_CPR_PATH.name}")
log.info(f"  Shape     : {calc_meta['height']} H x {calc_meta['width']} W")
log.info(f"  CRS       : {calc_meta['crs']}")
log.info(f"  Transform : {calc_meta['transform']}")
log.info(f"  Nodata    : {calc_meta['nodata']}")

log.info("\nOfficial aligned CPR:")
offic_meta = read_tiff_metadata(cfg.OFFICIAL_CPR_PATH)
log.info(f"  Path      : {cfg.OFFICIAL_CPR_PATH.name}")
log.info(f"  Shape     : {offic_meta['height']} H x {offic_meta['width']} W")
log.info(f"  CRS       : {str(offic_meta['crs'])[:80]}")
log.info(f"  Transform : {offic_meta['transform']}")
log.info(f"  Bounds    : {offic_meta['bounds']}")
log.info(f"  Nodata    : {offic_meta['nodata']}")

log.info("\nGeometry file discovery:")
found_geom = discover_geometry_files(cfg.GEOM_DIR)

# Use SLI geometry CSV (primary source for the SLI-derived CPR)
if "sli_csv" not in found_geom:
    log.error("SLI geometry CSV not found. Cannot georeference.")
    sys.exit(1)

geom_csv_path = found_geom["sli_csv"]
log.info(f"  Using SLI geometry CSV: {geom_csv_path.name}")


# ===========================================================================
# STEP 2  --  Parse geometry CSV
# ===========================================================================
section("STEP 2  |  Parse Geometry CSV", log)

t0 = Timer()
df, N_AZ, N_RNG = read_geometry_csv(geom_csv_path)
log.info(f"  Parsed in {t0}")

az_px, rng_px, lat_grid, lon_grid = build_tie_grids(
    df,
    N_AZ         = N_AZ,
    N_RNG        = N_RNG,
    scene_height = cfg.SLI_HEIGHT,
    scene_width  = cfg.SLI_WIDTH,
)


# ===========================================================================
# STEP 3  --  Create GCPs
# ===========================================================================
section("STEP 3  |  Create Ground Control Points", log)

gcps = create_gcps(
    az_px, rng_px, lat_grid, lon_grid,
    az_stride=cfg.GCP_AZ_STRIDE,
)

src_crs = CRS.from_wkt(cfg.MOON_GEO_WKT)
dst_crs = CRS.from_wkt(cfg.MOON_STEREO_WKT)

log.info(f"  src CRS: {src_crs.to_string()[:80]}")
log.info(f"  dst CRS: {dst_crs.to_string()[:80]}")

# Target transform matches the official CPR grid exactly
dst_transform = from_bounds(
    west  = cfg.TARGET_BOUNDS[0],
    south = cfg.TARGET_BOUNDS[1],
    east  = cfg.TARGET_BOUNDS[2],
    north = cfg.TARGET_BOUNDS[3],
    width  = cfg.TARGET_WIDTH,
    height = cfg.TARGET_HEIGHT,
)
log.info(f"  Target transform: {dst_transform}")


# ===========================================================================
# STEP 4  --  Reproject CPR to Moon South Pole Stereographic
# ===========================================================================
section("STEP 4  |  Warp CPR -> Moon Stereographic", log)

georef_path = cfg.OUT_DIR / cfg.GEOREF_OUTPUT_NAME

if georef_path.exists():
    log.info(f"  Georeferenced CPR already exists: {georef_path.name}")
    log.info("  Loading from disk (delete to re-run reprojection).")
    georef_arr, _ = read_tiff_data(georef_path, nodata_override=np.nan)
else:
    t0 = Timer()
    georef_arr = warp_with_gcps(
        src_path     = cfg.CALC_CPR_PATH,
        gcps         = gcps,
        src_crs      = src_crs,
        dst_crs      = dst_crs,
        dst_transform= dst_transform,
        dst_shape    = (cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH),
        output_path  = georef_path,
        src_nodata   = cfg.NODATA_CALC,
        dst_nodata   = np.nan,
    )

    if georef_arr is None:
        log.warning("  GCP warp returned empty result. Switching to scipy fallback.")
        t0 = Timer()
        georef_arr = warp_scipy_inverse(
            src_path     = cfg.CALC_CPR_PATH,
            az_px        = az_px,
            rng_px       = rng_px,
            lat_grid     = lat_grid,
            lon_grid     = lon_grid,
            scene_height = cfg.SLI_HEIGHT,
            scene_width  = cfg.SLI_WIDTH,
            dst_crs      = dst_crs,
            dst_transform= dst_transform,
            dst_shape    = (cfg.TARGET_HEIGHT, cfg.TARGET_WIDTH),
            output_path  = georef_path,
        )
        log.info(f"  Scipy fallback done in {t0}")
    else:
        log.info(f"  GCP warp done in {t0}")

log.info(
    f"  Georeferenced CPR: shape={georef_arr.shape}  "
    f"valid={np.sum(np.isfinite(georef_arr)):,}"
)


# ===========================================================================
# STEP 5  --  Load official CPR
# ===========================================================================
section("STEP 5  |  Load Official CPR", log)

t0 = Timer()
offic_arr, offic_profile = read_tiff_data(
    cfg.OFFICIAL_CPR_PATH,
    nodata_override=None,
)
log.info(f"  Loaded in {t0}")


# ===========================================================================
# STEP 6  --  Extract overlap
# ===========================================================================
section("STEP 6  |  Extract Overlap", log)

calc_1d, offic_1d, overlap_mask = extract_overlap(
    calc    = georef_arr,
    offic   = offic_arr,
    nodata_c = np.nan,
    nodata_o = offic_profile.get("nodata") or np.nan,
)

n_overlap = int(overlap_mask.sum())
n_total   = cfg.TARGET_HEIGHT * cfg.TARGET_WIDTH
overlap_pct = 100 * n_overlap / n_total

log.info(f"  Overlap pixels  : {n_overlap:,}")
log.info(f"  Target total    : {n_total:,}")
log.info(f"  Overlap fraction: {overlap_pct:.3f}%")

if n_overlap < 1000:
    log.error("  Fewer than 1000 overlap pixels. Check georeferencing.")
    log.error("  Possible issue: scene is outside the official CPR extent.")
    sys.exit(1)


# ===========================================================================
# STEP 7  --  Statistics for both rasters
# ===========================================================================
section("STEP 7  |  Per-Raster Statistics", log)

log.info("Georeferenced Calculated CPR:")
stats_calc = array_stats(georef_arr, nodata=np.nan, label="Calc CPR (georef)")
log_stats(stats_calc, log)

log.info("Official CPR:")
stats_offic = array_stats(offic_arr, nodata=offic_profile.get("nodata"),
                           label="Official CPR")
log_stats(stats_offic, log)

# Overlap-only statistics
log.info("Overlap pixels — Calculated:")
stats_calc_ol  = array_stats(calc_1d.reshape(-1,1),  label="Calc (overlap)")
log_stats(stats_calc_ol, log)
log.info("Overlap pixels — Official:")
stats_offic_ol = array_stats(offic_1d.reshape(-1,1), label="Official (overlap)")
log_stats(stats_offic_ol, log)


# ===========================================================================
# STEP 8  --  Compute all metrics
# ===========================================================================
section("STEP 8  |  Quantitative Metrics", log)

# Pack overlap pixels back into the metrics dict
import metrics as _m
metrics_res = _m.compute_all_metrics(georef_arr, offic_arr, overlap_mask)


# ===========================================================================
# STEP 9  --  Generate figures
# ===========================================================================
section("STEP 9  |  Generate Figures", log)

generate_all_plots(
    calc_2d  = georef_arr,
    offic_2d = offic_arr,
    metrics  = metrics_res,
    out_dir  = cfg.FIG_DIR,
)


# ===========================================================================
# STEP 10  --  Write validation report
# ===========================================================================
section("STEP 10  |  Write Validation Report", log)

report_path = cfg.OUT_DIR / cfg.REPORT_NAME
generate_report(
    metrics      = metrics_res,
    stats_calc   = stats_calc,
    stats_offic  = stats_offic,
    overlap_info = {
        "total_pixels":  n_total,
        "overlap_pixels": n_overlap,
        "overlap_pct":    overlap_pct,
        "target_shape":  f"{cfg.TARGET_HEIGHT} x {cfg.TARGET_WIDTH}",
    },
    out_path    = report_path,
    calc_path   = cfg.CALC_CPR_PATH,
    offic_path  = cfg.OFFICIAL_CPR_PATH,
    georef_path = georef_path,
)


# ===========================================================================
# Final summary
# ===========================================================================
section("FINAL SUMMARY", log)

r   = metrics_res.get("pearson_r",  float("nan"))
rm  = metrics_res.get("rmse",       float("nan"))
bi  = metrics_res.get("bias",       float("nan"))
r2  = metrics_res.get("r2",         float("nan"))
ss  = metrics_res.get("ssim",       float("nan"))
hi  = metrics_res.get("hist_intersection", float("nan"))

log.info(f"""
  ---------------------------------------------------------------
  Chandrayaan-2 DFSAR CPR Validation Summary
  ---------------------------------------------------------------
  Overlap pixels          : {n_overlap:,}  ({overlap_pct:.2f}% of target)

  Pearson r               : {r:.4f}
  Spearman r              : {metrics_res.get('spearman_r', float('nan')):.4f}
  RMSE                    : {rm:.4f}
  MAE                     : {metrics_res.get('mae', float('nan')):.4f}
  Bias (calc - official)  : {bi:+.4f}
  R²                      : {r2:.4f}
  SSIM                    : {ss:.4f}
  Histogram intersection  : {hi:.4f}
  Mutual information      : {metrics_res.get('mutual_information', float('nan')):.4f}

  Calculated CPR median   : {stats_calc.get('median', float('nan')):.4f}
  Official CPR median     : {stats_offic.get('median', float('nan')):.4f}

  Output files
  ---------------------------------------------------------------
  Georeferenced CPR       : {georef_path}
  Validation report       : {report_path}
  Figures                 : {cfg.FIG_DIR}
  Log                     : {cfg.LOG_DIR / 'validation.log'}

  Total runtime           : {T_total}
  ---------------------------------------------------------------
""")

log.info("Validation pipeline completed successfully.")
