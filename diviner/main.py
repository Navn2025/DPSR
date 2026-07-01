"""
main.py
=======
Entry point for the Diviner Thermal Integration Pipeline.

Run from the project root:
    python diviner/main.py

or from inside diviner/:
    python main.py

Pipeline steps
--------------
 1. Load all input datasets and print full metadata.
 2. Convert GRD files (Tmean, Pump) to float32 GeoTIFF.
 3+4. Reproject + resample every dataset to the reference grid.
 5. Generate quick-look preview PNGs for Tmean, ZIT, Pump.
 6. Compute descriptive statistics for every feature band.
 7. Build a 9-band Feature Stack GeoTIFF.
 8. Compute physics-based Ice Confidence Score.
 9. Save Ice_Confidence_Map.tif and Ice_Confidence_Map.png.
10. Generate publication-quality maps, histograms, correlation matrix,
    and scatter plots.
11. Write statistics_report.csv.
12. Generate a PDF summary report.

IMPORTANT — this pipeline NEVER modifies, renames, or deletes any
existing outputs from CPR, DOP, DPSR, or any other module.  All new
files are written exclusively to outputs/diviner/ and outputs/previews/.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import config` resolves
# correctly whether the script is run from inside diviner/ or from the
# project root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np
import rasterio
from rasterio.windows import Window

import config as cfg
from utils import (
    setup_logger, Timer, array_stats, percentile_stats,
    log_stats, memory_mb, MemoryTracker, section,
)
from loader import read_metadata, print_metadata, grd_to_tif
from aligner import (
    load_reference_grid, align_raster, compute_slope,
    save_slope, load_aligned,
)
from visualizer import (
    save_diviner_previews,
    save_feature_maps,
    save_feature_histograms,
    save_correlation_matrix,
    save_scatter_plots,
    save_ice_confidence_png,
)
from ice_score import compute_ice_confidence
from reporter import write_statistics_csv, write_pdf_report


# ===========================================================================
# Bootstrap — create output directories, initialise logger and timers
# ===========================================================================
for d in (cfg.OUT_DIR, cfg.PREVIEW_DIR, cfg.ALIGNED_DIR,
          cfg.REPORT_DIR, cfg.LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

log             = setup_logger(cfg.LOG_DIR)
pipeline_timer  = Timer()
mem             = MemoryTracker()


# ===========================================================================
# STEP 1 — Load all input datasets and print full metadata
# ===========================================================================
section("STEP 1  |  Load Input Datasets & Print Metadata", log)

diviner_inputs = {
    "Tmean (GRD)": cfg.TMEAN_GRD,
    "ZIT   (TIF)": cfg.ZIT_TIF,
    "Pump  (GRD)": cfg.PUMP_GRD,
}
log.info("  ── Diviner Thermal Datasets ──────────────────────────────────")
diviner_metas = {}
for label, path in diviner_inputs.items():
    if not path.exists():
        log.error(f"  MISSING input file: {path}")
        log.error("  Aborting pipeline.  "
                  "Please place the Diviner files in temp/data/ .")
        sys.exit(1)
    meta = read_metadata(path, nodata_override=cfg.NODATA)
    diviner_metas[label] = meta
    print_metadata(label, meta, log)

log.info("")
log.info("  ── Existing Pipeline Products (read-only) ────────────────────")
existing_inputs = {
    "DEM  (LOLA)": cfg.DEM_PATH,
    "PSR  (mask)": cfg.PSR_PATH,
    "DPSR (mask)": cfg.DPSR_PATH,
    "CPR  (DFSAR)": cfg.CPR_PATH,
    "DOP  (DFSAR)": cfg.DOP_PATH,
}
existing_metas = {}
for label, path in existing_inputs.items():
    if not path.exists():
        log.warning(f"  Optional file not found: {path.name}  "
                    f"(band will be excluded from Feature Stack)")
        existing_metas[label] = None
        continue
    meta = read_metadata(path, nodata_override=cfg.NODATA)
    existing_metas[label] = meta
    print_metadata(label, meta, log)

log.info(f"\n  STEP 1 complete in {pipeline_timer}")


# ===========================================================================
# STEP 2 — Convert GRD files to GeoTIFF
# ===========================================================================
section("STEP 2  |  Convert GRD → GeoTIFF", log)

t0 = Timer()

# ZIT is the only Diviner file with a proper CRS and affine transform.
# Tmean and Pump are from the same dataset and share the same 2533×2533
# polar-stereographic grid, but their GRD files carry no spatial reference.
# We copy ZIT's CRS/transform into the converted GeoTIFFs so the aligner
# can reproject them correctly.
log.info("  ZIT is already a GeoTIFF — no conversion needed.")
zit_tif_path = cfg.ZIT_TIF    # source TIF used directly

log.info("  Tmean .grd → GeoTIFF  (assigning spatial reference from ZIT) ...")
tmean_tif_path = grd_to_tif(
    cfg.TMEAN_GRD, cfg.OUT_DIR, cfg.TMEAN_TIF_NAME,
    nodata=cfg.NODATA, crs_ref_path=cfg.ZIT_TIF,
)

log.info("  Pump .grd → GeoTIFF  (assigning spatial reference from ZIT) ...")
pump_tif_path = grd_to_tif(
    cfg.PUMP_GRD, cfg.OUT_DIR, cfg.PUMP_TIF_NAME,
    nodata=cfg.NODATA, crs_ref_path=cfg.ZIT_TIF,
)

log.info(f"  STEP 2 complete in {t0}")


# ===========================================================================
# STEP 3 + 4 — Reproject + resample all datasets to the reference grid
# ===========================================================================
section("STEP 3+4  |  Reproject & Resample to Reference Grid", log)

t0  = Timer()
ref = load_reference_grid(cfg.REFERENCE_GRID)
log.info(f"  Reference grid: {cfg.REFERENCE_GRID.name}")
log.info(f"    CRS      : {ref['crs']}")
log.info(f"    Size     : {ref['width']} x {ref['height']}")
log.info(f"    Transform: {ref['transform']}")
log.info("")

# ── Diviner layers (continuous → bilinear) ──────────────────────────────────
log.info("  Aligning Diviner layers  (bilinear resampling) ...")

tmean_aligned_path = align_raster(
    tmean_tif_path, cfg.ALIGNED_DIR, cfg.TMEAN_ALIGNED_NAME,
    ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
)
zit_aligned_path = align_raster(
    zit_tif_path,   cfg.ALIGNED_DIR, cfg.ZIT_ALIGNED_NAME,
    ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
)
pump_aligned_path = align_raster(
    pump_tif_path,  cfg.ALIGNED_DIR, cfg.PUMP_ALIGNED_NAME,
    ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
)

# ── DEM (continuous) ────────────────────────────────────────────────────────
if cfg.DEM_PATH.exists():
    log.info("  Aligning DEM  (bilinear resampling, km → m ×1000) ...")
    dem_aligned_path = align_raster(
        cfg.DEM_PATH, cfg.ALIGNED_DIR, cfg.DEM_ALIGNED_NAME,
        ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
        scale_factor=1000.0,   # LOLA .lbl values are in km; convert to metres
    )
else:
    log.warning("  DEM not found — DEM and Slope bands will be omitted.")
    dem_aligned_path = None

# ── PSR + DPSR (binary masks → nearest neighbour) ───────────────────────────
log.info("  Aligning PSR and DPSR masks  (nearest-neighbour resampling) ...")
if cfg.PSR_PATH.exists():
    psr_aligned_path = align_raster(
        cfg.PSR_PATH, cfg.ALIGNED_DIR, cfg.PSR_ALIGNED_NAME,
        ref, resampling=cfg.RESAMPLE_MASK, nodata=cfg.NODATA,
    )
else:
    log.warning("  PSR_mask.tif not found — PSR band omitted.")
    psr_aligned_path = None

if cfg.DPSR_PATH.exists():
    # DPSR is the reference grid itself; copy or align trivially
    dpsr_aligned_path = align_raster(
        cfg.DPSR_PATH, cfg.ALIGNED_DIR, cfg.DPSR_ALIGNED_NAME,
        ref, resampling=cfg.RESAMPLE_MASK, nodata=cfg.NODATA,
    )
else:
    log.warning("  DPSR.tif not found — DPSR band omitted.")
    dpsr_aligned_path = None

# ── CPR (continuous, optional) ──────────────────────────────────────────────
if cfg.CPR_PATH.exists():
    log.info("  Aligning CPR  (bilinear resampling) ...")
    cpr_aligned_path = align_raster(
        cfg.CPR_PATH, cfg.ALIGNED_DIR, cfg.CPR_ALIGNED_NAME,
        ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
    )
else:
    log.warning(
        f"  CPR not found at {cfg.CPR_PATH.name} — CPR band omitted "
        f"from feature stack and ice score."
    )
    cpr_aligned_path = None

# ── DOP (continuous, optional) ──────────────────────────────────────────────
if cfg.DOP_PATH.exists():
    log.info("  Aligning DOP  (bilinear resampling) ...")
    dop_aligned_path = align_raster(
        cfg.DOP_PATH, cfg.ALIGNED_DIR, cfg.DOP_ALIGNED_NAME,
        ref, resampling=cfg.RESAMPLE_CONTINUOUS, nodata=cfg.NODATA,
    )
else:
    log.warning(
        f"  DOP not found at {cfg.DOP_PATH.name} — DOP band omitted."
    )
    dop_aligned_path = None

log.info(f"  STEP 3+4 complete in {t0}")


# ===========================================================================
# Load aligned arrays into memory
# ===========================================================================
section("Loading aligned arrays into memory", log)

t0 = Timer()

arr_tmean = load_aligned(tmean_aligned_path, cfg.NODATA)
arr_zit   = load_aligned(zit_aligned_path,   cfg.NODATA)
arr_pump  = load_aligned(pump_aligned_path,  cfg.NODATA)

arr_dem  = load_aligned(dem_aligned_path,  cfg.NODATA) if dem_aligned_path  else None
arr_psr  = load_aligned(psr_aligned_path,  cfg.NODATA) if psr_aligned_path  else None
arr_dpsr = load_aligned(dpsr_aligned_path, cfg.NODATA) if dpsr_aligned_path else None
arr_cpr  = load_aligned(cpr_aligned_path,  cfg.NODATA) if cpr_aligned_path  else None
arr_dop  = load_aligned(dop_aligned_path,  cfg.NODATA) if dop_aligned_path  else None

for name, arr in [("Tmean", arr_tmean), ("ZIT",  arr_zit),  ("Pump", arr_pump),
                  ("DEM",   arr_dem),   ("PSR",  arr_psr),  ("DPSR", arr_dpsr),
                  ("CPR",   arr_cpr),   ("DOP",  arr_dop)]:
    if arr is not None:
        mem.add(arr)
        log.info(f"  Loaded {name:<6}  {arr.shape}  {memory_mb(arr):.1f} MB")

log.info(f"  All arrays loaded in {t0}  (tracked: {mem.current_mb:.0f} MB)")


# ===========================================================================
# Derive Slope from aligned DEM
# ===========================================================================
arr_slope = None
if arr_dem is not None:
    section("Deriving Slope from aligned DEM", log)
    t0 = Timer()
    arr_slope = compute_slope(arr_dem, ref["transform"], ref["crs"], cfg.NODATA)
    mem.add(arr_slope)
    log.info(f"  Slope computed in {t0}  ({memory_mb(arr_slope):.1f} MB)")

    slope_path = save_slope(
        arr_slope, cfg.ALIGNED_DIR, cfg.SLOPE_ALIGNED_NAME, ref, cfg.NODATA
    )
    log.info(f"  Slope saved: {slope_path.name}")
else:
    log.warning("  DEM unavailable — Slope band omitted.")


# ===========================================================================
# STEP 5 — Quick-look previews for Diviner layers
# ===========================================================================
section("STEP 5  |  Quick-Look Previews (Tmean, ZIT, Pump)", log)

t0 = Timer()
save_diviner_previews(arr_tmean, arr_zit, arr_pump, cfg.PREVIEW_DIR, cfg.NODATA)
log.info(f"  STEP 5 complete in {t0}  — saved to {cfg.PREVIEW_DIR}")


# ===========================================================================
# STEP 6 — Statistics for every feature band
# ===========================================================================
section("STEP 6  |  Descriptive Statistics", log)

t0 = Timer()
pcts = cfg.REPORT_PERCENTILES

# Build an ordered dict of all available bands for consistent ordering
all_bands_ordered = [
    ("DEM",   arr_dem),
    ("Slope", arr_slope),
    ("PSR",   arr_psr),
    ("DPSR",  arr_dpsr),
    ("CPR",   arr_cpr),
    ("DOP",   arr_dop),
    ("Tmean", arr_tmean),
    ("ZIT",   arr_zit),
    ("Pump",  arr_pump),
]
available_bands = {
    name: arr for name, arr in all_bands_ordered if arr is not None
}

for name, arr in available_bands.items():
    stats = array_stats(arr, nodata=cfg.NODATA, label=name)
    log_stats(stats, log)

    pct_vals = percentile_stats(arr, cfg.NODATA, pcts)
    log.info("  Percentiles:")
    for p in pcts:
        log.info(f"    P{p:02d} = {pct_vals[p]:.6e}")
    log.info("")

log.info(f"  STEP 6 complete in {t0}")


# ===========================================================================
# STEP 7 — Build 9-band Feature Stack
# ===========================================================================
section("STEP 7  |  Build Feature Stack  (Feature_Stack.tif)", log)

stack_out_path = cfg.OUT_DIR / cfg.FEATURE_STACK_NAME

_stack_valid = False
if stack_out_path.exists():
    _file_size = stack_out_path.stat().st_size
    # Check BigTIFF magic bytes (II\x2b\x00 or MM\x00\x2b).
    # Files written before BIGTIFF=YES was added are standard TIFF (4 GB
    # hard limit) and will be corrupt when the stack exceeds that limit.
    _BIGTIFF_MAGIC = {b"II\x2b\x00", b"MM\x00\x2b"}
    try:
        with open(stack_out_path, "rb") as _f:
            _is_bigtiff = _f.read(4) in _BIGTIFF_MAGIC
    except OSError:
        _is_bigtiff = False
    try:
        with rasterio.open(stack_out_path) as _chk:
            _stack_valid = (_chk.count == len(available_bands) and _is_bigtiff)
    except Exception:
        _stack_valid = False
    if _stack_valid:
        log.info(f"  [skip] {cfg.FEATURE_STACK_NAME} already exists ({len(available_bands)} bands, {_file_size/1024**2:.0f} MB).")
    else:
        log.warning(
            f"  {cfg.FEATURE_STACK_NAME} is incomplete/corrupt "
            f"({_file_size/1024**2:.0f} MB, bigtiff={_is_bigtiff}) — deleting and re-writing."
        )
        stack_out_path.unlink()

if not _stack_valid:
    t0      = Timer()
    n_bands = len(available_bands)
    H, W    = ref["height"], ref["width"]

    log.info(f"  Writing {n_bands}-band Feature Stack  "
             f"({W} x {H}, float32, LZW) ...")

    stack_profile = {
        "driver":      "GTiff",
        "dtype":       "float32",
        "width":       W,
        "height":      H,
        "count":       n_bands,
        "crs":         ref["crs"],
        "transform":   ref["transform"],
        "nodata":      cfg.NODATA,
        "compress":    "lzw",
        "interleave":  "band",
        "BIGTIFF":     "YES",
    }

    with rasterio.open(stack_out_path, "w", **stack_profile) as dst:
        for band_idx, (name, arr) in enumerate(available_bands.items(), start=1):
            for row_start in range(0, H, cfg.WRITE_BLOCK):
                row_end = min(row_start + cfg.WRITE_BLOCK, H)
                n_rows  = row_end - row_start
                win     = Window(0, row_start, W, n_rows)
                dst.write(arr[row_start:row_end, :], band_idx, window=win)
            dst.set_band_description(band_idx, name)
            log.info(f"    Band {band_idx:>2}: {name}")

        dst.update_tags(
            PIPELINE    = "Diviner Thermal Integration — ISRO Hackathon",
            BAND_ORDER  = " | ".join(
                f"{i+1}:{n}" for i, n in enumerate(available_bands)
            ),
            REFERENCE   = str(cfg.REFERENCE_GRID.name),
            NODATA      = str(cfg.NODATA),
        )

    size_mb = stack_out_path.stat().st_size / 1024 ** 2
    log.info(f"  Feature Stack saved: {stack_out_path}  ({size_mb:.1f} MB)  in {t0}")


# ===========================================================================
# STEP 8 — Physics-based Ice Confidence Score
# ===========================================================================
section("STEP 8  |  Physics-Based Ice Confidence Score", log)

# Only pass bands that the ice-score module knows how to weight
ice_score_bands = {
    name: arr
    for name, arr in available_bands.items()
    if name in cfg.ICE_SCORE_WEIGHTS
}

log.info(f"  Contributing bands : {list(ice_score_bands.keys())}")
log.info(f"  Weights            : {cfg.ICE_SCORE_WEIGHTS}")
log.info(f"  Weight total       : {sum(cfg.ICE_SCORE_WEIGHTS.values()):.2f}")

t0 = Timer()
arr_ice, norm_bands = compute_ice_confidence(
    bands   = ice_score_bands,
    weights = cfg.ICE_SCORE_WEIGHTS,
    nodata  = cfg.NODATA,
    lo_pct  = cfg.PREVIEW_PERCENTILE_LO,
    hi_pct  = cfg.PREVIEW_PERCENTILE_HI,
)
mem.add(arr_ice)
log.info(f"  Ice Confidence Score computed in {t0}")


# ===========================================================================
# STEP 9 — Save Ice_Confidence_Map.tif and Ice_Confidence_Map.png
# ===========================================================================
section("STEP 9  |  Save Ice Confidence Map", log)

ice_tif_path = cfg.OUT_DIR / cfg.ICE_CONF_TIF_NAME
ice_png_path = cfg.OUT_DIR / cfg.ICE_CONF_PNG_NAME

if ice_tif_path.exists():
    log.info(f"  [skip] {cfg.ICE_CONF_TIF_NAME} already exists.")
else:
    t0  = Timer()
    H, W = arr_ice.shape
    ice_profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     W,
        "height":    H,
        "count":     1,
        "crs":       ref["crs"],
        "transform": ref["transform"],
        "nodata":    cfg.NODATA,
        "compress":  "lzw",
    }
    with rasterio.open(ice_tif_path, "w", **ice_profile) as dst:
        for row_start in range(0, H, cfg.WRITE_BLOCK):
            row_end = min(row_start + cfg.WRITE_BLOCK, H)
            n_rows  = row_end - row_start
            win     = Window(0, row_start, W, n_rows)
            dst.write(arr_ice[row_start:row_end, :], 1, window=win)

        dst.update_tags(
            PIPELINE   = "Physics-Based Ice Confidence Score (diviner/ice_score.py)",
            BANDS_USED = " | ".join(ice_score_bands.keys()),
            WEIGHTS    = str(cfg.ICE_SCORE_WEIGHTS),
            FORMULA    = "score = Σ(norm_i × w_i) / Σ(w_i available)",
            NODATA     = str(cfg.NODATA),
        )

    size_mb = ice_tif_path.stat().st_size / 1024 ** 2
    log.info(f"  Saved: {ice_tif_path}  ({size_mb:.1f} MB)  in {t0}")

# PNG
t0 = Timer()
save_ice_confidence_png(arr_ice, ice_png_path, cfg.NODATA)
log.info(f"  Ice Confidence PNG saved in {t0}")

ice_stats = array_stats(arr_ice, nodata=cfg.NODATA, label="IceConfidence")
log_stats(ice_stats, log)


# ===========================================================================
# STEP 10 — Publication-quality visualisations
# ===========================================================================
section("STEP 10  |  Generate Visualisations", log)

t0 = Timer()

# All bands including the Ice Confidence score
viz_bands = dict(available_bands)
viz_bands["IceConfidence"] = arr_ice

all_cmaps = dict(cfg.CMAPS)

log.info("  Saving feature maps ...")
save_feature_maps(viz_bands, all_cmaps, cfg.OUT_DIR, cfg.NODATA)

log.info("  Saving feature histograms ...")
save_feature_histograms(viz_bands, cfg.OUT_DIR, cfg.NODATA, cfg.HIST_BINS)

log.info("  Saving correlation matrix ...")
save_correlation_matrix(available_bands, cfg.OUT_DIR, cfg.NODATA)

log.info("  Saving scatter plots ...")
save_scatter_plots(available_bands, cfg.OUT_DIR, cfg.NODATA)

log.info(f"  STEP 10 complete in {t0}")


# ===========================================================================
# STEP 11 — CSV statistics report
# ===========================================================================
section("STEP 11  |  Write Statistics CSV Report", log)

csv_path = cfg.REPORT_DIR / cfg.STATS_CSV_NAME
all_stats_bands = dict(available_bands)
all_stats_bands["IceConfidence"] = arr_ice

t0 = Timer()
write_statistics_csv(
    bands             = all_stats_bands,
    out_path          = csv_path,
    nodata            = cfg.NODATA,
    percentile_levels = cfg.REPORT_PERCENTILES,
)
log.info(f"  STEP 11 complete in {t0}")


# ===========================================================================
# STEP 12 — PDF report
# ===========================================================================
section("STEP 12  |  Generate PDF Report", log)

# Collect all input paths for the report table
report_input_paths = {
    "Tmean (GRD)":  cfg.TMEAN_GRD,
    "ZIT   (TIF)":  cfg.ZIT_TIF,
    "Pump  (GRD)":  cfg.PUMP_GRD,
    "DEM   (LOLA)": cfg.DEM_PATH,
    "PSR   (mask)": cfg.PSR_PATH,
    "DPSR  (mask)": cfg.DPSR_PATH,
    "CPR   (DFSAR)": cfg.CPR_PATH,
    "DOP   (DFSAR)": cfg.DOP_PATH,
}

t0 = Timer()
pdf_path = cfg.REPORT_DIR / cfg.PDF_REPORT_NAME
write_pdf_report(
    out_path     = pdf_path,
    bands        = all_stats_bands,
    input_paths  = report_input_paths,
    aligned_dir  = cfg.ALIGNED_DIR,
    preview_dir  = cfg.PREVIEW_DIR,
    out_dir      = cfg.OUT_DIR,
    weights      = cfg.ICE_SCORE_WEIGHTS,
    nodata       = cfg.NODATA,
)
log.info(f"  STEP 12 complete in {t0}")


# ===========================================================================
# Final summary
# ===========================================================================
section("PIPELINE COMPLETE", log)

# Tally the total valid ice pixels at each confidence tier
if arr_ice is not None:
    valid_ice = arr_ice[(arr_ice != cfg.NODATA) & np.isfinite(arr_ice)]
    total_px  = arr_ice.size
    n_valid   = valid_ice.size
    n_high    = int((valid_ice >= 0.7).sum())
    n_med     = int(((valid_ice >= 0.4) & (valid_ice < 0.7)).sum())
    n_low     = int((valid_ice < 0.4).sum())
else:
    n_valid = n_high = n_med = n_low = total_px = 0

log.info(f"""
  ─────────────────────────────────────────────────────────────────
  Diviner Thermal Integration Pipeline — Final Summary
  ─────────────────────────────────────────────────────────────────
  Reference grid      : {cfg.REFERENCE_GRID.name}
  Grid size           : {ref['width']} x {ref['height']}  (W x H pixels)
  CRS                 : {ref['crs']}

  FEATURE STACK
  ─────────────────────────────────────────────────────────────────
  Bands in stack      : {list(available_bands.keys())}
  Output              : {cfg.OUT_DIR / cfg.FEATURE_STACK_NAME}

  ICE CONFIDENCE STATISTICS
  ─────────────────────────────────────────────────────────────────
  Valid pixels        : {n_valid:,} / {total_px:,}
  High confidence (≥ 0.7) : {n_high:,}  ({100*n_high/max(n_valid,1):.1f} %)
  Medium (0.4–0.7)        : {n_med:,}  ({100*n_med/max(n_valid,1):.1f} %)
  Low  (< 0.4)            : {n_low:,}  ({100*n_low/max(n_valid,1):.1f} %)

  OUTPUT LOCATIONS
  ─────────────────────────────────────────────────────────────────
  GeoTIFFs (aligned)  : {cfg.ALIGNED_DIR}
  Feature Stack       : {cfg.OUT_DIR / cfg.FEATURE_STACK_NAME}
  Ice Confidence      : {cfg.OUT_DIR / cfg.ICE_CONF_TIF_NAME}
  Ice Conf. PNG       : {cfg.OUT_DIR / cfg.ICE_CONF_PNG_NAME}
  Maps + Histograms   : {cfg.OUT_DIR}
  Previews            : {cfg.PREVIEW_DIR}
  Statistics CSV      : {csv_path}
  PDF Report          : {pdf_path}
  Log file            : {cfg.LOG_DIR / 'diviner_pipeline.log'}

  RESOURCE USAGE
  ─────────────────────────────────────────────────────────────────
  Peak tracked array memory : {mem.peak_mb:.0f} MB
  Total pipeline runtime    : {pipeline_timer}
  ─────────────────────────────────────────────────────────────────
""")

log.info("Diviner pipeline finished successfully.")
