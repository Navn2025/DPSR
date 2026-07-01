"""
main.py
=======
Entry point for the Chandrayaan-2 DFSAR Full-Pol SLI -> CPR pipeline.

Run from the project root:
    python cpr/main.py                # default: exact coherent SLI CPR (Putrevu et al. 2023)
    python cpr/main.py --research     # published co-pol-only mu_c formula (cpr.compute_cpr_research)

--research uses only HH and VV (HV/VH are never loaded, so it is also
faster/lighter than the default path) and writes a separate output
(Calculated_CPR_research.tif) so it never overwrites the default
result. See cpr.compute_cpr_research() for the formula and its
numerical caveats (the raw ratio is log10-rescaled onto a 0-2 range
since it saturates far above that on real terrain).

Pipeline steps (default mode)
------------------------------
1.  Read metadata from all four SLI GeoTIFFs (HH, HV, VH, VV).
2.  Validate geometric consistency (dimensions, CRS, transform).
3.  Load HH, VV complex SLC -- compute OC field = S_HH + S_VV.
4.  Load HV, VH complex SLC -- compute SC field = S_HH - S_VV + 2j*S_XP.
5.  Compute single-look power: P_SC = |SC_field|^2,  P_OC = |OC_field|^2.
6.  Apply multilook (uniform_filter) to P_SC and P_OC.
7.  Compute CPR = mean(P_SC) / mean(P_OC).
8.  Print CPR statistics + save histogram.
9.  Save preview PNGs in outputs/previews/.
10. Write Calculated_CPR.tif in outputs/cpr/.
11. Print diagnostic report.

In --research mode, steps 4-7 are replaced by: multilook |S_HH|^2 and
|S_VV|^2 directly, then apply cpr.compute_cpr_research().

Formula (default)
------------------
    S_RR = (S_HH - S_VV + 2j * S_HV) / 2    [SC, same-sense circular]
    S_RL = (S_HH + S_VV) / 2                 [OC, opposite-sense circular]
    CPR  = mean(|S_RR|^2) / mean(|S_RL|^2)  after multilooking

    The 1/2 factors cancel; internally we use
        SC_field = S_HH - S_VV + 2j * S_XP    (S_XP = (S_HV + S_VH)/2)
        OC_field = S_HH + S_VV

Reference: Putrevu et al. (2023), JGR Planets, DOI: 10.1029/2023JE007745
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
from rasterio.windows import Window

import config as cfg
from utils           import setup_logger, Timer, array_stats, log_stats, section, memory_mb
from reader          import read_metadata, print_metadata, load_complex
from validator       import validate_rasters
from complex_builder import complex_to_power, apply_multilook
from cpr             import compute_cpr, compute_cpr_research
from visualizer      import (
    save_power_preview,
    save_cpr_preview,
    save_cpr_histogram,
)

parser = argparse.ArgumentParser(description="Chandrayaan-2 DFSAR SLI -> CPR pipeline")
parser.add_argument(
    "--research", action="store_true",
    help="Use the published co-pol-only CPR(mu_c) formula (HH/VV only) "
         "instead of the default exact coherent SC/OC formula.",
)
args = parser.parse_args()
RESEARCH_MODE = args.research

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
for d in (cfg.PREV_DIR, cfg.CPR_DIR, cfg.LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = setup_logger(cfg.LOG_DIR)
pipeline_timer = Timer()
log.info(f"Mode: {'RESEARCH (published mu_c formula, HH/VV only)' if RESEARCH_MODE else 'DEFAULT (exact coherent SC/OC formula)'}")


# ===========================================================================
# STEP 1  --  Read metadata
# ===========================================================================
section("STEP 1  |  Read SLI Metadata", log)

metas: dict = {}
for pol, path in cfg.SLI_PATHS.items():
    meta = read_metadata(path)
    metas[pol] = meta
    log.info("-" * 50)
    print_metadata(pol, meta, log)

ref = metas["HH"]
H, W = ref["height"], ref["width"]
log.info(f"\n  Scene dimensions : {W} samples (range) x {H} lines (azimuth)")
log.info(f"  Pixel spacing    : {ref['res'][0]} m (range) x {ref['res'][1]} m (azimuth)")


# ===========================================================================
# STEP 2  --  Validate consistency
# ===========================================================================
section("STEP 2  |  Validate Raster Consistency", log)
validate_rasters(metas)


# ===========================================================================
# STEP 3  --  Load HH + VV, form OC field and partial SC field
# ===========================================================================
section("STEP 3  |  Load HH, VV -> Form OC = S_HH + S_VV", log)

log.info("Loading HH complex SLC ...")
t0 = Timer()
S_HH = load_complex(cfg.SLI_PATHS["HH"], "HH")
log.info(f"  HH loaded in {t0}  ({memory_mb(S_HH):.0f} MB)")

log.info("Loading VV complex SLC ...")
t0 = Timer()
S_VV = load_complex(cfg.SLI_PATHS["VV"], "VV")
log.info(f"  VV loaded in {t0}  ({memory_mb(S_VV):.0f} MB)")

research_info = None

if RESEARCH_MODE:
    # Only HH, VV are needed for the co-pol-only mu_c formula -- HV/VH
    # are never loaded, saving ~1 GB and two large reads vs default mode.
    P_HH_sl = complex_to_power(S_HH, "HH")
    P_VV_sl = complex_to_power(S_VV, "VV")
    del S_HH, S_VV
    log.info("  S_HH and S_VV freed from memory (research mode: HV/VH not needed).")
else:
    # OC field  =  S_HH + S_VV   (no cross-pol contribution for monostatic)
    OC_field = S_HH + S_VV
    log.info(f"  OC field formed  ({memory_mb(OC_field):.0f} MB complex64)")

    # Partial SC  =  S_HH - S_VV   (cross-pol contribution added in Step 4)
    diff = S_HH - S_VV

    del S_HH, S_VV      # free ~1 GB; we only need diff and OC_field forward
    log.info("  S_HH and S_VV freed from memory.")


# ===========================================================================
# STEP 4  --  Load HV + VH, complete SC field  (default mode only)
# ===========================================================================
if not RESEARCH_MODE:
    section("STEP 4  |  Load HV, VH -> Form SC = (S_HH - S_VV) + 2j * S_XP", log)

    log.info("Loading HV complex SLC ...")
    t0 = Timer()
    S_HV = load_complex(cfg.SLI_PATHS["HV"], "HV")
    log.info(f"  HV loaded in {t0}  ({memory_mb(S_HV):.0f} MB)")

    log.info("Loading VH complex SLC ...")
    t0 = Timer()
    S_VH = load_complex(cfg.SLI_PATHS["VH"], "VH")
    log.info(f"  VH loaded in {t0}  ({memory_mb(S_VH):.0f} MB)")

    # Averaged cross-pol (reciprocity: S_HV == S_VH for monostatic SAR)
    S_XP = (S_HV + S_VH) * 0.5
    del S_HV, S_VH
    log.info("  S_XP (averaged cross-pol) formed; HV and VH freed.")

    # SC field  =  (S_HH - S_VV) + 2j * S_XP
    # real(SC) = diff.real - 2 * XP.imag
    # imag(SC) = diff.imag + 2 * XP.real
    SC_field = np.empty_like(diff)
    SC_field.real[:] = diff.real - 2.0 * S_XP.imag
    SC_field.imag[:] = diff.imag + 2.0 * S_XP.real
    del diff, S_XP
    log.info(f"  SC field formed  ({memory_mb(SC_field):.0f} MB complex64)")

    log.info(
        f"  |OC| RMS = {float(np.sqrt(np.mean(OC_field.real**2 + OC_field.imag**2))):.3e}  "
        f"  |SC| RMS = {float(np.sqrt(np.mean(SC_field.real**2 + SC_field.imag**2))):.3e}"
    )


# ===========================================================================
# STEP 5  --  Single-look power
# ===========================================================================
section("STEP 5  |  Single-Look Power", log)

if RESEARCH_MODE:
    P_HH, P_VV = P_HH_sl, P_VV_sl
    for label, arr in (("P_HH (single-look)", P_HH), ("P_VV (single-look)", P_VV)):
        st = array_stats(arr, nodata=0.0, label=label)
        log_stats(st, log)
else:
    P_SC = complex_to_power(SC_field, "SC")
    del SC_field
    log.info(f"  P_SC formed  ({memory_mb(P_SC):.0f} MB float32)")

    P_OC = complex_to_power(OC_field, "OC")
    del OC_field
    log.info(f"  P_OC formed  ({memory_mb(P_OC):.0f} MB float32)")

    for label, arr in (("P_SC (single-look)", P_SC), ("P_OC (single-look)", P_OC)):
        st = array_stats(arr, nodata=0.0, label=label)
        log_stats(st, log)


# ===========================================================================
# STEP 6  --  Multilook (speckle reduction)
# ===========================================================================
section("STEP 6  |  Multilook Filtering", log)

az, rg = cfg.MULTILOOK_WINDOW
log.info(f"Applying box-car filter: ({az} az x {rg} rg) -- {az * rg} effective looks")
t0 = Timer()

if RESEARCH_MODE:
    ML_HH = apply_multilook(P_HH, cfg.MULTILOOK_WINDOW, "HH")
    ML_VV = apply_multilook(P_VV, cfg.MULTILOOK_WINDOW, "VV")
    del P_HH, P_VV
    log.info(f"  Multilook done in {t0}")
    for label, arr in (("ML_HH", ML_HH), ("ML_VV", ML_VV)):
        st = array_stats(arr, nodata=0.0, label=label)
        log_stats(st, log)
else:
    ML_SC = apply_multilook(P_SC, cfg.MULTILOOK_WINDOW, "SC")
    ML_OC = apply_multilook(P_OC, cfg.MULTILOOK_WINDOW, "OC")
    del P_SC, P_OC
    log.info(f"  Multilook done in {t0}")
    for label, arr in (("ML_SC", ML_SC), ("ML_OC", ML_OC)):
        st = array_stats(arr, nodata=0.0, label=label)
        log_stats(st, log)


# ===========================================================================
# STEP 7  --  Compute CPR
# ===========================================================================
if RESEARCH_MODE:
    section("STEP 7  |  Compute CPR(mu_c) -- published co-pol-only formula", log)
    log.info(f"  epsilon = {cfg.EPSILON}   nodata = {cfg.NODATA}   output range = {cfg.CPR_RESEARCH_VALID_RANGE}")
    t0 = Timer()
    cpr, research_info = compute_cpr_research(
        ML_HH, ML_VV, epsilon=cfg.EPSILON, nodata=cfg.NODATA,
        valid_range=cfg.CPR_RESEARCH_VALID_RANGE,
    )
    log.info(f"  CPR_research computation done in {t0}")
else:
    section("STEP 7  |  Compute CPR = ML_SC / ML_OC", log)
    log.info(
        "Formula: CPR = mean(|S_HH - S_VV + 2j*S_HV|^2) / mean(|S_HH + S_VV|^2)\n"
        "         (circular basis conversion, Putrevu et al. 2023)"
    )
    log.info(f"  epsilon = {cfg.EPSILON}   nodata = {cfg.NODATA}")
    t0 = Timer()
    cpr = compute_cpr(ML_SC, ML_OC, epsilon=cfg.EPSILON, nodata=cfg.NODATA)
    log.info(f"  CPR computation done in {t0}")


# ===========================================================================
# STEP 8  --  CPR statistics + histogram
# ===========================================================================
section("STEP 8  |  CPR Statistics & Histogram", log)

cpr_stats = array_stats(cpr, nodata=cfg.NODATA, label="CPR")
log_stats(cpr_stats, log)

valid_cpr = cpr[(cpr != cfg.NODATA) & np.isfinite(cpr)]
frac_gt1 = 0.0
if valid_cpr.size > 0:
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    log.info("  Percentiles:")
    for p in pcts:
        log.info(f"    P{p:02d} = {np.percentile(valid_cpr, p):.4f}")

    frac_gt1 = float(np.mean(valid_cpr > 1.0)) * 100.0
    if RESEARCH_MODE:
        log.info(
            f"  Pixels with CPR(mu_c) display > 1.0 : {frac_gt1:.2f}%  "
            f"(upper half of the log-rescaled range -- NOT directly comparable "
            f"to the default formula's CPR>1 ice threshold, see cpr.compute_cpr_research)"
        )
    else:
        log.info(f"  Pixels with CPR > 1.0 : {frac_gt1:.2f}%  (ice candidates)")

prev_dir = (cfg.PREV_DIR / "research") if RESEARCH_MODE else cfg.PREV_DIR
prev_dir.mkdir(parents=True, exist_ok=True)
save_cpr_histogram(cpr, prev_dir, nodata=cfg.NODATA)


# ===========================================================================
# STEP 9  --  Preview images
# ===========================================================================
section("STEP 9  |  Save Preview Images", log)

log.info(f"Downsample factor: 1:{cfg.PREVIEW_DOWNSAMPLE}")
if RESEARCH_MODE:
    save_power_preview(ML_HH, ML_VV, prev_dir,
                       downsample=cfg.PREVIEW_DOWNSAMPLE, nodata=0.0,
                       labels=("HH_power", "VV_power"))
    raw_lo = research_info.get("rescale_lo_raw", float("nan"))
    raw_hi = research_info.get("rescale_hi_raw", float("nan"))
    cpr_preview_title = (
        f"CPR(μ_c) log10-rescaled onto {cfg.CPR_RESEARCH_VALID_RANGE}\n"
        f"raw P1–P99 range: [{raw_lo:.3g}, {raw_hi:.3g}]  --  "
        f"NOT the literal unscaled formula value"
    )
else:
    save_power_preview(ML_SC, ML_OC, prev_dir,
                       downsample=cfg.PREVIEW_DOWNSAMPLE, nodata=0.0,
                       labels=("SC_power", "OC_power"))
    cpr_preview_title = None
save_cpr_preview(cpr, prev_dir,
                 downsample=cfg.PREVIEW_DOWNSAMPLE, nodata=cfg.NODATA,
                 title=cpr_preview_title)
log.info(f"  All previews saved to: {prev_dir}")


# ===========================================================================
# STEP 10  --  Write CPR GeoTIFF
# ===========================================================================
out_name = "Calculated_CPR_research.tif" if RESEARCH_MODE else cfg.CPR_OUTPUT_NAME
section(f"STEP 10  |  Save {out_name}", log)

out_tif = cfg.CPR_DIR / out_name
out_profile = ref["profile"].copy()
out_profile.update(
    dtype    = "float32",
    count    = 1,
    nodata   = cfg.NODATA,
    compress = "lzw",
)

log.info(f"Writing {out_tif.name} ({W}W x {H}H, float32, LZW) ...")
t0 = Timer()

with rasterio.open(out_tif, "w", **out_profile) as dst:
    for row_start in range(0, H, cfg.WRITE_BLOCK):
        row_end = min(row_start + cfg.WRITE_BLOCK, H)
        nrows   = row_end - row_start
        win     = Window(0, row_start, W, nrows)
        dst.write(cpr[row_start:row_end, :], 1, window=win)

    if RESEARCH_MODE:
        raw_med = research_info.get("raw_median", float("nan"))
        raw_p1  = research_info.get("rescale_lo_raw", float("nan"))
        raw_p99 = research_info.get("rescale_hi_raw", float("nan"))
        dst.update_tags(
            PIPELINE        = "Chandrayaan-2 DFSAR Full-Pol SLI -> CPR (research mode)",
            FORMULA         = "CPR(mu_c) = (sqrt(P_HH)+sqrt(P_VV))^2 / (sqrt(P_HH)-sqrt(P_VV))^2",
            RESCALE         = f"log10(raw) P1-P99 [{raw_p1:.4g}, {raw_p99:.4g}] -> {cfg.CPR_RESEARCH_VALID_RANGE}",
            RAW_MEDIAN      = f"{raw_med:.6g}",
            REFERENCE       = "published mu_c formula, log10-rescaled -- NOT literal formula value",
            MULTILOOK       = str(cfg.MULTILOOK_WINDOW),
            EPSILON         = str(cfg.EPSILON),
            NODATA          = str(cfg.NODATA),
            INPUT_HH        = cfg.SLI_PATHS["HH"].name,
            INPUT_VV        = cfg.SLI_PATHS["VV"].name,
            CPR_DISPLAY_MED = f"{cpr_stats['median']:.6f}",
            CPR_DISPLAY_MEA = f"{cpr_stats['mean']:.6f}",
        )
    else:
        dst.update_tags(
            PIPELINE    = "Chandrayaan-2 DFSAR Full-Pol SLI -> CPR",
            FORMULA     = "mean(|S_HH-S_VV+2j*S_HV|^2) / mean(|S_HH+S_VV|^2)",
            REFERENCE   = "Putrevu et al. (2023) JGR Planets 10.1029/2023JE007745",
            MULTILOOK   = str(cfg.MULTILOOK_WINDOW),
            EPSILON     = str(cfg.EPSILON),
            NODATA      = str(cfg.NODATA),
            INPUT_HH    = cfg.SLI_PATHS["HH"].name,
            INPUT_HV    = cfg.SLI_PATHS["HV"].name,
            INPUT_VH    = cfg.SLI_PATHS["VH"].name,
            INPUT_VV    = cfg.SLI_PATHS["VV"].name,
            CPR_MEDIAN  = f"{cpr_stats['median']:.6f}",
            CPR_MEAN    = f"{cpr_stats['mean']:.6f}",
        )

log.info(f"  Done in {t0}")
log.info(f"  Saved : {out_tif}")
log.info(f"  Size  : {out_tif.stat().st_size / 1024**2:.1f} MB")


# ===========================================================================
# STEP 11  --  Diagnostic report
# ===========================================================================
section("STEP 11  |  Diagnostic Report", log)

n_total  = H * W
n_valid  = int(cpr_stats["valid"])
n_nodata = int((cpr == cfg.NODATA).sum())
n_nan    = int(cpr_stats["nan"])
n_inf    = int(cpr_stats["inf"])

if RESEARCH_MODE:
    raw_med = research_info.get("raw_median", float("nan"))
    raw_p1  = research_info.get("rescale_lo_raw", float("nan"))
    raw_p99 = research_info.get("rescale_hi_raw", float("nan"))
    n_invalid_raw = research_info.get("n_invalid", 0)
    formula_lines = (
        "  Formula       : CPR(mu_c) = (sqrt(sigma_HH)+sqrt(sigma_VV))^2\n"
        "                              / (sqrt(sigma_HH)-sqrt(sigma_VV))^2\n"
        "                  Applied to multilooked |S_HH|^2 and |S_VV|^2\n"
        "                  (calibration constant cancels in HH/VV ratio)\n"
        "                  Raw output log10-RESCALED onto (0.0, 2.0) --\n"
        "                  NOT the literal unscaled formula value."
    )
    reference_line = "  Reference     : published mu_c formula (co-pol only)"
    inputs_line    = (
        f"  Inputs        : HH  {cfg.SLI_PATHS['HH'].name}\n"
        f"                  VV  {cfg.SLI_PATHS['VV'].name}\n"
        f"                  (HV/VH not loaded in research mode)"
    )
    rescale_section = (
        f"\n  LOG-RESCALE (raw mu_c -> display CPR)\n"
        f"  -------------------------------------------------------\n"
        f"  Raw median    : {raw_med:.4g}\n"
        f"  P1 raw        : {raw_p1:.4g}  --> display {cfg.CPR_RESEARCH_VALID_RANGE[0]}\n"
        f"  P99 raw       : {raw_p99:.4g}  --> display {cfg.CPR_RESEARCH_VALID_RANGE[1]}\n"
        f"  Denom~0 masked: {n_invalid_raw:,}"
    )
    ice_label      = "  CPR > 1.0     : {frac:.2f}%  (upper half of log-rescaled range;\n" \
                     "                  NOT directly comparable to default CPR>1 ice threshold)"
    ice_line       = ice_label.format(frac=frac_gt1)
    power_a_label  = "HH_power"
    power_b_label  = "VV_power"
else:
    formula_lines  = (
        "  Formula       : CPR = mean(|SC|^2) / mean(|OC|^2)\n"
        "                  SC = S_HH - S_VV + 2j*(S_HV+S_VH)/2\n"
        "                  OC = S_HH + S_VV"
    )
    reference_line = "  Reference     : Putrevu et al. (2023) JGR Planets"
    inputs_line    = (
        f"  Inputs        : HH  {cfg.SLI_PATHS['HH'].name}\n"
        f"                  HV  {cfg.SLI_PATHS['HV'].name}\n"
        f"                  VH  {cfg.SLI_PATHS['VH'].name}\n"
        f"                  VV  {cfg.SLI_PATHS['VV'].name}"
    )
    rescale_section = ""
    ice_line        = f"  CPR > 1 (ice) : {frac_gt1:.2f}%"
    power_a_label   = "SC_power"
    power_b_label   = "OC_power"

log.info(f"""
  -------------------------------------------------------
  Chandrayaan-2 DFSAR Full-Pol SLI -> CPR Pipeline
  Mode          : {"RESEARCH (published mu_c formula)" if RESEARCH_MODE else "DEFAULT (exact coherent SC/OC)"}
  -------------------------------------------------------
  Scene         : {cfg.SCENE}  ({cfg._DATE})
{formula_lines}
{reference_line}

  INPUT RASTERS
  -------------------------------------------------------
  Dimensions    : {W} x {H}  (range x azimuth)
  Bands         : 2 per file (Band1=Real, Band2=Imag)
  Dtype         : float32 (complex SLC, I/Q)
{inputs_line}

  PROCESSING
  -------------------------------------------------------
  Multilook     : {cfg.MULTILOOK_WINDOW}  az x rg  = {cfg.MULTILOOK_WINDOW[0]*cfg.MULTILOOK_WINDOW[1]} looks
  Epsilon       : {cfg.EPSILON}
{rescale_section}
  CPR STATISTICS ({"display, log-rescaled" if RESEARCH_MODE else "physical ratio"})
  -------------------------------------------------------
  Min           : {cpr_stats['min']:.6f}
  Max           : {cpr_stats['max']:.6f}
  Mean          : {cpr_stats['mean']:.6f}
  Median        : {cpr_stats['median']:.6f}
  Std           : {cpr_stats['std']:.6f}
  Valid pixels  : {n_valid:,} / {n_total:,}  ({100*n_valid/n_total:.2f}%)
{ice_line}
  NoData        : {n_nodata:,}
  NaN / Inf     : {n_nan:,} / {n_inf:,}
{"" if RESEARCH_MODE else chr(10) + "  EXPECTED RANGE (Putrevu et al. 2023)" + chr(10) + "  -------------------------------------------------------" + chr(10) + "  Bare regolith : 0.05 - 0.5" + chr(10) + "  Ice candidates: > 1.0" + chr(10) + "  Official mosaic median: ~0.21"}
  OUTPUT FILES
  -------------------------------------------------------
  CPR GeoTIFF   : {out_tif}
  {power_a_label} : {prev_dir / f"{power_a_label}.png"}
  {power_b_label} : {prev_dir / f"{power_b_label}.png"}
  CPR preview   : {prev_dir / "CPR.png"}
  CPR histogram : {prev_dir / "CPR_histogram.png"}
  Log file      : {cfg.LOG_DIR / "cpr_pipeline.log"}

  TOTAL RUNTIME : {pipeline_timer}
  -------------------------------------------------------
""")

log.info("Pipeline completed successfully.")
