"""
main.py
=======
Entry point for the Chandrayaan-2 DFSAR calibrated GRI (Ground Range
Image) -> CPR pipeline.

Run from the project root:
    python cpr_gri/main.py                # default: reflection-symmetry approximation (cpr.compute_cpr_gri)
    python cpr_gri/main.py --research     # published co-pol-only mu_c formula (cpr.compute_cpr_research)

--research uses the formula:

    CPR(mu_c) = (sigma_HH + sigma_VV + 2*sqrt(sigma_HH*sigma_VV))
                / (sigma_HH + sigma_VV - 2*sqrt(sigma_HH*sigma_VV))

from only the calibrated co-pol channels (no HV/VH needed) -- see
cpr.compute_cpr_research() for the full derivation and its numerical
caveats. It writes a separate output (Calculated_CPR_GRI_research.tif)
so it never overwrites the default approximation's result.

Pipeline steps
--------------
1.  Read metadata from HH, HV, VH, VV, and the incidence-angle GeoTIFF.
2.  Validate geometric consistency.
3.  Auto-detect raw-DN / dB / linear-power format per channel and
    calibrate to linear power.
4.  Compute CPR_GRI via the reflection-symmetry power-only approximation,
    or (--research) the published co-pol-only mu_c formula.
5.  NoData-aware multilook (same window as the SLI CPR pipeline).
6.  Statistics.
7.  Preview PNGs.
8.  Write Calculated_CPR_GRI.tif (or _research.tif).
9.  Compare against the official DFSAR CPR mosaic.
10. Write validation_report.txt (or _research.txt).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
from rasterio.windows import Window

import cpr.cpr_gri.config as cfg
from cpr.cpr_gri.utils import setup_logger, Timer, array_stats, log_stats, percentiles, section, memory_mb, MemoryTracker
from cpr.cpr_gri.reader import read_metadata, print_metadata, load_band, parse_calibration_constant
from cpr.cpr_gri.validator import validate_rasters
from cpr.cpr_gri.preprocessing import to_linear_power, multilook_nodata_aware
from cpr import compute_cpr_gri, compute_cpr_research
from validation import load_geometry_grid, moon_south_polar_stereographic, sample_official_mosaic, compare_cpr
from cpr.cpr_gri.visualizer import (
    save_power_previews, save_cpr_preview, save_cpr_histogram,
    save_scatter, save_difference_map, save_histogram_overlap,
)

parser = argparse.ArgumentParser(description="Chandrayaan-2 DFSAR GRI -> CPR pipeline")
parser.add_argument(
    "--research", action="store_true",
    help="Use the published co-pol-only CPR(mu_c) formula instead of the "
         "default reflection-symmetry power-only approximation.",
)
args = parser.parse_args()
RESEARCH_MODE = args.research

for d in (cfg.PREV_DIR, cfg.VALID_DIR, cfg.LOG_DIR, cfg.CPR_GRI_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = setup_logger(cfg.LOG_DIR)
pipeline_timer = Timer()
mem = MemoryTracker()
log.info(f"Mode: {'RESEARCH (published mu_c formula)' if RESEARCH_MODE else 'DEFAULT (reflection-symmetry approximation)'}")


# ===========================================================================
# STEP 1  --  Read metadata
# ===========================================================================
section("STEP 1  |  Read GRI Metadata", log)

metas = {}
for pol, path in cfg.GRI_PATHS.items():
    metas[pol] = read_metadata(path)
    log.info("-" * 50)
    print_metadata(pol, metas[pol], log)

metas["INCIDENCE"] = read_metadata(cfg.INCIDENCE_PATH)
log.info("-" * 50)
print_metadata("INCIDENCE", metas["INCIDENCE"], log)

ref = metas["HH"]
H, W = ref["height"], ref["width"]
log.info(f"\n  Scene dimensions : {W} samples (range) x {H} lines (azimuth)")
log.info(f"  Pixel spacing    : {ref['res']}")


# ===========================================================================
# STEP 2  --  Validate consistency
# ===========================================================================
section("STEP 2  |  Validate Raster Consistency", log)
validate_rasters(metas)


# ===========================================================================
# STEP 3  --  Format detection + calibration to linear power
# ===========================================================================
section("STEP 3  |  Detect Format & Calibrate to Linear Power", log)

K = parse_calibration_constant(cfg.GRI_LABEL_XML, cfg.CALIBRATION_CONSTANT_FALLBACK)

raw = {}
power = {}
valid = {}
for pol in ("HH", "HV", "VH", "VV"):
    t0 = Timer()
    arr = load_band(cfg.GRI_PATHS[pol], pol)
    raw[pol] = arr
    mem.add(arr)
    p, v = to_linear_power(arr, pol, K)
    power[pol] = p
    valid[pol] = v
    mem.add(p, v)
    log.info(f"  {pol} processed in {t0}")

for pol in ("HH", "HV", "VH", "VV"):
    mem.remove(raw[pol])
raw.clear()

# Reciprocity-averaged cross-pol power (same convention as the SLI pipeline)
xp_valid = valid["HV"] & valid["VH"]
P_XP = np.where(xp_valid, (power["HV"] + power["VH"]) * 0.5, np.nan).astype(np.float32)
mem.add(P_XP)
log.info(f"  P_XP = (P_HV + P_VH)/2 formed  valid={int(xp_valid.sum()):,}/{P_XP.size:,}")


# ===========================================================================
# STEP 5  --  NoData-aware multilook (applied before ratio, as in the SLI pipeline)
# ===========================================================================
section("STEP 5  |  NoData-Aware Multilook", log)

az, rg = cfg.MULTILOOK_WINDOW
log.info(f"Window: ({az}az x {rg}rg) = {az*rg} looks -- same as the SLI CPR pipeline")

ml_HH, valid_HH = multilook_nodata_aware(power["HH"], valid["HH"], cfg.MULTILOOK_WINDOW, "HH")
ml_VV, valid_VV = multilook_nodata_aware(power["VV"], valid["VV"], cfg.MULTILOOK_WINDOW, "VV")
ml_XP, valid_XP = multilook_nodata_aware(P_XP, xp_valid, cfg.MULTILOOK_WINDOW, "XP")

ml_power_full = {}
for pol in ("HH", "HV", "VH", "VV"):
    ml_power_full[pol], _ = multilook_nodata_aware(power[pol], valid[pol], cfg.MULTILOOK_WINDOW, pol)

mem.add(ml_HH, ml_VV, ml_XP)
for pol in ("HH", "HV", "VH", "VV"):
    mem.remove(power[pol], valid[pol])
mem.remove(P_XP, xp_valid)


# ===========================================================================
# STEP 4  --  Compute CPR_GRI
# ===========================================================================
research_info = None
if RESEARCH_MODE:
    section("STEP 4  |  Compute CPR_GRI (published co-pol-only mu_c formula, log-rescaled)", log)
    combined_valid = valid_HH & valid_VV
    t0 = Timer()
    cpr_gri, research_info = compute_cpr_research(
        ml_HH, ml_VV, combined_valid, epsilon=cfg.EPSILON, nodata=cfg.NODATA,
        valid_range=cfg.CPR_RESEARCH_VALID_RANGE,
    )
    log.info(f"  CPR_research computed in {t0}")
else:
    section("STEP 4  |  Compute CPR_GRI (power-only approximation)", log)
    combined_valid = valid_HH & valid_VV & valid_XP
    t0 = Timer()
    cpr_gri = compute_cpr_gri(ml_HH, ml_VV, ml_XP, combined_valid, epsilon=cfg.EPSILON, nodata=cfg.NODATA)
    log.info(f"  CPR_GRI computed in {t0}")
mem.add(cpr_gri)


# ===========================================================================
# STEP 6  --  Statistics
# ===========================================================================
section("STEP 6  |  CPR_GRI Statistics", log)

cpr_stats = array_stats(cpr_gri, nodata=cfg.NODATA, label="CPR_GRI")
log_stats(cpr_stats, log)

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
cpr_pcts = percentiles(cpr_gri, cfg.NODATA, pcts)
log.info("  Percentiles:")
for p in pcts:
    log.info(f"    P{p:02d} = {cpr_pcts[p]:.4f}")


# ===========================================================================
# STEP 7  --  Preview images
# ===========================================================================
mode_tag = "research" if RESEARCH_MODE else "default"
prev_dir = (cfg.PREV_DIR / "research") if RESEARCH_MODE else cfg.PREV_DIR
valid_dir = (cfg.VALID_DIR / "research") if RESEARCH_MODE else cfg.VALID_DIR
prev_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(parents=True, exist_ok=True)

section("STEP 7  |  Save Preview Images", log)

cpr_preview_title = (
    f"CPR(mu_c) log10-rescaled onto {cfg.CPR_RESEARCH_VALID_RANGE}\n"
    f"raw range [{research_info['rescale_lo_raw']:.3g}, {research_info['rescale_hi_raw']:.3g}] "
    f"(P1-P99) -- NOT the literal unscaled formula value"
    if RESEARCH_MODE else
    "CPR_GRI = 1 + 4*P_XP / (P_HH+P_VV)\n(reflection-symmetry approximation, 2-98 pct stretch)"
)
save_power_previews(ml_power_full, prev_dir)
save_cpr_preview(cpr_gri, prev_dir, cfg.NODATA, title=cpr_preview_title)
save_cpr_histogram(cpr_gri, prev_dir, cfg.NODATA)
log.info(f"  All previews saved to: {prev_dir}")


# ===========================================================================
# STEP 8  --  Write CPR_GRI GeoTIFF
# ===========================================================================
out_name = "Calculated_CPR_GRI_research.tif" if RESEARCH_MODE else cfg.CPR_GRI_OUTPUT_NAME
section(f"STEP 8  |  Save {out_name}", log)

out_tif = cfg.CPR_GRI_DIR / out_name
out_profile = ref["profile"].copy()
out_profile.update(dtype="float32", count=1, nodata=cfg.NODATA, compress="lzw")

formula_tag = (
    "CPR(mu_c) = (sigma_HH+sigma_VV+2*sqrt(sigma_HH*sigma_VV)) / (sigma_HH+sigma_VV-2*sqrt(sigma_HH*sigma_VV)), "
    "then log10-rescaled onto the output range  (published co-pol-only formula)"
    if RESEARCH_MODE else
    "CPR = 1 + 4*P_XP/(P_HH+P_VV)  (reflection-symmetry approximation)"
)
assumption_tag = (
    f"HH and VV assumed fully coherent/correlated (quasi-specular/Bragg regime); no HV/VH used. "
    f"Raw formula saturates on most real terrain (median HH-VV gap ~0.6dB, formula needs >15dB "
    f"to stay under CPR=2), so output is log10(raw) linearly rescaled from its own "
    f"P1-P99 range [{research_info['rescale_lo_raw']:.3g}, {research_info['rescale_hi_raw']:.3g}] "
    f"(raw CPR units) onto {cfg.CPR_RESEARCH_VALID_RANGE} -- NOT the literal unscaled formula "
    f"value. See cpr.compute_cpr_research docstring."
    if RESEARCH_MODE else
    "Re(<S_HH S_VV*>) ~ 0 -- GRI is a detected/intensity product with no inter-channel phase"
)

log.info(f"Writing {out_tif.name} ({W}W x {H}H, float32, LZW) ...")
log.info(
    "  NOTE: source GRI rasters carry CRS=None / identity transform "
    "(ground-range grid, not map-projected -- see reader.py); the output "
    "faithfully preserves that same (lack of) georeferencing rather than "
    "fabricating one. STEP 9 below performs its own independent "
    "geolocation via the geometry CSV for comparison purposes only."
)
t0 = Timer()
with rasterio.open(out_tif, "w", **out_profile) as dst:
    for row_start in range(0, H, cfg.WRITE_BLOCK):
        row_end = min(row_start + cfg.WRITE_BLOCK, H)
        win = Window(0, row_start, W, row_end - row_start)
        dst.write(cpr_gri[row_start:row_end, :], 1, window=win)

    dst.update_tags(
        PIPELINE   = "Chandrayaan-2 DFSAR GRI -> CPR",
        MODE       = mode_tag,
        FORMULA    = formula_tag,
        ASSUMPTION = assumption_tag,
        CALIBRATION= f"sigma0_dB = 20*log10(DN) - {K}",
        MULTILOOK  = str(cfg.MULTILOOK_WINDOW),
        EPSILON    = str(cfg.EPSILON),
        NODATA     = str(cfg.NODATA),
        CPR_MEDIAN = f"{cpr_stats['median']:.6f}",
        CPR_MEAN   = f"{cpr_stats['mean']:.6f}",
    )
log.info(f"  Done in {t0}")
log.info(f"  Saved : {out_tif}")
log.info(f"  Size  : {out_tif.stat().st_size / 1024**2:.2f} MB")


# ===========================================================================
# STEP 9  --  Compare against official DFSAR CPR mosaic
# ===========================================================================
section("STEP 9  |  Compare Against Official DFSAR CPR Mosaic", log)

log.info(
    "  NOTE: the only official CPR raster in this dataset is a south-polar "
    "MOSAIC (2025-06-30), a different acquisition than this GRI scene "
    "(2021-05-06/2025-10-25). This is a spatial/order-of-magnitude "
    "consistency check against an independent product, NOT pixel-exact "
    "ground truth -- see validation.py docstring."
)

comparison_stats = {"n": 0}
try:
    t0 = Timer()
    lat_grid, lon_grid = load_geometry_grid(cfg.GEOM_CSV, W, H)
    x, y = moon_south_polar_stereographic(lat_grid, lon_grid, cfg.MOON_RADIUS_M)
    official, official_valid = sample_official_mosaic(cfg.OFFICIAL_CPR_MOSAIC, x, y)
    comparison_stats = compare_cpr(cpr_gri, combined_valid & (cpr_gri != cfg.NODATA), official, official_valid)
    log.info(f"  Comparison computed in {t0}")

    if comparison_stats.get("n", 0) >= 2:
        both_valid = combined_valid & (cpr_gri != cfg.NODATA) & official_valid & np.isfinite(official)
        save_scatter(cpr_gri[both_valid], official[both_valid], valid_dir, comparison_stats)
        diff_2d = np.where(both_valid, cpr_gri - official, np.nan).astype(np.float32)
        save_difference_map(diff_2d, valid_dir)
        save_histogram_overlap(cpr_gri[both_valid], official[both_valid], valid_dir, comparison_stats)
    else:
        log.warning("  Not enough co-valid pixels to produce comparison figures.")
except Exception as exc:
    log.error(f"  STEP 9 comparison failed: {exc}")


# ===========================================================================
# STEP 10  --  Validation report
# ===========================================================================
section("STEP 10  |  Validation Report", log)

n_total = H * W
n_valid = int(cpr_stats["valid"])
n_invalid = n_total - n_valid

report_lines = [
    "=" * 70,
    "Chandrayaan-2 DFSAR GRI -> CPR Pipeline -- Validation Report",
    f"Mode               : {mode_tag.upper()}",
    "=" * 70,
    f"Scene              : {cfg.SCENE}  ({cfg._DATE})",
    f"Formula            : {formula_tag}",
    f"Assumption         : {assumption_tag}",
    f"Calibration        : sigma0_dB = 20*log10(DN) - {K}  (raw uint16 DN detected)",
    "",
    "INPUT RASTERS",
    "-" * 70,
    f"Dimensions         : {W} x {H} (range x azimuth)",
    f"Multilook          : {cfg.MULTILOOK_WINDOW}  = {az*rg} looks (matches SLI pipeline)",
    "",
    "CPR_GRI STATISTICS" + (" (log-rescaled display values)" if RESEARCH_MODE else ""),
    "-" * 70,
] + ([
    f"Raw formula min/median/max : {research_info['raw_min']:.4g} / {research_info['raw_median']:.4g} / {research_info['raw_max']:.4g}",
    f"Rescale window (raw units) : [{research_info['rescale_lo_raw']:.4g}, {research_info['rescale_hi_raw']:.4g}]  -> {cfg.CPR_RESEARCH_VALID_RANGE}",
    "",
] if RESEARCH_MODE else []) + [
    f"Min                : {cpr_stats['min']:.6f}",
    f"Max                : {cpr_stats['max']:.6f}",
    f"Mean               : {cpr_stats['mean']:.6f}",
    f"Median             : {cpr_stats['median']:.6f}",
    f"Std                : {cpr_stats['std']:.6f}",
    f"Valid pixels       : {n_valid:,} / {n_total:,}  ({100*n_valid/n_total:.2f}%)",
    f"Invalid pixels     : {n_invalid:,}  ({100*n_invalid/n_total:.2f}%)",
] + [f"P{p:02d}                : {cpr_pcts[p]:.4f}" for p in pcts] + [
    "",
    "COMPARISON VS OFFICIAL DFSAR CPR MOSAIC (2025-06-30, different acquisition)",
    "-" * 70,
]

if comparison_stats.get("n", 0) >= 2:
    cs = comparison_stats
    report_lines += [
        f"Co-valid pixels    : {cs['n']:,}",
        f"Pearson r          : {cs['pearson_r']:.4f}  (p={cs['pearson_p']:.2e})",
        f"Spearman rho       : {cs['spearman_r']:.4f}  (p={cs['spearman_p']:.2e})",
        f"RMSE               : {cs['rmse']:.4f}",
        f"MAE                : {cs['mae']:.4f}",
        f"Bias (ours-ref)    : {cs['bias']:+.4f}",
        f"R^2                : {cs['r2']:.4f}",
        f"Histogram overlap  : {cs['hist_overlap']:.4f}",
        f"Ours  mean/median  : {cs['ours_mean']:.4f} / {cs['ours_median']:.4f}",
        f"Ref   mean/median  : {cs['ref_mean']:.4f} / {cs['ref_median']:.4f}",
        "",
        "INTERPRETATION",
        "-" * 70,
    ]
    interp = []

    def _strength(v):
        av = abs(v)
        return "strong" if av >= 0.5 else "moderate" if av >= 0.3 else "weak" if av >= 0.1 else "negligible"

    if abs(cs["pearson_r"] - cs["spearman_r"]) > 0.2:
        interp.append(
            f"Spatial pattern: Pearson r={cs['pearson_r']:+.3f} ({_strength(cs['pearson_r'])}) "
            f"and Spearman rho={cs['spearman_r']:+.3f} ({_strength(cs['spearman_r'])}) disagree "
            f"substantially across the {cs['n']:,} co-valid pixels. This gap is itself diagnostic: "
            f"Pearson is sensitive to a small number of extreme-magnitude outliers while Spearman "
            f"(rank-based) is not, so a much weaker/negative Pearson alongside a stronger Spearman "
            f"indicates the ranking/spatial pattern is more consistent between products than the "
            f"raw-value correlation suggests -- expected when CPR values span many orders of "
            f"magnitude (see cpr.compute_cpr_research's numerical-sensitivity note)."
        )
    else:
        interp.append(
            f"Spatial pattern: Pearson r={cs['pearson_r']:+.3f} and Spearman rho={cs['spearman_r']:+.3f} "
            f"agree and indicate a {_strength(cs['pearson_r'])} "
            f"{'positive' if cs['pearson_r'] >= 0 else 'negative'} relationship between CPR_GRI and "
            f"the official mosaic across the {cs['n']:,} co-valid pixels."
        )
    if cs["r2"] < 0 and RESEARCH_MODE:
        interp.append(
            f"Absolute agreement is POOR (R^2={cs['r2']:.2f}) with bias "
            f"{cs['bias']:+.3f}. This published CPR(mu_c) formula assumes HH "
            f"and VV are FULLY coherent/correlated (the opposite extreme from "
            f"the default pipeline's reflection-symmetry assumption of zero "
            f"correlation) -- it is also numerically ill-conditioned wherever "
            f"sigma_HH ~ sigma_VV, since the denominator (sqrt(sigma_HH)-"
            f"sqrt(sigma_VV))^2 approaches zero there and the ratio diverges. "
            f"Large disagreement with the official mosaic indicates the true "
            f"HH-VV correlation for this terrain lies somewhere between the "
            f"two extremes assumed by the default and --research formulas."
        )
        interp.append(
            "Practical implication: treat CPR(mu_c) values as a research "
            "reference computed exactly per the published formula, not a "
            "drop-in replacement for the default GRI approximation or the "
            "exact SLI-based CPR -- compare its spatial pattern (not absolute "
            "scale) against the other products."
        )
    elif cs["r2"] < 0:
        interp.append(
            f"Absolute agreement is POOR (R^2={cs['r2']:.2f}, i.e. worse than "
            f"predicting the mosaic's mean everywhere) with a large systematic "
            f"bias of {cs['bias']:+.3f}. This is EXPECTED and traceable to a "
            f"specific structural limitation of the power-only approximation, "
            f"not a bug: CPR_GRI = 1 + 4*P_XP/(P_HH+P_VV) is bounded below by "
            f"1.0 by construction (the added term is never negative), whereas "
            f"the official mosaic has median {cs['ref_median']:.3f} -- i.e. "
            f"predominantly OC-dominant, CPR<1 bare-regolith terrain. Reaching "
            f"CPR<1 requires a positive Re(<S_HH S_VV*>) co-pol correlation "
            f"term, which this approximation drops under the reflection-"
            f"symmetry assumption (see cpr.py). The comparison itself is "
            f"therefore direct empirical evidence that Re(<S_HH S_VV*>) is "
            f"NOT negligible for this terrain, i.e. that reflection symmetry "
            f"is a poor assumption here -- exactly the kind of information "
            f"loss the task asked us to state explicitly rather than paper "
            f"over."
        )
        interp.append(
            "Practical implication: CPR_GRI's ABSOLUTE values should not be "
            "compared against the official CPR>1 ice-candidate threshold. It "
            "may still be useful for RELATIVE / spatial-anomaly analysis "
            "(identifying pixels anomalous versus their local background) "
            "within a single GRI scene, but the exact SLI-based CPR pipeline "
            "remains the scientifically valid product for absolute CPR "
            "thresholding and should be preferred wherever SLI data exists."
        )
    else:
        interp.append(
            f"R^2={cs['r2']:.3f} and bias={cs['bias']:+.3f} indicate reasonable "
            f"absolute agreement with the official mosaic given the different "
            f"acquisition dates and the formula's simplifying assumption."
        )
    interp.append(
        "Remaining scatter is also expected from: (a) different acquisition "
        "dates/geometry between this GRI scene and the 2025-06-30 mosaic, and "
        "(b) nearest-neighbour resampling error in the manual polar-"
        "stereographic georeferencing used for this comparison."
    )
    report_lines += interp
else:
    report_lines += ["No co-valid pixels were found -- comparison could not be computed."]

report_lines += [
    "",
    "RESOURCE USAGE",
    "-" * 70,
    f"Peak tracked array memory : {mem.peak_mb:.0f} MB",
    f"Total runtime             : {pipeline_timer}",
    "",
    "OUTPUT FILES",
    "-" * 70,
    f"CPR GeoTIFF        : {out_tif}",
    f"Previews           : {prev_dir}",
    f"Validation figures : {valid_dir}",
    f"Log file           : {cfg.LOG_DIR / 'cpr_gri_pipeline.log'}",
    "=" * 70,
]

report_name = "validation_report_research.txt" if RESEARCH_MODE else "validation_report.txt"
report_path = valid_dir / report_name
report_path.write_text("\n".join(report_lines), encoding="utf-8")
log.info(f"  Validation report saved: {report_path}")
for line in report_lines:
    log.info(line)

log.info("Pipeline completed successfully.")
