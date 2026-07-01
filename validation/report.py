"""
report.py  --  Generate validation_report.txt from computed metrics.
"""
import datetime
import logging
from pathlib import Path
from typing import Dict

import numpy as np

log = logging.getLogger("validation.report")


_INTERPRETATION_THRESHOLDS = {
    "pearson_r": {
        0.90: "Excellent agreement (r >= 0.90)",
        0.70: "Good agreement (0.70 <= r < 0.90)",
        0.50: "Moderate agreement (0.50 <= r < 0.70)",
        0.0:  "Poor agreement (r < 0.50)",
    },
    "ssim": {
        0.80: "High structural similarity (SSIM >= 0.80)",
        0.60: "Moderate structural similarity (0.60 <= SSIM < 0.80)",
        0.0:  "Low structural similarity (SSIM < 0.60)",
    },
    "hist_intersection": {
        0.80: "Very similar CPR distributions (HI >= 0.80)",
        0.60: "Moderately similar distributions (0.60 <= HI < 0.80)",
        0.0:  "Dissimilar distributions (HI < 0.60)",
    },
}


def _interpret(metric_name: str, value: float) -> str:
    thresholds = _INTERPRETATION_THRESHOLDS.get(metric_name, {})
    for thresh, desc in sorted(thresholds.items(), reverse=True):
        if value >= thresh:
            return desc
    return "Unknown"


def generate_report(
    metrics:        Dict,
    stats_calc:     Dict,
    stats_offic:    Dict,
    overlap_info:   Dict,
    out_path:       Path,
    calc_path:      Path,
    offic_path:     Path,
    georef_path:    Path,
) -> None:
    lines = []

    def w(s: str = ""):
        lines.append(s)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w("=" * 72)
    w("  CHANDRAYAAN-2 DFSAR CPR VALIDATION REPORT")
    w("=" * 72)
    w(f"  Generated : {now}")
    w(f"  Calculated CPR    : {calc_path.name}")
    w(f"  Georeferenced CPR : {georef_path.name}")
    w(f"  Official CPR      : {offic_path.name}")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  1. GEOREFERENCING")
    w("-" * 72)
    w(f"  Method          : GCP reprojection (rasterio.warp.reproject)")
    w(f"  GCP source      : DFSAR SLI geometry CSV (8521 az x 9 rng tie pts)")
    w(f"  GCP subset used : every 10th azimuth tie row x 9 range cols = 7668 GCPs")
    w(f"  Src CRS         : Moon Geographic (sphere R=1737400m)")
    w(f"  Dst CRS         : Moon South Pole Stereographic (lat_origin=-90)")
    w(f"  Target grid     : {overlap_info.get('target_shape','?')}")
    w(f"  Target res      : 20 m/pixel")
    w(f"  Target bounds   : ±151680 m (all four sides)")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  2. RASTER STATISTICS")
    w("-" * 72)
    for label, st in (("Calculated CPR (georef.)", stats_calc),
                      ("Official CPR (Putrevu 2023)", stats_offic)):
        w(f"\n  --- {label} ---")
        for k in ("valid", "min", "max", "mean", "median", "std",
                  "p01", "p05", "p25", "p50", "p75", "p95", "p99"):
            if k in st:
                v = st[k]
                fmt = f"{v:,.6f}" if isinstance(v, float) else f"{v:,}"
                w(f"    {k:10s}: {fmt}")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  3. OVERLAP STATISTICS")
    w("-" * 72)
    n   = metrics.get("n_pixels", 0)
    tot = overlap_info.get("total_pixels", 1)
    w(f"  Valid overlap pixels     : {n:,}")
    w(f"  Target grid total pixels : {tot:,}")
    w(f"  Overlap fraction         : {100*n/tot:.3f}%")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  4. QUANTITATIVE METRICS")
    w("-" * 72)
    metric_rows = [
        ("Pearson r",             "pearson_r",          ".4f"),
        ("Pearson p-value",       "pearson_pval",       ".2e"),
        ("Spearman r",            "spearman_r",         ".4f"),
        ("Spearman p-value",      "spearman_pval",      ".2e"),
        ("RMSE",                  "rmse",               ".6f"),
        ("MAE",                   "mae",                ".6f"),
        ("Bias (calc - offic)",   "bias",               "+.6f"),
        ("R²",                    "r2",                 ".4f"),
        ("SSIM",                  "ssim",               ".4f"),
        ("Histogram intersection","hist_intersection",  ".4f"),
        ("Mutual information",    "mutual_information", ".4f"),
    ]
    for label, key, fmt in metric_rows:
        val = metrics.get(key, float("nan"))
        if np.isfinite(val):
            w(f"  {label:30s}: {val:{fmt}}")
        else:
            w(f"  {label:30s}: N/A")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  5. INTERPRETATION")
    w("-" * 72)
    for key in ("pearson_r", "ssim", "hist_intersection"):
        val  = metrics.get(key, float("nan"))
        if np.isfinite(val):
            desc = _interpret(key, val)
            w(f"  {key:30s}: {desc}")
    w()

    bias_val = metrics.get("bias", float("nan"))
    if np.isfinite(bias_val):
        if abs(bias_val) < 0.05:
            w("  Bias               : Negligible (|bias| < 0.05)")
        elif abs(bias_val) < 0.15:
            w("  Bias               : Small (|bias| < 0.15)")
        else:
            w("  Bias               : Substantial (|bias| >= 0.15)")
            if bias_val > 0:
                w("    -> Calculated CPR is systematically HIGHER than official.")
                w("       Possible causes: different look count, multilook window,")
                w("       processing date difference (different SAR pass).")
            else:
                w("    -> Calculated CPR is systematically LOWER than official.")
    w()

    r_val = metrics.get("pearson_r", float("nan"))
    if np.isfinite(r_val) and r_val >= 0.7:
        w("  OVERALL CONCLUSION: The calculated CPR is in good quantitative")
        w("  agreement with the official DFSAR CPR product. The circular-basis")
        w("  formula (Putrevu et al. 2023) produces physically consistent results.")
    elif np.isfinite(r_val):
        w("  OVERALL CONCLUSION: The calculated CPR shows moderate agreement.")
        w("  Remaining differences may be due to: (a) different SAR acquisition")
        w("  date and orbit geometry, (b) different multilook window, (c) partial")
        w("  georeferencing error near the scene edges.")
    w()

    # ------------------------------------------------------------------
    w("-" * 72)
    w("  6. REFERENCE")
    w("-" * 72)
    w("  Putrevu, D. et al. (2023). Chandrayaan-2 DFSAR Full Polarimetric")
    w("  observations of the Lunar South Pole. Journal of Geophysical Research:")
    w("  Planets. DOI: 10.1029/2023JE007745")
    w()
    w("=" * 72)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Report written: {out_path.name}")
