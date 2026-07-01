"""
Chandrayaan-2 DFSAR Full-Pol GRI -- Circular Polarization Ratio (CPR)
======================================================================

DIAGNOSTIC REPORT (run once; findings embedded as comments)
------------------------------------------------------------
1. Image dimensions  : 194 samples (range) x 14350 lines (azimuth)
   -> THIN-STRIP BUG: matplotlib auto-fit a 194x14350 array into a square
      figure, making it look like a 1-pixel sliver.  Fix: aspect='auto'
      or figure width proportional to image columns.

2. Data type         : uint16 (UnsignedLSB2) -- these are AMPLITUDE DN,
                       NOT intensity/power.
   -> FORMULA BUG: the previous code applied sqrt(HH*VV) treating DN as
      power.  Amplitude must be squared first to obtain intensity:
      sigma0 ~ DN^2 / K^2   (K = calibration_constant = 70.308868)

3. CPR formula used  : (HH + VV + 2*sqrt(HH*VV)) / (HH + VV - 2*sqrt(HH*VV))
   -> When DN are amplitude, sqrt(sigma_HH * sigma_VV) = HH*VV/K^2, so
      the denominator simplifies to (HH - VV)^2 / K^2.
      For distributed scatterers (most of the lunar surface) HH ~ VV,
      making the denominator ~ 0 and CPR ~ +inf for the majority of pixels.

4. CORRECT formula for full-pol data (using cross-pol channels):
      CPR = (sigma_HH + sigma_VV + 2*(sigma_HV + sigma_VH))
            / (sigma_HH + sigma_VV)
   This is always >= 1, never blows up, and is physically motivated:
   it is the ratio of same-sense circular backscatter to opposite-sense
   circular backscatter under the random-phase assumption
   (Cloude & Pottier 1996; O'Brien & Byrne 2022).
   The K^2 calibration constant cancels in the ratio so raw amplitude
   DN^2 can be used directly.

Calibration constant K = 70.308868 (from XML metadata).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "data", "calibrated", "20251025")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PFX = "ch2_sar_ncxl_20251025t211236510_d_gri_xx_fp"
PATHS = {
    "HH": os.path.join(DATA_DIR, f"{PFX}_hh_d18.tif"),
    "HV": os.path.join(DATA_DIR, f"{PFX}_hv_d18.tif"),
    "VH": os.path.join(DATA_DIR, f"{PFX}_vh_d18.tif"),
    "VV": os.path.join(DATA_DIR, f"{PFX}_vv_d18.tif"),
}
NODATA = -9999.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def read_band(path):
    with rasterio.open(path) as src:
        arr  = src.read(1).astype(np.float64)
        meta = {
            "width":     src.width,
            "height":    src.height,
            "crs":       src.crs,
            "count":     src.count,
            "dtype":     src.dtypes[0],
            "nodata":    src.nodata,
            "res":       src.res,
            "transform": src.transform,
        }
    return arr, meta


def print_metadata(name, meta):
    print(f"\n  [{name}]")
    print(f"    Width x Height : {meta['width']} x {meta['height']}")
    print(f"    CRS            : {meta['crs']}")
    print(f"    Bands          : {meta['count']}")
    print(f"    Dtype          : {meta['dtype']}")
    print(f"    NoData         : {meta['nodata']}")
    print(f"    Resolution     : {meta['res']}")
    print(f"    Transform      :\n      {meta['transform']}")


def print_stats(name, arr):
    valid = arr[np.isfinite(arr) & (arr > 0)]
    print(f"\n  [{name}]")
    print(f"    Min  (non-zero valid) : {valid.min():.4f}")
    print(f"    Max                   : {valid.max():.4f}")
    print(f"    Mean                  : {valid.mean():.4f}")
    print(f"    Std                   : {valid.std():.4f}")
    print(f"    NaN  count            : {np.isnan(arr).sum()}")
    print(f"    Inf  count            : {np.isinf(arr).sum()}")
    print(f"    Neg  count            : {(arr < 0).sum()}")
    print(f"    Zero count            : {(arr == 0).sum()}")


def pct_stretch(arr, lo=2, hi=98):
    valid = arr[np.isfinite(arr)]
    return tuple(np.percentile(valid, [lo, hi]))


def show_band(ax, arr, title, cmap="gray", lo=2, hi=98):
    disp = arr.copy()
    disp[~np.isfinite(disp)] = np.nan
    vmin, vmax = pct_stretch(disp, lo, hi)
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="none")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Range (samples)")
    ax.set_ylabel("Azimuth (lines)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# 1. Read
# ---------------------------------------------------------------------------
print_section("1. READING ALL POLARIZATION BANDS")
bands, metas = {}, {}
profile = None
for pol, path in PATHS.items():
    arr, meta = read_band(path)
    bands[pol] = arr
    metas[pol] = meta
    print(f"  Loaded {pol}: shape {arr.shape}  dtype {meta['dtype']}")
    if profile is None:
        with rasterio.open(path) as src:
            profile = src.profile.copy()

# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------
print_section("2. METADATA")
for pol in PATHS:
    print_metadata(pol, metas[pol])

# ---------------------------------------------------------------------------
# 3. Raw amplitude statistics
# ---------------------------------------------------------------------------
print_section("3. RAW AMPLITUDE DN STATISTICS")
for pol in PATHS:
    print_stats(pol, bands[pol])

# ---------------------------------------------------------------------------
# 4. Data type determination
# ---------------------------------------------------------------------------
print_section("4. DATA TYPE DETERMINATION")
print("""
  Source   : XML metadata (d_gri_xx_fp_xx_d18.xml)
  DataType : UnsignedLSB2 (uint16)
  Values   : 0 -- 65535 (HH max ~13865, VV max ~11890)
  Units    : AMPLITUDE Digital Numbers (DN)
             -- NOT intensity, NOT dB, NOT calibrated sigma0

  To convert to intensity (sigma0 proportional):
      sigma0_proportional = DN^2 / K^2     K = 70.308868 (calibration constant)
  For a ratio (CPR), K^2 cancels:
      sigma0 is proportional to DN^2
""")

# ---------------------------------------------------------------------------
# 5. Consistency check
# ---------------------------------------------------------------------------
print_section("5. CONSISTENCY CHECK (dimensions, CRS, transform)")
ref = metas["HH"]
all_ok = True
for pol in ["HV", "VH", "VV"]:
    m = metas[pol]
    dim_ok = (m["width"] == ref["width"]) and (m["height"] == ref["height"])
    crs_ok = (m["crs"] == ref["crs"])
    tf_ok  = (m["transform"] == ref["transform"])
    status = "OK" if (dim_ok and crs_ok and tf_ok) else "MISMATCH"
    print(f"  HH vs {pol}: dims={dim_ok}  CRS={crs_ok}  transform={tf_ok}  --> {status}")
    if status != "OK":
        all_ok = False
print(f"\n  All bands consistent: {all_ok}")

# ---------------------------------------------------------------------------
# 6. Invalid value detection
# ---------------------------------------------------------------------------
print_section("6. INVALID VALUE DETECTION")
for pol in PATHS:
    arr = bands[pol]
    print(f"  {pol}: NaN={np.isnan(arr).sum()}  Inf={np.isinf(arr).sum()}  "
          f"Neg={(arr<0).sum()}  Zero={(arr==0).sum()}")

# ---------------------------------------------------------------------------
# 7. Convert amplitude DN -> intensity (sigma proportional)
# ---------------------------------------------------------------------------
print_section("7. AMPLITUDE -> INTENSITY (squaring DN)")
print("  sigma proportional = DN^2  (K^2 cancels in CPR ratio)")
I = {pol: bands[pol] ** 2 for pol in PATHS}

# Build a zero-pixel mask (where any channel is zero = invalid)
zero_mask = (bands["HH"] == 0) | (bands["VV"] == 0)
print(f"  Zero/invalid pixels: {zero_mask.sum()}")

print_section("8. INTENSITY STATISTICS (sigma proportional = DN^2)")
for pol in PATHS:
    arr = I[pol].copy()
    arr[zero_mask] = np.nan
    print_stats(pol, arr)

# ---------------------------------------------------------------------------
# 9. DIAGNOSE the old formula
# ---------------------------------------------------------------------------
print_section("9. DIAGNOSIS OF ORIGINAL CPR FORMULA")
print("""
  Formula used:
    CPR = (HH + VV + 2*sqrt(HH*VV)) / (HH + VV - 2*sqrt(HH*VV))

  Problem 1 (wrong input units):
    HH and VV were amplitude DN, not intensity.
    Correct intensity = DN^2.  When you substitute:
      numerator   = HH^2 + VV^2 + 2*HH*VV = (HH + VV)^2
      denominator = HH^2 + VV^2 - 2*HH*VV = (HH - VV)^2
    So the formula collapses to (HH+VV)^2 / (HH-VV)^2.

  Problem 2 (denominator singularity):
    For distributed scatterers (most of the lunar surface) HH ~ VV,
    so (HH - VV) ~ 0 and CPR -> +inf.
    This explains the median CPR ~1195 and max CPR ~1.5e8.

  Problem 3 (missing cross-pol):
    The formula ignores HV/VH entirely.  For full-pol data the
    cross-pol term is essential; it prevents the denominator singularity.
""")

# ---------------------------------------------------------------------------
# 10. CORRECT CPR formula (full-pol, cross-pol included)
# ---------------------------------------------------------------------------
print_section("10. COMPUTING CORRECT CPR")
print("""
  CPR = (sigma_HH + sigma_VV + 2*(sigma_HV + sigma_VH))
        / (sigma_HH + sigma_VV)

  Reference: Cloude & Pottier (1996), O'Brien & Byrne (2022)
  Physical meaning:
    Numerator  ~ same-sense circular backscatter (SC)  x4
    Denominator~ opposite-sense circular backscatter (OC) x4
    CPR > 1 indicates enhanced same-sense return (ice / volume scattering)
    CPR ~ 1 for specular / smooth surface
""")

sigma_HH = I["HH"].copy()
sigma_VV = I["VV"].copy()
sigma_HV = I["HV"].copy()
sigma_VH = I["VH"].copy()

eps = 1e-10
numerator   = sigma_HH + sigma_VV + 2.0 * (sigma_HV + sigma_VH)
denominator = sigma_HH + sigma_VV + eps
CPR = (numerator / denominator).astype(np.float32)

# Mask zero pixels
CPR[zero_mask] = NODATA

# ---------------------------------------------------------------------------
# 11. CPR statistics
# ---------------------------------------------------------------------------
print_section("11. CPR STATISTICS")
valid_cpr = CPR[CPR != NODATA]
print(f"  Min    : {valid_cpr.min():.6f}")
print(f"  Max    : {valid_cpr.max():.6f}")
print(f"  Mean   : {valid_cpr.mean():.6f}")
print(f"  Std    : {valid_cpr.std():.6f}")
print(f"  Median : {np.median(valid_cpr):.6f}")
print(f"\n  Values > 1.0 (potential ice candidate): "
      f"{(valid_cpr > 1.0).sum()} / {valid_cpr.size} pixels "
      f"({100*(valid_cpr>1.0).mean():.1f}%)")

# ---------------------------------------------------------------------------
# 12. Save CPR GeoTIFF
# ---------------------------------------------------------------------------
print_section("12. SAVING CPR.tif")
profile.update(dtype="float32", count=1, nodata=NODATA)
cpr_tif = os.path.join(OUT_DIR, "CPR.tif")
with rasterio.open(cpr_tif, "w", **profile) as dst:
    dst.write(CPR, 1)
print(f"  Saved: {cpr_tif}")

# ---------------------------------------------------------------------------
# 13. Visualise HH, VV, CPR
#     FIX: image is 194 wide x 14350 tall (aspect ~74:1)
#          use aspect='auto' and a tall narrow figure layout
# ---------------------------------------------------------------------------
print_section("13. GENERATING CPR.png")

fig, axes = plt.subplots(1, 3, figsize=(10, 20))

cpr_display = CPR.copy().astype(np.float64)
cpr_display[CPR == NODATA] = np.nan

show_band(axes[0], bands["HH"].astype(np.float64),
          "HH Amplitude (DN)\n194 x 14350 px", cmap="gray")
show_band(axes[1], bands["VV"].astype(np.float64),
          "VV Amplitude (DN)\n194 x 14350 px", cmap="gray")
show_band(axes[2], cpr_display,
          "CPR  (SC / OC)\n= (HH+VV+2(HV+VH)) / (HH+VV)", cmap="viridis")

fig.suptitle(
    "Ch-2 DFSAR Full-Pol GRI -- 2025-10-25\n"
    "L-band, incidence 26 deg, South Pole region",
    fontsize=10
)
plt.tight_layout()
cpr_png = os.path.join(OUT_DIR, "CPR.png")
plt.savefig(cpr_png, dpi=150, bbox_inches="tight")
print(f"  Saved: {cpr_png}")

# ---------------------------------------------------------------------------
# 14. Summary diagnostic report
# ---------------------------------------------------------------------------
print_section("14. DIAGNOSTIC SUMMARY")
print(f"""
  BUG 1 -- Thin vertical strip appearance
  ----------------------------------------
  Cause   : Image dimensions are 194 (range) x 14350 (azimuth).
            Aspect ratio = 73.97:1.
            matplotlib's default imshow with a square (10x10) figure
            compresses 14350 azimuth lines into ~10 inches while 194
            range samples occupy only ~0.14 inches -- it appears as a
            1-pixel-wide white strip against a blank background.
  Fix     : aspect='auto' in imshow + tall figure (1x3 subplots, 10x20 inches).

  BUG 2 -- Colorbar saturating at ~50,000 (physically unrealistic)
  ----------------------------------------------------------------
  Cause   : Two compounding errors:
    (a) DN are amplitude, not intensity.  The formula requires sigma0
        (power proportional to DN^2), so the missing square inflated
        CPR by making sqrt(HH*VV) = HH*VV/K^2 instead of HH*VV.
    (b) Denominator = (HH - VV)^2 -> 0 for typical terrain where
        HH ~ VV, causing CPR -> +inf for most pixels.
  Fix     : Square DNs to get intensity, then use the cross-pol formula.

  BUG 3 -- Formula inappropriate for full-pol data
  -------------------------------------------------
  Old formula: (sigma_HH + sigma_VV +/- 2*sqrt(sigma_HH*sigma_VV))
               -- valid only for same-sense/opposite-sense decomposition
               -- singular when HH == VV
  Correct    : (sigma_HH + sigma_VV + 2*(sigma_HV+sigma_VH))
               / (sigma_HH + sigma_VV)
               -- uses all available polarizations
               -- always >= 1, never singular
               -- matches O'Brien & Byrne (2022) for full-pol mode
""")
