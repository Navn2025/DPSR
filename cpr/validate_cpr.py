"""
validate_cpr.py
===============
Compare our computed Calculated_CPR.tif with the official DFSAR CPR product.

Since the SLI-derived CPR lacks a geographic CRS (identity transform), a
pixel-by-pixel spatial comparison is not possible.  This script performs:

    1. Load both CPR products and report their metadata.
    2. Compute statistical summaries for each.
    3. Compare value distributions (histogram overlay, percentile table).
    4. If both products cover the same area (geo-referenced), resample and
       compute pixel-wise Pearson r, RMSE, and mean bias.
    5. Save a four-panel figure:
         top-left  : our CPR (downsampled)
         top-right : official CPR (downsampled)
         bottom-left  : overlaid histograms
         bottom-right : scatter plot (when spatial alignment is possible)

Run:
    python cpr/validate_cpr.py

Outputs are written to:
    cpr/outputs/previews/validation_*.png
    cpr/outputs/logs/validation.log
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import rasterio
from rasterio.warp import reproject, Resampling

import config as cfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUR_CPR_PATH = cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME
OFFIC_CPR_PATH = (
    Path(__file__).resolve().parents[1]
    / "DFSAR" / "data_pipeline" / "outputs" / "aligned" / "CPR.tif"
)
PREV_DIR = cfg.PREV_DIR
LOG_DIR  = cfg.LOG_DIR

for d in (PREV_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log = logging.getLogger("validate_cpr")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")

_fh = logging.FileHandler(LOG_DIR / "validation.log", mode="w", encoding="utf-8")
_fh.setFormatter(fmt)
log.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(fmt)
log.addHandler(_sh)


# ===========================================================================
# Helper functions
# ===========================================================================

def load_valid(path: Path, nodata_override: float = None) -> tuple:
    """Return (array_float32, profile, nodata_value)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nd = nodata_override if nodata_override is not None else src.nodata
    mask = np.full(arr.shape, False)
    if nd is not None:
        mask |= (arr == nd)
    mask |= ~np.isfinite(arr)
    mask |= (arr < 0)
    arr[mask] = np.nan
    return arr, profile, nd


def describe(name: str, arr: np.ndarray) -> dict:
    """Print percentile table for valid (non-NaN) pixels."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        log.warning(f"  {name}: NO VALID PIXELS")
        return {}
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {
        "name":   name,
        "count":  valid.size,
        "min":    float(valid.min()),
        "max":    float(valid.max()),
        "mean":   float(valid.mean()),
        "median": float(np.median(valid)),
        "std":    float(valid.std()),
        "pcts":   {p: float(np.percentile(valid, p)) for p in pcts},
    }
    log.info(f"\n  {name}:")
    log.info(f"    Valid pixels : {valid.size:,}")
    log.info(f"    Min / Max    : {stats['min']:.4f}  /  {stats['max']:.4f}")
    log.info(f"    Mean         : {stats['mean']:.4f}")
    log.info(f"    Median       : {stats['median']:.4f}")
    log.info(f"    Std          : {stats['std']:.4f}")
    log.info("    Percentiles  :")
    for p, v in stats["pcts"].items():
        log.info(f"      P{p:02d} = {v:.4f}")
    return stats


def downsample_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average 2-D array by integer factor."""
    h, w = arr.shape
    h2 = h // factor * factor
    w2 = w // factor * factor
    return arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


# ===========================================================================
# STEP 1  --  Load both products
# ===========================================================================
log.info("=" * 60)
log.info("  CPR VALIDATION")
log.info("=" * 60)

if not OUR_CPR_PATH.exists():
    log.error(f"Our CPR not found: {OUR_CPR_PATH}")
    log.error("Run main.py first to generate Calculated_CPR.tif")
    sys.exit(1)

if not OFFIC_CPR_PATH.exists():
    log.warning(f"Official CPR not found: {OFFIC_CPR_PATH}")
    log.warning("Proceeding with statistics-only comparison (no spatial alignment).")
    official_available = False
else:
    official_available = True

log.info(f"\nOur CPR    : {OUR_CPR_PATH}")
our_cpr, our_profile, our_nd = load_valid(OUR_CPR_PATH)
log.info(f"  Shape  : {our_cpr.shape}")
log.info(f"  CRS    : {our_profile.get('crs')}")
log.info(f"  Nodata : {our_nd}")

if official_available:
    log.info(f"\nOfficial   : {OFFIC_CPR_PATH}")
    off_cpr, off_profile, off_nd = load_valid(OFFIC_CPR_PATH)
    log.info(f"  Shape  : {off_cpr.shape}")
    log.info(f"  CRS    : {off_profile.get('crs')}")
    log.info(f"  Nodata : {off_nd}")


# ===========================================================================
# STEP 2  --  Per-product statistics
# ===========================================================================
log.info("\n" + "=" * 60)
log.info("  STATISTICAL SUMMARIES")
log.info("=" * 60)

our_stats = describe("Our CPR  (circular basis)", our_cpr)
if official_available:
    off_stats = describe("Official CPR (Putrevu 2023)", off_cpr)


# ===========================================================================
# STEP 3  --  Spatial pixel-wise comparison (only if both are georeferenced)
# ===========================================================================
can_compare_spatially = False

if official_available:
    our_crs   = our_profile.get("crs")
    off_crs   = off_profile.get("crs")

    if our_crs is None or str(our_crs) in ("", "None"):
        log.info("\nSpatial comparison skipped: our CPR has no CRS (SLI identity transform).")
        log.info("  -> Run georeferencing from geometry CSV before spatial comparison.")
    else:
        log.info("\nBoth products have CRS -- attempting spatial alignment and comparison.")
        can_compare_spatially = True

        # Reproject official CPR onto our grid
        dest = np.full(our_cpr.shape, np.nan, dtype=np.float32)
        with rasterio.open(OFFIC_CPR_PATH) as src:
            reproject(
                source      = rasterio.band(src, 1),
                destination = dest,
                src_transform  = src.transform,
                src_crs        = src.crs,
                dst_transform  = our_profile["transform"],
                dst_crs        = our_profile["crs"],
                resampling     = Resampling.bilinear,
                src_nodata     = off_nd,
                dst_nodata     = np.nan,
            )

        our_v = our_cpr.ravel()
        off_v = dest.ravel()

        valid_mask = np.isfinite(our_v) & np.isfinite(off_v)
        our_v = our_v[valid_mask]
        off_v = off_v[valid_mask]

        log.info(f"  Overlapping valid pixels : {valid_mask.sum():,}")

        if our_v.size > 10:
            r, pval = pearsonr(our_v[:500_000], off_v[:500_000])   # cap for speed
            rmse    = float(np.sqrt(np.mean((our_v - off_v) ** 2)))
            bias    = float(np.mean(our_v - off_v))
            log.info(f"  Pearson r    : {r:.4f}  (p={pval:.2e})")
            log.info(f"  RMSE         : {rmse:.4f}")
            log.info(f"  Mean bias    : {bias:.4f}  (ours - official)")
        else:
            log.warning("  Too few overlapping pixels for correlation.")
            can_compare_spatially = False


# ===========================================================================
# STEP 4  --  Figures
# ===========================================================================
log.info("\n" + "=" * 60)
log.info("  GENERATING FIGURES")
log.info("=" * 60)

DS = 50   # downsample factor for display

our_ds  = downsample_2d(np.where(np.isfinite(our_cpr), our_cpr, 0), DS)
our_ds[our_ds <= 0] = np.nan

# Percentile stretch
def pct_lim(arr, lo=2, hi=98):
    v = arr[np.isfinite(arr) & (arr > 0)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0, 1)

# -----------------------------------------------------------------------
# Figure 1 : side-by-side CPR maps
# -----------------------------------------------------------------------
n_panels = 2 if official_available else 1
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 10))
if n_panels == 1:
    axes = [axes]

vmin_o, vmax_o = pct_lim(our_ds)
im0 = axes[0].imshow(our_ds, cmap="viridis", vmin=vmin_o, vmax=vmax_o,
                     aspect="auto", interpolation="nearest")
cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cb0.set_label("CPR", fontsize=8)
axes[0].set_title(
    f"Our CPR (circular basis)\n"
    f"median={our_stats['median']:.3f}  mean={our_stats['mean']:.3f}",
    fontsize=9,
)
axes[0].set_xlabel("Range samples / " + str(DS), fontsize=7)
axes[0].set_ylabel("Azimuth lines / " + str(DS), fontsize=7)

if official_available:
    off_ds = downsample_2d(np.where(np.isfinite(off_cpr), off_cpr, 0), DS)
    off_ds[off_ds <= 0] = np.nan
    vmin_f, vmax_f = pct_lim(off_ds)
    im1 = axes[1].imshow(off_ds, cmap="viridis", vmin=vmin_f, vmax=vmax_f,
                         aspect="auto", interpolation="nearest")
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("CPR", fontsize=8)
    axes[1].set_title(
        f"Official CPR (Putrevu et al. 2023)\n"
        f"median={off_stats['median']:.3f}  mean={off_stats['mean']:.3f}",
        fontsize=9,
    )
    axes[1].set_xlabel("Range samples / " + str(DS), fontsize=7)

plt.tight_layout()
map_path = PREV_DIR / "validation_maps.png"
plt.savefig(map_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"  Saved: {map_path}")

# -----------------------------------------------------------------------
# Figure 2 : overlaid histograms
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

our_valid = our_cpr[np.isfinite(our_cpr) & (our_cpr > 0)]
p2, p98 = np.percentile(our_valid, [2, 98])
ax.hist(our_valid[(our_valid >= p2) & (our_valid <= p98)],
        bins=200, density=True, alpha=0.6, color="steelblue", label="Our CPR")

if official_available:
    off_valid = off_cpr[np.isfinite(off_cpr) & (off_cpr > 0)]
    op2, op98 = np.percentile(off_valid, [2, 98])
    ax.hist(off_valid[(off_valid >= op2) & (off_valid <= op98)],
            bins=200, density=True, alpha=0.5, color="tomato", label="Official CPR")

ax.axvline(1.0, color="green", lw=1.2, ls="--", label="CPR = 1.0")
ax.axvline(float(np.median(our_valid)), color="steelblue", lw=1.5, ls="-",
           label=f"Our median = {float(np.median(our_valid)):.3f}")
if official_available:
    ax.axvline(float(np.median(off_valid)), color="tomato", lw=1.5, ls="-",
               label=f"Official median = {float(np.median(off_valid)):.3f}")

ax.set_xlabel("CPR (dimensionless)", fontsize=11)
ax.set_ylabel("Probability density", fontsize=11)
ax.set_title("CPR Distribution Comparison  (2nd-98th pct range shown)", fontsize=10)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
plt.tight_layout()

hist_path = PREV_DIR / "validation_histogram.png"
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"  Saved: {hist_path}")

# -----------------------------------------------------------------------
# Figure 3 : scatter plot (if spatial comparison was possible)
# -----------------------------------------------------------------------
if can_compare_spatially and our_v.size > 100:
    n_scat = min(200_000, our_v.size)
    idx = np.random.choice(our_v.size, n_scat, replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(off_v[idx], our_v[idx], s=0.5, alpha=0.3, color="steelblue")
    lim = max(off_v.max(), our_v.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.0, label="1:1 line")
    ax.set_xlabel("Official CPR", fontsize=11)
    ax.set_ylabel("Our CPR", fontsize=11)
    ax.set_title(
        f"Scatter: Our vs Official CPR\n"
        f"r = {r:.4f}   RMSE = {rmse:.4f}   bias = {bias:+.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    scat_path = PREV_DIR / "validation_scatter.png"
    plt.savefig(scat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {scat_path}")

# ===========================================================================
# Summary
# ===========================================================================
log.info("\n" + "=" * 60)
log.info("  VALIDATION SUMMARY")
log.info("=" * 60)
log.info(f"  Our CPR   median : {our_stats.get('median', 'N/A'):.4f}")
if official_available:
    log.info(f"  Official  median : {off_stats.get('median', 'N/A'):.4f}")
    log.info(f"  Expected range   : 0.03 - 2.37 (Putrevu et al. 2023)")
    our_med = our_stats.get("median", 0.0)
    off_med = off_stats.get("median", 0.0)
    ratio   = our_med / off_med if off_med > 0 else float("inf")
    log.info(f"  Ratio our/official median : {ratio:.3f}")
    if 0.5 <= ratio <= 2.0:
        log.info("  ASSESSMENT: Median within 2x of official -> formula plausible.")
    else:
        log.warning("  ASSESSMENT: Median differs by more than 2x from official.")
        log.warning("  Possible causes: different scene geometry, look angle, multilook window,")
        log.warning("  or residual calibration difference.")
else:
    log.info("  Official CPR not available -- statistical comparison only.")

log.info("\nValidation complete.")
