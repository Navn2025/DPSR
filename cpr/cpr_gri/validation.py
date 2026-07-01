"""
validation.py
==============
STEP 9 -- Compare Calculated_CPR_GRI.tif against the official DFSAR CPR
product.

Important caveat on "official product"
-----------------------------------------
No official CPR raster exists in this dataset for the exact 2021-05-06
Faustini (or 2025-10-25 south-pole) SLI/GRI pass used elsewhere in this
project. The only real official CPR raster available is a south-polar
MOSAIC product:

    DFSAR/.../ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif

acquired/composited 2025-06-30, in Moon 2000 South Polar Stereographic
projection (25 m/px). It covers the whole south-polar region and
therefore DOES spatially cover the Faustini scene, but it is a
DIFFERENT acquisition (different pass, possibly mosaicked from multiple
passes) -- so this comparison is a spatial / order-of-magnitude
consistency check against an independent product, not a pixel-exact
ground-truth validation. That distinction is carried through into the
validation report.

Georeferencing approach
-------------------------
The GRI product itself carries no CRS/map projection (identity
transform -- see reader.py), so pixel-for-pixel comparison requires
manually georeferencing each GRI pixel:

    1. Interpolate per-pixel (lat, lon) from the sparse geometry CSV
       tie-point table onto the full GRI pixel grid.
    2. Project (lat, lon) to the mosaic's Moon 2000 South Polar
       Stereographic (x, y) using the closed-form spherical polar
       stereographic equations (Snyder, "Map Projections: A Working
       Manual", USGS PP 1395, p.161) -- implemented directly in numpy,
       matching the mosaic's declared CRS parameters exactly
       (latitude_of_origin=-90, central_meridian=0, false easting/
       northing=0, sphere radius 1,737,400 m) so no PROJ/pyproj
       dependency is needed.
    3. Invert the mosaic's affine transform (a simple axis-aligned
       scale+offset, verified from its declared transform -- no
       rotation terms) to get (row, col) into the mosaic, and
       nearest-neighbour sample the official CPR value at every GRI
       pixel.

This produces a full 2-D "official CPR resampled onto the GRI grid"
image, which supports difference maps, scatter plots and correlation
statistics.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator
from scipy import stats as sp_stats

log = logging.getLogger("cpr_gri_pipeline.validation")


# ---------------------------------------------------------------------------
# Geolocation
# ---------------------------------------------------------------------------

def load_geometry_grid(geom_csv: Path, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the sparse (Latitude, Longitude, Range) tie-point table
    onto the full (height, width) GRI pixel grid.

    Returns
    -------
    (lat_grid, lon_grid), each float64 shape (height, width)
    """
    data = np.genfromtxt(geom_csv, delimiter=",", skip_header=1)
    lat_col, lon_col, range_col = data[:, 0], data[:, 1], data[:, 2]

    diffs = np.diff(range_col)
    resets = np.where(diffs < -100.0)[0]
    if resets.size == 0:
        raise ValueError(f"Could not auto-detect tie-point grid shape from {geom_csv}")
    n_rng = int(resets[0] + 1)
    n_az = len(data) // n_rng
    log.info(f"  Geometry tie points: N_AZ={n_az}, N_RNG={n_rng}  (total rows={len(data)})")

    lat_grid = lat_col[: n_az * n_rng].reshape(n_az, n_rng)
    lon_grid = lon_col[: n_az * n_rng].reshape(n_az, n_rng)

    az_ties = np.linspace(0, height - 1, n_az)
    rng_ties = np.linspace(0, width - 1, n_rng)

    lat_interp = RegularGridInterpolator((az_ties, rng_ties), lat_grid, bounds_error=False, fill_value=None)
    lon_interp = RegularGridInterpolator((az_ties, rng_ties), lon_grid, bounds_error=False, fill_value=None)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pts = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    lat_full = lat_interp(pts).reshape(height, width)
    lon_full = lon_interp(pts).reshape(height, width)

    log.info(
        f"  Full-grid lat range: [{lat_full.min():.4f}, {lat_full.max():.4f}]  "
        f"lon range: [{lon_full.min():.4f}, {lon_full.max():.4f}]"
    )
    return lat_full, lon_full


def moon_south_polar_stereographic(lat_deg: np.ndarray, lon_deg: np.ndarray, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical South Polar Stereographic projection (tangent at the pole,
    true scale at -90 deg), matching the official mosaic's declared CRS:

        GEOGCS  : sphere, radius R (Moon_2000_IAU_IAG)
        PROJCS  : Polar_Stereographic, latitude_of_origin=-90,
                  central_meridian=0, false_easting=0, false_northing=0

    Formulas (Snyder 1987, eq. for south polar aspect, spherical, k0=1):

        rho = 2*R*tan(pi/4 + phi/2)
        x   = rho * sin(lambda)
        y   = rho * cos(lambda)

    where phi, lambda are latitude/longitude in radians. rho -> 0 as
    phi -> -90 deg (the pole maps to the projection origin), matching
    the mosaic's near-zero-offset extent centred on the pole.
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    rho = 2.0 * R * np.tan(np.pi / 4.0 + phi / 2.0)
    x = rho * np.sin(lam)
    y = rho * np.cos(lam)
    return x, y


# ---------------------------------------------------------------------------
# Sampling the official mosaic
# ---------------------------------------------------------------------------

def sample_official_mosaic(mosaic_path: Path, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nearest-neighbour sample the official CPR mosaic at projected (x, y)
    points, without loading the full (multi-GB) mosaic into memory --
    only the small bounding window covering our points is read.

    Returns
    -------
    (sampled_values float32, sampled_valid bool), same shape as x/y.
    """
    with rasterio.open(mosaic_path) as src:
        transform = src.transform
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        if abs(b) > 1e-9 or abs(d) > 1e-9:
            raise ValueError("Official mosaic transform has rotation terms; the fast-path inverse assumes none.")

        col_f = (x - c) / a
        row_f = (y - f) / e
        col_i = np.round(col_f).astype(np.int64)
        row_i = np.round(row_f).astype(np.int64)

        in_bounds = (col_i >= 0) & (col_i < src.width) & (row_i >= 0) & (row_i < src.height)

        r0, r1 = int(row_i[in_bounds].min()), int(row_i[in_bounds].max())
        c0, c1 = int(col_i[in_bounds].min()), int(col_i[in_bounds].max())
        margin = 2
        r0, r1 = max(0, r0 - margin), min(src.height - 1, r1 + margin)
        c0, c1 = max(0, c0 - margin), min(src.width - 1, c1 + margin)

        log.info(
            f"  Official mosaic window: rows [{r0}:{r1}] cols [{c0}:{c1}]  "
            f"({r1-r0+1} x {c1-c0+1} px) of full {src.height} x {src.width}"
        )
        win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        sub = src.read(1, window=win).astype(np.float32)
        nodata = src.nodata

    local_row = row_i - r0
    local_col = col_i - c0
    sampled = np.full(x.shape, np.nan, dtype=np.float32)
    valid = np.zeros(x.shape, dtype=bool)

    lr = local_row[in_bounds]
    lc = local_col[in_bounds]
    vals = sub[lr, lc]
    if nodata is not None:
        finite = np.isfinite(vals) & (vals != nodata)
    else:
        finite = np.isfinite(vals)

    sampled_flat = sampled[in_bounds]
    sampled_flat[finite] = vals[finite]
    sampled[in_bounds] = sampled_flat

    valid_flat = valid[in_bounds]
    valid_flat[finite] = True
    valid[in_bounds] = valid_flat

    log.info(f"  Sampled official CPR: {int(valid.sum()):,} / {valid.size:,} points valid")
    return sampled, valid


# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------

def compare_cpr(
    cpr_gri: np.ndarray,
    valid_gri: np.ndarray,
    official: np.ndarray,
    valid_official: np.ndarray,
) -> dict:
    """
    Compute Pearson correlation, Spearman rank correlation, RMSE, MAE,
    bias, R^2 and histogram overlap between our CPR_GRI and the
    resampled official CPR, over pixels valid in both.

    Spearman (rank-based) is reported alongside Pearson because some CPR
    formulations (e.g. cpr.compute_cpr_research, the published mu_c
    formula) can produce extreme outliers by construction wherever their
    denominator approaches zero -- Pearson r is not robust to that and
    can be driven to ~0 or an arbitrary sign by a handful of huge values
    even when the two products track each other well in rank/pattern.
    """
    both_valid = valid_gri & valid_official & np.isfinite(cpr_gri) & np.isfinite(official)
    n = int(both_valid.sum())
    if n < 2:
        log.warning("  Fewer than 2 co-valid pixels -- comparison statistics undefined.")
        return {"n": n}

    ours = cpr_gri[both_valid].astype(np.float64)
    ref = official[both_valid].astype(np.float64)

    r, pvalue = sp_stats.pearsonr(ours, ref)
    rho, rho_pvalue = sp_stats.spearmanr(ours, ref)
    diff = ours - ref
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    hi = float(np.percentile(np.concatenate([ours, ref]), 99))
    bins = np.linspace(0.0, max(hi, 1e-6), 100)
    h_ours, _ = np.histogram(ours, bins=bins, density=False)
    h_ref, _ = np.histogram(ref, bins=bins, density=False)
    h_ours = h_ours / max(h_ours.sum(), 1)
    h_ref = h_ref / max(h_ref.sum(), 1)
    hist_overlap = float(np.sum(np.minimum(h_ours, h_ref)))

    result = {
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(pvalue),
        "spearman_r": float(rho),
        "spearman_p": float(rho_pvalue),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "hist_overlap": hist_overlap,
        "ours_mean": float(ours.mean()), "ours_median": float(np.median(ours)),
        "ref_mean": float(ref.mean()), "ref_median": float(np.median(ref)),
    }
    log.info(
        f"  Comparison (n={n:,}): Pearson r={r:.4f}  Spearman rho={rho:.4f}  "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:+.4f}  R2={r2:.4f}  "
        f"HistOverlap={hist_overlap:.4f}"
    )
    return result
