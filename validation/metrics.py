"""
metrics.py
==========
All quantitative comparison metrics between the calculated and official CPR.
"""
import logging
from typing import Dict, Tuple

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as _ssim

log = logging.getLogger("validation.metrics")


# ---------------------------------------------------------------------------
# Overlap extraction
# ---------------------------------------------------------------------------

def extract_overlap(
    calc:    np.ndarray,
    offic:   np.ndarray,
    nodata_c: float,
    nodata_o: float,
    cpr_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return matched 1-D arrays of valid pixels that are finite in BOTH rasters.

    Parameters
    ----------
    calc, offic : full (H, W) float32 arrays
    nodata_c    : nodata value in calc (NaN for our georeferenced product)
    nodata_o    : nodata value in official product
    cpr_max     : upper physical limit; values above are masked as outliers

    Returns
    -------
    a, b       : matched 1-D valid arrays (both arrays, same pixels)
    mask_2d    : (H, W) boolean of the overlap region
    """
    m_c = np.isfinite(calc)
    m_o = np.isfinite(offic)

    if not np.isnan(nodata_c):
        m_c &= (calc != nodata_c)
    if not np.isnan(nodata_o):
        m_o &= (offic != nodata_o)

    m_c &= (calc > 0)  & (calc <= cpr_max)
    m_o &= (offic > 0) & (offic <= cpr_max)

    mask_2d = m_c & m_o
    a = calc[mask_2d].ravel()
    b = offic[mask_2d].ravel()

    log.info(
        f"  Overlap pixels: {mask_2d.sum():,}  "
        f"(calc valid={m_c.sum():,}, official valid={m_o.sum():,})"
    )
    return a, b, mask_2d


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> Dict:
    r, pval = stats.pearsonr(a, b)
    return {"pearson_r": float(r), "pearson_pval": float(pval)}


def spearman_r(a: np.ndarray, b: np.ndarray) -> Dict:
    # Cap at 2M points for speed
    if len(a) > 2_000_000:
        idx = np.random.default_rng(42).choice(len(a), 2_000_000, replace=False)
        a, b = a[idx], b[idx]
    r, pval = stats.spearmanr(a, b)
    return {"spearman_r": float(r), "spearman_pval": float(pval)}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def bias(a: np.ndarray, b: np.ndarray) -> float:
    """Mean signed difference: calc - official."""
    return float(np.mean(a - b))


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot (using b as reference)."""
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def ssim_metric(
    calc_2d:   np.ndarray,
    offic_2d:  np.ndarray,
    mask_2d:   np.ndarray,
    win_size:  int = 7,
) -> float:
    """
    Structural Similarity Index (SSIM) on the overlap bounding box.
    Crops to the minimum bounding rectangle of the overlap mask.
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return float("nan")

    r_min, r_max = int(rows.argmax()), int(len(rows) - rows[::-1].argmax() - 1)
    c_min, c_max = int(cols.argmax()), int(len(cols) - cols[::-1].argmax() - 1)

    a_crop = calc_2d[r_min:r_max+1, c_min:c_max+1].copy()
    b_crop = offic_2d[r_min:r_max+1, c_min:c_max+1].copy()
    m_crop = mask_2d[r_min:r_max+1, c_min:c_max+1]

    # Replace nodata with mean so SSIM doesn't penalise nodata edges
    mean_a = float(np.nanmean(a_crop[m_crop]))
    mean_b = float(np.nanmean(b_crop[m_crop]))
    a_crop[~m_crop] = mean_a
    b_crop[~m_crop] = mean_b
    a_crop[~np.isfinite(a_crop)] = mean_a
    b_crop[~np.isfinite(b_crop)] = mean_b

    # Downsample if very large
    MAX_DIM = 4096
    if max(a_crop.shape) > MAX_DIM:
        step = max(a_crop.shape) // MAX_DIM + 1
        a_crop = a_crop[::step, ::step]
        b_crop = b_crop[::step, ::step]

    data_range = max(float(b_crop.max() - b_crop.min()), 1e-6)
    win = min(win_size, a_crop.shape[0], a_crop.shape[1])
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")

    try:
        val = _ssim(a_crop, b_crop, data_range=data_range, win_size=win)
        return float(val)
    except Exception as e:
        log.warning(f"  SSIM computation failed: {e}")
        return float("nan")


def histogram_intersection(
    a: np.ndarray, b: np.ndarray,
    bins: int = 200,
    range_: Tuple[float, float] = (0.0, 3.0),
) -> float:
    """
    Histogram intersection: sum(min(hist_a[i], hist_b[i])) / n.
    Returns 0 (no overlap) to 1 (identical distributions).
    """
    ha, edges = np.histogram(a, bins=bins, range=range_, density=False)
    hb, _     = np.histogram(b, bins=bins, range=range_, density=False)
    ha = ha / ha.sum() if ha.sum() > 0 else ha
    hb = hb / hb.sum() if hb.sum() > 0 else hb
    return float(np.sum(np.minimum(ha, hb)))


def mutual_information(
    a: np.ndarray, b: np.ndarray,
    bins: int = 100,
    range_: Tuple[float, float] = (0.0, 3.0),
) -> float:
    """Normalised mutual information via 2-D histogram. Returns value in [0, 1]."""
    h2d, _, _ = np.histogram2d(a, b, bins=bins, range=[range_, range_])
    total = h2d.sum()
    if total == 0:
        return 0.0
    h2d = h2d / total          # joint probability P(i, j)

    ha = h2d.sum(axis=1)       # marginal P_a(i)
    hb = h2d.sum(axis=0)       # marginal P_b(j)

    # MI = sum_{i,j} P(i,j) * log( P(i,j) / (P_a(i) * P_b(j)) )
    outer = np.outer(ha, hb)   # (bins, bins) of P_a(i) * P_b(j)
    nz    = (h2d > 0) & (outer > 0)
    mi    = float(np.sum(h2d[nz] * np.log(h2d[nz] / outer[nz])))

    # Normalise by sqrt(H(a) * H(b))  (Strehl-Ghosh normalisation)
    ha_nz = ha[ha > 0]
    hb_nz = hb[hb > 0]
    h_a   = float(-np.sum(ha_nz * np.log(ha_nz)))
    h_b   = float(-np.sum(hb_nz * np.log(hb_nz)))
    denom = np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0
    return float(mi / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    calc_2d:  np.ndarray,
    offic_2d: np.ndarray,
    mask_2d:  np.ndarray,
) -> Dict:
    """
    Run all metrics on the matched overlap pixels.

    Returns a flat dict of all computed values.
    """
    a, b, _ = extract_overlap.__wrapped__ if hasattr(extract_overlap, '__wrapped__') \
              else (None, None, None)

    # We re-extract 1-D arrays from the 2-D mask
    a = calc_2d[mask_2d].ravel()
    b = offic_2d[mask_2d].ravel()

    if len(a) < 30:
        log.error("  Fewer than 30 overlap pixels — metrics not reliable.")
        return {}

    log.info(f"  Computing metrics on {len(a):,} overlap pixels ...")

    m = {}
    m["n_pixels"] = len(a)

    m.update(pearson_r(a, b))
    m.update(spearman_r(a, b))
    m["rmse"]  = rmse(a, b)
    m["mae"]   = mae(a, b)
    m["bias"]  = bias(a, b)
    m["r2"]    = r_squared(a, b)

    log.info("  Computing SSIM (may take a moment) ...")
    m["ssim"]  = ssim_metric(calc_2d, offic_2d, mask_2d)

    log.info("  Computing histogram intersection ...")
    m["hist_intersection"] = histogram_intersection(a, b)

    log.info("  Computing mutual information ...")
    m["mutual_information"] = mutual_information(a, b)

    # Difference arrays
    diff     = calc_2d.copy()
    abs_diff = calc_2d.copy()
    diff[mask_2d]     = a - b
    abs_diff[mask_2d] = np.abs(a - b)
    diff[~mask_2d]    = np.nan
    abs_diff[~mask_2d] = np.nan

    m["diff_arr"]     = diff
    m["abs_diff_arr"] = abs_diff
    m["overlap_mask"] = mask_2d
    m["calc_1d"]      = a
    m["offic_1d"]     = b

    for k, v in m.items():
        if isinstance(v, float):
            log.info(f"    {k:25s}: {v:.6f}")
        elif isinstance(v, int):
            log.info(f"    {k:25s}: {v:,}")

    return m
