"""
ice_score.py
============
STEP 8 of the Diviner integration pipeline.

Computes a physics-based Ice Confidence Score in [0, 1].

No machine-learning model is used.  Every normalisation and weighting
decision is grounded in published literature and explained below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHYSICAL INDICATORS
-------------------
Eight indicators are normalised independently, then combined as a
weighted sum.

  Indicator   Weight   Ice-positive sense   Physical basis
  ─────────── ──────   ──────────────────── ─────────────────────────────────
  CPR          0.20    HIGH → more ice      Volume / double-bounce scattering
                                            from ice-grain aggregates
                                            (Nozette 1996; Campbell 2006)
  DOP          0.12    LOW  → more ice      Depolarising, volumetric targets
                                            lower the overall degree of
                                            polarisation (van Zyl & Kim 2011)
  Tmean        0.20    LOW  → more ice      Thermally stable cold environments
                                            preserve volatile ice
                                            (Paige et al. 2010)
  ZIT          0.15    LOW  → more ice      Zero-incidence temperature is the
                                            best proxy for the coldest surface
                                            temperature at a PSR pixel
                                            (Hayne et al. 2015)
  Pump         0.13    HIGH → more ice      Efficient volatile cold-trapping
                                            proxy; high pump → ice delivery
                                            more likely (Schorghofer 2014)
  PSR          0.10    1    → more ice      Permanently shadowed → no solar
                                            heating → thermally stable ice
                                            (Watson et al. 1961)
  DPSR         0.05    1    → more ice      Doubly shadowed → extreme cold
                                            trap → extra ice stability
                                            (O'Brien & Byrne 2022)
  Slope        0.05    LOW  → more ice      Flat terrain favours ice
                                            accumulation and retention
                                            (Prettyman et al. 2012)

  Total weight = 1.00

NORMALISATION
-------------
Each continuous band (CPR, DOP, Tmean, ZIT, Pump, Slope) is clipped to
the [p₂, p₉₈] range of its valid pixel distribution to suppress outliers,
then linearly scaled to [0, 1].

  • "high → more ice" bands: keep direction  (max = 1)
  • "low  → more ice" bands: invert          (min = 1, i.e. 1 − normed)

Binary masks (PSR, DPSR) are cast directly to 0.0 / 1.0.

COMBINATION
-----------
Score at each pixel = Σ(normalised_band_i × weight_i) / Σ(weight_i valid)

Dividing by the total weight of AVAILABLE bands at each pixel means that
pixels where some optional bands are nodata still get a meaningful score
from the bands that are present, rather than being forced to zero.

REFERENCES
----------
Nozette et al. (1996) Science 274, 1495.
Campbell et al. (2006) Nature 443, 835.
Paige et al. (2010) Science 330, 479.
Hayne et al. (2015) Icarus 255, 58.
Schorghofer (2014) Astrophys. J. 788, 169.
Watson et al. (1961) JGR 66, 3033.
O'Brien & Byrne (2022) Planet. Space Sci. 221, 105566.
van Zyl & Kim (2011) SAR Polarimetry, JPL.
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger("diviner_pipeline.ice_score")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_continuous(
    arr:    np.ndarray,
    lo_pct: float,
    hi_pct: float,
    nodata: float  = -9999.0,
    invert: bool   = False,
    label:  str    = "",
) -> np.ndarray:
    """
    Clip *arr* to [lo_pct, hi_pct] percentiles of valid pixels, then
    scale to [0, 1].  If *invert* is True the direction is flipped so
    that the MINIMUM of the original data maps to 1 (used for bands where
    LOWER = more ice evidence).

    Nodata / NaN / Inf pixels receive nodata in the output.
    """
    valid_mask = np.isfinite(arr) & (arr != nodata)
    valid      = arr[valid_mask]

    if valid.size == 0:
        log.warning(f"  Normalise [{label}]: no valid pixels — returning nodata.")
        return np.full_like(arr, nodata, dtype=np.float32)

    p_lo = float(np.percentile(valid, lo_pct))
    p_hi = float(np.percentile(valid, hi_pct))
    log.debug(f"  Normalise [{label}]: p{lo_pct}={p_lo:.4e}  p{hi_pct}={p_hi:.4e}"
              f"  invert={invert}")

    if p_hi <= p_lo:
        log.warning(
            f"  Normalise [{label}]: p{lo_pct}={p_lo:.4e} >= p{hi_pct}={p_hi:.4e} "
            f"— constant band; setting all valid pixels to 0.5."
        )
        out = np.where(valid_mask, 0.5, nodata).astype(np.float32)
        return out

    clipped = np.clip(arr[valid_mask].astype(np.float64), p_lo, p_hi)
    normed  = (clipped - p_lo) / (p_hi - p_lo)   # strictly in [0, 1]
    if invert:
        normed = 1.0 - normed   # flip: smallest → 1, largest → 0

    out = np.full_like(arr, nodata, dtype=np.float32)
    out[valid_mask] = normed.astype(np.float32)
    return out


def _normalise_mask(
    arr:    np.ndarray,
    nodata: float = -9999.0,
    label:  str   = "",
) -> np.ndarray:
    """
    Cast a binary mask (0 = outside, non-zero = inside) to float32 [0, 1].
    Nodata pixels remain nodata.
    """
    valid_mask = np.isfinite(arr) & (arr != nodata)
    out = np.full_like(arr, nodata, dtype=np.float32)
    # Treat any strictly positive value as "inside" (1.0)
    out[valid_mask] = np.where(arr[valid_mask] > 0, 1.0, 0.0).astype(np.float32)
    log.debug(
        f"  Normalise [{label}] (mask): "
        f"{int((out == 1.0).sum()):,} inside / {int(valid_mask.sum()):,} valid"
    )
    return out


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_ice_confidence(
    bands:   Dict[str, np.ndarray],
    weights: Dict[str, float],
    nodata:  float = -9999.0,
    lo_pct:  float = 2.0,
    hi_pct:  float = 98.0,
) -> np.ndarray:
    """
    Compute the physics-based Ice Confidence Score.

    Parameters
    ----------
    bands   : dict of aligned 2-D float32 arrays keyed by band name.
              Expected keys: CPR, DOP, Tmean, ZIT, Pump, PSR, DPSR, Slope.
              Missing optional keys (e.g. CPR, DOP) are silently skipped.
    weights : dict mapping band name → relative weight (should sum to 1.0).
    nodata  : nodata sentinel shared by all bands and the output.
    lo_pct  : lower percentile for robust normalisation (default 2).
    hi_pct  : upper percentile for robust normalisation (default 98).

    Returns
    -------
    ice_score : 2-D float32 array in [0, 1]; nodata where no band was valid.
    norm_bands: dict of normalised band arrays (for diagnostics / PDF report).
    """
    H, W = next(iter(bands.values())).shape
    log.info(f"  Scene size: {W} x {H}  nodata={nodata}")

    # ── Combined normalise-and-accumulate pass ───────────────────────────────
    # Each band is normalised, immediately added to score/weight_sum, then
    # freed.  This avoids holding all 8 normalised arrays (~7 GB) in memory
    # simultaneously alongside the original bands (~8 GB).
    #
    # score[p]      = Σ_i  norm_i[p] × w_i
    # weight_sum[p] = Σ_i  w_i
    # final[p]      = score[p] / weight_sum[p]   (nodata where sum = 0)
    #
    # All arrays stay float32; boolean indexing avoids full-array temporaries.

    score      = np.zeros((H, W), dtype=np.float32)   # 879 MB
    weight_sum = np.zeros((H, W), dtype=np.float32)   # 879 MB

    # Band configuration: (name, invert, is_binary_mask)
    _BAND_CFG = [
        ("CPR",   False, False, "high → more ice, keep direction"),
        ("DOP",   True,  False, "low → more ice, inverted"),
        ("Tmean", True,  False, "low → more ice, inverted"),
        ("ZIT",   True,  False, "low → more ice, inverted"),
        ("Pump",  False, False, "high → more ice, keep direction"),
        ("PSR",   False, True,  "binary 0/1 mask"),
        ("DPSR",  False, True,  "binary 0/1 mask"),
        ("Slope", True,  False, "low → more ice, inverted"),
    ]

    for band_name, invert, is_mask, desc in _BAND_CFG:
        if band_name not in bands:
            continue
        if band_name not in weights:
            log.warning(f"  No weight for '{band_name}' — skipped.")
            continue

        log.info(f"  Normalising {band_name}  ({desc})")
        if is_mask:
            normed = _normalise_mask(bands[band_name], nodata, label=band_name)
        else:
            normed = _normalise_continuous(
                bands[band_name], lo_pct, hi_pct, nodata,
                invert=invert, label=band_name,
            )

        w_f32      = np.float32(weights[band_name])
        valid_mask = np.isfinite(normed) & (normed != np.float32(nodata))
        n_valid    = int(valid_mask.sum())

        # Boolean-index only valid pixels to avoid a full-array temporary
        score[valid_mask]      += normed[valid_mask] * w_f32
        weight_sum[valid_mask] += w_f32
        del normed              # free 879 MB immediately

        log.info(f"    Added [{band_name}] weight={weights[band_name]:.2f}  "
                 f"valid_px={n_valid:,}")

    # ── Final score ──────────────────────────────────────────────────────────
    valid_out = weight_sum > 0
    ice_score = np.full((H, W), fill_value=nodata, dtype=np.float32)
    np.divide(score, weight_sum, out=ice_score, where=valid_out)  # in-place

    n_valid = int(valid_out.sum())
    if n_valid > 0:
        sc_min  = float(ice_score[valid_out].min())
        sc_max  = float(ice_score[valid_out].max())
        sc_mean = float(ice_score[valid_out].mean())
        log.info(
            f"  Ice Confidence: {n_valid:,} / {H*W:,} valid pixels  "
            f"min={sc_min:.4f}  max={sc_max:.4f}  mean={sc_mean:.4f}"
        )
        if sc_min < -0.001 or sc_max > 1.001:
            log.warning(
                f"  Score out of [0,1] range ({sc_min:.4f}–{sc_max:.4f}) — "
                f"check normalisation."
            )
    else:
        log.warning("  Ice Confidence: NO valid pixels in output!")

    # norm_bands is no longer built to avoid 7 GB peak memory overhead;
    # the return slot is kept for API compatibility.
    return ice_score, {}
