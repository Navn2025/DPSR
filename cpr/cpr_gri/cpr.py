"""
cpr.py
======
STEP 4 -- Compute the Circular Polarization Ratio (CPR) from calibrated,
multilooked GRI linear-power channels.

Why the exact SLI-pipeline formula cannot be used here
--------------------------------------------------------
The SLI CPR pipeline (cpr/complex_builder.py, cpr/cpr.py) works from
COMPLEX single-look-complex data and forms the true circular-basis
fields coherently:

    SC_field = S_HH - S_VV + 2j*S_HV      (same-sense circular)
    OC_field = S_HH + S_VV                 (opposite-sense circular)
    CPR      = mean(|SC|^2) / mean(|OC|^2)

This requires the COMPLEX cross term S_HH * conj(S_VV) -- i.e. the
relative PHASE between the HH and VV returns.

GRI products are detected/intensity products: each of HH, HV, VH, VV is
delivered as backscattered POWER only (|S_pq|^2-equivalent, see
preprocessing.py). The phase relationship between channels is destroyed
by detection and is NOT recoverable from GRI alone. Exact reconstruction
of SC_field/OC_field, and therefore of the exact CPR, is IMPOSSIBLE from
GRI intensity data. This is stated explicitly per the task's STEP 4
requirement, rather than silently approximated.

Closest physically valid approximation
-----------------------------------------
Expand the exact power definitions:

    |SC|^2 = |S_HH - S_VV + 2j*S_HV|^2
           = |S_HH|^2 + |S_VV|^2 + 4|S_HV|^2 - 2*Re(S_HH * conj(S_VV))
    |OC|^2 = |S_HH + S_VV|^2
           = |S_HH|^2 + |S_VV|^2 + 2*Re(S_HH * conj(S_VV))

Both expressions differ from the available power-only quantities only by
the unknown co-pol correlation term Re(<S_HH S_VV*>). Under the
REFLECTION SYMMETRY assumption widely used for natural, azimuthally
random media (Cloude & Pottier, 2009; Lee & Pottier, "Polarimetric Radar
Imaging") -- i.e. no preferred coherent alignment between the HH and VV
returns for a rough, randomly-oriented regolith surface -- this
correlation term is small and is dropped:

    Re(<S_HH S_VV*>) ~ 0   (reflection symmetry approximation)

which gives:

    <|SC|^2>_approx = P_HH + P_VV + 4*P_XP
    <|OC|^2>_approx = P_HH + P_VV

    CPR_GRI = <|SC|^2>_approx / <|OC|^2>_approx = 1 + 4*P_XP / (P_HH + P_VV)

where P_XP = (P_HV + P_VH) / 2 is the reciprocity-averaged cross-pol
power (same averaging convention as the SLI pipeline).

This approximation is directionally and physically correct:
    - A smooth, non-depolarizing (single-bounce) surface has P_XP -> 0,
      giving CPR -> 1 (matches the physical definition: OC-dominant,
      "flat" reflector).
    - Increasing depolarization / volume scattering (higher cross-pol
      power relative to co-pol) drives CPR above 1, consistent with the
      same "SC-dominant -> rough / possible subsurface scatterer"
      interpretation used for the exact SLI CPR.
    - It reduces to the *exact* formula whenever the true HH-VV
      correlation genuinely vanishes, and is a documented approximation
      (not the exact ratio) whenever it does not.

All inputs here are POWER quantities that have already been multilooked
by preprocessing.multilook_nodata_aware (STEP 5), consistent with the
requirement not to compute this from single-look/undersampled pixels.
"""

import logging

import numpy as np

log = logging.getLogger("cpr_gri_pipeline.cpr")


def compute_cpr_gri(
    P_HH: np.ndarray,
    P_VV: np.ndarray,
    P_XP: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = 1e-10,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    CPR_GRI = 1 + 4 * P_XP / (P_HH + P_VV)   (reflection-symmetry approximation)

    Parameters
    ----------
    P_HH, P_VV, P_XP : float32 multilooked linear-power arrays (H, W)
    valid_mask       : bool array (H, W), True where all contributing
                       channels were valid through calibration+multilook
    epsilon          : guards the denominator against division by zero
    nodata           : fill value for invalid output pixels

    Returns
    -------
    float32 CPR array (H, W); invalid pixels set to `nodata`.

    Invalid pixel criteria
    -----------------------
    - valid_mask == False (upstream NoData/invalid in any channel)
    - P_HH + P_VV <= 0  (non-physical / zero co-pol denominator)
    - negative power inputs (clamped to 0 before use, flagged if present)
    - NaN or Inf anywhere in the computation
    """
    hh = np.asarray(P_HH, dtype=np.float64)
    vv = np.asarray(P_VV, dtype=np.float64)
    xp = np.asarray(P_XP, dtype=np.float64)

    negative_input = (np.nan_to_num(hh, nan=0.0) < 0) | (np.nan_to_num(vv, nan=0.0) < 0) | (np.nan_to_num(xp, nan=0.0) < 0)
    hh = np.clip(hh, 0.0, None)
    vv = np.clip(vv, 0.0, None)
    xp = np.clip(xp, 0.0, None)

    denom = hh + vv
    finite = np.isfinite(hh) & np.isfinite(vv) & np.isfinite(xp)
    nonpositive = ~(denom > 0.0)

    cpr = (1.0 + 4.0 * xp / (denom + epsilon)).astype(np.float32)

    bad = ~finite | nonpositive | negative_input | ~valid_mask | ~np.isfinite(cpr)
    cpr[bad] = nodata

    n_bad = int(bad.sum())
    n_tot = int(cpr.size)
    log.info(
        f"  CPR_GRI computed: total={n_tot:,}  invalid={n_bad:,} "
        f"({100 * n_bad / n_tot:.3f}%)  valid={n_tot - n_bad:,}"
    )

    valid = cpr[~bad]
    if valid.size > 0:
        log.info(
            f"  CPR_GRI valid: min={valid.min():.4f}  max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  median={np.median(valid):.4f}"
        )

    return cpr


# ---------------------------------------------------------------------------
# Published-formula CPR (co-pol only, "--research" mode)
# ---------------------------------------------------------------------------

def compute_cpr_research(
    P_HH: np.ndarray,
    P_VV: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = 1e-10,
    nodata: float = -9999.0,
    valid_range=(0.0, 2.0),
    rescale_percentiles=(1.0, 99.0),
) -> tuple:
    """
    Published co-pol-only CPR formula (the mu_c formulation):

        CPR(mu_c) = (sigma_HH + sigma_VV + 2*sqrt(sigma_HH*sigma_VV))
                    / (sigma_HH + sigma_VV - 2*sqrt(sigma_HH*sigma_VV))

    This is algebraically two perfect squares:

        numerator   = (sqrt(sigma_HH) + sqrt(sigma_VV))^2
        denominator = (sqrt(sigma_HH) - sqrt(sigma_VV))^2

    Unlike compute_cpr_gri() above (which needs the cross-pol channels
    and drops the unknown HH-VV phase correlation under a reflection-
    symmetry assumption), this formula uses ONLY the co-pol backscatter
    coefficients sigma_HH, sigma_VV -- it implicitly assumes the HH and
    VV returns are fully coherent/correlated (the square-root cross term
    stands in for Re(<S_HH S_VV*>) directly, rather than dropping it),
    which is the standard assumption for a dominant quasi-specular/Bragg
    single-bounce surface-scattering regime.

    Numerical note (important, confirmed on real data): solving the
    formula for CPR(mu_c) = 2 shows the co-pol channels must differ by
    >15.3 dB for the raw ratio to stay under 2 (CPR=5 needs >8.4 dB,
    CPR=10 needs >5.7 dB). On the real Faustini GRI scene, the median
    HH-VV gap is only ~0.6 dB (91% of pixels within +-3 dB) -- so the
    raw formula saturates to very large values (hundreds to hundreds of
    thousands) on essentially the ENTIRE scene, not as occasional
    outliers. A hard clip to `valid_range` was tried first and produces
    a perfectly flat, information-free image (every valid pixel pinned
    to the clip ceiling) -- confirmed on this data and useless as a
    product despite technically satisfying the numeric range.

    Chosen fix: LOG-RESCALE. The raw ratio is log10-transformed (which
    is the natural domain for a quantity spanning multiple orders of
    magnitude) and then linearly mapped from its own
    [rescale_percentiles[0], rescale_percentiles[1]] percentile range
    onto `valid_range`, clipping only the extreme tails. This preserves
    the RELATIVE spatial pattern (which pixels are more/less "mu_c-like"
    relative to the rest of the scene) inside the requested numeric
    range, instead of collapsing all spatial information as a hard clip
    does. The exact rescale bounds used (in raw CPR units) are returned
    in the diagnostics dict so the transform is fully reproducible and
    the output is never mistaken for the literal unscaled formula value.

    Parameters
    ----------
    P_HH, P_VV          : float32 multilooked, calibrated linear-power arrays (H, W)
    valid_mask          : bool array (H, W), True where HH/VV were valid through
                          calibration + multilook
    epsilon             : denominator floor guarding against exact division by zero
    nodata              : fill value for invalid output pixels
    valid_range         : (min, max) output range for the log-rescaled product
    rescale_percentiles : (lo, hi) percentiles of log10(raw CPR), over valid
                          pixels, used to define the rescale window

    Returns
    -------
    (cpr_display, info) where:
        cpr_display : float32 array (H, W), log-rescaled into `valid_range`,
                      invalid pixels set to `nodata`
        info        : dict with the raw (unscaled) formula's statistics and
                      the exact rescale bounds used, for reporting/tags
    """
    hh = np.clip(np.asarray(P_HH, dtype=np.float64), 0.0, None)
    vv = np.clip(np.asarray(P_VV, dtype=np.float64), 0.0, None)

    sqrt_hh = np.sqrt(hh)
    sqrt_vv = np.sqrt(vv)
    numerator = (sqrt_hh + sqrt_vv) ** 2
    denominator = (sqrt_hh - sqrt_vv) ** 2

    finite = np.isfinite(hh) & np.isfinite(vv)
    denom_too_small = denominator < epsilon

    cpr_raw = numerator / (denominator + epsilon)
    bad = ~finite | denom_too_small | ~valid_mask | ~np.isfinite(cpr_raw) | (cpr_raw <= 0)

    n_tot = int(cpr_raw.size)
    n_bad = int(bad.sum())
    lo, hi = valid_range
    info = {
        "n_total": n_tot, "n_invalid": n_bad,
        "raw_min": float("nan"), "raw_max": float("nan"), "raw_median": float("nan"),
        "rescale_lo_raw": float("nan"), "rescale_hi_raw": float("nan"),
    }

    if n_bad >= n_tot:
        log.warning("  CPR_research: no valid pixels -- nothing to rescale.")
        return np.full(cpr_raw.shape, nodata, dtype=np.float32), info

    raw_valid = cpr_raw[~bad]
    info["raw_min"] = float(raw_valid.min())
    info["raw_max"] = float(raw_valid.max())
    info["raw_median"] = float(np.median(raw_valid))
    log.info(
        f"  CPR_research (mu_c) RAW: min={info['raw_min']:.4f}  max={info['raw_max']:.4f}  "
        f"median={info['raw_median']:.4f}  invalid={n_bad:,}/{n_tot:,} "
        f"[denom~0 diverging: {int((denom_too_small & finite & valid_mask).sum()):,}]"
    )

    log_raw = np.log10(raw_valid)
    p_lo, p_hi = rescale_percentiles
    log_lo, log_hi = np.percentile(log_raw, [p_lo, p_hi])
    info["rescale_lo_raw"] = float(10 ** log_lo)
    info["rescale_hi_raw"] = float(10 ** log_hi)
    log.info(
        f"  CPR_research log-rescale window (P{p_lo:g}-P{p_hi:g} of log10(raw)): "
        f"[{info['rescale_lo_raw']:.4f}, {info['rescale_hi_raw']:.4f}] raw CPR -> {valid_range}"
    )

    log_cpr_full = np.full(cpr_raw.shape, np.nan, dtype=np.float64)
    log_cpr_full[~bad] = np.log10(cpr_raw[~bad])

    if log_hi > log_lo:
        scaled = lo + (log_cpr_full - log_lo) * (hi - lo) / (log_hi - log_lo)
    else:
        scaled = np.full(cpr_raw.shape, (lo + hi) / 2.0)
    cpr_display = np.clip(scaled, lo, hi).astype(np.float32)
    cpr_display[bad] = nodata

    valid = cpr_display[~bad]
    log.info(
        f"  CPR_research display (log-rescaled): min={valid.min():.4f}  max={valid.max():.4f}  "
        f"mean={valid.mean():.4f}  median={np.median(valid):.4f}"
    )

    return cpr_display, info
