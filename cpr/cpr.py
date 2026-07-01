"""
cpr.py
======
Compute the Circular Polarization Ratio (CPR) from multilooked power
images of the same-sense circular (SC) and opposite-sense circular (OC)
components derived from the full-pol linear scattering matrix.

Formula
-------
    S_RR = (S_HH - S_VV + 2j * S_HV) / 2   [SC, same-sense]
    S_RL = (S_HH + S_VV) / 2                [OC, opposite-sense]

    sigma_SC = mean( |S_RR|^2 )   after multilooking
    sigma_OC = mean( |S_RL|^2 )   after multilooking

    CPR = sigma_SC / sigma_OC

The 1/2 normalisation cancels in the ratio; complex_builder.py forms
    SC_field = S_HH - S_VV + 2j * S_HV
    OC_field = S_HH + S_VV
so what arrives here is already the post-multilook power of those fields.

Physical interpretation
-----------------------
    CPR < 1   -->  OC dominates; surface (Bragg) scattering (bare regolith)
    CPR ~ 1   -->  comparable SC and OC power
    CPR > 1   -->  SC dominates; volume / subsurface scattering (ice candidate)
    CPR >> 1  -->  strong retro-reflection from ice deposits

Expected range for lunar south pole
------------------------------------
    Bare regolith  : 0.05 -- 0.5
    Mixed terrain  : 0.5  -- 1.0
    Ice candidates : > 1.0
    Official mosaic median (Putrevu et al. 2023): ~0.21

Reference
---------
    Putrevu et al. (2023), JGR Planets, DOI: 10.1029/2023JE007745
"""

import logging

import numpy as np

log = logging.getLogger("cpr_pipeline.cpr")


def compute_cpr(
    ml_sc_power: np.ndarray,
    ml_oc_power: np.ndarray,
    epsilon: float = 1e-10,
    nodata:  float = -9999.0,
) -> np.ndarray:
    """
    Compute pixel-wise CPR = sigma_SC / sigma_OC.

    Parameters
    ----------
    ml_sc_power : float32 array (H, W)
        Multilooked same-sense circular power = mean(|S_HH - S_VV + 2j*S_HV|^2)
    ml_oc_power : float32 array (H, W)
        Multilooked opposite-sense circular power = mean(|S_HH + S_VV|^2)
    epsilon     : small constant added to denominator to prevent div-by-zero
    nodata      : fill value for invalid pixels in the output

    Returns
    -------
    float32 CPR array (H, W); invalid pixels set to `nodata`.

    Invalid pixel criteria
    ----------------------
    - ml_oc_power <= 0  (zero or negative denominator)
    - NaN or Inf in either input
    - CPR is NaN or Inf after division
    """
    sc = np.asarray(ml_sc_power, dtype=np.float64)
    oc = np.asarray(ml_oc_power, dtype=np.float64)

    bad_in = (
        ~np.isfinite(sc) |
        ~np.isfinite(oc) |
        (oc <= 0.0)       # OC power zero -> undefined ratio
    )

    sc = np.maximum(sc, 0.0)   # clamp any float32 underflow artifacts
    oc = np.maximum(oc, 0.0)

    cpr = (sc / (oc + epsilon)).astype(np.float32)

    bad_out = ~np.isfinite(cpr)
    bad     = bad_in | bad_out

    cpr[bad] = nodata

    n_bad = int(bad.sum())
    n_tot = int(cpr.size)
    log.debug(
        f"  CPR computed: total={n_tot}  bad/nodata={n_bad}  "
        f"valid={n_tot - n_bad}"
    )

    valid = cpr[~bad]
    if valid.size > 0:
        log.debug(
            f"  CPR valid: min={valid.min():.4f}  max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  median={np.median(valid):.4f}"
        )

    return cpr


# ---------------------------------------------------------------------------
# Published-formula CPR (co-pol only, "--research" mode)
# ---------------------------------------------------------------------------

def compute_cpr_research(
    P_HH: np.ndarray,
    P_VV: np.ndarray,
    epsilon: float = 1e-10,
    nodata: float = -9999.0,
    valid_range=(0.0, 2.0),
    rescale_percentiles=(1.0, 99.0),
) -> tuple:
    """
    Published co-pol-only CPR formula (the mu_c formulation), applied to
    SLI-derived multilooked power instead of GRI-calibrated sigma0 --
    see cpr_gri/cpr.py's compute_cpr_research for the full derivation
    and the numerical-sensitivity findings (identical here since this is
    the same formula):

        CPR(mu_c) = (sigma_HH + sigma_VV + 2*sqrt(sigma_HH*sigma_VV))
                    / (sigma_HH + sigma_VV - 2*sqrt(sigma_HH*sigma_VV))
                  = (sqrt(sigma_HH)+sqrt(sigma_VV))^2 / (sqrt(sigma_HH)-sqrt(sigma_VV))^2

    Calibration-invariance note: this formula is a function of the ratio
    sigma_HH/sigma_VV only (verify: dividing num/denom by sigma_VV shows
    it depends only on r = sigma_HH/sigma_VV). The DFSAR calibration
    constant is a single global value shared by every polarisation
    channel (see cpr/config.py CALIBRATION_K), so it cancels exactly in
    this ratio just as it does in the default CPR = ML_SC/ML_OC formula.
    That means P_HH, P_VV can be the raw multilooked |S_HH|^2, |S_VV|^2
    single-look-complex power (uncalibrated DN-equivalent units) without
    first converting to physical sigma0 -- no calibration step is needed
    for this pipeline's research mode.

    Same numerical behaviour as the GRI version: the raw ratio only
    stays under CPR=2 when HH and VV differ by >15.3 dB, which is rare
    for natural terrain (co-pol channels are usually within a few dB of
    each other), so the raw formula saturates to very large values
    almost everywhere on real data. This implementation therefore
    LOG-RESCALES the raw ratio from its own [rescale_percentiles] range
    onto `valid_range`, exactly as the GRI pipeline does, instead of
    hard-clipping (which was confirmed to collapse to a flat,
    information-free image).

    Parameters
    ----------
    P_HH, P_VV          : float32 multilooked power arrays (H, W); any
                          consistent linear units work (see note above)
    epsilon             : denominator floor guarding against exact division by zero
    nodata              : fill value for invalid output pixels
    valid_range         : (min, max) output range for the log-rescaled product
    rescale_percentiles : (lo, hi) percentiles of log10(raw CPR), over valid
                          pixels, used to define the rescale window

    Returns
    -------
    (cpr_display, info) -- see cpr_gri/cpr.py's compute_cpr_research.
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
    bad = ~finite | denom_too_small | ~np.isfinite(cpr_raw) | (cpr_raw <= 0)

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
        f"[denom~0 diverging: {int((denom_too_small & finite).sum()):,}]"
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
