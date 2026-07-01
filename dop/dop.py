"""
dop.py
======
STEP 7 -- Compute the Degree of Polarization (DOP) from the Stokes
parameters:

    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0

A small epsilon is added to the denominator to guard against
division-by-zero, and pixels are masked invalid under the criteria
listed in `compute_dop` below.

Reference: Born & Wolf, "Principles of Optics", Ch. 10 (degree of
polarization of a partially polarized wave); van Zyl & Kim (2011) for
the radar-specific formulation via the Stokes scattering operator.
"""

import logging

import numpy as np

log = logging.getLogger("dop_pipeline.dop")


def compute_dop(
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    epsilon: float = 1e-10,
    nodata: float = -9999.0,
    input_invalid_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute pixel-wise DOP = sqrt(S1^2+S2^2+S3^2) / (S0 + epsilon).

    Parameters
    ----------
    S0, S1, S2, S3      : float32 Stokes parameter arrays (H, W)
    epsilon             : small constant preventing division-by-zero
    nodata              : fill value written to invalid pixels
    input_invalid_mask  : optional bool array (H, W), True where the
                           original SLI inputs were NoData/non-finite
                           before any polarimetric processing (STEP 7:
                           "Handle ... NoData correctly")

    Returns
    -------
    float32 DOP array (H, W), physically bounded to [0, 1]; invalid
    pixels are set to `nodata`.

    Invalid pixel criteria
    -----------------------
    - S0 <= 0                      (non-physical / zero total power)
    - NaN or Inf in any of S0..S3  ("Handle NaN / Inf ... correctly")
    - Negative power inputs        (S0 < 0 is caught above; S1/S2/S3
                                     are signed by definition and are
                                     NOT flagged just for being negative)
    - Numerically inconsistent Stokes vector (|S1,S2,S3| > S0), which
      would produce DOP > 1 -- physically impossible, clamps to invalid
    - input_invalid_mask == True   (upstream NoData in the SLI rasters)
    """
    s0 = np.asarray(S0, dtype=np.float64)
    s1 = np.asarray(S1, dtype=np.float64)
    s2 = np.asarray(S2, dtype=np.float64)
    s3 = np.asarray(S3, dtype=np.float64)

    finite = np.isfinite(s0) & np.isfinite(s1) & np.isfinite(s2) & np.isfinite(s3)

    # S0 is a sum of two mean powers and must be >= 0 physically; a
    # non-positive value is either NoData fill or numerical noise.
    nonpositive_s0 = ~(s0 > 0.0)

    mag = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
    dop_raw = mag / (s0 + epsilon)

    # Physically DOP in [0, 1]; a value > 1 signals numerical noise
    # (near-zero S0) rather than a real depolarisation state.
    out_of_range = dop_raw > 1.0 + 1e-6

    bad = ~finite | nonpositive_s0 | out_of_range
    if input_invalid_mask is not None:
        bad = bad | input_invalid_mask

    dop = np.clip(dop_raw, 0.0, 1.0).astype(np.float32)
    dop[bad] = nodata

    n_bad = int(bad.sum())
    n_tot = int(dop.size)
    log.info(
        f"  DOP computed: total={n_tot:,}  invalid={n_bad:,} "
        f"({100 * n_bad / n_tot:.3f}%)  valid={n_tot - n_bad:,}"
    )

    valid = dop[~bad]
    if valid.size > 0:
        log.info(
            f"  DOP valid: min={valid.min():.4f}  max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  median={np.median(valid):.4f}"
        )

    return dop
