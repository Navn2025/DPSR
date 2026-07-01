"""
complex_builder.py
==================
Construct complex scattering matrix elements from I/Q bands, convert from
the linear (HH/HV/VH/VV) polarisation basis to the circular (RR/RL) basis,
and compute multilooked power images for CPR.

Circular basis conversion (monostatic reciprocal SAR, HV = VH)
---------------------------------------------------------------
    S_RR = (S_HH - S_VV + 2j * S_HV) / 2     [same-sense,     SC]
    S_RL = (S_HH + S_VV) / 2                   [opposite-sense, OC]

The factor 1/2 cancels in CPR = |S_RR|^2 / |S_RL|^2, so we drop it
for efficiency.  The un-normalised forms are:

    SC_field = S_HH - S_VV + 2j * S_HV
    OC_field = S_HH + S_VV

Reference: Putrevu et al. (2023), JGR Planets, DOI 10.1029/2023JE007745
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

log = logging.getLogger("cpr_pipeline.complex_builder")


# ---------------------------------------------------------------------------
# Complex SLC construction
# ---------------------------------------------------------------------------

def build_complex(real: np.ndarray, imag: np.ndarray, label: str) -> np.ndarray:
    """
    Form S = real + j*imag as a complex64 array.

    Parameters
    ----------
    real  : float32 Band-1 (I channel)
    imag  : float32 Band-2 (Q channel)
    label : polarisation label for logging

    Returns
    -------
    complex64 array (height, width)
    """
    slc = (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(np.complex64)
    log.debug(
        f"  SLC {label}: shape={slc.shape}  dtype={slc.dtype}  "
        f"mem={slc.nbytes / 1024**2:.1f} MB"
    )
    return slc


# ---------------------------------------------------------------------------
# Linear -> Circular polarisation basis conversion
# ---------------------------------------------------------------------------

def linear_to_circular(
    S_HH: np.ndarray,
    S_HV: np.ndarray,
    S_VH: np.ndarray,
    S_VV: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the full-pol linear scattering matrix to the circular basis.

    For monostatic SAR the reciprocity condition holds (S_HV = S_VH).
    Both cross-pol channels are averaged here to reduce noise.

    SC (same-sense circular, RR):
        SC_field = S_HH - S_VV + 2j * S_XP
    OC (opposite-sense circular, RL):
        OC_field = S_HH + S_VV

    The 1/2 normalisation factor is dropped because CPR = |SC|^2 / |OC|^2
    and it cancels exactly.

    Parameters
    ----------
    S_HH, S_HV, S_VH, S_VV : complex64 arrays (H, W)

    Returns
    -------
    (SC_field, OC_field) : complex64 arrays (H, W)
        Single-look complex fields before multilooking.
    """
    S_XP = (S_HV + S_VH) * 0.5     # averaged cross-pol (reciprocity)

    # OC = S_HH + S_VV  (no cross-pol contribution for monostatic)
    OC_field = S_HH + S_VV          # complex64

    # SC = S_HH - S_VV + 2j * S_XP
    #   real part: (HH_r - VV_r) - 2 * XP_i
    #   imag part: (HH_i - VV_i) + 2 * XP_r
    diff = S_HH - S_VV              # complex64
    SC_field = np.empty_like(diff)
    SC_field.real[:] = diff.real - 2.0 * S_XP.imag
    SC_field.imag[:] = diff.imag + 2.0 * S_XP.real

    log.info(
        f"  Circular conversion done.  "
        f"|SC| mean={np.mean(SC_field.real**2 + SC_field.imag**2)**0.5:.3e}  "
        f"|OC| mean={np.mean(OC_field.real**2 + OC_field.imag**2)**0.5:.3e}"
    )
    return SC_field, OC_field


def complex_to_power(field: np.ndarray, label: str) -> np.ndarray:
    """Return |field|^2 as float32."""
    power = (field.real ** 2 + field.imag ** 2).astype(np.float32)
    log.debug(
        f"  Power {label}: min={power.min():.4e}  max={power.max():.4e}  "
        f"mean={power.mean():.4e}"
    )
    return power


# ---------------------------------------------------------------------------
# Multilook (speckle reduction)
# ---------------------------------------------------------------------------

def apply_multilook(
    power: np.ndarray,
    window: Tuple[int, int],
    label: str,
) -> np.ndarray:
    """
    Box-car spatial averaging of a single-look power image.

    Parameters
    ----------
    power  : float32 array (H, W)
    window : (azimuth_lines, range_samples) averaging kernel
    label  : label for logging

    Returns
    -------
    float32 multilooked power array (same shape)

    Notes
    -----
    Uses scipy.ndimage.uniform_filter with 'reflect' padding to avoid
    edge artefacts.  The effective number of looks is az * rg.
    """
    az, rg = window
    if az == 1 and rg == 1:
        log.debug(f"  Multilook {label}: window=(1,1) -- skipped")
        return power

    ml = uniform_filter(power.astype(np.float64), size=(az, rg)).astype(np.float32)
    log.info(
        f"  Multilook {label}: window=({az}az x {rg}rg)  "
        f"effective looks={az * rg}  "
        f"ml_min={ml.min():.4e}  ml_max={ml.max():.4e}"
    )
    return ml
