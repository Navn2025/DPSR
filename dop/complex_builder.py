"""
complex_builder.py
==================
STEP 3 -- Construct the complex scattering matrix S from the four
Chandrayaan-2 DFSAR full-pol SLI channels:

        | S_HH  S_HV |
    S = |            |
        | S_VH  S_VV |

STEP 4 -- Derive amplitude, power and phase for each polarisation:

    Amplitude(S) = |S|            = sqrt(Re(S)^2 + Im(S)^2)
    Power(S)     = |S|^2          = Re(S)^2 + Im(S)^2
    Phase(S)     = arg(S)         = atan2(Im(S), Re(S))     in radians
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

log = logging.getLogger("dop_pipeline.complex_builder")


# ---------------------------------------------------------------------------
# STEP 3 -- Complex scattering matrix construction
# ---------------------------------------------------------------------------

def build_complex(real: np.ndarray, imag: np.ndarray, label: str) -> np.ndarray:
    """Form S = real + j*imag as a complex64 array."""
    slc = (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(np.complex64)
    log.debug(
        f"  SLC {label}: shape={slc.shape}  dtype={slc.dtype}  "
        f"mem={slc.nbytes / 1024**2:.1f} MB"
    )
    return slc


def build_scattering_matrix(
    S_HH: np.ndarray, S_HV: np.ndarray, S_VH: np.ndarray, S_VV: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Package the four complex scattering elements into the full 2x2
    Sinclair scattering matrix S, keyed by element name.

        S = [[S_HH, S_HV],
             [S_VH, S_VV]]

    Returned as a dict rather than a literal 2x2 array-of-arrays so each
    element stays a simple (H, W) complex64 raster.
    """
    for name, arr in (("HH", S_HH), ("HV", S_HV), ("VH", S_VH), ("VV", S_VV)):
        if arr.dtype != np.complex64:
            raise ValueError(f"S_{name} must be complex64, got {arr.dtype}")

    log.info(
        f"  Scattering matrix S assembled: shape={S_HH.shape}  "
        f"elements=[HH, HV, VH, VV]"
    )
    return {"HH": S_HH, "HV": S_HV, "VH": S_VH, "VV": S_VV}


# ---------------------------------------------------------------------------
# STEP 4 -- Amplitude / Power / Phase per polarisation
# ---------------------------------------------------------------------------

def amplitude_power_phase(S: np.ndarray, label: str) -> Dict[str, np.ndarray]:
    """
    Decompose a complex scattering channel into amplitude, power and phase.

    Parameters
    ----------
    S     : complex64 array (H, W) -- one polarisation channel
    label : polarisation label, used for logging only

    Returns
    -------
    dict with keys "amplitude", "power", "phase" (all float32)
    """
    amp   = np.abs(S).astype(np.float32)
    power = (S.real.astype(np.float32) ** 2 + S.imag.astype(np.float32) ** 2)
    phase = np.angle(S).astype(np.float32)   # radians, range (-pi, pi]

    log.info(
        f"  {label}: amplitude[min={amp.min():.4e}, max={amp.max():.4e}]  "
        f"power[min={power.min():.4e}, max={power.max():.4e}]  "
        f"phase[min={phase.min():.3f}, max={phase.max():.3f}] rad"
    )
    return {"amplitude": amp, "power": power.astype(np.float32), "phase": phase}


def multilook_power(power: np.ndarray, window: Tuple[int, int], label: str) -> np.ndarray:
    """
    Box-car multilook of a single-look power image (used only for the
    per-channel preview images in STEP 10, independent of the C3 matrix
    multilook applied in covariance.py).
    """
    az, rg = window
    if az == 1 and rg == 1:
        return power
    ml = uniform_filter(power.astype(np.float64), size=(az, rg)).astype(np.float32)
    log.debug(f"  Multilook {label}: ({az}az x {rg}rg)  min={ml.min():.4e}  max={ml.max():.4e}")
    return ml
