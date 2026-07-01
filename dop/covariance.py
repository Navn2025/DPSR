"""
covariance.py
=============
STEP 5 -- Construct the multilooked polarimetric covariance matrix (C3)
from the full quad-pol scattering matrix S, using the standard
lexicographic-basis formulation (Lee & Pottier, 2009, "Polarimetric
Radar Imaging: From Basics to Applications", CRC Press, Ch. 3).

Monostatic reciprocity
-----------------------
Chandrayaan-2 DFSAR is a monostatic radar, so the reciprocity theorem
S_HV = S_VH holds up to noise. The two cross-pol measurements are
therefore averaged into a single, lower-noise cross-pol channel:

    S_XP = (S_HV + S_VH) / 2

Lexicographic scattering vector
--------------------------------
    k_L = [ S_HH , sqrt(2) * S_XP , S_VV ]^T

The factor sqrt(2) preserves total power: |k_L|^2 = |S_HH|^2 + 2|S_XP|^2
+ |S_VV|^2, matching span(S) = |S_HH|^2 + |S_HV|^2 + |S_VH|^2 + |S_VV|^2
under reciprocity.

Covariance matrix
-----------------
    C3 = < k_L . k_L^H >

        | <|HH|^2>            sqrt2<HH.XP*>       <HH.VV*>       |
    C3= | sqrt2<XP.HH*>       2<|XP|^2>            sqrt2<XP.VV*>  |
        | <VV.HH*>            sqrt2<VV.XP*>        <|VV|^2>       |

C3 is Hermitian, so only the diagonal (3 real values) and the upper
triangle (3 complex values) are computed and stored.

CRITICAL CORRECTNESS NOTE
--------------------------
The ensemble average <.> MUST be applied to the per-pixel Hermitian
products (k_i * conj(k_j)) themselves, NOT to k_i and k_j separately
before multiplying. A single-look product k_i*conj(k_j) is a coherent
(rank-1) quantity; averaging the *products* over a multilook window is
what produces a physically meaningful, generally rank-3 (partially
polarized) covariance matrix. Averaging the fields first and only then
forming products would collapse the matrix back to rank-1 and make the
resulting DOP identically 1 everywhere, defeating the purpose of this
pipeline (see STEP 8 in dop.py / main.py).
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

log = logging.getLogger("dop_pipeline.covariance")

SQRT2 = float(np.sqrt(2.0))


def _boxcar(arr: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Real-valued boxcar (uniform) filter with reflect padding."""
    az, rg = window
    if az == 1 and rg == 1:
        return arr.astype(np.float32)
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)


def _boxcar_complex(arr: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Boxcar filter applied independently to real and imaginary parts."""
    out = np.empty_like(arr, dtype=np.complex64)
    out.real[:] = _boxcar(arr.real, window)
    out.imag[:] = _boxcar(arr.imag, window)
    return out


def build_covariance_matrix(
    S_HH: np.ndarray,
    S_HV: np.ndarray,
    S_VH: np.ndarray,
    S_VV: np.ndarray,
    window: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Build the multilooked 3x3 Hermitian covariance matrix C3.

    Parameters
    ----------
    S_HH, S_HV, S_VH, S_VV : complex64 arrays (H, W)
    window : (azimuth_lines, range_samples) multilook window,
             must match the CPR pipeline's MULTILOOK_WINDOW.

    Returns
    -------
    dict with keys:
        "C11", "C22", "C33"  -- float32, real diagonal elements
        "C12", "C13", "C23"  -- complex64, upper-triangle elements
        "span"               -- float32, total power = C11+C22+C33
    """
    az, rg = window
    log.info(
        f"Building covariance matrix C3 (lexicographic basis), "
        f"multilook window=({az}az x {rg}rg) -> {az * rg} looks"
    )

    # Reciprocity-averaged cross-pol channel.
    S_XP = (S_HV + S_VH) * 0.5

    # --- Single-look Hermitian outer-product elements ---------------------
    sl_c11 = (S_HH.real ** 2 + S_HH.imag ** 2)                      # |HH|^2
    sl_c22 = 2.0 * (S_XP.real ** 2 + S_XP.imag ** 2)                # 2|XP|^2
    sl_c33 = (S_VV.real ** 2 + S_VV.imag ** 2)                      # |VV|^2

    sl_c12 = SQRT2 * (S_HH * np.conj(S_XP))   # sqrt2<HH.XP*>
    sl_c13 = S_HH * np.conj(S_VV)             # <HH.VV*>
    sl_c23 = SQRT2 * (S_XP * np.conj(S_VV))   # sqrt2<XP.VV*>

    # --- Multilook (ensemble average) applied to the PRODUCTS -------------
    C11 = _boxcar(sl_c11, window)
    C22 = _boxcar(sl_c22, window)
    C33 = _boxcar(sl_c33, window)
    C12 = _boxcar_complex(sl_c12, window)
    C13 = _boxcar_complex(sl_c13, window)
    C23 = _boxcar_complex(sl_c23, window)

    span = (C11 + C22 + C33).astype(np.float32)

    log.info(
        f"  C11 (|HH|^2)  mean={C11.mean():.4e}   "
        f"C22 (2|XP|^2) mean={C22.mean():.4e}   "
        f"C33 (|VV|^2)  mean={C33.mean():.4e}"
    )
    log.info(f"  span = trace(C3) mean={span.mean():.4e}")

    return {
        "C11": C11, "C22": C22, "C33": C33,
        "C12": C12, "C13": C13, "C23": C23,
        "span": span,
    }
