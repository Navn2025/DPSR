"""
stokes.py
=========
STEP 6 -- Compute the modified Stokes parameters (S0, S1, S2, S3) from the
multilooked covariance matrix C3 built in covariance.py.

Derivation (full, not simplified)
----------------------------------
The classical Stokes parameters describe the second-order coherence of a
two-component (Jones) field. For a fixed illumination polarisation, the
radar receive channels form exactly such a two-component field, and the
2x2 Wolf coherency (mutual coherence) matrix

    J = | <Eh Eh*>   <Eh Ev*> |   = | J11  J12 |
        | <Ev Eh*>   <Ev Ev*> |     | J21  J22 |

maps onto the four real Stokes parameters via the standard optics
definitions (Born & Wolf, "Principles of Optics", Ch. 10):

    S0 = J11 + J22                       (total power)
    S1 = J11 - J22                       (linear H/V power imbalance)
    S2 = J12 + J21           = 2 Re(J12) (linear +-45deg component)
    S3 = i (J21 - J12)        = 2 Im(J12) (circular component)

This pipeline evaluates this at the reference illumination used
throughout monostatic full-pol SAR polarimetry -- horizontal (H)
transmit -- which is the standard, textbook radar specialisation of the
optical Stokes formalism (van Zyl & Kim, 2011, "Synthetic Aperture Radar
Polarimetry", JPL Space Science and Technology Series, Ch. 2; Lee &
Pottier, 2009, Ch. 2). Under H-transmit illumination:

    Eh = S_HH   (co-polarised receive channel)
    Ev = S_XP   (cross-polarised receive channel, reciprocity-averaged)

so:

    J11 = <|S_HH|^2>              = C11
    J22 = <|S_XP|^2>              = C22 / 2      (C22 = 2<|XP|^2>, see covariance.py)
    J12 = <S_HH . S_XP*>          = C12 / sqrt(2) (C12 = sqrt2<HH.XP*>)

Substituting directly into the Stokes definitions above:

    S0 = C11 + C22/2
    S1 = C11 - C22/2
    S2 = 2 Re(J12) = 2 Re(C12/sqrt2) = sqrt(2) * Re(C12)
    S3 = 2 Im(J12) = 2 Im(C12/sqrt2) = sqrt(2) * Im(C12)

No further simplification (e.g. dropping cross-terms, small-angle
approximations, or assuming reflection symmetry) is applied.

Physical interpretation
------------------------
    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0   (computed in dop.py)

    DOP ~ 1  -->  target behaves as a coherent, single (specular /
                  single-bounce Fresnel) scatterer -- smooth surface.
    DOP < 1  -->  depolarisation from multiple/volume scattering
                  (rough surfaces, subsurface heterogeneity) -- a
                  necessary (not sufficient) signature consistent with
                  buried volatile deposits, complementary to CPR.
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger("dop_pipeline.stokes")

SQRT2 = float(np.sqrt(2.0))


def compute_stokes_parameters(C3: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute S0, S1, S2, S3 from the multilooked covariance matrix C3.

    Parameters
    ----------
    C3 : dict as returned by covariance.build_covariance_matrix
         (must contain "C11", "C22", "C12")

    Returns
    -------
    dict with keys "S0", "S1", "S2", "S3" (all float32, same shape as C3 elements)
    """
    C11 = C3["C11"].astype(np.float64)
    C22 = C3["C22"].astype(np.float64)
    C12 = C3["C12"].astype(np.complex128)

    S0 = C11 + C22 / 2.0
    S1 = C11 - C22 / 2.0
    S2 = SQRT2 * C12.real
    S3 = SQRT2 * C12.imag

    stokes = {
        "S0": S0.astype(np.float32),
        "S1": S1.astype(np.float32),
        "S2": S2.astype(np.float32),
        "S3": S3.astype(np.float32),
    }

    for k, v in stokes.items():
        finite = v[np.isfinite(v)]
        if finite.size:
            log.info(f"  {k}: min={finite.min():.4e}  max={finite.max():.4e}  mean={finite.mean():.4e}")
        else:
            log.warning(f"  {k}: no finite values")

    # Sanity check inherent to the Stokes formalism: S0 >= sqrt(S1^2+S2^2+S3^2)
    # must hold for a physically valid partially-polarised wave. Log (do not
    # raise on) violations -- they indicate numerical noise near S0 ~ 0 and
    # are masked out downstream in dop.py.
    mag = np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2)
    violations = np.isfinite(S0) & np.isfinite(mag) & (mag > S0 * 1.0001)
    n_viol = int(violations.sum())
    if n_viol:
        log.warning(
            f"  Stokes inequality S0 >= |S1,S2,S3| violated at {n_viol} pixels "
            f"({100 * n_viol / S0.size:.4f}%) -- numerical noise near S0~0, "
            f"will be masked as invalid in DOP computation."
        )

    return stokes
