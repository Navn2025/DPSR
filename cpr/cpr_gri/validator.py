"""
validator.py
============
STEP 2 -- Geometric consistency checks across the four polarisation
channels and the incidence-angle raster. Raises ValueError with a
descriptive message on any mismatch.
"""

import logging
from typing import Dict

log = logging.getLogger("cpr_gri_pipeline.validator")


def validate_rasters(metas: Dict[str, dict]) -> None:
    """
    Verify that all rasters share identical:
        - dimensions  (width, height)
        - CRS
        - affine transform
        - pixel resolution
        - bounds

    Parameters
    ----------
    metas : dict mapping channel label -> metadata dict
            (as returned by reader.read_metadata)

    Raises
    ------
    ValueError if any mismatch is found.
    """
    keys = list(metas.keys())
    if not keys:
        raise ValueError("No metadata to validate.")

    ref_key = keys[0]
    ref     = metas[ref_key]
    errors  = []

    for label in keys[1:]:
        m = metas[label]

        if (m["width"], m["height"]) != (ref["width"], ref["height"]):
            errors.append(
                f"{label} dimensions {m['width']}x{m['height']} != "
                f"{ref_key} {ref['width']}x{ref['height']}"
            )

        if m["crs"] != ref["crs"]:
            errors.append(f"{label} CRS [{m['crs']}] != {ref_key} CRS [{ref['crs']}]")

        if m["transform"] != ref["transform"]:
            errors.append(f"{label} affine transform != {ref_key} transform")

        if m["res"] != ref["res"]:
            errors.append(f"{label} resolution {m['res']} != {ref_key} {ref['res']}")

        if tuple(m["bounds"]) != tuple(ref["bounds"]):
            errors.append(f"{label} bounds {m['bounds']} != {ref_key} bounds {ref['bounds']}")

    if errors:
        msg = "Raster consistency check FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(msg)
        raise ValueError(msg)

    log.info(
        f"Consistency check PASSED: all {len(keys)} rasters share "
        f"identical dimensions ({ref['width']}W x {ref['height']}H), "
        f"CRS, transform, resolution, bounds."
    )
