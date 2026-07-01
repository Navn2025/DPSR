"""
validator.py
============
Geometric consistency checks across all four polarisation channels
(STEP 2). Raises ValueError with a descriptive message on any mismatch.
"""

import logging
from typing import Dict

log = logging.getLogger("dop_pipeline.validator")


def validate_rasters(metas: Dict[str, dict]) -> None:
    """
    Verify that all raster bands share identical:
        - dimensions  (width, height)
        - CRS
        - affine transform
        - pixel resolution
        - band count

    Parameters
    ----------
    metas : dict mapping polarisation label -> metadata dict
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

    for pol in keys[1:]:
        m = metas[pol]

        if (m["width"], m["height"]) != (ref["width"], ref["height"]):
            errors.append(
                f"{pol} dimensions {m['width']}x{m['height']} != "
                f"{ref_key} {ref['width']}x{ref['height']}"
            )

        if m["crs"] != ref["crs"]:
            errors.append(
                f"{pol} CRS [{m['crs']}] != {ref_key} CRS [{ref['crs']}]"
            )

        if m["transform"] != ref["transform"]:
            errors.append(
                f"{pol} affine transform != {ref_key} transform"
            )

        if m["res"] != ref["res"]:
            errors.append(
                f"{pol} resolution {m['res']} != {ref_key} {ref['res']}"
            )

        if m["count"] != ref["count"]:
            errors.append(
                f"{pol} band count {m['count']} != {ref_key} {ref['count']}"
            )

    if errors:
        msg = "Raster consistency check FAILED:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        log.error(msg)
        raise ValueError(msg)

    log.info(
        f"Consistency check PASSED: all {len(keys)} bands share "
        f"identical dimensions ({ref['width']}W x {ref['height']}H), "
        f"CRS, transform, resolution."
    )
