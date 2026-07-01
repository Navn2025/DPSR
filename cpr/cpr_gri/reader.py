"""
reader.py
=========
STEP 1 -- Read Chandrayaan-2 DFSAR calibrated GRI (Ground Range Image)
GeoTIFFs. Unlike SLI, each GRI file is a single-band intensity raster
(no separate real/imaginary bands).

Also parses the scene's radiometric calibration constant out of the
accompanying PDS4 XML label (needed by preprocessing.py to convert raw
digital numbers to calibrated backscatter -- see that module's docstring
for why this step turned out to be necessary).
"""

import logging
import re
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("cpr_gri_pipeline.reader")


# ---------------------------------------------------------------------------
# Metadata extraction  (STEP 1)
# ---------------------------------------------------------------------------

def read_metadata(path: Path) -> dict:
    """
    Open a GeoTIFF and return a metadata dict WITHOUT loading raster data.
    Raises FileNotFoundError if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"GRI file not found: {path}")

    with rasterio.open(path) as src:
        meta = {
            "path":      path,
            "filename":  path.name,
            "width":     src.width,
            "height":    src.height,
            "count":     src.count,
            "crs":       src.crs,
            "transform": src.transform,
            "res":       src.res,
            "bounds":    src.bounds,
            "dtypes":    src.dtypes,
            "nodata":    src.nodata,
            "profile":   src.profile.copy(),
        }
    log.debug(f"Metadata read: {path.name}  ({src.width}W x {src.height}H)")
    return meta


def print_metadata(label: str, meta: dict, logger: logging.Logger) -> None:
    """Log all key metadata fields for one raster channel."""
    logger.info(f"  Channel      : {label}")
    logger.info(f"  File         : {meta['filename']}")
    logger.info(f"  Width        : {meta['width']} samples")
    logger.info(f"  Height       : {meta['height']} lines")
    logger.info(f"  Data type    : {meta['dtypes']}")
    logger.info(f"  CRS          : {meta['crs']}")
    logger.info(f"  Resolution   : {meta['res']}")
    logger.info(f"  Transform    : {meta['transform']}")
    logger.info(f"  Bounds       : {meta['bounds']}")
    logger.info(f"  NoData value : {meta['nodata']}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_band(path: Path, label: str) -> np.ndarray:
    """
    Load band 1 of a single-band GRI GeoTIFF, preserving its native dtype
    (STEP 3's format-detection logic needs to see the real dtype, e.g.
    uint16 raw digital numbers vs. float32 physical units).
    """
    if not path.exists():
        raise FileNotFoundError(f"GRI file not found: {path}")

    with rasterio.open(path) as src:
        arr = src.read(1)

    log.info(
        f"  Loaded {label}: shape={arr.shape}  dtype={arr.dtype}  "
        f"mem={arr.nbytes / 1024**2:.1f} MB"
    )
    return arr


# ---------------------------------------------------------------------------
# Radiometric calibration constant (from PDS4 XML label)
# ---------------------------------------------------------------------------

_CAL_CONST_RE = re.compile(r"<isda:calibration_constant>\s*([-\d.eE]+)\s*</isda:calibration_constant>")


def parse_calibration_constant(xml_path: Path, fallback: float) -> float:
    """
    Extract the scene-wide radiometric calibration constant from the PDS4
    label (<isda:calibration_constant>), used by the standard DFSAR L1B
    calibration equation:

        sigma0_dB = 20 * log10(DN) - calibration_constant

    Uses a plain regex over the raw XML text rather than a full XML parser
    (e.g. xml.etree.ElementTree) to avoid dealing with the PDS4 namespace
    plumbing for a single well-known, uniquely-named tag; xml.etree is
    standard library (no extra install) but this is simpler and just as
    robust for this one value.

    Falls back to `fallback` (with a warning) if the label is missing or
    the tag cannot be found.
    """
    if not xml_path.exists():
        log.warning(
            f"PDS4 label not found ({xml_path.name}); using fallback "
            f"calibration constant = {fallback}"
        )
        return fallback

    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    m = _CAL_CONST_RE.search(text)
    if not m:
        log.warning(
            f"<isda:calibration_constant> not found in {xml_path.name}; "
            f"using fallback = {fallback}"
        )
        return fallback

    value = float(m.group(1))
    log.info(f"  Calibration constant parsed from label: {value}")
    return value
