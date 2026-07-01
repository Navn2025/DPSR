"""
reader.py
=========
Read Chandrayaan-2 DFSAR SLI (complex SLC) GeoTIFFs.

Each file contains:
    Band 1  ->  Real      component (I channel), float32
    Band 2  ->  Imaginary component (Q channel), float32
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio

log = logging.getLogger("cpr_pipeline.reader")


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def read_metadata(path: Path) -> dict:
    """
    Open a GeoTIFF and return a metadata dict WITHOUT loading raster data.
    Raises FileNotFoundError if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"SLI file not found: {path}")

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
            "dtypes":    src.dtypes,
            "nodata":    src.nodata,
            "profile":   src.profile.copy(),
        }
    log.debug(f"Metadata read: {path.name}  ({src.width}W x {src.height}H)")
    return meta


def print_metadata(pol: str, meta: dict, logger: logging.Logger) -> None:
    """Log all key metadata fields for one polarisation channel."""
    logger.info(f"  Polarisation : {pol}")
    logger.info(f"  File         : {meta['filename']}")
    logger.info(f"  Width        : {meta['width']} samples")
    logger.info(f"  Height       : {meta['height']} lines")
    logger.info(f"  Bands        : {meta['count']}  "
                f"(Band 1 = Real, Band 2 = Imaginary)")
    logger.info(f"  CRS          : {meta['crs']}")
    logger.info(f"  Transform    : {meta['transform']}")
    logger.info(f"  Resolution   : {meta['res']}")
    logger.info(f"  Data type    : {meta['dtypes']}")
    logger.info(f"  NoData value : {meta['nodata']}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_complex(path: Path, pol: str) -> np.ndarray:
    """
    Load both bands of an SLI GeoTIFF and return a complex64 array.

    Parameters
    ----------
    path : Path  -- file to read
    pol  : str   -- polarisation label used for log messages

    Returns
    -------
    np.ndarray, shape (height, width), dtype complex64
    """
    if not path.exists():
        raise FileNotFoundError(f"SLI file not found: {path}")

    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError(
                f"{path.name}: expected 2 bands (I/Q), found {src.count}"
            )
        real = src.read(1).astype(np.float32)
        imag = src.read(2).astype(np.float32)

    slc = real + 1j * imag
    log.info(
        f"  Loaded {pol}: shape={slc.shape}  dtype=complex64  "
        f"mem={slc.nbytes / 1024**2:.1f} MB"
    )
    return slc.astype(np.complex64)


def load_power(path: Path, pol: str) -> np.ndarray:
    """
    Load an SLI GeoTIFF and return backscatter power |S|^2
    WITHOUT keeping the full complex array in memory.

    Power = real^2 + imag^2 (equivalent to |S|^2)
    """
    if not path.exists():
        raise FileNotFoundError(f"SLI file not found: {path}")

    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError(
                f"{path.name}: expected 2 bands (I/Q), found {src.count}"
            )
        real = src.read(1).astype(np.float32)
        imag = src.read(2).astype(np.float32)

    power = real ** 2 + imag ** 2   # |S|^2 = I^2 + Q^2
    del real, imag                  # free memory immediately

    log.info(
        f"  Power {pol}: shape={power.shape}  "
        f"min={power.min():.4e}  max={power.max():.4e}  "
        f"mem={power.nbytes / 1024**2:.1f} MB"
    )
    return power
