"""
step01_load.py — Load DEM, illumination, and PSR mask into RAM.

Inputs
------
  DEM file     : PDS .lbl / .img  (15 168 × 15 168 float32)
  Illumination : GeoTIFF uint8    (same grid)
  PSR mask     : GeoTIFF uint8    (same grid; 1 = PSR pixel)

Outputs
-------
  elevation    : float32 C-contiguous ndarray  (H, W)  metres
  illumination : uint8   C-contiguous ndarray  (H, W)
  psr_mask     : uint8   C-contiguous ndarray  (H, W)
  meta         : dict    rasterio write-metadata (CRS, transform, …)
  psr_rows     : int32   ndarray  (P,)  row indices of PSR pixels
  psr_cols     : int32   ndarray  (P,)  col indices of PSR pixels

Complexity
----------
  O(H × W) — single pass over every pixel for each array.

Optimizations
-------------
  • float32 (not float64) halves elevation memory  (920 MB vs 1.84 GB)
  • np.ascontiguousarray ensures C-order; critical for cache-friendly
    row traversal in the inner ray loop.
  • All rasters are read once and kept in RAM for the full pipeline —
    no repeated disk I/O inside the ray-casting loop.
  • np.where used (not Python list comprehension) to extract PSR indices.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import rasterio
from rasterio.enums import Resampling

from pipeline.utils import DEM_PATH, ILLUMINATION_PATH, PSR_MASK_PATH, CELLSIZE, get_logger

log = get_logger(__name__)


def load_dem() -> tuple[np.ndarray, dict]:
    """
    Load the LOLA DEM into a float32 array.

    Input  : DEM_PATH (PDS .lbl pointing to .img, or GeoTIFF)
    Output : (elevation [float32 metres], rasterio meta dict)
    """
    with rasterio.open(DEM_PATH) as ds:
        raw  = ds.read(1, out_dtype=np.float32)   # read as float32 directly
        meta = ds.meta.copy()
        meta.update(dtype="uint8", count=1, compress="lzw")

    # LOLA LDEM stores values in kilometres → convert to metres
    elevation = np.ascontiguousarray(raw * 1000.0, dtype=np.float32)

    log.info(
        "DEM loaded  shape=%s  dtype=%s  RAM=%.0f MB",
        elevation.shape, elevation.dtype, elevation.nbytes / 1e6,
    )
    return elevation, meta


def load_illumination() -> np.ndarray:
    """
    Load the binary illumination map.

    Input  : ILLUMINATION_PATH (uint8 GeoTIFF; 1=illuminated 0=shadow)
    Output : uint8 C-contiguous ndarray (H, W)
    """
    with rasterio.open(ILLUMINATION_PATH) as ds:
        arr = ds.read(1, out_dtype=np.uint8)

    illum = np.ascontiguousarray(arr, dtype=np.uint8)
    log.info("Illumination loaded  shape=%s  RAM=%.0f MB",
             illum.shape, illum.nbytes / 1e6)
    return illum


def load_psr_mask() -> np.ndarray:
    """
    Load the rasterised PSR mask.

    Input  : PSR_MASK_PATH (uint8 GeoTIFF; 1=PSR, 0=non-PSR)
    Output : uint8 C-contiguous ndarray (H, W)
    """
    with rasterio.open(PSR_MASK_PATH) as ds:
        arr = ds.read(1, out_dtype=np.uint8)

    mask = np.ascontiguousarray(arr, dtype=np.uint8)
    log.info("PSR mask loaded  shape=%s  PSR pixels=%s",
             mask.shape, f"{np.count_nonzero(mask):,}")
    return mask


def extract_psr_indices(psr_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract row and column indices of every PSR pixel.

    Input
    -----
    psr_mask : uint8 (H, W) — 1 where pixel is inside a PSR

    Output
    ------
    psr_rows : int32 (P,) — row index of each PSR pixel
    psr_cols : int32 (P,) — col index of each PSR pixel

    Complexity : O(H × W)  —  single np.where call, no Python loop.
    """
    rows, cols = np.where(psr_mask == 1)
    psr_rows   = np.ascontiguousarray(rows, dtype=np.int32)
    psr_cols   = np.ascontiguousarray(cols, dtype=np.int32)
    log.info("PSR indices extracted  count=%s", f"{len(psr_rows):,}")
    return psr_rows, psr_cols


def validate_grids(elevation: np.ndarray,
                   illumination: np.ndarray,
                   psr_mask: np.ndarray) -> None:
    """
    Assert all three grids share the same shape.

    Raises ValueError if shapes differ (prevents silent misalignment bugs).
    """
    if not (elevation.shape == illumination.shape == psr_mask.shape):
        raise ValueError(
            f"Grid shape mismatch: elevation={elevation.shape}  "
            f"illumination={illumination.shape}  psr_mask={psr_mask.shape}"
        )
    log.info("Grid shapes validated: %s", elevation.shape)


def load_all() -> dict:
    """
    Convenience wrapper: load everything and return a single dict.

    Returns
    -------
    {
        "elevation"    : float32 (H, W),
        "illumination" : uint8   (H, W),
        "psr_mask"     : uint8   (H, W),
        "psr_rows"     : int32   (P,),
        "psr_cols"     : int32   (P,),
        "meta"         : dict,
    }
    """
    elevation,   meta = load_dem()
    illumination      = load_illumination()
    psr_mask          = load_psr_mask()

    validate_grids(elevation, illumination, psr_mask)

    psr_rows, psr_cols = extract_psr_indices(psr_mask)

    return dict(
        elevation    = elevation,
        illumination = illumination,
        psr_mask     = psr_mask,
        psr_rows     = psr_rows,
        psr_cols     = psr_cols,
        meta         = meta,
    )
