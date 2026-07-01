"""
step01_load_dem.py  —  Load the LOLA south-polar DEM into memory.

Purpose
-------
Read the LOLA 20 m/px south-polar elevation model from its PDS3 label file
(.lbl + .img pair) or GeoTIFF, convert from kilometres to metres, and return
a C-contiguous float32 array together with the rasterio metadata dictionary
needed for writing output rasters.

Inputs
------
DEM_PATH : Path to the PDS3 label file  (defined in utils.py)
           Expects the associated binary .img in the same directory.
           rasterio reads LOLA PDS3 labels natively via GDAL's PDS driver.

           DEM specification:
             Dataset   : LOLA 20 m/px south polar DEM (Smith et al. 2010)
             Coverage  : 85°S – 90°S (full south polar region)
             Shape     : ~15 168 × 15 168 pixels
             Projection: Moon South Polar Stereographic (ESRI:104903 or custom)
             Units     : kilometres in file → converted to metres here
             No-data   : LOLA uses -32768 (km) for missing; treated as-is

Outputs
-------
elevation : np.ndarray, float32, shape (H, W), C-contiguous
            Elevation in metres above the LOLA reference sphere.
meta      : dict
            rasterio write-metadata (driver, CRS, transform, count, dtype …)
            Used by all downstream steps to write co-registered output rasters.

Mathematical basis
------------------
No computation is performed here.  The raw pixel values are multiplied by
1000 to convert km → m.  All subsequent elevation angle calculations
(step04_visibility.py) require elevations in metres to be consistent with
the horizontal distances, also in metres.

Time complexity  : O(H × W)  —  single read pass over all pixels
Memory complexity: O(H × W)  —  ~15 168² × 4 bytes ≈ 920 MB for float32

Optimisation strategy
---------------------
• Read directly into float32 (not the file's int16) via rasterio's out_dtype
  parameter — avoids a separate cast allocation.
• np.ascontiguousarray ensures C-order row layout, critical for cache-friendly
  row traversal in the Numba ray-casting kernel.
• Data is read once at startup and kept in RAM for the entire pipeline; no
  repeated disk I/O in inner loops.

Reference
---------
Smith, D.E. et al. (2010). The Lunar Orbiter Laser Altimeter Investigation
on the Lunar Reconnaissance Orbiter Mission. Space Science Reviews, 150, 209.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dpsr.utils import DEM_PATH, CELLSIZE, get_logger

log = get_logger(__name__)


def load_dem(dem_path: Path = DEM_PATH) -> tuple[np.ndarray, dict]:
    """
    Load the LOLA DEM and return elevation array + rasterio metadata.

    Parameters
    ----------
    dem_path : Path to the DEM label or GeoTIFF file.

    Returns
    -------
    elevation : float32 ndarray (H, W), elevation in metres, C-contiguous
    meta      : dict — rasterio metadata for writing output GeoTIFFs

    Raises
    ------
    FileNotFoundError : if dem_path does not exist
    rasterio.errors.RasterioIOError : if GDAL cannot open the file

    Notes
    -----
    LOLA PDS3 .lbl files are opened transparently by rasterio/GDAL using the
    PDS3 driver.  The .img binary must be in the same directory.
    """
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM not found: {dem_path}\n"
            "Download LOLA 20 m/px south-polar DEM from:\n"
            "  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        )

    with rasterio.open(dem_path) as ds:
        raw  = ds.read(1, out_dtype=np.float32)
        meta = ds.meta.copy()

    elevation = np.ascontiguousarray(raw * 1000.0, dtype=np.float32)

    meta.update(dtype="float32", count=1, compress="lzw", driver="GTiff")

    log.info(
        "DEM loaded  shape=(%d, %d)  cellsize=%.0f m  RAM=%.0f MB  "
        "elev_range=[%.0f, %.0f] m",
        elevation.shape[0], elevation.shape[1],
        CELLSIZE,
        elevation.nbytes / 1e6,
        float(elevation.min()), float(elevation.max()),
    )

    return elevation, meta


# ---------------------------------------------------------------------------
# CLI entry point — run as: python step01_load_dem.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    elev, meta = load_dem()
    print(f"\n  Shape     : {elev.shape}")
    print(f"  dtype     : {elev.dtype}")
    print(f"  RAM       : {elev.nbytes / 1e6:.0f} MB")
    print(f"  Elev min  : {elev.min():.0f} m")
    print(f"  Elev max  : {elev.max():.0f} m")
    print(f"  Elev mean : {elev.mean():.0f} m")
    print(f"  CRS       : {meta.get('crs', 'unknown')}")
    print(f"  Transform : {meta.get('transform', 'unknown')}")
