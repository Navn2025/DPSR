"""
step_illumination.py — Physically based solar illumination / shadow map.

Replaces the hillshade-threshold proxy with a proper shadow-casting model.

Algorithm
---------
For each pixel P, cast a ray in the direction of the sun along the DEM.
P is illuminated iff no terrain along that ray exceeds the sun's elevation
angle:

    max { (h_d - h_P) / dist_d  :  d along ray toward sun } < tan(E)

where E is the sun elevation angle and dist_d is horizontal distance to step d.

This is the exact same horizon-angle test used in the DPSR visibility kernel
(step03), now applied in the sun direction instead of 72 directions.  It is
mathematically equivalent to GRASS r.sunmask and GDAL viewshed, both of which
use the same derivation.

Why this is better than hillshade > threshold
---------------------------------------------
Hillshade computes  I = cos(incidence_angle), a surface-normal dot product.
A pixel with a sun-facing slope has high hillshade, but it may still be
in shadow if a ridge between it and the sun blocks the ray.
The shadow-casting algorithm explicitly checks for that ridge, hillshade
cannot.

Annual illumination (recommended for DPSR)
------------------------------------------
DPSR definition: "never illuminated by the Sun at any point during the year
(or over a representative orbital period)."

Equivalently, the illumination input to the DPSR kernel should mark a pixel
as "possibly illuminated" if there EXISTS at least one sun position during
the year for which the pixel is not in shadow.

compute_annual_illumination() samples the sun at N azimuths (uniform over
360°) at the peak solar elevation and returns a binary mask of pixels that
receive sunlight from at least one direction.  This is a conservative
approximation; a full ephemeris gives the exact set.

For the lunar south pole (~89.5°S):
  - Max solar elevation: ~1.54°  (90° - latitude)
  - Solar azimuth: cycles 0→360° over one lunar month (27.3 days)
  - A pixel illuminated from any azimuth at that elevation is "ever lit"

Usage
-----
    from pipeline.step_illumination import (
        compute_solar_illumination,
        compute_annual_illumination,
    )

    # Single epoch
    illum = compute_solar_illumination(elevation, sun_az_deg=45.0, sun_el_deg=1.54)

    # Annual (recommended)
    illum = compute_annual_illumination(elevation, n_azimuths=72, sun_el_deg=1.54)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import math
import time

import numpy as np
from numba import njit, prange

from pipeline.utils import CELLSIZE, MAX_DISTANCE, SUN_ELEVATION, SUN_AZIMUTH, get_logger

log = get_logger("illumination")


# ── Ray precomputation ────────────────────────────────────────────────────────

def _sun_ray(sun_az_deg: float, max_dist: int, cellsize: float):
    """
    Compute integer (row, col) offsets and distances along the sun azimuth.

    Parameters
    ----------
    sun_az_deg : float
        Sun azimuth in degrees, geographic convention: 0 = North, 90 = East.
        Ray is cast FROM each pixel TOWARD the sun (i.e., in the direction
        the sun is located, not away from it).

    Returns
    -------
    dr   : int32  (max_dist,)  row offsets toward sun
    dc   : int32  (max_dist,)  col offsets toward sun
    dist : float32 (max_dist,) Euclidean horizontal distances in metres
    """
    az_rad = math.radians(sun_az_deg)

    # Geographic → raster: row+ = South, col+ = East
    #   sun azimuth A° from north means the sun is in direction
    #   (sin A) east, (cos A) north  →  raster row change = -(cos A)
    dr_unit = -math.cos(az_rad)
    dc_unit =  math.sin(az_rad)

    steps = np.arange(1, max_dist + 1, dtype=np.float64)
    dr    = np.round(dr_unit * steps).astype(np.int32)
    dc    = np.round(dc_unit * steps).astype(np.int32)
    dist  = (np.sqrt(dr.astype(np.float64)**2 +
                     dc.astype(np.float64)**2) * cellsize).astype(np.float32)
    dist  = np.where(dist < 1.0, np.float32(1.0), dist)
    return dr, dc, dist


# ── Numba shadow kernel ───────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _shadow_kernel(
    elevation: np.ndarray,   # float32  (H, W)
    sun_dr:    np.ndarray,   # int32    (max_dist,)
    sun_dc:    np.ndarray,   # int32    (max_dist,)
    sun_dist:  np.ndarray,   # float32  (max_dist,)
    sun_tan:   float,        # tan(sun_elevation_angle)
) -> np.ndarray:             # uint8    (H, W)  1=illuminated  0=shadow
    """
    Classify each DEM pixel as illuminated (1) or in shadow (0).

    A pixel P is in shadow if any terrain along the ray toward the sun
    has a terrain angle >= sun elevation angle:

        (h_d - h_P) / dist_d  >=  tan(sun_elevation)

    Identical derivation to the DPSR LOS kernel.
    """
    H, W     = elevation.shape
    max_dist = sun_dr.shape[0]
    result   = np.ones((H, W), dtype=np.uint8)    # default: illuminated

    for row in prange(H):
        for col in range(W):
            cur_h = elevation[row, col]

            for d in range(max_dist):
                r = row + sun_dr[d]
                c = col + sun_dc[d]

                if r < 0 or r >= H or c < 0 or c >= W:
                    break

                terrain_tan = (elevation[r, c] - cur_h) / sun_dist[d]

                if terrain_tan >= sun_tan:
                    result[row, col] = 0   # blocked → in shadow
                    break

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def compute_solar_illumination(
    elevation:   np.ndarray,
    sun_az_deg:  float = SUN_AZIMUTH,
    sun_el_deg:  float = SUN_ELEVATION,
    cellsize:    float = CELLSIZE,
    max_dist:    int   = MAX_DISTANCE,
) -> np.ndarray:
    """
    Compute a binary illumination map for a single sun position.

    Parameters
    ----------
    elevation   : float32 DEM array (H, W), metres
    sun_az_deg  : sun azimuth, degrees, 0=North 90=East
    sun_el_deg  : sun elevation above horizon, degrees
    cellsize    : metres per pixel
    max_dist    : maximum ray length in pixels

    Returns
    -------
    illumination : uint8 (H, W)  1=illuminated  0=shadow
    """
    sun_tan = math.tan(math.radians(sun_el_deg))
    dr, dc, dist = _sun_ray(sun_az_deg, max_dist, cellsize)

    log.info(
        "Shadow cast: az=%.1f deg  el=%.2f deg  sun_tan=%.4f  "
        "ray_len=%d px (%.0f km)",
        sun_az_deg, sun_el_deg, sun_tan, max_dist, max_dist * cellsize / 1000,
    )
    t0 = time.perf_counter()
    illum = _shadow_kernel(elevation, dr, dc, dist, sun_tan)
    log.info(
        "  done in %.1f s  illuminated=%.1f%%",
        time.perf_counter() - t0,
        100.0 * illum.sum() / illum.size,
    )
    return illum


def compute_annual_illumination(
    elevation:   np.ndarray,
    n_azimuths:  int   = 36,
    sun_el_deg:  float = SUN_ELEVATION,
    cellsize:    float = CELLSIZE,
    max_dist:    int   = MAX_DISTANCE,
) -> np.ndarray:
    """
    Approximate annual illumination by sweeping the sun through all azimuths.

    Returns a binary mask: 1 if the pixel is illuminated from AT LEAST ONE
    sun azimuth (i.e., it receives sunlight at some point during the year),
    0 if it is in shadow from all directions (permanent shadow = PSR candidate).

    This is a conservative approximation of the true annual illumination,
    which would require integration over the actual solar ephemeris.  For the
    lunar south pole the sun circles the horizon at near-constant elevation,
    so uniform azimuth sampling is a good approximation.

    Parameters
    ----------
    n_azimuths : number of azimuth samples (72 = every 5°, recommended)
    """
    azimuths = np.linspace(0.0, 360.0, n_azimuths, endpoint=False)
    combined = np.zeros(elevation.shape, dtype=np.uint8)

    log.info(
        "Annual illumination: %d azimuths  el=%.2f deg  "
        "ray_len=%d px (%.0f km)",
        n_azimuths, sun_el_deg, max_dist, max_dist * cellsize / 1000,
    )
    t_total = time.perf_counter()

    for i, az in enumerate(azimuths):
        illum    = compute_solar_illumination(
            elevation,
            sun_az_deg=float(az),
            sun_el_deg=sun_el_deg,
            cellsize=cellsize,
            max_dist=max_dist,
        )
        combined = np.maximum(combined, illum)   # union: ever illuminated
        pct      = 100.0 * combined.sum() / combined.size
        log.info("  az %5.1f deg  [%2d/%2d]  ever_lit=%.1f%%",
                 az, i + 1, n_azimuths, pct)

    log.info(
        "Annual illumination complete in %.0f s  "
        "ever_illuminated=%.1f%%  permanent_shadow=%.1f%%",
        time.perf_counter() - t_total,
        100.0 * combined.sum() / combined.size,
        100.0 * (combined == 0).sum() / combined.size,
    )
    return combined


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, rasterio
    from pipeline.utils import DEM_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Compute solar shadow map")
    parser.add_argument("--az",      type=float, default=SUN_AZIMUTH,
                        help="Sun azimuth degrees (0=North, 90=East)")
    parser.add_argument("--el",      type=float, default=SUN_ELEVATION,
                        help="Sun elevation degrees above horizon")
    parser.add_argument("--annual",  action="store_true",
                        help="Sweep all azimuths (annual illumination)")
    parser.add_argument("--n-az",    type=int,   default=72,
                        help="Number of azimuth samples for --annual")
    args = parser.parse_args()

    log.info("Loading DEM ...")
    with rasterio.open(DEM_PATH) as ds:
        elevation = ds.read(1).astype(np.float32) * 1000.0   # km → m
        meta      = ds.meta.copy()
        meta.update(dtype="uint8", count=1, compress="lzw")

    if args.annual:
        illum = compute_annual_illumination(elevation, n_azimuths=args.n_az,
                                            sun_el_deg=args.el)
        out   = OUTPUT_DIR / "illumination_annual.tif"
    else:
        illum = compute_solar_illumination(elevation, sun_az_deg=args.az,
                                           sun_el_deg=args.el)
        out   = OUTPUT_DIR / "illumination.tif"

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(illum, 1)
    log.info("Saved: %s", out)
