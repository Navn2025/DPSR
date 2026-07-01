"""
reader.py  --  Read CPR GeoTIFFs, geometry CSV, and discover geometry files.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

log = logging.getLogger("validation.reader")


# ---------------------------------------------------------------------------
# Raster reading
# ---------------------------------------------------------------------------

def read_tiff_metadata(path: Path) -> dict:
    """Open a GeoTIFF and return its metadata without loading pixel data."""
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")
    with rasterio.open(path) as src:
        meta = {
            "path":      path,
            "width":     src.width,
            "height":    src.height,
            "count":     src.count,
            "crs":       src.crs,
            "transform": src.transform,
            "bounds":    src.bounds,
            "res":       src.res,
            "dtypes":    src.dtypes,
            "nodata":    src.nodata,
            "profile":   src.profile.copy(),
            "tags":      src.tags(),
        }
    log.debug(f"  Metadata: {path.name}  {meta['width']}W x {meta['height']}H  crs={meta['crs']}")
    return meta


def read_tiff_data(
    path: Path,
    nodata_override: Optional[float] = None,
    band: int = 1,
) -> Tuple[np.ndarray, dict]:
    """
    Load a raster band as float32.

    Returns (array, profile).
    nodata pixels are set to np.nan (for float) or kept as-is.
    """
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")
    with rasterio.open(path) as src:
        arr  = src.read(band).astype(np.float32)
        profile = src.profile.copy()
        nd = nodata_override if nodata_override is not None else src.nodata

    if nd is not None:
        if np.isnan(nd):
            arr[np.isnan(arr)] = np.nan
        else:
            arr[arr == nd] = np.nan

    log.info(
        f"  Loaded {path.name}: shape={arr.shape}  "
        f"valid={np.sum(np.isfinite(arr)):,}  nodata={nd}"
    )
    return arr, profile


# ---------------------------------------------------------------------------
# Geometry file discovery
# ---------------------------------------------------------------------------

def discover_geometry_files(geom_dir: Path) -> dict:
    """
    Search geom_dir for all available geometry inputs.
    Returns a dict of detected file types and paths.
    """
    found = {}
    if not geom_dir.exists():
        log.warning(f"Geometry directory not found: {geom_dir}")
        return found

    for f in sorted(geom_dir.iterdir()):
        name = f.name.lower()
        if "sli" in name and f.suffix == ".csv":
            found["sli_csv"] = f
        elif "gri" in name and f.suffix == ".csv":
            found["gri_csv"] = f
        elif "sri" in name and f.suffix == ".csv":
            found["sri_csv"] = f
        elif "oat" in name and f.suffix == ".csv":
            found["oat_csv"] = f
        elif f.suffix == ".xml":
            found.setdefault("xml_files", []).append(f)
        elif f.suffix in (".tif", ".tiff"):
            n = f.stem.lower()
            for key in ("lat", "lon", "latitude", "longitude",
                        "incidence", "slant_range"):
                if key in n:
                    found[key + "_raster"] = f
                    break

    log.info(f"Geometry files discovered in {geom_dir}:")
    for k, v in found.items():
        if isinstance(v, list):
            for vv in v:
                log.info(f"  [{k}] {vv.name}")
        else:
            log.info(f"  [{k}] {v.name}")
    return found


# ---------------------------------------------------------------------------
# Geometry CSV parsing
# ---------------------------------------------------------------------------

def read_geometry_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read the DFSAR geometry CSV for the SLI product.

    The CSV has no explicit row/col index columns. Each group of N_RNG_TIES rows
    belongs to one azimuth tie-line; the group size is detected automatically.

    Returns a DataFrame with columns:
        lat, lon, slant_range, incidence_angle, az_tie_idx, rng_tie_idx
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "lat" in cl:
            col_map[c] = "lat"
        elif "lon" in cl:
            col_map[c] = "lon"
        elif "slant" in cl or "range" in cl.split("(")[0]:
            col_map[c] = "slant_range"
        elif "incidence" in cl:
            col_map[c] = "incidence_angle"
    df = df.rename(columns=col_map)

    # Detect N_RNG_TIES by finding how many rows until slant_range resets
    sr = df["slant_range"].values
    diffs = np.diff(sr)
    # Count consecutive positives before first large negative (reset)
    reset_idx = np.where(diffs < -500)[0]
    if len(reset_idx) > 0:
        N_RNG = int(reset_idx[0]) + 1
    else:
        raise ValueError("Could not detect range tie group size from Slant_Range resets")

    N_ROWS = len(df)
    N_AZ   = N_ROWS // N_RNG
    if N_ROWS % N_RNG != 0:
        log.warning(f"CSV rows ({N_ROWS}) not evenly divisible by N_RNG ({N_RNG})")

    df["az_tie_idx"]  = np.arange(N_ROWS) // N_RNG
    df["rng_tie_idx"] = np.arange(N_ROWS) %  N_RNG

    log.info(
        f"  SLI geometry CSV: {N_AZ} azimuth tie rows x {N_RNG} range tie cols"
        f" = {N_ROWS} total tie points"
    )
    return df, N_AZ, N_RNG
