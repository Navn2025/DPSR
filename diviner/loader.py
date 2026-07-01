"""
loader.py
=========
STEP 1 + STEP 2 of the Diviner integration pipeline.

STEP 1 — Open each input file with rasterio and log:
          CRS, resolution, bounds, width, height, dtype, nodata, statistics.

STEP 2 — Convert .grd (GMT binary / NetCDF) files to float32 GeoTIFF.
          Original files are NEVER modified.
          Output TIFs go to outputs/diviner/.

All rasterio reads use a single-band read so the module stays compatible
with both the Diviner GeoTIFFs and PDS3 .img files loaded elsewhere.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

log = logging.getLogger("diviner_pipeline.loader")

# Moon ellipsoid semi-major axis (used when CRS is None/unknown to assign
# a placeholder geographic CRS for the Moon).
_MOON_PROJ = "+proj=longlat +R=1737400 +no_defs +type=crs"


# ---------------------------------------------------------------------------
# STEP 1 — Metadata reader
# ---------------------------------------------------------------------------

def read_metadata(path: Path, nodata_override: Optional[float] = None) -> dict:
    """
    Open a raster file with rasterio and return a rich metadata dict.

    Works for GeoTIFF, GRD (GMT NetCDF / binary), and PDS3 .img files.
    Band 1 is always read for statistics.

    Parameters
    ----------
    path            : path to the input raster.
    nodata_override : if the file has no nodata tag, treat this value as
                      nodata when computing statistics.

    Returns
    -------
    dict with keys: path, crs, transform, width, height, res, bounds,
    dtype, nodata, count, profile, stats.
    """
    with rasterio.open(path) as src:
        raw = src.read(1, masked=False)
        data = raw.astype(np.float64)

        nodata = src.nodata if src.nodata is not None else nodata_override

        # Build valid pixel mask
        valid_mask = np.isfinite(data)
        if nodata is not None:
            valid_mask &= data != float(nodata)
        valid = data[valid_mask]

        # For very large rasters (e.g. 15k×15k PSR/DEM) numpy's std()
        # internally promotes to float64 and needs ~2×nbytes.  Cap the
        # sample at 5M pixels to avoid OOM; results remain representative.
        _MAX_STAT_PX = 5_000_000
        valid_sample = valid
        if valid.size > _MAX_STAT_PX:
            idx = np.random.default_rng(seed=0).choice(
                valid.size, _MAX_STAT_PX, replace=False
            )
            valid_sample = valid[idx]

        stats: dict = {
            "min":   float(valid_sample.min())                  if valid_sample.size else float("nan"),
            "max":   float(valid_sample.max())                  if valid_sample.size else float("nan"),
            "mean":  float(valid_sample.mean())                 if valid_sample.size else float("nan"),
            "std":   float(valid_sample.std(dtype=np.float64))  if valid_sample.size else float("nan"),
            "valid": int(valid.size),
            "total": int(data.size),
        }

        crs = src.crs
        if crs is None:
            log.warning(
                f"  {path.name}: CRS is None — assigning Moon geographic "
                f"placeholder ({_MOON_PROJ}).  Verify before publishing results."
            )

        return {
            "path":      path,
            "crs":       crs,
            "transform": src.transform,
            "width":     src.width,
            "height":    src.height,
            "res":       src.res,
            "bounds":    src.bounds,
            "dtype":     src.dtypes[0],
            "nodata":    nodata,
            "count":     src.count,
            "profile":   src.profile.copy(),
            "stats":     stats,
        }


def print_metadata(label: str, meta: dict, logger: logging.Logger) -> None:
    """Log every metadata field for one dataset."""
    s = meta["stats"]
    logger.info(f"  ── {label} ──────────────────────────────────")
    logger.info(f"    File    : {meta['path'].name}")
    logger.info(f"    CRS     : {meta['crs']}")
    logger.info(f"    Res     : {meta['res'][0]:.8g} x {meta['res'][1]:.8g}  (x, y  CRS units)")
    logger.info(f"    Bounds  : {meta['bounds']}")
    logger.info(f"    Width   : {meta['width']}")
    logger.info(f"    Height  : {meta['height']}")
    logger.info(f"    Dtype   : {meta['dtype']}")
    logger.info(f"    NoData  : {meta['nodata']}")
    logger.info(f"    Bands   : {meta['count']}")
    logger.info(
        f"    Stats   : Min={s['min']:.4e}  Max={s['max']:.4e}  "
        f"Mean={s['mean']:.4e}  Std={s['std']:.4e}"
    )
    logger.info(f"    Valid   : {s['valid']:,} / {s['total']:,}  "
                f"({100 * s['valid'] / max(s['total'], 1):.2f} %)")


# ---------------------------------------------------------------------------
# STEP 2 — GRD → GeoTIFF conversion
# ---------------------------------------------------------------------------

def grd_to_tif(
    grd_path:     Path,
    out_dir:      Path,
    out_name:     str,
    nodata:       float           = -9999.0,
    crs_ref_path: Optional[Path] = None,
) -> Path:
    """
    Convert any GDAL-readable grid file (GMT .grd, NetCDF, etc.) to a
    float32 GeoTIFF with LZW compression.

    The original file is NEVER modified.  If the output already exists the
    function returns its path immediately without re-processing.

    Parameters
    ----------
    grd_path     : path to the source grid file.
    out_dir      : directory to write the converted TIF into.
    out_name     : output filename (e.g. "Tmean_converted.tif").
    nodata       : sentinel value written into the output for missing pixels.
    crs_ref_path : optional path to a georeferenced TIF whose CRS and
                   affine transform are assigned to the output.  Use this
                   when the GRD has no spatial reference (identity
                   transform) but is known to share the same grid as an
                   existing GeoTIFF (e.g. ZIT shares its grid with Tmean
                   and Pump from the same Diviner dataset).

    Returns
    -------
    Path to the converted GeoTIFF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists():
        log.info(f"  [skip] {out_name} already exists — conversion skipped.")
        return out_path

    log.info(f"  Converting  {grd_path.name}  →  {out_name}")

    with rasterio.open(grd_path) as src:
        data       = src.read(1).astype(np.float32)
        profile    = src.profile.copy()
        src_nodata = src.nodata

    # Normalise any pre-existing nodata to our sentinel
    if src_nodata is not None:
        bad_nd = np.float32(src_nodata)
        # Large fill-value sentinels (e.g. 9.97e+36) may not compare exactly
        # in float32; compare with tolerance for safety.
        data[np.abs(data - bad_nd) < 1e30] = np.float32(nodata)
    data[~np.isfinite(data)] = np.float32(nodata)

    # Borrow CRS and transform from a reference GeoTIFF when the GRD has none
    ref_crs       = None
    ref_transform = None
    if crs_ref_path is not None and crs_ref_path.exists():
        with rasterio.open(crs_ref_path) as ref:
            ref_crs       = ref.crs
            ref_transform = ref.transform
            ref_h, ref_w  = ref.height, ref.width

        if data.shape != (ref_h, ref_w):
            log.warning(
                f"  CRS reference shape ({ref_h}×{ref_w}) != "
                f"GRD shape {data.shape} — spatial reference may be wrong."
            )
        log.info(
            f"  Assigning CRS from {crs_ref_path.name}: "
            f"CRS={ref_crs}  transform={ref_transform}"
        )

    profile.update(
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        nodata    = nodata,
        compress  = "lzw",
        tiled     = False,
    )
    if ref_crs is not None:
        profile["crs"]       = ref_crs
    if ref_transform is not None:
        profile["transform"] = ref_transform

    # Remove format-specific keys that cannot be written to GTiff
    for key in ("AREA_OR_POINT", "scale_factor", "add_offset"):
        profile.pop(key, None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            SOURCE      = str(grd_path.name),
            PIPELINE    = "Diviner GRD→GeoTIFF conversion (diviner/loader.py)",
            NODATA      = str(nodata),
            CRS_SOURCE  = str(crs_ref_path.name) if crs_ref_path else "from GRD",
        )

    size_mb = out_path.stat().st_size / 1024 ** 2
    log.info(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
    return out_path
