"""
setup_dem.py
============
One-time utility: convert the LOLA PDS3 DEM (.img + .lbl) to a GeoTIFF
so rasterio can read it normally throughout the pipeline.

Run once before main.py:
    python setup_dem.py

Input  : ../../data/ldem_85s_20m_float.img   (PDS3 binary, float32 LE)
Output : datasets/DEM/LOLA_DEM_20m.tif        (GeoTIFF, float32, Moon CRS)

The CRS and affine transform are derived from the PDS3 label and cross-
validated against the DPSR raster (which uses the identical projection).
"""
from __future__ import annotations

import sys
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent        # data_pipeline/
ISRO_ROOT    = PIPELINE_DIR.parent.parent             # ISRO_Hackathon/

DEM_IMG    = ISRO_ROOT / "data" / "ldem_85s_20m_float.img"
DEM_LBL    = ISRO_ROOT / "data" / "ldem_85s_20m_float.lbl"
DPSR_TIF   = ISRO_ROOT / "results" / "DPSR.tif"
OUTPUT_DIR = PIPELINE_DIR / "datasets" / "DEM"
OUTPUT_TIF = OUTPUT_DIR / "LOLA_DEM_20m.tif"

# ── PDS3 grid parameters (from the .lbl file) ─────────────────────────────────
# Derived from:  LINES = 15168, LINE_SAMPLES = 15168, MAP_SCALE = 20 m/pix
# LINE_PROJECTION_OFFSET = SAMPLE_PROJECTION_OFFSET = 7583.5 pix (pole at centre)
LINES        = 15168
LINE_SAMPLES = 15168
MAP_SCALE_M  = 20.0                      # metres per pixel
HALF_EXT     = (LINES / 2) * MAP_SCALE_M  # = 151 680 m from pole to edge

# Bounds of the raster in South Polar Stereographic metres
LEFT, RIGHT  = -HALF_EXT, +HALF_EXT
BOTTOM, TOP  =  -HALF_EXT, +HALF_EXT


def build_moon_crs_from_dpsr() -> CRS:
    """
    Read the CRS from the DPSR raster (same projection as the DEM).
    Falls back to a hand-crafted WKT if the DPSR file is missing.
    """
    if DPSR_TIF.exists():
        with rasterio.open(DPSR_TIF) as src:
            if src.crs:
                print(f"  CRS taken from DPSR: {src.crs.to_string()[:60]}…")
                return src.crs

    # Fallback: explicit WKT for Moon 2000 South Pole Stereographic
    wkt = (
        'PROJCS["Moon_2000_South_Pole_Stereographic",'
        'GEOGCS["GCS_Moon_2000",'
        'DATUM["D_Moon_2000",'
        'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
        'PRIMEM["Reference_Meridian",0],'
        'UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Polar_Stereographic"],'
        'PARAMETER["latitude_of_origin",-90],'
        'PARAMETER["central_meridian",0],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1]]'
    )
    print("  CRS: using fallback Moon 2000 South Pole Stereographic WKT.")
    return CRS.from_wkt(wkt)


def read_pds3_float32(img_path: Path) -> np.ndarray:
    """
    Read a PDS3 PC_REAL 32-bit binary image as a float32 NumPy array.

    PC_REAL = IEEE 754 single-precision, little-endian byte order.
    The label says RECORD_BYTES = 60672 = 15168 × 4, FILE_RECORDS = 15168,
    so the file is a raw (LINES × LINE_SAMPLES) float32 array with no header.
    """
    expected_bytes = LINES * LINE_SAMPLES * 4
    actual_bytes   = img_path.stat().st_size

    print(f"  File size : {actual_bytes:,} bytes  (expected {expected_bytes:,})")

    # Some PDS3 files have a small header before the image data.
    # Compute offset = file_size − image_size.
    offset = actual_bytes - expected_bytes
    if offset < 0:
        raise ValueError(
            f"File too small: {actual_bytes} bytes, need {expected_bytes}."
        )
    if offset > 0:
        print(f"  Header offset detected: {offset} bytes — skipping.")

    raw = np.fromfile(img_path, dtype="<f4", offset=offset)

    if raw.size != LINES * LINE_SAMPLES:
        raise ValueError(
            f"Element count mismatch: got {raw.size}, "
            f"expected {LINES * LINE_SAMPLES}."
        )

    dem = raw.reshape(LINES, LINE_SAMPLES)
    print(f"  Raw value range: min={dem.min():.2f}  max={dem.max():.2f}  "
          f"mean={dem.mean():.2f}")

    # Detect unit: if max < 50, values are probably in km -> convert to m
    if dem.max() < 50:
        print("  Values appear to be in km — converting to metres (×1000).")
        dem = dem * 1000.0
    else:
        print("  Values appear to be in metres — no unit conversion needed.")

    return dem.astype("float32")


def convert() -> None:
    print("\n" + "=" * 62)
    print("  LOLA DEM Converter  (PDS3 -> GeoTIFF)")
    print("=" * 62)

    # ── Pre-flight checks ─────────────────────────────────────────
    if not DEM_IMG.exists():
        print(f"\n[ERROR] DEM not found: {DEM_IMG}")
        sys.exit(1)

    if OUTPUT_TIF.exists():
        print(f"\n[SKIP] Output already exists: {OUTPUT_TIF}")
        print("  Delete it and re-run if you want to regenerate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── CRS ───────────────────────────────────────────────────────
    crs = build_moon_crs_from_dpsr()

    # ── Affine transform ──────────────────────────────────────────
    # from_bounds(left, bottom, right, top, width, height) derives:
    #   Affine(x_pixel, 0, left, 0, -y_pixel, top)
    transform = from_bounds(LEFT, BOTTOM, RIGHT, TOP, LINE_SAMPLES, LINES)
    print(f"  Affine transform: {transform}")

    # ── Read binary DEM ───────────────────────────────────────────
    print(f"\nReading PDS3 binary: {DEM_IMG.name}")
    dem = read_pds3_float32(DEM_IMG)

    # ── NoData check ──────────────────────────────────────────────
    # LOLA DEMs use 0 or very large values for missing pixels.
    # Apply a generous threshold: ignore extreme outliers.
    fill_val = dem.min()
    if fill_val < -20000 or fill_val > 20000:
        print(f"  Masking fill value: {fill_val:.2f}")
        dem[dem == fill_val] = np.nan

    print(f"\nFinal DEM stats (metres):")
    finite = dem[np.isfinite(dem)]
    print(f"  min   = {finite.min():.2f} m")
    print(f"  max   = {finite.max():.2f} m")
    print(f"  mean  = {finite.mean():.2f} m")
    print(f"  valid = {finite.size:,} / {dem.size:,} pixels")

    # ── Save GeoTIFF ──────────────────────────────────────────────
    print(f"\nSaving -> {OUTPUT_TIF}")
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=LINE_SAMPLES,
        height=LINES,
        count=1,
        crs=crs,
        transform=transform,
        nodata=float("nan"),
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(dem, 1)

    size_mb = OUTPUT_TIF.stat().st_size / 1_048_576
    print(f"  Done!  {size_mb:.0f} MB  ->  {OUTPUT_TIF}")
    print("\nYou can now run:  python main.py\n")


if __name__ == "__main__":
    convert()
