"""
main.py — Single entry point for the entire DPSR pipeline.

Usage
-----
    python main.py              # auto-detect GPU; fall back to CPU
    python main.py --cpu        # force CPU (Numba parallel)
    python main.py --gpu        # force GPU (Numba CUDA)
    python main.py --redo       # recompute all intermediate files
    python main.py --annual     # use annual illumination (all azimuths, ~72x slower)

Pipeline steps (skipped automatically if output already exists)
---------------------------------------------------------------
    [1]  Load DEM + PSR shapefile
    [2]  Rasterize PSR shapefile   →  results/PSR_mask.tif
    [3]  Compute slope + aspect    →  results/slope.tif, aspect.tif
    [4]  Compute hillshade         →  results/hillshade.tif  (visualisation only)
    [5]  Solar shadow map          →  results/illumination.tif  ← physically based
    [6]  DPSR ray-casting          →  results/DPSR.tif
    [7]  Validate against known points (rim vs. interior)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.crs import CRS

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
(ROOT / "images").mkdir(exist_ok=True)

DEM_PATH          = DATA_DIR / "ldem_85s_20m_float.lbl"
PSR_SHP_PATH      = DATA_DIR / "LPSR_80S_20MPP_ADJ.shp"

PSR_MASK_PATH     = OUT_DIR / "PSR_mask.tif"
SLOPE_PATH        = OUT_DIR / "slope.tif"
ASPECT_PATH       = OUT_DIR / "aspect.tif"
HILLSHADE_PATH    = OUT_DIR / "hillshade.tif"
ILLUMINATION_PATH = OUT_DIR / "illumination.tif"
DPSR_PATH         = OUT_DIR / "DPSR.tif"

# ── Sun geometry ──────────────────────────────────────────────────────────────
# Peak solar elevation at the lunar south pole (~89.5°S) is ~1.54°.
# Azimuth for a single-epoch run; use --annual to sweep all azimuths.
SUN_AZIMUTH   = 0.0    # degrees, 0=North 90=East — change per epoch
SUN_ELEVATION = 1.54   # degrees — peak solar elevation at 89.5°S

# Search radius: 2500 px × 20 m = 50 km, covers Amundsen crater (~103 km diam)
MAX_DISTANCE = 2500
CELLSIZE     = 20.0    # metres per pixel


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def skip(path: Path) -> bool:
    if path.exists():
        print(f"  [SKIP] {path.name} already exists")
        return True
    return False

def done(path: Path, t0: float) -> None:
    mb = path.stat().st_size / 1e6
    print(f"  [DONE] {path.name}  ({mb:.1f} MB)  {time.perf_counter()-t0:.1f} s")


def save_preview(tif_path: Path, cmap: str = "gray", label: str = None,
                 vmin=None, vmax=None, pct_clip: float = 2.0) -> None:
    """Downsample TIF to ≤1024px and save PNG to images/."""
    if not tif_path.exists():
        return
    from rasterio.enums import Resampling
    png_path = ROOT / "images" / (tif_path.stem + ".png")
    with rasterio.open(tif_path) as ds:
        scale  = min(1.0, 1024 / max(ds.width, ds.height))
        out_h  = max(1, int(ds.height * scale))
        out_w  = max(1, int(ds.width  * scale))
        arr    = ds.read(1, out_shape=(out_h, out_w),
                         resampling=Resampling.average).astype(np.float32)
    if vmin is None:
        vmin = float(np.nanpercentile(arr, pct_clip))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 100 - pct_clip))
    if vmin == vmax:
        vmin, vmax = float(arr.min()), float(arr.max())

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=13, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview  -> images/{png_path.name}")


# ── Step 1: Load DEM ───────────────────────────────────────────────────────────

def load_dem():
    print("  Loading DEM …")
    with rasterio.open(DEM_PATH) as ds:
        elevation = ds.read(1, out_dtype=np.float32) * 1000.0   # km → m
        meta      = ds.meta.copy()
        meta.update(dtype="float32", count=1, compress="lzw", driver="GTiff")
    print(f"  DEM shape : {elevation.shape}   RAM : {elevation.nbytes/1e6:.0f} MB")
    return elevation, meta


# ── Step 2: PSR mask ───────────────────────────────────────────────────────────

def make_psr_mask(elevation: np.ndarray, meta: dict) -> np.ndarray:
    if skip(PSR_MASK_PATH):
        with rasterio.open(PSR_MASK_PATH) as ds:
            return ds.read(1)

    t0 = time.perf_counter()
    print("  Loading PSR shapefile …")
    psr = gpd.read_file(PSR_SHP_PATH)

    # Read the DEM's CRS and transform.  rasterio reads both correctly from the
    # PDS3 label (POLAR_STEREOGRAPHIC MOON, Affine(20,0,-151680,0,-20,151680)).
    with rasterio.open(DEM_PATH) as dem_ds:
        dem_crs       = dem_ds.crs
        dem_transform = dem_ds.transform

    # The shapefile has no .prj — its coordinates are already in the same
    # Lunar South Polar Stereographic metres as the DEM.
    if psr.crs is None:
        psr = psr.set_crs(dem_crs)
    elif psr.crs != dem_crs:
        psr = psr.to_crs(dem_crs)

    print("  Rasterizing PSR shapefile …")
    mask = rasterize(
        [(geom, 1) for geom in psr.geometry],
        out_shape = elevation.shape,
        transform = dem_transform,
        fill      = 0,
        dtype     = "uint8",
    )

    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, nodata=None)
    with rasterio.open(PSR_MASK_PATH, "w", **out_meta) as dst:
        dst.write(mask, 1)

    done(PSR_MASK_PATH, t0)
    print(f"  PSR pixels : {mask.sum():,}")
    return mask


# ── Step 3: Slope + Aspect ────────────────────────────────────────────────────

def make_slope_aspect(elevation: np.ndarray, meta: dict):
    if skip(SLOPE_PATH) and skip(ASPECT_PATH):
        with rasterio.open(SLOPE_PATH) as s, rasterio.open(ASPECT_PATH) as a:
            return s.read(1), a.read(1)

    t0 = time.perf_counter()
    print("  Computing gradient (may take a moment for 15k×15k DEM) …")
    dz_dy, dz_dx = np.gradient(elevation, CELLSIZE)

    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
    aspect    = np.degrees(np.arctan2(-dz_dx, dz_dy))
    aspect    = ((aspect + 360) % 360).astype(np.float32)

    out_meta = meta.copy()
    out_meta.update(dtype="float32", count=1)
    with rasterio.open(SLOPE_PATH,  "w", **out_meta) as dst: dst.write(slope_deg, 1)
    with rasterio.open(ASPECT_PATH, "w", **out_meta) as dst: dst.write(aspect,    1)

    done(SLOPE_PATH, t0)
    return slope_deg, aspect


# ── Step 4: Hillshade (visualisation only, not used for illumination) ─────────

def make_hillshade(elevation: np.ndarray, meta: dict) -> None:
    if skip(HILLSHADE_PATH):
        return
    t0 = time.perf_counter()
    print("  Computing hillshade (visualisation) …")
    dz_dy, dz_dx = np.gradient(elevation, CELLSIZE)
    slope  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az_rad = np.radians(SUN_AZIMUTH)
    el_rad = np.radians(SUN_ELEVATION)
    hillshade = (
        np.sin(el_rad) * np.cos(slope)
        + np.cos(el_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    ).clip(0, 1).astype(np.float32)
    out_meta = meta.copy()
    out_meta.update(dtype="float32", count=1)
    with rasterio.open(HILLSHADE_PATH, "w", **out_meta) as dst:
        dst.write(hillshade, 1)
    done(HILLSHADE_PATH, t0)


# ── Step 5: Solar shadow map (replaces hillshade threshold) ───────────────────
#
# Previous approach:  hillshade > 0.20  (approximation — ignores terrain blocking)
# Current approach:   ray-cast from each pixel toward the sun and check whether
#                     any terrain exceeds the sun elevation angle.
#
# Use --annual to sweep all azimuths (best for DPSR); default uses SUN_AZIMUTH.

def make_illumination(elevation: np.ndarray, meta: dict,
                      annual: bool = False,
                      force_cpu: bool = False,
                      force_gpu: bool = False) -> np.ndarray:
    if skip(ILLUMINATION_PATH):
        with rasterio.open(ILLUMINATION_PATH) as ds:
            return ds.read(1)

    from pipeline.step_illumination import (
        compute_solar_illumination,
        compute_annual_illumination,
        _CUDA_AVAILABLE,
    )

    # Determine backend
    if force_gpu and not _CUDA_AVAILABLE:
        raise RuntimeError("--gpu requested but CUDA is not available.")
    use_gpu = True  if force_gpu else \
              False if force_cpu  else \
              _CUDA_AVAILABLE

    backend_str = "GPU (CUDA)" if use_gpu else "CPU (Numba parallel)"

    t0 = time.perf_counter()
    if annual:
        print(f"  Solar shadow — annual sweep  72 azimuths  el={SUN_ELEVATION}°"
              f"  [{backend_str}] …")
        illum = compute_annual_illumination(
            elevation, n_azimuths=72,
            sun_el_deg=SUN_ELEVATION, cellsize=CELLSIZE, max_dist=MAX_DISTANCE,
            use_gpu=use_gpu,
        )
    else:
        print(f"  Solar shadow — single epoch  az={SUN_AZIMUTH}°  el={SUN_ELEVATION}°"
              f"  [{backend_str}] …")
        illum = compute_solar_illumination(
            elevation, sun_az_deg=SUN_AZIMUTH, sun_el_deg=SUN_ELEVATION,
            cellsize=CELLSIZE, max_dist=MAX_DISTANCE,
            use_gpu=use_gpu,
        )

    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, compress="lzw", nodata=None)
    with rasterio.open(ILLUMINATION_PATH, "w", **out_meta) as dst:
        dst.write(illum, 1)

    done(ILLUMINATION_PATH, t0)
    print(f"  Illuminated pixels : {illum.sum():,} / {illum.size:,}"
          f"  ({100*illum.mean():.1f}%)")
    return illum


# ── Validation: spot-check known points ───────────────────────────────────────
#
# A simple sanity check before trusting the DPSR map:
#   •  A point on the sunlit crater rim should see illuminated terrain.
#   •  A point deep in the crater interior should not.
#
# Row/col coordinates below are approximate for the LOLA 20 m south-polar DEM.
# Replace with exact coordinates from your PSR shapefile or QGIS inspection.

def validate_spot_checks(elevation, illumination, dpsr_raster, psr_mask):
    print("\n  Spot-check validation:")
    # Coordinates verified by debug_validation.py:
    #   Shackleton PSR centroid (233.7 km2 polygon nearest south pole):
    #     projected (8449, -6649) m  →  row=7916, col=8006
    #     psr=1, illum=0 confirmed by polygon intersection test
    #   Rim point: 10 km north of centroid along col=8006, outside PSR polygon
    # Per O'Brien & Byrne (2022) Table 1, DPSRs within Shackleton are
    # small sub-craters on the open floor, not the floor itself.
    # Open PSR floor is PSR but NOT necessarily DPSR.
    checks = [
        ("Shackleton rim (should see sunlight)",        7416, 8006, False),
        ("Shackleton PSR floor (PSR, not DPSR floor)",  7916, 8006, False),
    ]
    for label, row, col, expect_dpsr in checks:
        h = elevation[row, col]
        lit = illumination[row, col]
        dpsr = dpsr_raster[row, col]
        psr  = psr_mask[row, col]
        ok = (bool(dpsr) == expect_dpsr)
        status = "OK" if ok else "MISMATCH"
        print(f"    [{status}] {label}")
        print(f"           elev={h:.0f} m  illum={lit}  psr={psr}  dpsr={dpsr}"
              f"  (expected dpsr={int(expect_dpsr)})")


# ── Step 6: DPSR  (O'Brien & Byrne 2022) ─────────────────────────────────────
#
# Implements "Double Shadows at the Lunar Poles", PSJ 3:258.
#
# A pixel is DPSR iff:
#   1. It is permanently shadowed  (psr_mask == 1)
#   2. It has NO line of sight to ANY non-PSR surface  (psr_mask == 0)
#      along any of 360 azimuth directions (paper: 720).
#
# Key differences from naive approach:
#   - Visibility target = psr_mask == 0   (NOT illumination raster)
#   - Curvature-corrected elevation angle (Eq. A4 from paper Appendix)
#   - Post-processing: remove components < 5 pixels (8-connected)

def make_dpsr(elevation: np.ndarray, psr_mask: np.ndarray, meta: dict,
              force_cpu: bool = False, force_gpu: bool = False,
              n_angles: int = 360, max_dist: int = 2500) -> np.ndarray:
    """
    Compute DPSR map using O'Brien & Byrne (2022) method.

    Parameters
    ----------
    n_angles  : azimuth directions  (paper: 720; default: 360 at 1 deg)
    max_dist  : max ray pixels      (paper: 7500=150km; default: 2500=50km)
    """
    if skip(DPSR_PATH):
        with rasterio.open(DPSR_PATH) as ds:
            return ds.read(1)

    from pipeline.step_dpsr_obrien import compute_dpsr, _CUDA_AVAILABLE

    if force_gpu and not _CUDA_AVAILABLE:
        raise RuntimeError("--gpu requested but CUDA is not available.")
    use_gpu = True  if force_gpu else \
              False if force_cpu  else \
              _CUDA_AVAILABLE

    t0 = time.perf_counter()
    dpsr_raster = compute_dpsr(
        elevation, psr_mask,
        n_angles=n_angles,
        max_dist=max_dist,
        cellsize=CELLSIZE,
        use_gpu=use_gpu,
        min_component=5,
    )

    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, compress="lzw", nodata=None)
    with rasterio.open(DPSR_PATH, "w", **out_meta) as dst:
        dst.write(dpsr_raster, 1)

    done(DPSR_PATH, t0)
    return dpsr_raster


# ── Master pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPSR full pipeline")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU mode")
    parser.add_argument("--gpu",    action="store_true", help="Force GPU mode")
    parser.add_argument("--redo",   action="store_true",
                        help="Recompute all intermediate files")
    parser.add_argument("--annual", action="store_true",
                        help="Use annual illumination (sweep 72 azimuths)")
    args = parser.parse_args()

    if args.redo:
        for p in [PSR_MASK_PATH, SLOPE_PATH, ASPECT_PATH,
                  HILLSHADE_PATH, ILLUMINATION_PATH, DPSR_PATH]:
            if p.exists():
                p.unlink()
                print(f"  Removed {p.name}")

    t_total = time.perf_counter()

    banner("STEP 1 / 7  —  Load DEM")
    elevation, meta = load_dem()

    banner("STEP 2 / 7  —  PSR Mask")
    psr_mask = make_psr_mask(elevation, meta)
    save_preview(PSR_MASK_PATH, cmap="gray", label="PSR Mask", vmin=0, vmax=1)

    banner("STEP 3 / 7  —  Slope + Aspect")
    slope, aspect = make_slope_aspect(elevation, meta)
    del slope, aspect
    save_preview(SLOPE_PATH,  cmap="terrain", label="Slope (°)",    pct_clip=1)
    save_preview(ASPECT_PATH, cmap="hsv",     label="Aspect (°)",   vmin=0, vmax=360)

    banner("STEP 4 / 7  —  Hillshade  (visualisation only)")
    make_hillshade(elevation, meta)
    save_preview(HILLSHADE_PATH, cmap="gray", label="Hillshade", vmin=0, vmax=1)

    banner("STEP 5 / 7  —  Solar Shadow Map  (physically based)")
    illumination = make_illumination(elevation, meta, annual=args.annual,
                                     force_cpu=args.cpu, force_gpu=args.gpu)
    save_preview(ILLUMINATION_PATH, cmap="gray",
                 label="Illumination (annual)" if args.annual else f"Illumination (az={SUN_AZIMUTH}°)",
                 vmin=0, vmax=1)

    banner("STEP 6 / 7  —  DPSR  (O'Brien & Byrne 2022)")
    t6 = time.perf_counter()
    dpsr = make_dpsr(elevation, psr_mask, meta,
                     force_cpu=args.cpu, force_gpu=args.gpu)
    done(DPSR_PATH, t6)
    save_preview(DPSR_PATH, cmap="hot", label="DPSR (O'Brien & Byrne 2022)", vmin=0, vmax=1)

    banner("STEP 7 / 7  —  Validation  (spot checks)")
    validate_spot_checks(elevation, illumination, dpsr, psr_mask)

    # ── Final summary plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    illum_label = "Illumination (annual)" if args.annual else f"Illumination (az={SUN_AZIMUTH}°)"
    axes[0].imshow(psr_mask,     cmap="gray"); axes[0].set_title("PSR Mask")
    axes[1].imshow(illumination,  cmap="gray"); axes[1].set_title(illum_label)
    axes[2].imshow(dpsr,         cmap="hot");  axes[2].set_title(
        f"DPSR  ({dpsr.sum():,} pixels)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    fig_path = ROOT / "images" / "DPSR_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Summary plot -> {fig_path}")

    total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f} s ({total/60:.1f} min)")
    print(f"  Output → {DPSR_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
