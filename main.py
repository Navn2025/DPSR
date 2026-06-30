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
PSR_SHP_PATH      = DATA_DIR / "LOLA_PSR_75S_120M_82S_060M_5KM2_FINAL.shp"

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
    # These are placeholder coordinates — update after inspecting the DEM
    checks = [
        ("Shackleton rim (should see sunlight)",  7580, 7580, False),
        ("Shackleton interior (should be DPSR)",  7584, 7584, True),
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


# ── Step 6: DPSR ray-casting ──────────────────────────────────────────────────

def make_dpsr(elevation: np.ndarray, illumination: np.ndarray,
              psr_mask: np.ndarray, meta: dict,
              force_cpu: bool, force_gpu: bool) -> np.ndarray:

    from pipeline.step01_load        import extract_psr_indices
    from pipeline.step02_precompute_rays import precompute_rays

    psr_rows, psr_cols = extract_psr_indices(psr_mask)
    ray_dr, ray_dc, ray_dist, ray_len = precompute_rays()

    # ── Decide CPU vs GPU ──────────────────────────────────────────────────────
    use_gpu = False
    if not force_cpu:
        try:
            from numba import cuda
            use_gpu = cuda.is_available()
        except Exception:
            pass
    if force_gpu and not use_gpu:
        raise RuntimeError("--gpu requested but no CUDA GPU found.")

    if use_gpu:
        print("  Backend : GPU (Numba CUDA)")
        dpsr_flags = _run_gpu(elevation, illumination, psr_rows, psr_cols,
                               ray_dr, ray_dc, ray_dist, ray_len)
    else:
        print("  Backend : CPU (Numba parallel, 12 cores)")
        dpsr_flags = _run_cpu(elevation, illumination, psr_rows, psr_cols,
                               ray_dr, ray_dc, ray_dist, ray_len)

    # Scatter flags back to raster
    dpsr_raster = np.zeros(elevation.shape, dtype=np.uint8)
    mask_idx    = dpsr_flags == 1
    dpsr_raster[psr_rows[mask_idx], psr_cols[mask_idx]] = 1

    # Save
    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, compress="lzw", nodata=None)
    with rasterio.open(DPSR_PATH, "w", **out_meta) as dst:
        dst.write(dpsr_raster, 1)

    print(f"  DPSR pixels : {dpsr_raster.sum():,}")
    return dpsr_raster


def _run_cpu(elevation, illumination, psr_rows, psr_cols,
             ray_dr, ray_dc, ray_dist, ray_len):
    from pipeline.step03_numba_raytrace import classify_psr_pixels, warmup
    print("  Compiling Numba kernel …")
    warmup(ray_dr, ray_dc, ray_dist, ray_len)
    print(f"  Processing {len(psr_rows):,} pixels …")
    t0 = time.perf_counter()
    flags = classify_psr_pixels(
        elevation, illumination, psr_rows, psr_cols,
        ray_dr, ray_dc, ray_dist, ray_len,
    )
    elapsed = time.perf_counter() - t0
    print(f"  CPU done : {elapsed:.1f} s ({elapsed/60:.1f} min)  "
          f"speed={len(psr_rows)/elapsed:,.0f} px/s")
    return flags


def _run_gpu(elevation, illumination, psr_rows, psr_cols,
             ray_dr, ray_dc, ray_dist, ray_len):
    import math
    from numba import cuda

    THREADS = 256

    @cuda.jit(cache=True)
    def _kernel(elev, illum, rows, cols, dr, dc, dist, rlen, out):
        i = cuda.grid(1)
        if i >= rows.shape[0]:
            return
        n_rows = elev.shape[0]; n_cols = elev.shape[1]
        n_ang  = dr.shape[0]
        row = rows[i]; col = cols[i]; cur_h = elev[row, col]
        is_dpsr = True
        for a in range(n_ang):
            ht = -1.0e18; ns = rlen[a]
            for d in range(ns):
                r = row + dr[a, d]; c = col + dc[a, d]
                if r < 0 or r >= n_rows or c < 0 or c >= n_cols: break
                tt = (elev[r, c] - cur_h) / dist[a, d]
                if illum[r, c] == 1 and tt >= ht:
                    is_dpsr = False; break
                if tt > ht: ht = tt
            if not is_dpsr: break
        out[i] = 1 if is_dpsr else 0

    print(f"  Transferring to GPU …")
    d_elev = cuda.to_device(elevation);    d_ill  = cuda.to_device(illumination)
    d_rows = cuda.to_device(psr_rows);    d_cols = cuda.to_device(psr_cols)
    d_dr   = cuda.to_device(ray_dr);      d_dc   = cuda.to_device(ray_dc)
    d_dist = cuda.to_device(ray_dist);    d_rlen = cuda.to_device(ray_len)
    d_out  = cuda.device_array(len(psr_rows), dtype=np.uint8)

    n_blocks = math.ceil(len(psr_rows) / THREADS)
    print(f"  Launching kernel  blocks={n_blocks}  threads={THREADS} …")
    t0 = time.perf_counter()
    _kernel[n_blocks, THREADS](d_elev, d_ill, d_rows, d_cols,
                                d_dr, d_dc, d_dist, d_rlen, d_out)
    cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  GPU done : {elapsed:.1f} s  speed={len(psr_rows)/elapsed:,.0f} px/s")
    return d_out.copy_to_host()


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

    banner("STEP 3 / 7  —  Slope + Aspect")
    slope, aspect = make_slope_aspect(elevation, meta)
    del slope, aspect

    banner("STEP 4 / 7  —  Hillshade  (visualisation only)")
    make_hillshade(elevation, meta)

    banner("STEP 5 / 7  —  Solar Shadow Map  (physically based)")
    illumination = make_illumination(elevation, meta, annual=args.annual,
                                     force_cpu=args.cpu, force_gpu=args.gpu)

    banner("STEP 6 / 7  —  DPSR Ray-Casting")
    t6 = time.perf_counter()
    dpsr = make_dpsr(elevation, illumination, psr_mask, meta,
                     force_cpu=args.cpu, force_gpu=args.gpu)
    done(DPSR_PATH, t6)

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
    print(f"\n  Summary plot → {fig_path}")

    total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f} s ({total/60:.1f} min)")
    print(f"  Output → {DPSR_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
