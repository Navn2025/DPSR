"""
DPSR Extraction — Self-contained optimised implementation
=========================================================

Scientific algorithm: unchanged.
  A PSR pixel is DPSR if, in every horizontal direction, all
  illuminated terrain lies below the terrain horizon (no line-of-sight
  to sunlit ground from any azimuth).

Illumination model
------------------
  Previous approach: hillshade > threshold   (approximation)
  Current approach:  solar shadow casting    (physically based)

  For each pixel P, a ray is cast TOWARD the sun along the DEM.
  P is in shadow if any terrain along that ray satisfies:

      (h_d - h_P) / dist_d  >=  tan(sun_elevation)

  This is the same horizon-angle test as the DPSR kernel, applied
  in the sun direction rather than all 72 directions.  It correctly
  accounts for terrain that blocks sunlight even when the slope
  "faces" the sun (hillshade cannot detect this).

  Use ANNUAL_MODE = True to sweep all azimuths and produce the
  union illumination map (best for DPSR accuracy).

Search radius
-------------
  MAX_DISTANCE = 2500 px × 20 m = 50 km
  Justification:
    Shackleton  ~21 km diam → rim up to 10.5 km from interior
    Haworth     ~51 km diam → rim up to 25.5 km
    Amundsen   ~103 km diam → rim up to 51.5 km
  A 10 km radius (500 px) misses the illuminated rim of Haworth
  and larger craters.  Early exit keeps actual cost proportional
  to the blocking distance, not MAX_DISTANCE.

Computational complexity
------------------------
Let  P = PSR pixels (~1.5 M)
     A = azimuth rays  (72, one every 5°)
     D = max ray steps (2500 pixels × 20 m = 50 km)

  ┌──────────────────────────────────────────┬──────────────────┐
  │ Technique                                │ Speedup (approx) │
  ├──────────────────────────────────────────┼──────────────────┤
  │ Numba @njit — compiled machine code      │    50 – 100 ×    │
  │ prange — all CPU cores                   │     N_cores ×    │
  │ tan comparison instead of arctan/degrees │       1.5 ×      │
  │ Precomputed ray offsets (no trig/loop)   │         2 ×      │
  │ Early exit once visible terrain found    │    up to 36 ×    │
  ├──────────────────────────────────────────┼──────────────────┤
  │ TOTAL (conservative)                     │  ~1 000 – 2 000× │
  └──────────────────────────────────────────┴──────────────────┘

  Wall-clock estimate (DPSR step only): 7 days → 5 – 30 minutes
"""

import math
import time
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from numba import njit, prange

# ── Configuration ─────────────────────────────────────────────────────────────
N_ANGLES     = 72      # DPSR rays every 5°  (360 / 5 = 72)
MAX_DISTANCE = 2500    # max ray length in pixels  (2500 × 20 m = 50 km)
CELLSIZE     = 20.0    # DEM pixel size in metres

# Sun geometry (geographic: 0 = North, 90 = East)
# Peak solar elevation at lunar south pole (~89.5°S) is ~1.54°.
SUN_ELEVATION = 1.54   # degrees above horizon
SUN_AZIMUTH   = 0.0    # degrees — change per epoch or use ANNUAL_MODE

# Set True to sweep all azimuths for annual illumination (slower, more accurate)
ANNUAL_MODE   = False
N_AZ_ANNUAL   = 72     # azimuth samples when ANNUAL_MODE is True

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "results"
IMG_DIR  = BASE_DIR / "images"
OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Solar shadow / illumination map
# ═══════════════════════════════════════════════════════════════════════════════

def _sun_ray_offsets(sun_az_deg: float, max_dist: int, cellsize: float):
    """Integer (row, col) offsets and distances along the sun azimuth direction."""
    az_rad  = math.radians(sun_az_deg)
    # Geographic azimuth → raster  (row+ = South, col+ = East)
    dr_unit = -math.cos(az_rad)
    dc_unit =  math.sin(az_rad)
    steps   = np.arange(1, max_dist + 1, dtype=np.float64)
    dr      = np.round(dr_unit * steps).astype(np.int32)
    dc      = np.round(dc_unit * steps).astype(np.int32)
    dist    = (np.sqrt(dr.astype(np.float64)**2 +
                       dc.astype(np.float64)**2) * cellsize).astype(np.float32)
    dist    = np.where(dist < 1.0, np.float32(1.0), dist)
    return dr, dc, dist


@njit(parallel=True, cache=True, fastmath=True)
def _shadow_kernel(elevation, sun_dr, sun_dc, sun_dist, sun_tan):
    """
    Binary illumination map via solar shadow casting.

    Pixel P is in shadow if any terrain along the ray toward the sun
    exceeds the sun elevation angle:

        (h_d - h_P) / dist_d  >=  tan(sun_elevation)

    Identical derivation to the DPSR LOS kernel.
    Returns uint8 array: 1 = illuminated, 0 = shadow.
    """
    H, W     = elevation.shape
    max_dist = sun_dr.shape[0]
    result   = np.ones((H, W), dtype=np.uint8)   # default: illuminated

    for row in prange(H):
        for col in range(W):
            cur_h = elevation[row, col]
            for d in range(max_dist):
                r = row + sun_dr[d]
                c = col + sun_dc[d]
                if r < 0 or r >= H or c < 0 or c >= W:
                    break
                if (elevation[r, c] - cur_h) / sun_dist[d] >= sun_tan:
                    result[row, col] = 0   # blocked
                    break
    return result


def compute_solar_illumination(elevation, sun_az_deg=SUN_AZIMUTH,
                               sun_el_deg=SUN_ELEVATION):
    """
    Shadow map for a single sun position.
    Returns uint8 (H, W): 1=illuminated, 0=shadow.
    """
    sun_tan       = math.tan(math.radians(sun_el_deg))
    dr, dc, dist  = _sun_ray_offsets(sun_az_deg, MAX_DISTANCE, CELLSIZE)
    t0 = time.perf_counter()
    result = _shadow_kernel(elevation, dr, dc, dist, sun_tan)
    print(f"      shadow cast: az={sun_az_deg:.0f}° el={sun_el_deg:.2f}°  "
          f"lit={result.mean()*100:.1f}%  {time.perf_counter()-t0:.1f} s")
    return result


def compute_annual_illumination(elevation, n_azimuths=N_AZ_ANNUAL,
                                sun_el_deg=SUN_ELEVATION):
    """
    Annual illumination: union of shadow maps at all azimuths.
    A pixel is 1 (ever illuminated) if lit from AT LEAST ONE azimuth.
    """
    azimuths = np.linspace(0.0, 360.0, n_azimuths, endpoint=False)
    combined = np.zeros(elevation.shape, dtype=np.uint8)
    print(f"      annual illumination: {n_azimuths} azimuths, el={sun_el_deg:.2f}°")
    for i, az in enumerate(azimuths):
        illum    = compute_solar_illumination(elevation, float(az), sun_el_deg)
        combined = np.maximum(combined, illum)
        if (i + 1) % 12 == 0 or i == 0:
            print(f"        [{i+1}/{n_azimuths}]  ever_lit={combined.mean()*100:.1f}%")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DPSR ray-casting kernel
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_ray_offsets(n_angles=N_ANGLES, max_dist=MAX_DISTANCE,
                           cellsize=CELLSIZE):
    """
    Integer (row, col) offsets and distances for all DPSR ray directions.
    Called once; the result is fed to _compute_dpsr().
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    steps  = np.arange(1, max_dist + 1, dtype=np.float64)
    sin_a  = np.sin(angles)
    cos_a  = np.cos(angles)
    dr     = np.round(np.outer(sin_a, steps)).astype(np.int32)
    dc     = np.round(np.outer(cos_a, steps)).astype(np.int32)
    dist   = (np.sqrt(dr.astype(np.float64)**2 +
                      dc.astype(np.float64)**2) * cellsize).astype(np.float32)
    dist   = np.where(dist < 1.0, np.float32(1.0), dist)
    return dr, dc, dist


@njit(parallel=True, cache=True, fastmath=True)
def _compute_dpsr(elevation, illumination, psr_rows, psr_cols,
                  ray_dr, ray_dc, ray_dist):
    """
    Classify each PSR pixel as DPSR (1) or not (0).

    For each azimuth direction:
      Walk outward along the ray, tracking highest terrain angle.
      Visibility check BEFORE horizon update — illuminated terrain is
      visible iff terrain_tan >= highest_tan seen in all closer steps.
      This is equivalent to exact LOS analysis (see proof in docs).
    """
    n_psr  = psr_rows.shape[0]
    n_ang  = ray_dr.shape[0]
    n_dist = ray_dr.shape[1]
    n_rows = elevation.shape[0]
    n_cols = elevation.shape[1]
    result = np.zeros(n_psr, dtype=np.uint8)

    for i in prange(n_psr):
        row     = psr_rows[i]
        col     = psr_cols[i]
        cur_h   = elevation[row, col]
        is_dpsr = True

        for a in range(n_ang):
            highest_tan = -1.0e9

            for d in range(n_dist):
                r = row + ray_dr[a, d]
                c = col + ray_dc[a, d]

                if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                    break

                terrain_tan = (elevation[r, c] - cur_h) / ray_dist[a, d]

                # Visibility check BEFORE horizon update
                if illumination[r, c] == 1 and terrain_tan >= highest_tan:
                    is_dpsr = False
                    break

                if terrain_tan > highest_tan:
                    highest_tan = terrain_tan

            if not is_dpsr:
                break

        if is_dpsr:
            result[i] = 1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_dpsr():
    print("=" * 60)
    print("DPSR Extraction — dpsr_fast.py")
    print(f"  MAX_DISTANCE = {MAX_DISTANCE} px  ({MAX_DISTANCE*CELLSIZE/1000:.0f} km)")
    print(f"  SUN_ELEVATION = {SUN_ELEVATION}°   ANNUAL_MODE = {ANNUAL_MODE}")
    print("=" * 60)

    # ── 1. Load DEM ───────────────────────────────────────────────────────────
    print("\n[1/7] Loading DEM ...")
    with rasterio.open(BASE_DIR / "data" / "ldem_85s_20m_float.lbl") as ds:
        elevation     = ds.read(1).astype(np.float32) * 1000.0   # km → m
        dem_crs       = ds.crs
        dem_transform = ds.transform
    print(f"      shape={elevation.shape}  RAM={elevation.nbytes/1e6:.0f} MB")

    # ── 2. Solar illumination map (physically based) ──────────────────────────
    illum_cache = OUT_DIR / ("illumination_annual.tif" if ANNUAL_MODE
                             else "illumination.tif")
    if illum_cache.exists():
        print(f"[2/7] Loading cached illumination: {illum_cache.name} ...")
        illumination = rasterio.open(illum_cache).read(1)
    else:
        print("[2/7] Computing solar shadow map ...")
        if ANNUAL_MODE:
            illumination = compute_annual_illumination(elevation)
        else:
            illumination = compute_solar_illumination(elevation)
        # Cache to disk
        meta = {
            "driver": "GTiff", "dtype": "uint8", "count": 1,
            "crs": dem_crs, "transform": dem_transform, "compress": "lzw",
            "height": elevation.shape[0], "width": elevation.shape[1],
        }
        with rasterio.open(illum_cache, "w", **meta) as dst:
            dst.write(illumination, 1)
        print(f"      cached → {illum_cache.name}")
    print(f"      illuminated = {illumination.mean()*100:.1f}% of DEM")

    # ── 3. Load PSR mask ──────────────────────────────────────────────────────
    print("[3/7] Loading PSR mask ...")
    with rasterio.open(OUT_DIR / "PSR_mask.tif") as ds:
        psr_mask = ds.read(1)
    psr_rows, psr_cols = np.where(psr_mask == 1)
    psr_rows = psr_rows.astype(np.int32)
    psr_cols = psr_cols.astype(np.int32)
    print(f"      PSR pixels : {len(psr_rows):,}")

    # ── 4. Precompute ray offsets ─────────────────────────────────────────────
    print("[4/7] Precomputing ray offsets ...")
    ray_dr, ray_dc, ray_dist = precompute_ray_offsets()
    print(f"      shape : {ray_dr.shape}  ({N_ANGLES} angles × {MAX_DISTANCE} steps)")

    # ── 5. JIT warm-up ────────────────────────────────────────────────────────
    print("[5/7] Compiling Numba kernels (first run only, ~10-30 s) ...")
    _e   = np.zeros((4, 4), dtype=np.float32)
    _ill = np.zeros((4, 4), dtype=np.uint8)
    _r   = np.array([1], dtype=np.int32)
    _c   = np.array([1], dtype=np.int32)
    t_jit = time.perf_counter()
    _compute_dpsr(_e, _ill, _r, _c,
                  ray_dr[:, :2], ray_dc[:, :2], ray_dist[:, :2])
    print(f"      compiled in {time.perf_counter()-t_jit:.1f} s")

    # ── 6. DPSR classification ────────────────────────────────────────────────
    print(f"[6/7] Classifying {len(psr_rows):,} PSR pixels ...")
    t0 = time.perf_counter()
    dpsr_flags = _compute_dpsr(
        elevation, illumination, psr_rows, psr_cols,
        ray_dr, ray_dc, ray_dist,
    )
    elapsed = time.perf_counter() - t0
    print(f"      done in {elapsed:.1f} s ({elapsed/60:.1f} min)"
          f"  throughput={len(psr_rows)/elapsed:,.0f} px/s"
          f"  DPSR={int(dpsr_flags.sum()):,}")

    dpsr_raster = np.zeros_like(psr_mask, dtype=np.uint8)
    dpsr_raster[psr_rows[dpsr_flags == 1],
                psr_cols[dpsr_flags == 1]] = 1

    # ── 7. Save + plot ────────────────────────────────────────────────────────
    print("[7/7] Saving results ...")
    out_path = OUT_DIR / "DPSR.tif"
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=dpsr_raster.shape[0], width=dpsr_raster.shape[1],
                       count=1, dtype="uint8", crs=dem_crs,
                       transform=dem_transform, compress="lzw") as dst:
        dst.write(dpsr_raster, 1)
    print(f"      saved → {out_path}")

    illum_label = "Illumination (annual)" if ANNUAL_MODE else f"Illumination (az={SUN_AZIMUTH}°)"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(psr_mask,     cmap="gray"); axes[0].set_title("PSR Mask")
    axes[1].imshow(illumination,  cmap="gray"); axes[1].set_title(illum_label)
    axes[2].imshow(dpsr_raster,  cmap="hot");  axes[2].set_title(
        f"DPSR  ({dpsr_raster.sum():,} pixels)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "DPSR_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Spot-check validation ─────────────────────────────────────────────────
    # Update these row/col values after inspecting the DEM in QGIS.
    # Rule: rim pixel should NOT be DPSR; deep interior pixel should BE DPSR.
    print("\n  Spot-check validation:")
    checks = [
        ("Shackleton rim (expect NOT dpsr)",      7580, 7580, False),
        ("Shackleton interior (expect DPSR)",     7584, 7584, True),
    ]
    for label, row, col, expect_dpsr in checks:
        dpsr_val = bool(dpsr_raster[row, col])
        illum_val = int(illumination[row, col])
        psr_val   = int(psr_mask[row, col])
        elev_val  = float(elevation[row, col])
        ok = (dpsr_val == expect_dpsr)
        print(f"    [{'OK' if ok else 'CHECK'}] {label}")
        print(f"           elev={elev_val:.0f}m  illum={illum_val}"
              f"  psr={psr_val}  dpsr={int(dpsr_val)}"
              f"  (expected dpsr={int(expect_dpsr)})")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  DPSR raster  → {out_path}")
    print(f"  Summary plot → {IMG_DIR / 'DPSR_comparison.png'}")
    print("=" * 60)


if __name__ == "__main__":
    run_dpsr()
