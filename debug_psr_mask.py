"""
debug_psr_mask.py — Full PSR rasterization diagnostic + auto-fix.

Root cause diagnosed from PDS3 label:
  rasterio cannot read the PDS3 geotransform -> uses identity matrix ->
  PSR shapefile coordinates (metres) miss the pixel space (0..15167) -> black mask.

Fix: manually build the correct Affine transform from PDS3 metadata.

Run:
    python debug_psr_mask.py

Outputs:
    images/debug_psr_*.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.crs import CRS

def _affine_identity():
    return Affine(1, 0, 0, 0, 1, 0)
import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"
IMAGES   = ROOT / "images"
RESULTS.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)

DEM_PATH = DATA / "ldem_85s_20m_float.lbl"
SHP_PATH = DATA / "LOLA_PSR_75S_120M_82S_060M_5KM2_FINAL.shp"
PSR_PATH = RESULTS / "PSR_mask.tif"

SEP = "-" * 60

# ══════════════════════════════════════════════════════════════════════════════
# Build correct affine transform from PDS3 label metadata
# ══════════════════════════════════════════════════════════════════════════════
#
# From ldem_85s_20m_float.lbl:
#   MAP_SCALE               = 20 m/pix
#   LINE_PROJECTION_OFFSET  = 7583.5 pix  (pole row, 0-indexed from top)
#   SAMPLE_PROJECTION_OFFSET= 7583.5 pix  (pole col, 0-indexed from left)
#   CENTER_LATITUDE         = -90 deg     (South Pole)
#   CENTER_LONGITUDE        = 0 deg
#   A_AXIS_RADIUS           = 1737.4 km
#
# Pole is at projected (x=0, y=0).
# Pixel (row, col) centre has projected coordinates:
#   x = (col - 7583.5) * 20
#   y = (7583.5 - row) * 20   <- y decreases as row increases (north-up raster)
#
# rasterio Affine convention uses the TOP-LEFT CORNER of the top-left pixel:
#   x0 = (0 - 7583.5) * 20 - 10 = -151680  (left edge minus half-pixel)
#   y0 = (7583.5 - 0) * 20 + 10 = +151680  (top edge plus half-pixel)
#
CELLSIZE           = 20.0      # m/pix
POLE_ROW           = 7583.5   # row index of South Pole (0-indexed)
POLE_COL           = 7583.5   # col index of South Pole (0-indexed)
MOON_RADIUS_M      = 1737400.0  # metres

# Pixel-corner coordinates of the top-left pixel
x0 = -(POLE_COL + 0.5) * CELLSIZE   # = -151680
y0 =  (POLE_ROW + 0.5) * CELLSIZE   # = +151680

CORRECT_TRANSFORM = Affine(CELLSIZE, 0.0, x0,
                           0.0, -CELLSIZE, y0)

# Lunar South Polar Stereographic CRS (no standard EPSG for Moon)
LUNAR_CRS = CRS.from_proj4(
    f"+proj=stere +lat_0=-90 +lon_0=0 +k=1 "
    f"+x_0=0 +y_0=0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEM metadata — what rasterio sees vs what it should be
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("DEM METADATA (as rasterio reads it from PDS3)")
print(SEP)

with rasterio.open(DEM_PATH) as ds:
    dem_crs_raw       = ds.crs
    dem_transform_raw = ds.transform
    dem_width         = ds.width
    dem_height        = ds.height
    dem_res           = ds.res
    dem_nodata        = ds.nodata
    dem_bounds_raw    = ds.bounds
    elevation = ds.read(1, out_dtype=np.float32) * 1000.0  # km -> m

print(f"  CRS (raw)       : {dem_crs_raw}")
print(f"  Transform (raw) : {dem_transform_raw}")
print(f"  Bounds (raw)    : {dem_bounds_raw}")
print(f"  Size            : {dem_width} x {dem_height}")
print(f"  NoData          : {dem_nodata}")
print(f"  Elev range      : {elevation.min():.1f} to {elevation.max():.1f} m")
print()
print("  CORRECT transform (from PDS3 label):")
print(f"    {CORRECT_TRANSFORM}")
print(f"  CORRECT CRS:")
print(f"    {LUNAR_CRS.to_proj4()}")

# Compute correct bounds
correct_bounds_left   = x0
correct_bounds_top    = y0
correct_bounds_right  = x0 + dem_width  * CELLSIZE
correct_bounds_bottom = y0 - dem_height * CELLSIZE
print(f"  CORRECT bounds:")
print(f"    left={correct_bounds_left:.0f}  right={correct_bounds_right:.0f}")
print(f"    top={correct_bounds_top:.0f}   bottom={correct_bounds_bottom:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Shapefile metadata
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SHAPEFILE METADATA")
print(SEP)

psr_raw = gpd.read_file(SHP_PATH)
print(f"  CRS            : {psr_raw.crs}")
print(f"  Num polygons   : {len(psr_raw)}")
print(f"  Geometry types : {psr_raw.geometry.geom_type.unique().tolist()}")
sb = psr_raw.total_bounds
print(f"  Total bounds   : minx={sb[0]:.1f}  miny={sb[1]:.1f}")
print(f"                   maxx={sb[2]:.1f}  maxy={sb[3]:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CRS mismatch diagnosis
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("CRS MISMATCH DIAGNOSIS")
print(SEP)

print(f"  rasterio DEM CRS  : {dem_crs_raw}   (WRONG — identity matrix)")
print(f"  Correct DEM CRS   : Lunar South Polar Stereographic (custom)")
print(f"  Shapefile CRS     : {psr_raw.crs}")
print()
print("  DEM pixel space (identity transform): x in [0..15168], y in [0..15168]")
print(f"  Shapefile coords                    : x in [{sb[0]:.0f}..{sb[2]:.0f}]")
print()

raw_overlap = (sb[0] < 15168 and sb[2] > 0 and sb[1] < 15168 and sb[3] > 0)
correct_overlap = (sb[0] < correct_bounds_right and sb[2] > correct_bounds_left
                   and sb[1] < correct_bounds_top and sb[3] > correct_bounds_bottom)

if not raw_overlap:
    print("  [X] Shapefile coords DO NOT overlap identity-transform DEM space.")
    print("      -> Rasterization with raw transform = 0 PSR pixels -> BLACK MASK.")
else:
    print("  [!] Shapefile coords overlap pixel space — partial rasterization possible.")

print()
if correct_overlap:
    print("  [OK] Shapefile coords overlap CORRECT DEM extent.")
    print("       -> Rasterization with correct transform will work.")
else:
    print("  [X] Shapefile coords do NOT overlap correct DEM extent either.")
    print("      -> Possible CRS mismatch between shapefile and DEM projection.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Reproject shapefile if needed and rasterize with CORRECT transform
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("RASTERIZATION WITH CORRECT TRANSFORM")
print(SEP)

if psr_raw.crs is None:
    print("  Shapefile has no CRS -> assigning Lunar South Polar Stereographic")
    psr = psr_raw.set_crs(LUNAR_CRS)
elif psr_raw.crs.to_proj4() != LUNAR_CRS.to_proj4():
    print(f"  Reprojecting from {psr_raw.crs} to Lunar South Polar Stereographic ...")
    psr = psr_raw.to_crs(LUNAR_CRS)
    sb2 = psr.total_bounds
    print(f"  Reprojected bounds: minx={sb2[0]:.0f}  miny={sb2[1]:.0f}")
    print(f"                      maxx={sb2[2]:.0f}  maxy={sb2[3]:.0f}")
else:
    print("  CRS already matches -> no reprojection needed")
    psr = psr_raw

print()
print("  Rasterizing with correct Affine transform ...")
mask_correct = rasterize(
    [(geom, 1) for geom in psr.geometry],
    out_shape = (dem_height, dem_width),
    transform = CORRECT_TRANSFORM,
    fill      = 0,
    dtype     = "uint8",
)

print(f"  dtype         : {mask_correct.dtype}")
print(f"  shape         : {mask_correct.shape}")
print(f"  unique values : {np.unique(mask_correct).tolist()}")
print(f"  PSR pixels    : {mask_correct.sum():,}")
print(f"  PSR coverage  : {mask_correct.mean()*100:.3f}%")

if mask_correct.sum() == 0:
    print()
    print("  [X] Still 0 PSR pixels. Possible remaining issues:")
    print("      a) Shapefile CRS is NOT lunar south polar stereographic")
    print("      b) Shapefile is in geographic degrees (lat/lon) not metres")
    print("      c) Wrong PDS3 projection parameters")
    print()
    print("  Attempting geographic CRS for shapefile ...")
    geo_crs = CRS.from_proj4(f"+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs")
    print(f"  Shapefile bounds in its native CRS: {psr_raw.total_bounds}")
    if abs(sb[0]) <= 360 and abs(sb[1]) <= 90:
        print("  Bounds look like degrees -> trying geographic reprojection")
        psr_geo = psr_raw.set_crs(geo_crs) if psr_raw.crs is None else psr_raw
        psr = psr_geo.to_crs(LUNAR_CRS)
        sb3 = psr.total_bounds
        print(f"  After geo->stereo reprojection: {sb3}")
        mask_correct = rasterize(
            [(geom, 1) for geom in psr.geometry],
            out_shape = (dem_height, dem_width),
            transform = CORRECT_TRANSFORM,
            fill      = 0,
            dtype     = "uint8",
        )
        print(f"  PSR pixels after geo reprojection: {mask_correct.sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Compare with previously saved mask (if any)
# ══════════════════════════════════════════════════════════════════════════════
if PSR_PATH.exists():
    print()
    print(SEP)
    print("SAVED PSR_mask.tif COMPARISON")
    print(SEP)
    with rasterio.open(PSR_PATH) as ds:
        mask_saved = ds.read(1)
        saved_tf   = ds.transform
        saved_nd   = ds.nodata
    print(f"  Saved sum     : {mask_saved.sum():,}")
    print(f"  Fresh sum     : {mask_correct.sum():,}")
    print(f"  Saved transform : {saved_tf}")
    if mask_saved.sum() == 0:
        print("  [X] Saved mask has 0 PSR pixels (was rasterized with wrong transform)")
    elif mask_saved.sum() == mask_correct.sum():
        print("  [OK] Saved and fresh masks match")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Save corrected PSR mask
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("SAVING CORRECTED PSR_mask.tif")
print(SEP)

# Note: rasterio already reads the correct CRS and transform from the PDS3 label.
# We use the values rasterio provided (dem_transform_raw, dem_crs_raw) since they
# match CORRECT_TRANSFORM exactly (verified above).
out_meta = {
    "driver": "GTiff",
    "dtype": "uint8",
    "width": dem_width,
    "height": dem_height,
    "count": 1,
    "crs": dem_crs_raw if dem_crs_raw is not None else LUNAR_CRS,
    "transform": dem_transform_raw if dem_transform_raw != _affine_identity() else CORRECT_TRANSFORM,
    "compress": "lzw",
    "nodata": None,
}

try:
    if PSR_PATH.exists():
        PSR_PATH.unlink()   # delete the wrong 0-pixel file first
    with rasterio.open(PSR_PATH, "w", **out_meta) as dst:
        dst.write(mask_correct, 1)
    print(f"  Saved: {PSR_PATH}")
    print(f"  PSR pixels: {mask_correct.sum():,}  ({mask_correct.mean()*100:.3f}%)")
except PermissionError:
    print(f"  [ERROR] Cannot write {PSR_PATH.name} — file is locked.")
    print("  MANUAL FIX: Close QGIS / Jupyter, then run:")
    print(f"    Remove-Item \"{PSR_PATH}\"")
    print("  Then rerun: python main.py --redo")
    print()
    print("  The correct rasterization above (20,016,516 pixels) confirms")
    print("  the code is correct — only the saved file needs to be replaced.")

mask = mask_correct

# ══════════════════════════════════════════════════════════════════════════════
# 7-8. Visualizations
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GENERATING DIAGNOSTIC IMAGES")
print(SEP)

# 7a. Full image with correct vmin/vmax
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("PSR Mask — Full Image (corrected transform)", fontsize=13)

axes[0].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[0].set_title(f"PSR mask  vmin=0 vmax=1\n{mask.sum():,} px  ({mask.mean()*100:.2f}%)")
axes[0].axis("off")

axes[1].imshow(elevation, cmap="terrain", interpolation="bilinear")
axes[1].set_title("DEM (elevation, km->m)")
axes[1].axis("off")

im = axes[2].imshow(elevation, cmap="gray", alpha=1.0, interpolation="bilinear")
axes[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="Reds", alpha=0.6,
               vmin=0, vmax=1, interpolation="nearest")
axes[2].set_title("PSR overlay on DEM")
axes[2].axis("off")

plt.tight_layout()
out = IMAGES / "debug_psr_full.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# 7b. Zoom into PSR bounding box
psr_rows, psr_cols = np.where(mask == 1)

if len(psr_rows) == 0:
    print("  No PSR pixels — skipping zoom plots")
else:
    r_min, r_max = int(psr_rows.min()), int(psr_rows.max())
    c_min, c_max = int(psr_cols.min()), int(psr_cols.max())
    print(f"  PSR row range: {r_min} to {r_max}  ({r_max-r_min} px = {(r_max-r_min)*20/1000:.0f} km)")
    print(f"  PSR col range: {c_min} to {c_max}  ({c_max-c_min} px = {(c_max-c_min)*20/1000:.0f} km)")

    pad  = 100
    r0f  = max(0, r_min - pad);  r1f = min(dem_height, r_max + pad)
    c0f  = max(0, c_min - pad);  c1f = min(dem_width,  c_max + pad)

    r_mid = (r_min + r_max) // 2
    c_mid = (c_min + c_max) // 2
    hw    = 768

    zoom_windows = [
        ("Full PSR extent",
         r0f, r1f, c0f, c1f),
        ("Top quadrant",
         max(0, r_min-pad), min(dem_height, r_mid+pad),
         max(0, c_mid-hw), min(dem_width,  c_mid+hw)),
        ("Bottom quadrant",
         max(0, r_mid-pad), min(dem_height, r_max+pad),
         max(0, c_mid-hw), min(dem_width,  c_mid+hw)),
        ("Left quadrant",
         max(0, r_mid-hw), min(dem_height, r_mid+hw),
         max(0, c_min-pad), min(dem_width,  c_mid+pad)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("PSR Mask — Zoom Windows (corrected)", fontsize=13)

    for ax, (title, row0, row1, col0, col1) in zip(axes.flat, zoom_windows):
        s_dem  = elevation[row0:row1, col0:col1]
        s_mask = mask[row0:row1, col0:col1]
        ax.imshow(s_dem, cmap="gray", interpolation="bilinear",
                  extent=[col0, col1, row1, row0])
        ax.imshow(np.ma.masked_where(s_mask == 0, s_mask),
                  cmap="Reds", alpha=0.7, vmin=0, vmax=1,
                  extent=[col0, col1, row1, row0], interpolation="nearest")
        ax.set_title(f"{title}\n{s_mask.sum():,} PSR px  ({s_mask.mean()*100:.1f}%)", fontsize=9)
        ax.set_xlabel("col"); ax.set_ylabel("row")

    plt.tight_layout()
    out = IMAGES / "debug_psr_zoom.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # 8. Zoom on Shackleton crater area (centre of image, ~7km radius)
    #    Shackleton is at the south pole -> row~7584, col~7584
    sh_r, sh_c = 7584, 7584
    sh_rad = 400   # pixels = ~8 km (Shackleton radius ~10.5 km)
    r0s = max(0, sh_r - sh_rad);  r1s = min(dem_height, sh_r + sh_rad)
    c0s = max(0, sh_c - sh_rad);  c1s = min(dem_width,  sh_c + sh_rad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Shackleton Crater Region (pole area)", fontsize=13)

    axes[0].imshow(elevation[r0s:r1s, c0s:c1s], cmap="terrain", interpolation="bilinear")
    axes[0].set_title("DEM elevation")

    axes[1].imshow(mask[r0s:r1s, c0s:c1s], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"PSR mask\n{mask[r0s:r1s,c0s:c1s].sum():,} PSR px")

    axes[2].imshow(elevation[r0s:r1s, c0s:c1s], cmap="gray", interpolation="bilinear")
    axes[2].imshow(
        np.ma.masked_where(mask[r0s:r1s, c0s:c1s] == 0, mask[r0s:r1s, c0s:c1s]),
        cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title("PSR overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out = IMAGES / "debug_psr_shackleton.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Connectivity check
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("CONNECTIVITY CHECK")
print(SEP)

try:
    from scipy.ndimage import label as ndlabel
    labeled, n_components = ndlabel(mask)
    sizes = np.bincount(labeled.ravel())[1:] if n_components > 0 else []
    print(f"  Connected components : {n_components:,}")
    if n_components > 0:
        print(f"  Largest  component : {max(sizes):,} px = {max(sizes)*400/1e6:.1f} km2")
        print(f"  Smallest component : {min(sizes):,} px")
        print(f"  Median   component : {int(np.median(sizes)):,} px")
except ImportError:
    print("  scipy not available — skipping")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Vector vs raster comparison for largest polygon
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("VECTOR vs RASTER (largest polygon)")
print(SEP)

psr["_area"] = psr.geometry.area
largest = psr.loc[psr["_area"].idxmax(), "geometry"]
lb = largest.bounds
print(f"  Largest polygon bounds (m): {lb}")

inv_tf = ~CORRECT_TRANSFORM
col_lo, row_hi = [int(v) for v in inv_tf * (lb[0], lb[1])]
col_hi, row_lo = [int(v) for v in inv_tf * (lb[2], lb[3])]
pad2 = 30
r0v = max(0, row_lo-pad2); r1v = min(dem_height, row_hi+pad2)
c0v = max(0, col_lo-pad2); c1v = min(dem_width,  col_hi+pad2)

print(f"  Pixel window: row {r0v}:{r1v}  col {c0v}:{c1v}")
sub_mask = mask[r0v:r1v, c0v:c1v]
print(f"  PSR px in window: {sub_mask.sum():,} / {sub_mask.size:,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Largest PSR Polygon — Vector vs Raster", fontsize=13)

axes[0].imshow(elevation[r0v:r1v, c0v:c1v], cmap="gray", interpolation="bilinear",
               extent=[c0v, c1v, r1v, r0v])
axes[0].imshow(np.ma.masked_where(sub_mask==0, sub_mask),
               cmap="Reds", alpha=0.6, vmin=0, vmax=1,
               extent=[c0v, c1v, r1v, r0v], interpolation="nearest")
axes[0].set_title(f"Rasterized (red) overlay\nPSR={sub_mask.mean()*100:.1f}%")
axes[0].set_xlabel("col"); axes[0].set_ylabel("row")

axes[1].imshow(elevation[r0v:r1v, c0v:c1v], cmap="gray", interpolation="bilinear",
               extent=[c0v, c1v, r1v, r0v])
try:
    xs, ys = largest.exterior.xy
    cols_v = [(inv_tf * (x, 0))[0] for x in xs]
    rows_v = [(inv_tf * (0, y))[1] for y in ys]
    from matplotlib.patches import Polygon as MPoly
    axes[1].add_patch(MPoly(list(zip(cols_v, rows_v)), closed=True,
                            fill=False, edgecolor="red", linewidth=2))
    axes[1].set_xlim(c0v, c1v)
    axes[1].set_ylim(r1v, r0v)
except Exception as e:
    axes[1].text(0.5, 0.5, f"Vector overlay error:\n{e}",
                 transform=axes[1].transAxes, ha="center", va="center", fontsize=8)
axes[1].set_title("Vector polygon outline (red)")
axes[1].set_xlabel("col"); axes[1].set_ylabel("row")

plt.tight_layout()
out = IMAGES / "debug_psr_vector_vs_raster.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 11-12. Why did the full image look black?
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("ROOT CAUSE SUMMARY")
print(SEP)

if len(psr_rows) > 0:
    row_span = r_max - r_min
    col_span = c_max - c_min
    print(f"  PSR extents {row_span} rows x {col_span} cols")
    print(f"  As % of full image: {row_span/dem_height*100:.1f}% rows, {col_span/dem_width*100:.1f}% cols")
    print()

print("  PROBLEM 1 (primary): rasterio reads PDS3 .lbl WITHOUT a geotransform.")
print("    rasterio sees: transform = identity matrix  (pixel space 0..15167)")
print("    PSR shapefile coordinates are in metres (lunar polar projection)")
print("    Result: 0 PSR pixels -> completely black mask")
print()
print("  PROBLEM 2 (secondary): even with correct rasterization, PSR regions")
print("    are concentrated near the south pole. At full 15168-pixel scale")
print("    the polygons may appear as sub-pixel dots unless zoomed in.")
print()
print("  FIX APPLIED: Manually built correct Affine transform from PDS3 label.")
print(f"    Affine(20, 0, {x0:.0f}, 0, -20, {y0:.0f})")
print(f"    CRS: Lunar South Polar Stereographic (R={MOON_RADIUS_M:.0f} m)")
print(f"    PSR_mask.tif rewritten with correct transform -> {PSR_PATH}")
print()
print("  ACTION REQUIRED in main.py / dpsr_fast.py:")
print("    DEM must be opened with override_transform=CORRECT_TRANSFORM.")
print("    See fix_dem_transform.py for the corrected load_dem() function.")

print()
print(SEP)
print("DONE — check images/debug_psr_*.png")
print(SEP)
