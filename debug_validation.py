"""
debug_validation.py — Verify Shackleton crater validation coordinates.

Checks:
  1. Shackleton published lat/lon → projected coords → DEM row/col
  2. 21×21 neighbourhood on every raster
  3. PSR polygon intersection + nearest-polygon distance
  4. 4-panel plot (DEM / PSR / illumination / DPSR) with marker
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import rasterio.transform
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer
from shapely.geometry import Point

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "results"
IMG_DIR   = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)

DEM_PATH          = DATA_DIR / "ldem_85s_20m_float.lbl"
PSR_SHP_PATH      = DATA_DIR / "LPSR_80S_20MPP_ADJ.shp"
PSR_MASK_PATH     = OUT_DIR  / "PSR_mask.tif"
ILLUMINATION_PATH = OUT_DIR  / "illumination.tif"
DPSR_PATH         = OUT_DIR  / "DPSR.tif"

# ── Known Shackleton crater points (lunar lat/lon, degrees) ───────────────────
# Shackleton center: USGS / IAU ~89.67°S, 0°E
# Rim high point:  ~89.5°S, 0°E  (rim is ~21 km diameter)
# Deep interior:   89.9°S, 0°E   (deeper inside the floor)
POINTS = {
    "Shackleton_center": (-89.67, 0.0),
    "Shackleton_rim_N":  (-89.50, 0.0),
    "Shackleton_floor":  (-89.90, 0.0),
}

SEP = "=" * 66


def sep(title=""):
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP)


# ── 1. Read DEM metadata ───────────────────────────────────────────────────────
sep("CHECK 1  -  DEM metadata")
with rasterio.open(DEM_PATH) as ds:
    dem_crs   = ds.crs
    dem_tf    = ds.transform
    H, W      = ds.height, ds.width
    dem_arr   = ds.read(1).astype(np.float32) * 1000.0   # km -> m
    dem_nodata = ds.nodata

print(f"  Shape     : {H} x {W}")
print(f"  Transform : {dem_tf}")
print(f"  CRS       : {dem_crs.to_string()[:80]}…")
print(f"  Elev range: {dem_arr.min():.1f} – {dem_arr.max():.1f} m")
print(f"  Nodata    : {dem_nodata}")

# Bounding box in projected metres
left   = dem_tf.c
top    = dem_tf.f
right  = left + dem_tf.a * W
bottom = top  + dem_tf.e * H
print(f"  Extent    : X [{left:.0f}, {right:.0f}]  Y [{bottom:.0f}, {top:.0f}] m")


# ── 2. Lat/lon -> projected -> row/col ────────────────────────────────────────
sep("CHECK 2  -  Coordinate conversion")

# Build a transformer: WGS-84 lunar (EPSG:4326 equivalent) -> DEM projection
# The DEM CRS is Lunar South Polar Stereographic with Moon ellipsoid.
# pyproj needs the CRS string; we use the one rasterio read from the PDS3 label.
try:
    crs_wkt = dem_crs.to_wkt()
    # Lunar geographic CRS (same datum, no projection)
    from pyproj import CRS as pCRS
    lunar_geo = pCRS.from_dict({
        "proj": "latlong",
        "a":    1737400,
        "b":    1737400,
        "no_defs": True,
    })
    lunar_proj = pCRS.from_wkt(crs_wkt)
    transformer = Transformer.from_crs(lunar_geo, lunar_proj, always_xy=True)
    print("  Transformer: lunar latlong -> DEM projection  [OK]")
except Exception as e:
    print(f"  Transformer error: {e}")
    raise

results = {}
for name, (lat, lon) in POINTS.items():
    x, y   = transformer.transform(lon, lat)          # always_xy: lon first
    row, col = rasterio.transform.rowcol(dem_tf, x, y)
    row = int(row); col = int(col)

    # Clamp to valid range for sampling
    row_c = max(0, min(H - 1, row))
    col_c = max(0, min(W - 1, col))

    elev = dem_arr[row_c, col_c]
    results[name] = dict(lat=lat, lon=lon, x=x, y=y,
                         row=row, col=col, elev=elev)

    print(f"\n  {name}")
    print(f"    lat={lat}  lon={lon}")
    print(f"    projected : X={x:.1f}  Y={y:.1f} m")
    print(f"    row={row}  col={col}  (clamped: {row_c},{col_c})")
    in_bounds = (0 <= row < H and 0 <= col < W)
    print(f"    in_bounds : {in_bounds}")
    print(f"    elevation : {elev:.1f} m")


# ── 3. Read all rasters at each point ─────────────────────────────────────────
sep("CHECK 3  -  Raster values at each point")

rasters = {}
for rpath, label in [
    (PSR_MASK_PATH,     "psr"),
    (ILLUMINATION_PATH, "illum"),
    (DPSR_PATH,         "dpsr"),
]:
    if rpath.exists():
        with rasterio.open(rpath) as ds:
            rasters[label] = ds.read(1)
    else:
        print(f"  MISSING: {rpath.name}")
        rasters[label] = None

for name, info in results.items():
    r, c = info["row"], info["col"]
    rc = max(0, min(H-1, r)); cc = max(0, min(W-1, c))
    vals = {}
    for label, arr in rasters.items():
        vals[label] = int(arr[rc, cc]) if arr is not None else "N/A"
    print(f"\n  {name}  (row={r}, col={c})")
    print(f"    elev={info['elev']:.1f} m  "
          f"psr={vals.get('psr','?')}  "
          f"illum={vals.get('illum','?')}  "
          f"dpsr={vals.get('dpsr','?')}")


# ── 4. 21x21 neighbourhood ─────────────────────────────────────────────────────
sep("CHECK 4  -  21x21 neighbourhood")

HALF = 10

for name, info in results.items():
    r, c = info["row"], info["col"]
    print(f"\n  {name}  row={r}  col={c}")
    for label, arr in rasters.items():
        if arr is None:
            continue
        r0 = max(0, r - HALF); r1 = min(H, r + HALF + 1)
        c0 = max(0, c - HALF); c1 = min(W, c + HALF + 1)
        patch = arr[r0:r1, c0:c1]
        total = patch.size
        ones  = int(patch.sum())
        print(f"    {label:8s}: {ones}/{total} = {100*ones/total:.1f}%  "
              f"unique={np.unique(patch).tolist()}")

    # Print elev neighbourhood as ASCII grid (every 4th px for readability)
    r0 = max(0, r - HALF); r1 = min(H, r + HALF + 1)
    c0 = max(0, c - HALF); c1 = min(W, c + HALF + 1)
    patch = dem_arr[r0:r1, c0:c1]
    print(f"    elev (m) range: [{patch.min():.0f}, {patch.max():.0f}]  "
          f"centre={dem_arr[max(0,min(H-1,r)), max(0,min(W-1,c))]:.0f} m")


# ── 5. PSR polygon intersection ───────────────────────────────────────────────
sep("CHECK 5  -  PSR polygon intersection")

psr = gpd.read_file(PSR_SHP_PATH)
if psr.crs is None:
    psr = psr.set_crs(dem_crs)

print(f"  PSR dataset: {len(psr)} polygons")
print(f"  CRS        : {psr.crs}")

for name, info in results.items():
    pt = Point(info["x"], info["y"])
    inside = psr[psr.geometry.contains(pt)]
    print(f"\n  {name}  ({info['lat']}°,{info['lon']}°)")
    if len(inside) > 0:
        for _, row_g in inside.iterrows():
            area_km2 = row_g.geometry.area / 1e6
            print(f"    INSIDE polygon  area={area_km2:.1f} km2")
    else:
        # Nearest polygon
        dists = psr.geometry.distance(pt)
        idx   = dists.idxmin()
        d_m   = dists[idx]
        area  = psr.geometry.iloc[idx].area / 1e6
        print(f"    NOT inside any polygon")
        print(f"    Nearest polygon: distance={d_m:.1f} m ({d_m/1000:.2f} km)  "
              f"area={area:.1f} km2")


# ── 6. 4-panel plot ───────────────────────────────────────────────────────────
sep("CHECK 6  -  4-panel validation plot")

datasets = [
    (dem_arr,              "DEM elevation (m)",     "terrain"),
    (rasters.get("psr"),   "PSR Mask",              "gray"),
    (rasters.get("illum"), "Illumination (annual)", "gray"),
    (rasters.get("dpsr"),  "DPSR",                  "hot"),
]

# Use Shackleton floor as primary validation point
primary = results["Shackleton_floor"]
r_p, c_p = primary["row"], primary["col"]

CROP = 200   # half-size of crop window in pixels
r0c = max(0,   r_p - CROP);  r1c = min(H, r_p + CROP)
c0c = max(0,   c_p - CROP);  c1c = min(W, c_p + CROP)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flat

for ax, (arr, title, cmap) in zip(axes, datasets):
    if arr is None:
        ax.text(0.5, 0.5, f"{title}\n(missing)", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        continue

    crop = arr[r0c:r1c, c0c:c1c]
    lo   = np.nanpercentile(crop, 2)
    hi   = np.nanpercentile(crop, 98)
    if lo == hi:
        lo, hi = crop.min(), crop.max()

    ax.imshow(crop, cmap=cmap, vmin=lo, vmax=hi, origin="upper",
              extent=[c0c, c1c, r1c, r0c])
    ax.set_title(title, fontsize=11)

    # Mark all validation points
    markers = {"Shackleton_center": ("o", "cyan",  10),
               "Shackleton_rim_N":  ("^", "lime",  10),
               "Shackleton_floor":  ("*", "red",   14)}
    for pname, (sym, col, sz) in markers.items():
        info = results[pname]
        ax.plot(info["col"], info["row"], sym, color=col, markersize=sz,
                markeredgecolor="black", markeredgewidth=0.8,
                label=pname.replace("_", " "))

    ax.set_xlabel("col"); ax.set_ylabel("row")

# Legend on first panel
axes[0].legend(loc="upper right", fontsize=8, framealpha=0.8)

plt.suptitle(
    f"Shackleton validation  (floor row={r_p}, col={c_p})\n"
    f"crop: rows [{r0c},{r1c}]  cols [{c0c},{c1c}]  "
    f"({(r1c-r0c)*20/1000:.0f} km × {(c1c-c0c)*20/1000:.0f} km)",
    fontsize=12)
plt.tight_layout()

out_plot = IMG_DIR / "validation_shackleton.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_plot}")


# ── 7. Summary ────────────────────────────────────────────────────────────────
sep("SUMMARY")
primary = results["Shackleton_floor"]
r, c = primary["row"], primary["col"]
rc = max(0, min(H-1, r)); cc = max(0, min(W-1, c))
psr_val   = int(rasters["psr"][rc, cc])   if rasters["psr"]   is not None else "N/A"
illum_val = int(rasters["illum"][rc, cc]) if rasters["illum"] is not None else "N/A"
dpsr_val  = int(rasters["dpsr"][rc, cc])  if rasters["dpsr"]  is not None else "N/A"

print(f"\n  Shackleton floor (89.9°S, 0°E)")
print(f"    row={r}  col={c}")
print(f"    elev={primary['elev']:.0f} m")
print(f"    psr={psr_val}  illum={illum_val}  dpsr={dpsr_val}")

if psr_val == 0:
    print("\n  DIAGNOSIS: psr=0 at Shackleton floor.")
    print("  This means either:")
    print("    (a) The point is genuinely outside the PSR polygons [most likely]")
    print("    (b) The PSR dataset does not include this area")
    print("    (c) There is a coordinate projection bug")
    print("  -> Check the polygon intersection result in CHECK 5 above.")
else:
    print("\n  PSR=1 confirmed at Shackleton floor — coordinate conversion is correct.")
    if illum_val == 1:
        print("  WARNING: illum=1 inside PSR — illumination map may need review.")

print(f"\n  Plot saved: images/validation_shackleton.png")
print(f"\n{SEP}\n")
