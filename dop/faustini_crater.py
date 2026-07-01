"""
faustini_crater.py
------------------
Extract and display the DOP strip covering the Faustini crater.

Crater parameters (USGS / Wikipedia):
  Center  : 87.18 S,  84.31 E
  Diameter: 42.48 km
  Radius  : 21.24 km

SLI pixel spacing:
  Azimuth : ~9.4 m / px
  Range   : ~25.0 m / px

The swath is only 244 px wide (= 6.1 km), so only a narrow E-W slice
of the 42.48-km crater is imaged. We extract the full along-track
extent covering the crater (diameter / 9.4 m ~ 4520 az lines), using
the SAME geometry CSV (lat/lon tie points) and crater geometry as the
CPR pipeline's cpr/faustini_crater.py, so the two products line up
pixel-for-pixel over the same ground footprint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Faustini crater parameters
# ---------------------------------------------------------------------------
CRATER_LAT   = -87.18    # degrees S
CRATER_LON   = 84.31     # degrees E
CRATER_DIAM  = 42.48     # km
CRATER_RAD   = CRATER_DIAM / 2.0  # 21.24 km

AZ_PX        = 9.4       # m per azimuth pixel
RG_PX        = 25.0      # m per range pixel
SCENE_H      = 252825

# How many az lines cover the crater diameter (add 20% margin)
HALF_LINES   = int(np.ceil(CRATER_RAD * 1000 / AZ_PX * 1.20))   # ~2713

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GEOM_CSV = (
    cfg.CPR_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)
DOP_PATH = cfg.DOP_DIR / cfg.DOP_OUTPUT_NAME
OUT_DIR  = cfg.PREV_DIR
OUT_PATH = OUT_DIR / "faustini_crater_dop.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"   # DOP is naturally bounded [0,1] -- no custom stretch needed

# ---------------------------------------------------------------------------
# Step 1 - geometry CSV -> lat/lon interpolation
# ---------------------------------------------------------------------------
print("Reading geometry CSV ...")
df = pd.read_csv(GEOM_CSV)

lat_col = [c for c in df.columns if "Latitude"    in c][0]
lon_col = [c for c in df.columns if "Longitude"   in c][0]
slr_col = [c for c in df.columns if "Slant_Range" in c][0]

# Auto-detect N_RNG
slant = df[slr_col].values
diffs = np.diff(slant)
n_rng = int(np.where(diffs < -500)[0][0] + 1)
n_az  = len(df) // n_rng
print(f"  N_AZ={n_az}, N_RNG={n_rng}")

lat_grid = df[lat_col].values[:n_az * n_rng].reshape(n_az, n_rng)
lon_grid = df[lon_col].values[:n_az * n_rng].reshape(n_az, n_rng)

mid      = n_rng // 2
lat_ties = lat_grid[:, mid]
lon_ties = lon_grid[:, mid]
az_px    = np.linspace(0, SCENE_H - 1, n_az)

print(f"  Scene lat : {lat_ties.min():.4f} to {lat_ties.max():.4f} deg")
print(f"  Scene lon : {lon_ties.min():.4f} to {lon_ties.max():.4f} deg")

lat_fn  = interp1d(az_px, lat_ties, kind="linear", fill_value="extrapolate")
lon_fn  = interp1d(az_px, lon_ties, kind="linear", fill_value="extrapolate")
all_px  = np.arange(SCENE_H, dtype=np.float64)
lat_all = lat_fn(all_px)
lon_all = lon_fn(all_px)

# ---------------------------------------------------------------------------
# Step 2 - find crater centre (nearest lat+lon over the full 2-D tie-point
# grid, not just nearest latitude)
# ---------------------------------------------------------------------------
# The geometry CSV's Longitude column is entirely negative (-166.7 to -71.9
# deg here), which does NOT match the crater's East-longitude gazetteer
# value (84.31 E) directly -- planetary longitude conventions vary (East
# 0-360 vs signed -180/180 vs West-positive). Rather than assume one
# convention, we test all plausible representations of the target longitude
# against the tie-point grid and keep whichever gives the smallest great-
# circle distance -- this is self-diagnosing and will not silently mislocate
# the crater if a convention guess is wrong.
MOON_RADIUS_KM = 1737.4

def haversine_km(lat1, lon1, lat2, lon2, R=MOON_RADIUS_KM):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

lon_candidates = {
    "East 0-360, as given"        : CRATER_LON,
    "East 0-360 -> signed -180/180": CRATER_LON - 360.0,
    "sign-flipped (West-positive)" : -CRATER_LON,
}

best_dist, best_idx, best_label, best_lon = None, None, None, None
for label, lon_cand in lon_candidates.items():
    dist = haversine_km(lat_grid, lon_grid, CRATER_LAT, lon_cand)
    idx  = np.unravel_index(int(np.argmin(dist)), dist.shape)
    d    = float(dist[idx])
    print(f"  Candidate lon convention '{label}': target={lon_cand:.2f} deg  "
          f"nearest tie-point distance={d:.2f} km")
    if best_dist is None or d < best_dist:
        best_dist, best_idx, best_label, best_lon = d, idx, label, lon_cand

tie_row, tie_col = best_idx
centre_row = int(round(float(az_px[tie_row])))
lat_at_centre = float(lat_all[centre_row])
lon_at_centre = float(lon_grid[tie_row, tie_col])

print(f"\nCrater centre requested : {abs(CRATER_LAT):.2f} S, {CRATER_LON:.2f} E")
print(f"Best-matching convention: '{best_label}'  (target lon {best_lon:.2f} deg)")
print(f"Closest scene point     : row={centre_row:,}  lat={lat_at_centre:.5f} deg  "
      f"lon={lon_at_centre:.5f} deg  distance={best_dist:.2f} km")
if best_dist > 30.0:
    print(
        f"  WARNING: nearest tie-point is {best_dist:.1f} km from the crater "
        f"centre (crater radius is {CRATER_RAD:.1f} km) -- this pass may only "
        f"clip the crater edge, or may not intersect it at all."
    )

# Clamp extract window to scene
r0 = max(0, centre_row - HALF_LINES)
r1 = min(SCENE_H - 1, centre_row + HALF_LINES)
print(f"Extract window          : rows {r0:,} to {r1:,}  ({r1-r0+1:,} lines)")
print(f"  lat at r0 : {lat_all[r0]:.5f}")
print(f"  lat at r1 : {lat_all[r1]:.5f}")

az_extent_km = (r1 - r0 + 1) * AZ_PX / 1000.0
rg_extent_km = 244 * RG_PX / 1000.0
print(f"  Ground extent : {rg_extent_km:.2f} km (range) x {az_extent_km:.2f} km (azimuth)")
print(f"  Crater diam   : {CRATER_DIAM:.2f} km")

# Latitude grid lines every 0.25 deg inside the strip
lat_strip = lat_all[r0:r1+1]
lat_ticks = np.arange(
    np.ceil(lat_strip.min() / 0.25) * 0.25,
    np.floor(lat_strip.max() / 0.25) * 0.25 + 0.001,
    0.25
)
tick_rows = [int(np.argmin(np.abs(lat_strip - lt))) for lt in lat_ticks]

# ---------------------------------------------------------------------------
# Step 3 - load DOP strip
# ---------------------------------------------------------------------------
print("\nLoading DOP strip ...")
with rasterio.open(DOP_PATH) as src:
    win      = Window(0, r0, src.width, r1 - r0 + 1)
    dop_strip = src.read(1, window=win).astype(np.float32)
dop_strip[dop_strip == cfg.NODATA] = np.nan

n_valid  = int(np.isfinite(dop_strip).sum())
print(f"  Strip shape : {dop_strip.shape}")
print(f"  Valid px    : {n_valid:,}")
print(f"  DOP median  : {float(np.nanmedian(dop_strip)):.4f}")
print(f"  DOP mean    : {float(np.nanmean(dop_strip)):.4f}")
print(f"  DOP min/max : {float(np.nanmin(dop_strip)):.4f} / {float(np.nanmax(dop_strip)):.4f}")

# ---------------------------------------------------------------------------
# Step 4 - resize strip to square for display (block-average in azimuth)
# ---------------------------------------------------------------------------
from scipy.ndimage import zoom as nd_zoom

def resize_square(patch, target):
    """Block-average patch (H, W) -> (target, target) for display."""
    h, w = patch.shape
    nan_mask = np.isnan(patch)
    fill     = float(np.nanmean(patch)) if np.any(~nan_mask) else 0.0
    filled   = np.where(nan_mask, fill, patch)
    resized  = nd_zoom(filled,   (target / h, target / w), order=1)
    nan_res  = nd_zoom(nan_mask.astype(np.float32), (target / h, target / w), order=1)
    resized[nan_res > 0.4] = np.nan
    return resized.astype(np.float32)

DISPLAY_SZ = 600    # square display pixels
print(f"\nResizing ({dop_strip.shape[0]}x{dop_strip.shape[1]}) -> ({DISPLAY_SZ}x{DISPLAY_SZ}) ...")
disp = resize_square(dop_strip, DISPLAY_SZ)

# Rescale annotation rows/ticks to display coordinates
scale = DISPLAY_SZ / dop_strip.shape[0]
tick_rows_d  = [int(r * scale) for r in tick_rows]
centre_row_d = int((centre_row - r0) * scale)
edge_rows_d  = []
for sign in [-1, 1]:
    er = int((centre_row - r0 + sign * CRATER_RAD * 1000 / AZ_PX) * scale)
    edge_rows_d.append(np.clip(er, 0, DISPLAY_SZ - 1))

# ---------------------------------------------------------------------------
# Step 5 - plot square image
# ---------------------------------------------------------------------------
print("Plotting ...")

vmin, vmax = 0.0, 1.0

fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="black")
ax.set_facecolor("black")

im = ax.imshow(
    disp,
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
    interpolation="nearest",
)

# Latitude gridlines
for lt, tr in zip(lat_ticks, tick_rows_d):
    col = "yellow" if abs(lt - CRATER_LAT) < 0.13 else "white"
    ax.axhline(tr, color=col, lw=0.8, ls="--", alpha=0.75)
    ax.text(-6, tr, f"{abs(lt):.2f}S",
            color=col, fontsize=8, va="center", ha="right")

# Crater centre line
ax.axhline(centre_row_d, color="yellow", lw=1.8, ls="-", alpha=1.0)

# Crater edge lines
for er in edge_rows_d:
    ax.axhline(er, color="cyan", lw=1.2, ls=":", alpha=0.9)

ax.set_xlim(-1, DISPLAY_SZ)
ax.set_ylim(DISPLAY_SZ, -1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines[:].set_visible(False)

# Colorbar
cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015)
cb.set_ticks([0.0, 0.5, 1.0])
cb.set_ticklabels(["0.0", "0.5", "1.0"], color="white", fontsize=9)
cb.set_label("DOP", color="white", fontsize=10)
cb.outline.set_edgecolor("white")
plt.setp(cb.ax.yaxis.get_ticklines(), color="white")

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], color="yellow", lw=1.8,
           label=f"Crater centre ({abs(CRATER_LAT):.2f}S)"),
    Line2D([0],[0], color="cyan",   lw=1.2, ls=":",
           label=f"Crater edge (+/-{CRATER_RAD:.1f} km)"),
    Line2D([0],[0], color="white",  lw=0.8, ls="--",
           label="Lat gridline (0.25 deg)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=7.5,
          facecolor="black", edgecolor="white", labelcolor="white")

ax.set_title(
    f"Faustini Crater  |  DOP  |  Chandrayaan-2 DFSAR  |  2021-05-06\n"
    f"Center {abs(CRATER_LAT):.2f}S, {CRATER_LON:.2f}E  |  Diam {CRATER_DIAM:.1f} km  |  "
    f"Median DOP: {float(np.nanmedian(dop_strip)):.3f}\n"
    f"Extent: {az_extent_km:.1f} km along-track x {rg_extent_km:.1f} km cross-track",
    color="white", fontsize=9, pad=10, linespacing=1.5,
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
            facecolor="black", pad_inches=0.15)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
