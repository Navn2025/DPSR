"""
geo_extract.py
--------------
Map latitude coordinates to CPR pixel rows for the Faustini scene,
then extract and display the CPR image for a given lat range.

Usage:
    python cpr/geo_extract.py

Outputs:
    - Printed row/lat mapping table
    - CPR strip image cropped to the requested lat range
      saved to faustini/outputs/previews/faustini_lat_strip.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LAT_MIN  = -87.0    # southernmost lat of interest (degrees)
LAT_MAX  = -84.0    # northernmost lat  (scene max is ~-85.71, so this clips)

NODATA   = cfg.NODATA
N_AZ     = 7902     # azimuth tie-point rows  (Faustini scene)
N_RNG    = 9        # range tie-point columns
SCENE_H  = 252825   # total azimuth lines in SLI

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)
CPR_PATH = cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME
OUT_DIR  = cfg.PREV_DIR
OUT_PATH = OUT_DIR / "faustini_lat_strip.png"

# ---------------------------------------------------------------------------
# Colormap (same as ice_zoom.py)
# ---------------------------------------------------------------------------
def make_cmap():
    nodes = [
        (0.00, "#000000"),
        (0.10, "#003300"),
        (0.30, "#00cc00"),
        (0.52, "#0000ff"),
        (0.74, "#ff00ff"),
        (0.90, "#ff5566"),
        (1.00, "#ffffff"),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cpr_paper", [(v, c) for v, c in nodes]
    )
    cmap.set_bad("black")
    return cmap

CMAP = make_cmap()

# ---------------------------------------------------------------------------
# Step 1: Read geometry CSV, interpolate lat for every pixel row
# ---------------------------------------------------------------------------
print("Reading geometry CSV ...")
df = pd.read_csv(GEOM_CSV)
print(f"  CSV rows: {len(df)}  (expected {N_AZ * N_RNG} = {N_AZ}az x {N_RNG}rng)")

# Auto-detect N_RNG from first Slant_Range reset
lat_col = [c for c in df.columns if "Latitude" in c][0]
lon_col = [c for c in df.columns if "Longitude" in c][0]
slr_col = [c for c in df.columns if "Slant_Range" in c][0]

slant = df[slr_col].values
diffs = np.diff(slant)
reset = np.where(diffs < -500)[0]
if len(reset) > 0:
    n_rng = int(reset[0] + 1)
else:
    n_rng = N_RNG
print(f"  Auto-detected N_RNG = {n_rng}")

n_az = len(df) // n_rng
lat_grid = df[lat_col].values[:n_az * n_rng].reshape(n_az, n_rng)
lon_grid = df[lon_col].values[:n_az * n_rng].reshape(n_az, n_rng)

# Middle range column for representative latitude per azimuth line
mid = n_rng // 2
lat_ties = lat_grid[:, mid]   # shape (n_az,)
lon_ties = lon_grid[:, mid]

# Pixel positions of the azimuth tie points
az_px = np.linspace(0, SCENE_H - 1, n_az)

print(f"  Lat range in ties: {lat_ties.min():.4f} to {lat_ties.max():.4f} deg")
print(f"  Lon range in ties: {lon_ties.min():.4f} to {lon_ties.max():.4f} deg")

# Interpolate to every pixel row
lat_fn = interp1d(az_px, lat_ties, kind="linear", fill_value="extrapolate")
lon_fn = interp1d(az_px, lon_ties, kind="linear", fill_value="extrapolate")

all_rows = np.arange(SCENE_H, dtype=np.float64)
lat_all  = lat_fn(all_rows)
lon_all  = lon_fn(all_rows)

# ---------------------------------------------------------------------------
# Step 2: Find rows in requested lat range
# ---------------------------------------------------------------------------
# Clamp to scene extent
lat_lo = max(LAT_MIN, float(lat_ties.min()))
lat_hi = min(LAT_MAX, float(lat_ties.max()))
print(f"\nRequested lat range : [{LAT_MIN}, {LAT_MAX}]")
print(f"Available in scene  : [{lat_ties.min():.4f}, {lat_ties.max():.4f}]")
print(f"Effective lat range : [{lat_lo:.4f}, {lat_hi:.4f}]")

# lat decreases with increasing row (scene goes S→N as az increases? check)
# Determine row direction
if lat_all[0] < lat_all[-1]:   # lat increases with row → S at top
    mask = (lat_all >= lat_lo) & (lat_all <= lat_hi)
else:                           # lat decreases with row → N at top
    mask = (lat_all >= lat_lo) & (lat_all <= lat_hi)

rows_in = np.where(mask)[0]

if len(rows_in) == 0:
    print("\nNo rows found in the requested lat range within this scene.")
    sys.exit(0)

r0, r1 = int(rows_in[0]), int(rows_in[-1])
print(f"\nPixel rows          : {r0:,} to {r1:,}  ({r1-r0+1:,} lines)")
print(f"Lat at row {r0:>7,}  : {lat_all[r0]:.5f} deg")
print(f"Lat at row {r1:>7,}  : {lat_all[r1]:.5f} deg")
print(f"Ground extent (az)  : {(r1-r0+1)*9.4/1000:.1f} km  (@9.4m/px)")

# Lat/lon at 5 evenly-spaced reference rows
print("\nCoordinate reference table:")
print(f"  {'Row':>8}  {'Latitude':>10}  {'Longitude':>12}")
print(f"  {'-'*8}  {'-'*10}  {'-'*12}")
ref_rows = np.linspace(r0, r1, 7, dtype=int)
for rr in ref_rows:
    print(f"  {rr:>8,}  {lat_all[rr]:>10.5f}  {lon_all[rr]:>12.5f}")

# ---------------------------------------------------------------------------
# Step 3: Load CPR strip
# ---------------------------------------------------------------------------
print("\nLoading CPR strip from disk ...")
with rasterio.open(CPR_PATH) as src:
    from rasterio.windows import Window
    win = Window(0, r0, src.width, r1 - r0 + 1)
    cpr_strip = src.read(1, window=win).astype(np.float32)
cpr_strip[cpr_strip == NODATA] = np.nan
print(f"  Strip shape: {cpr_strip.shape}  (rows x cols)")
print(f"  Valid pixels: {np.isfinite(cpr_strip).sum():,}")
print(f"  CPR > 1.0 (ice): {(cpr_strip > 1.0).sum():,}  ({100*float(np.nanmean(cpr_strip>1.0)):.2f}%)")

# ---------------------------------------------------------------------------
# Step 4: Plot
# ---------------------------------------------------------------------------
print("\nPlotting ...")
lat_strip = lat_all[r0:r1+1]

# Determine tick positions (lat lines every 0.5 deg)
tick_lats = np.arange(np.ceil(lat_lo * 2) / 2, np.floor(lat_hi * 2) / 2 + 0.01, 0.5)
tick_rows  = []
for tl in tick_lats:
    idx = np.argmin(np.abs(lat_strip - tl))
    tick_rows.append(idx)

# Display: resize to ~800 wide for sensible aspect
strip_h, strip_w = cpr_strip.shape
display_w = strip_w     # 244
scale_h   = display_w / strip_h if strip_h > display_w else 1.0
# Use aspect to get a reasonable display height
display_h_in = 10.0      # fixed plot height (inches)
display_w_in = display_h_in * (display_w / strip_h) * (9.4 / 25.0) * 1.5
display_w_in = max(display_w_in, 3.0)

vmax = float(np.nanpercentile(cpr_strip, 99))
vmax = max(vmax, 1.1)

fig, ax = plt.subplots(1, 1, figsize=(display_w_in + 1.0, display_h_in),
                        facecolor="black")
ax.set_facecolor("black")

im = ax.imshow(
    cpr_strip,
    cmap=CMAP,
    vmin=0.0,
    vmax=vmax,
    aspect="auto",
    interpolation="nearest",
)

# Lat gridlines and tick labels
for tl, tr in zip(tick_lats, tick_rows):
    ax.axhline(tr, color="white", lw=0.6, ls="--", alpha=0.6)
    ax.text(-4, tr, f"{tl:.1f}S",
            color="white", fontsize=8, va="center", ha="right")

ax.set_yticks([])
ax.set_xticks([])
ax.spines[:].set_visible(False)

# Colorbar
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_ticks([0, 1.0, vmax])
cb.set_ticklabels(["0", "1.0", f"{vmax:.2f}"], color="white", fontsize=8)
cb.set_label("CPR", color="white", fontsize=9)
cb.outline.set_edgecolor("white")
cb.ax.axhline(1.0 / vmax, color="white", lw=0.8, ls="--")
plt.setp(cb.ax.yaxis.get_ticklines(), color="white")

ax.set_title(
    f"Faustini  CPR  |  lat [{lat_all[r0]:.2f}, {lat_all[r1]:.2f}] deg\n"
    f"Rows {r0:,} - {r1:,}  |  {(r1-r0+1)*9.4/1000:.1f} km along-track  |  CPR>1: {100*float(np.nanmean(cpr_strip>1.0)):.2f}%",
    color="white", fontsize=9, pad=6,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
            facecolor="black", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
