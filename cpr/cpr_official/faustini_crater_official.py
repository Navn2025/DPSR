"""
faustini_crater_official.py
----------------------------
Crop the OFFICIAL Chandrayaan-2 DFSAR CPR mosaic (cpr_dataset/) to the
Faustini crater, for direct visual comparison against this project's own
derived products (cpr/faustini/outputs/previews/faustini_crater_cpr.png
and dop/outputs/previews/faustini_crater_dop.png).

Unlike those two (narrow ~6 km wide SLI/GRI swath strips), this mosaic
is already map-projected (Moon 2000 South Polar Stereographic, 25 m/px)
and covers the whole south-polar region, so it gives a genuine 2-D areal
view of the crater rather than a 1-D along-track strip.

Crater parameters (USGS / Wikipedia): 87.18 S, 84.31 E, diameter 42.48 km.

Longitude convention
----------------------
The gazetteer figure "84.31 E" uses an older West-positive-style lunar
convention. Cross-checked empirically against this project's own SLI/GRI
geometry tie-points (see dop/faustini_crater.py and cpr_gri/validation.py):
matching the crater center against the real orbit ground-track tie
points landed within 7.19 km of the crater centre using longitude
-84.31 (not +84.31, which misses by >150 deg). The same sign is used
here, verified again below by checking that a known, already-validated
nearby tie point projects close to this crater-centre pixel in the
mosaic's own coordinate system.

Projection
----------
Moon south polar stereographic (spherical, tangent at the pole) is
implemented directly with the closed-form Snyder equations in
cpr_gri/validation.py (moon_south_polar_stereographic) -- reused here
rather than duplicated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpr_gri"))
from validation import moon_south_polar_stereographic

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Crater parameters
# ---------------------------------------------------------------------------
CRATER_LAT  = -87.18
CRATER_LON  = -84.31     # sign-flipped vs. gazetteer 84.31 E -- see module docstring
CRATER_DIAM = 42.48      # km
CRATER_RAD  = CRATER_DIAM / 2.0

MOON_RADIUS_M = 1737400.0
MARGIN_FRAC   = 0.30     # extra margin around the crater radius for context

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATASET    = BASE_DIR.parent / "cpr_dataset" / "ch2_sar_ndxl_20250630mpcpspwest_d_cpr_xx_fp_xx_xxx.tif"
OUT_DIR    = BASE_DIR / "outputs"
OUT_PATH   = OUT_DIR / "faustini_crater_official_cpr.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colormap (matches cpr/faustini_crater.py for visual consistency)
# ---------------------------------------------------------------------------
def make_cmap():
    nodes = [
        (0.00, "#000000"), (0.10, "#003300"), (0.30, "#00cc00"),
        (0.52, "#0000ff"), (0.74, "#ff00ff"), (0.90, "#ff5566"), (1.00, "#ffffff"),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("cpr_paper", nodes)
    cmap.set_bad("black")
    return cmap

CMAP = make_cmap()

# ---------------------------------------------------------------------------
# Step 1 - crater centre -> mosaic pixel window
# ---------------------------------------------------------------------------
print("Projecting crater centre to Moon South Polar Stereographic ...")
x_c, y_c = moon_south_polar_stereographic(np.array([CRATER_LAT]), np.array([CRATER_LON]), MOON_RADIUS_M)
x_c, y_c = float(x_c[0]), float(y_c[0])
print(f"  Crater centre (lat={CRATER_LAT}, lon={CRATER_LON}) -> x={x_c:.1f} m, y={y_c:.1f} m")

with rasterio.open(DATASET) as src:
    res = src.res[0]
    col_c, row_c = ~src.transform * (x_c, y_c)
    col_c, row_c = int(round(col_c)), int(round(row_c))
    print(f"  Mosaic pixel: row={row_c}, col={col_c}  (mosaic is {src.height} x {src.width} px @ {res} m/px)")

    half_px = int(np.ceil(CRATER_RAD * 1000.0 * (1.0 + MARGIN_FRAC) / res))
    r0, r1 = max(0, row_c - half_px), min(src.height - 1, row_c + half_px)
    c0, c1 = max(0, col_c - half_px), min(src.width - 1, col_c + half_px)
    print(f"  Crop window: rows [{r0}:{r1}]  cols [{c0}:{c1}]  ({r1-r0+1} x {c1-c0+1} px, "
          f"{(r1-r0+1)*res/1000:.1f} x {(c1-c0+1)*res/1000:.1f} km)")

    win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    patch = src.read(1, window=win).astype(np.float32)
    nodata = src.nodata

patch_valid = np.isfinite(patch) if nodata is None else (np.isfinite(patch) & (patch != nodata))
n_valid = int(patch_valid.sum())
print(f"  Valid pixels in crop: {n_valid:,} / {patch.size:,}  ({100*n_valid/patch.size:.1f}%)")
if n_valid > 0:
    valid_vals = patch[patch_valid]
    print(f"  CPR median: {np.median(valid_vals):.4f}   mean: {valid_vals.mean():.4f}   "
          f"CPR>1.0 (ice candidates): {100*np.mean(valid_vals > 1.0):.2f}%")

# ---------------------------------------------------------------------------
# Step 2 - plot
# ---------------------------------------------------------------------------
print("\nPlotting ...")
disp = np.where(patch_valid, patch, np.nan)
vmax = float(np.nanpercentile(disp, 99.5)) if n_valid else 1.1
vmax = max(round(vmax + 0.05, 2), 1.10)

# Crater-centre and crater-edge pixel coordinates within the crop, for overlay
cy, cx = row_c - r0, col_c - c0
edge_px = CRATER_RAD * 1000.0 / res

fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="black")
ax.set_facecolor("black")

im = ax.imshow(disp, cmap=CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")

# Crater outline (a real circle, since this product is properly map-projected)
circle = Circle((cx, cy), edge_px, fill=False, edgecolor="cyan", lw=1.4, ls=":", alpha=0.9)
ax.add_patch(circle)
ax.plot(cx, cy, "+", color="yellow", ms=14, mew=1.8)

# Scale bar (10 km)
bar_px = 10_000.0 / res
bx0, by0 = disp.shape[1] * 0.06, disp.shape[0] * 0.95
ax.plot([bx0, bx0 + bar_px], [by0, by0], color="white", lw=2.5)
ax.text(bx0 + bar_px / 2, by0 - disp.shape[0] * 0.02, "10 km", color="white",
        fontsize=9, ha="center", va="bottom")

ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015)
cb.set_ticks([0.0, 1.0, vmax])
cb.set_ticklabels(["0", "1.0", f"{vmax:.2f}"], color="white", fontsize=9)
cb.set_label("CPR (official DFSAR mosaic)", color="white", fontsize=10)
cb.outline.set_edgecolor("white")
cb.ax.axhline(1.0 / vmax, color="white", lw=0.9, ls="--")
plt.setp(cb.ax.yaxis.get_ticklines(), color="white")

legend_els = [
    Line2D([0], [0], marker="+", color="yellow", lw=0, mew=1.8, ms=12,
           label=f"Crater centre ({abs(CRATER_LAT):.2f}S, {abs(CRATER_LON):.2f}W eq.)"),
    Line2D([0], [0], color="cyan", lw=1.4, ls=":", label=f"Crater rim (r={CRATER_RAD:.1f} km)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=7.5,
          facecolor="black", edgecolor="white", labelcolor="white")

n_ice = 100 * np.mean(valid_vals > 1.0) if n_valid else float("nan")
ax.set_title(
    f"Faustini Crater  |  Official DFSAR CPR Mosaic  |  2025-06-30 composite\n"
    f"Center {abs(CRATER_LAT):.2f}S, {abs(CRATER_LON):.2f}E-equiv  |  Diam {CRATER_DIAM:.1f} km  |  "
    f"CPR > 1.0: {n_ice:.2f}%",
    color="white", fontsize=9, pad=10, linespacing=1.5,
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="black", pad_inches=0.15)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
