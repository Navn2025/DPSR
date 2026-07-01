"""
faustini_combined.py
--------------------
Single figure: Faustini crater CPR map (left) + CPR vs DOP scatter (right).
Crater: 87.18 S, diam 42.48 km  |  Chandrayaan-2 DFSAR 2021-05-06
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
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter, zoom as nd_zoom
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CRATER_LAT  = -87.18
CRATER_DIAM = 42.48          # km
CRATER_RAD  = CRATER_DIAM / 2.0
AZ_PX       = 9.4            # m/px
RG_PX       = 25.0
SCENE_H     = 252825
HALF_LINES  = int(np.ceil(CRATER_RAD * 1000 / AZ_PX * 1.20))  # ~2713 lines
MULTILOOK   = cfg.MULTILOOK_WINDOW   # (19, 3)
EPS         = 1e-10
DISPLAY_SZ  = 500            # square px for CPR map
MAX_SCATTER = 10000
ICE_THR     = 1.0
DOP_ICE_THR = 0.13

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)
OUT_PATH = cfg.PREV_DIR / "faustini_combined.png"
cfg.PREV_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------
def make_cmap():
    nodes = [
        (0.00, "#000000"), (0.10, "#003300"), (0.30, "#00cc00"),
        (0.52, "#0000ff"), (0.74, "#ff00ff"), (0.90, "#ff5566"),
        (1.00, "#ffffff"),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cpr_paper", [(v, c) for v, c in nodes])
    cmap.set_bad("black")
    return cmap
CMAP = make_cmap()

# ---------------------------------------------------------------------------
# Step 1 — find crater centre row via geometry CSV
# ---------------------------------------------------------------------------
print("Reading geometry CSV ...")
df = pd.read_csv(GEOM_CSV)
lat_col = [c for c in df.columns if "Latitude"    in c][0]
lon_col = [c for c in df.columns if "Longitude"   in c][0]
slr_col = [c for c in df.columns if "Slant_Range" in c][0]
slant   = df[slr_col].values
n_rng   = int(np.where(np.diff(slant) < -500)[0][0] + 1)
n_az    = len(df) // n_rng
lat_ties = df[lat_col].values[:n_az*n_rng].reshape(n_az, n_rng)[:, n_rng//2]
az_px    = np.linspace(0, SCENE_H-1, n_az)
lat_fn   = interp1d(az_px, lat_ties, kind="linear", fill_value="extrapolate")
lat_all  = lat_fn(np.arange(SCENE_H, dtype=np.float64))

centre_row = int(np.argmin(np.abs(lat_all - CRATER_LAT)))
r0 = max(0, centre_row - HALF_LINES)
r1 = min(SCENE_H, centre_row + HALF_LINES)
print(f"  Centre row: {centre_row:,}  lat={lat_all[centre_row]:.4f}")
print(f"  Window: [{r0:,}, {r1:,}]  ({r1-r0} lines = {(r1-r0)*AZ_PX/1000:.1f} km)")

# ---------------------------------------------------------------------------
# Step 2 — load CPR strip & resize to square
# ---------------------------------------------------------------------------
print("Loading CPR strip ...")
with rasterio.open(cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME) as src:
    cpr_strip = src.read(1, window=Window(0, r0, src.width, r1-r0)).astype(np.float32)
cpr_strip[cpr_strip == cfg.NODATA] = np.nan

def resize_sq(patch, sz):
    h, w   = patch.shape
    nm     = np.isnan(patch)
    fill   = float(np.nanmean(patch)) if np.any(~nm) else 0.0
    res    = nd_zoom(np.where(nm, fill, patch), (sz/h, sz/w), order=1)
    nm_r   = nd_zoom(nm.astype(np.float32), (sz/h, sz/w), order=1)
    res[nm_r > 0.4] = np.nan
    return res.astype(np.float32)

disp = resize_sq(cpr_strip, DISPLAY_SZ)

# Lat tick positions in display coords
scale       = DISPLAY_SZ / cpr_strip.shape[0]
lat_strip   = lat_all[r0:r1]
lat_ticks   = np.arange(np.ceil(lat_strip.min()/0.25)*0.25,
                         np.floor(lat_strip.max()/0.25)*0.25+0.001, 0.25)
tick_rows_d = [int(np.argmin(np.abs(lat_strip - lt)) * scale) for lt in lat_ticks]
centre_d    = int((centre_row - r0) * scale)
edge_d      = [int((centre_row - r0 + s * CRATER_RAD*1000/AZ_PX) * scale)
               for s in [-1, 1]]

# ---------------------------------------------------------------------------
# Step 3 — compute CPR & DOP from SLC for scatter
# ---------------------------------------------------------------------------
def load_win(pol):
    with rasterio.open(cfg.SLI_PATHS[pol]) as src:
        win  = Window(0, r0, src.width, r1-r0)
        real = src.read(1, window=win).astype(np.float32)
        imag = src.read(2, window=win).astype(np.float32)
    return real + 1j*imag

def ml(arr):
    az, rg = MULTILOOK
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)

print("Computing CPR & DOP from SLC ...")
S_HH = load_win("HH"); S_VV = load_win("VV")
OC   = S_HH + S_VV;   diff = S_HH - S_VV;   del S_HH, S_VV
S_HV = load_win("HV"); S_VH = load_win("VH")
XP   = (S_HV + S_VH) * 0.5;                  del S_HV, S_VH
SC      = np.empty_like(diff)
SC.real = diff.real - 2.0*XP.imag
SC.imag = diff.imag + 2.0*XP.real;            del diff, XP

P_SC = SC.real**2 + SC.imag**2
P_OC = OC.real**2 + OC.imag**2
CR   = SC.real*OC.real + SC.imag*OC.imag
CI   = SC.imag*OC.real - SC.real*OC.imag;     del SC, OC

ML_SC = ml(P_SC); ML_OC = ml(P_OC)
ML_CR = ml(CR);   ML_CI = ml(CI);             del P_SC, P_OC, CR, CI

cpr_arr = ML_SC / (ML_OC + EPS)
A = ML_SC + ML_OC
B = ML_OC - ML_SC
dop_arr = np.sqrt(B**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A + EPS)
dop_arr = np.clip(dop_arr, 0.0, 1.0)

mask    = (ML_OC > EPS) & (cpr_arr > 0) & (cpr_arr <= 2.5) & np.isfinite(dop_arr)
cpr_1d  = cpr_arr[mask].ravel()
dop_1d  = dop_arr[mask].ravel()

mean_cpr    = float(np.mean(cpr_1d))
mean_dop    = float(np.mean(dop_1d))
n_ice       = int((cpr_1d > ICE_THR).sum())
ice_pct     = 100.0 * n_ice / len(cpr_1d)
joint_mask  = (cpr_1d > ICE_THR) & (dop_1d < DOP_ICE_THR)
n_joint     = int(joint_mask.sum())
joint_pct   = 100.0 * n_joint / len(cpr_1d)
print(f"  n={len(cpr_1d):,}  mean_CPR={mean_cpr:.3f}  mean_DOP={mean_dop:.3f}  "
      f"ice={ice_pct:.2f}%  joint={joint_pct:.2f}%")

# subsample for display, preserving joint mask
rng = np.random.default_rng(42)
if len(cpr_1d) > MAX_SCATTER:
    idx      = rng.choice(len(cpr_1d), MAX_SCATTER, replace=False)
    cp, dp   = cpr_1d[idx], dop_1d[idx]
    ij_plot  = joint_mask[idx]
else:
    cp, dp   = cpr_1d, dop_1d
    ij_plot  = joint_mask

# ---------------------------------------------------------------------------
# Step 4 — plot
# ---------------------------------------------------------------------------
print("Plotting ...")
vmax = float(np.nanpercentile(disp, 99.5))
vmax = max(round(vmax + 0.05, 2), 1.10)

fig, (ax_map, ax_sc) = plt.subplots(
    1, 2, figsize=(13, 7),
    gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.35},
    facecolor="white",
)

# ── LEFT: CPR map ──────────────────────────────────────────────────────────
ax_map.set_facecolor("black")
im = ax_map.imshow(disp, cmap=CMAP, vmin=0.0, vmax=vmax,
                   aspect="auto", interpolation="nearest")

# Lat gridlines
for lt, tr in zip(lat_ticks, tick_rows_d):
    col = "yellow" if abs(lt - CRATER_LAT) < 0.13 else "white"
    ax_map.axhline(tr, color=col, lw=0.7, ls="--", alpha=0.7)
    ax_map.text(-6, tr, f"{abs(lt):.2f}S",
                color=col, fontsize=7, va="center", ha="right")

# Crater centre & edges
ax_map.axhline(centre_d, color="yellow", lw=2.0, ls="-")
for er in edge_d:
    ax_map.axhline(np.clip(er, 0, DISPLAY_SZ-1),
                   color="cyan", lw=1.2, ls=":")

ax_map.set_xlim(-1, DISPLAY_SZ); ax_map.set_ylim(DISPLAY_SZ, -1)
ax_map.axis("off")

cb = fig.colorbar(im, ax=ax_map, fraction=0.042, pad=0.02)
cb.set_ticks([0.0, 1.0, vmax])
cb.set_ticklabels(["0", "1.0", f"{vmax:.2f}"], color="white", fontsize=8)
cb.set_label("CPR", color="white", fontsize=10)
cb.outline.set_edgecolor("white")
cb.ax.axhline(1.0/vmax, color="white", lw=0.9, ls="--")
plt.setp(cb.ax.yaxis.get_ticklines(), color="white")
ax_map.set_facecolor("black")
fig.patch.set_facecolor("white")

from matplotlib.lines import Line2D
ax_map.legend(handles=[
    Line2D([0],[0], color="yellow", lw=2.0,
           label=f"Centre ({abs(CRATER_LAT):.2f}S)"),
    Line2D([0],[0], color="cyan",   lw=1.2, ls=":",
           label=f"Edge (+/-{CRATER_RAD:.1f} km)"),
], loc="lower right", fontsize=7,
   facecolor="black", edgecolor="white", labelcolor="white")

ax_map.set_title("Faustini Crater — CPR\n"
                 f"Diam {CRATER_DIAM:.1f} km  |  CPR>1: {ice_pct:.2f}%  |  "
                 f"CPR>1&DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
                 color="white", fontsize=9, pad=6,
                 bbox=dict(facecolor="black", pad=4, edgecolor="none"))
ax_map.set_facecolor("black")
ax_map.figure.set_facecolor("white")

# ── RIGHT: CPR vs DOP scatter with dual constraint ─────────────────────────
ax_sc.set_facecolor("white")

# Ice-candidate quadrant shading
ax_sc.axvspan(0.0, DOP_ICE_THR, ymin=ICE_THR / 2.0, ymax=1.0,
              color="red", alpha=0.08, zorder=0)

not_ij = ~ij_plot
if not_ij.sum() > 0:
    ax_sc.scatter(dp[not_ij], cp[not_ij],
                  s=2, c="black", alpha=0.28, linewidths=0, rasterized=True)
if ij_plot.sum() > 0:
    ax_sc.scatter(dp[ij_plot], cp[ij_plot],
                  s=4, c="red", alpha=0.70, linewidths=0, rasterized=True,
                  zorder=4, label=f"CPR>1 & DOP<{DOP_ICE_THR}")

ax_sc.plot(mean_dop, mean_cpr, marker="*", color="blue",
           markersize=14, zorder=5,
           label=f"Mean ({mean_dop:.3f}, {mean_cpr:.3f})")
ax_sc.axhline(ICE_THR,     color="gray",      lw=0.8, ls="--", alpha=0.65)
ax_sc.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)

ax_sc.set_xlim(0.0, 1.0); ax_sc.set_ylim(0.0, 2.0)
ax_sc.set_xlabel("DOP", fontsize=12)
ax_sc.set_ylabel("CPR", fontsize=12)
ax_sc.tick_params(labelsize=10)
ax_sc.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_sc.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax_sc.spines[["top", "right"]].set_visible(False)
ax_sc.legend(fontsize=8, loc="upper left", framealpha=0.85, edgecolor="lightgray")
ax_sc.set_title(f"Faustini — CPR vs DOP\n"
                f"n = {len(cpr_1d):,}  |  ice (CPR>1): {ice_pct:.2f}%",
                fontsize=9, fontweight="bold")
ax_sc.text(0.97, 0.02,
           f"CPR>1: {ice_pct:.1f}%\n"
           f"CPR>1 & DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
           transform=ax_sc.transAxes, fontsize=8,
           va="bottom", ha="right", color="dimgray", linespacing=1.4,
           bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

# ── Main title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Faustini Crater  |  Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06  |  L-band",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
