"""
combined_figure.py
------------------
Single publication figure combining:
  TOP HALF  — 3×3 CPR spatial maps  (ice_zoom style)
  BOTTOM HALF — 3×3 CPR vs DOP scatter plots

Regions R1-R9 are the 9 ice-candidate clusters found by ice_zoom.py.

Usage:
    python cpr/combined_figure.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter, zoom as nd_zoom
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
AZ_PX          = 9.4
RG_PX          = 25.0
SWATH_W        = 244
PATCH_AZ       = int(round(SWATH_W * RG_PX / AZ_PX))   # 649 lines ~ 6.1 km
DISPLAY_SZ     = 200        # px per CPR map panel (square)
MIN_DIST       = PATCH_AZ + 50
DENSITY_SMOOTH = 150
N_PANELS       = 9
SCENE_H        = 252825
MULTILOOK      = cfg.MULTILOOK_WINDOW   # (19, 3)
EPS            = 1e-10
MAX_SCATTER    = 6000
RNG_SEED       = 42
ICE_THR        = 1.0
DOP_ICE_THR    = 0.13

OUT_DIR  = cfg.PREV_DIR
OUT_PATH = OUT_DIR / "combined_cpr_dop.png"

# ---------------------------------------------------------------------------
# Colormap (CPR maps)
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
# Step 1 – find cluster centres & load CPR
# ---------------------------------------------------------------------------
print("Loading CPR ...")
with rasterio.open(cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME) as src:
    cpr_full = src.read(1).astype(np.float32)
cpr_full[cpr_full == cfg.NODATA] = np.nan

ice  = (cpr_full > ICE_THR) & np.isfinite(cpr_full)
dens = uniform_filter(ice.astype(np.float32),
                      size=(DENSITY_SMOOTH, 1)).mean(axis=1)
peaks, _ = find_peaks(dens, height=0.003, distance=MIN_DIST)
order     = np.argsort(-dens[peaks])
centres   = sorted(peaks[order[:N_PANELS]].tolist())
labels    = [f"R{i+1}" for i in range(len(centres))]
print(f"  Cluster centres: {centres}")

# ---------------------------------------------------------------------------
# Step 2 – extract & resize CPR patches
# ---------------------------------------------------------------------------
def resize_sq(patch, sz=DISPLAY_SZ):
    h, w   = patch.shape
    nm     = np.isnan(patch)
    fill   = float(np.nanmean(patch)) if np.any(~nm) else 0.0
    filled = np.where(nm, fill, patch)
    res    = nd_zoom(filled,   (sz/h, sz/w), order=1)
    nm_r   = nd_zoom(nm.astype(np.float32), (sz/h, sz/w), order=1)
    res[nm_r > 0.4] = np.nan
    return res.astype(np.float32)

patches = []
half = PATCH_AZ // 2
for az in centres:
    r0 = max(0, az - half)
    r1 = min(SCENE_H, az + half)
    patches.append(resize_sq(cpr_full[r0:r1, :]))

del cpr_full

# ---------------------------------------------------------------------------
# Step 3 – compute CPR & DOP per region from SLC
# ---------------------------------------------------------------------------
def load_win(pol, r0, r1):
    with rasterio.open(cfg.SLI_PATHS[pol]) as src:
        win  = Window(0, r0, src.width, r1-r0)
        real = src.read(1, window=win).astype(np.float32)
        imag = src.read(2, window=win).astype(np.float32)
    return real + 1j*imag

def ml(arr):
    az, rg = MULTILOOK
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)

def get_cpr_dop(r0, r1):
    S_HH = load_win("HH", r0, r1); S_VV = load_win("VV", r0, r1)
    OC   = S_HH + S_VV;            diff = S_HH - S_VV
    del S_HH, S_VV
    S_HV = load_win("HV", r0, r1); S_VH = load_win("VH", r0, r1)
    XP   = (S_HV + S_VH) * 0.5;   del S_HV, S_VH
    SC      = np.empty_like(diff)
    SC.real = diff.real - 2.0*XP.imag
    SC.imag = diff.imag + 2.0*XP.real
    del diff, XP
    P_SC = SC.real**2 + SC.imag**2
    P_OC = OC.real**2 + OC.imag**2
    CR   = SC.real*OC.real + SC.imag*OC.imag
    CI   = SC.imag*OC.real - SC.real*OC.imag
    del SC, OC
    ML_SC = ml(P_SC); ML_OC = ml(P_OC)
    ML_CR = ml(CR);   ML_CI = ml(CI)
    del P_SC, P_OC, CR, CI
    cpr = ML_SC / (ML_OC + EPS)
    A   = ML_SC + ML_OC
    B   = ML_OC - ML_SC
    dop = np.sqrt(B**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A + EPS)
    dop = np.clip(dop, 0.0, 1.0)
    mask = (ML_OC > EPS) & (cpr > 0) & (cpr <= 2.5) & np.isfinite(dop)
    return cpr[mask].ravel(), dop[mask].ravel()

rng = np.random.default_rng(RNG_SEED)
scatter_data = []
print("Computing CPR & DOP ...")
for label, az in zip(labels, centres):
    r0 = max(0, az - half); r1 = min(SCENE_H, az + half)
    print(f"  {label} ...", end=" ", flush=True)
    cpr_1d, dop_1d = get_cpr_dop(r0, r1)
    mean_c    = float(np.mean(cpr_1d)); mean_d = float(np.mean(dop_1d))
    n_total_f = len(cpr_1d)
    n_ice     = int((cpr_1d > ICE_THR).sum())
    ij_full   = (cpr_1d > ICE_THR) & (dop_1d < DOP_ICE_THR)
    n_joint   = int(ij_full.sum())
    if n_total_f > MAX_SCATTER:
        idx    = rng.choice(n_total_f, MAX_SCATTER, replace=False)
        cpr_1d = cpr_1d[idx]; dop_1d = dop_1d[idx]; ij_full = ij_full[idx]
    n_total = len(cpr_1d)
    ice_pct   = 100.0 * n_ice   / n_total_f if n_total_f > 0 else 0.0
    joint_pct = 100.0 * n_joint / n_total_f if n_total_f > 0 else 0.0
    scatter_data.append(dict(c=cpr_1d, d=dop_1d, ij=ij_full,
                             mc=mean_c, md=mean_d,
                             ice=ice_pct, joint=joint_pct,
                             n_ice=n_ice, n_joint=n_joint, n_total=n_total_f))
    print(f"CPR={mean_c:.3f}  DOP={mean_d:.3f}  ice={ice_pct:.1f}%  "
          f"joint={joint_pct:.2f}%")

# ---------------------------------------------------------------------------
# Step 4 – build combined figure
# ---------------------------------------------------------------------------
print("\nBuilding combined figure ...")

ncols = 3
nrows_top = int(np.ceil(N_PANELS / ncols))   # 3
nrows_bot = nrows_top                          # 3

# GridSpec: top section = CPR maps (with thin cb cols), bottom = scatter
# col layout for top: [img, cb, gap, img, cb, gap, img, cb]
col_widths = []
for _ in range(ncols):
    col_widths += [1.0, 0.055, 0.04]
col_widths = col_widths[:-1]   # drop last gap → 8 cols

fig = plt.figure(figsize=(5.0*ncols, 4.2*nrows_top + 3.8*nrows_bot + 0.8),
                 facecolor="white")

# Two main row groups via nested GridSpec
outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[4.2*nrows_top, 0.4, 3.8*nrows_bot],
    hspace=0.0,
)

# Section title row (middle slot used for section labels)
ax_title_top = fig.add_subplot(outer[0])
ax_title_top.axis("off")
ax_title_bot = fig.add_subplot(outer[2])
ax_title_bot.axis("off")

# Top GridSpec inside outer[0]
gs_top = gridspec.GridSpecFromSubplotSpec(
    nrows_top, len(col_widths),
    subplot_spec=outer[0],
    width_ratios=col_widths,
    hspace=0.08, wspace=0.0,
)

# Bottom GridSpec inside outer[2]
gs_bot = gridspec.GridSpecFromSubplotSpec(
    nrows_bot, ncols,
    subplot_spec=outer[2],
    hspace=0.45, wspace=0.35,
)

for idx in range(N_PANELS):
    row = idx // ncols
    col = idx % ncols
    img_col = col * 3
    cb_col  = img_col + 1

    # --- CPR map panel ---
    ax  = fig.add_subplot(gs_top[row, img_col])
    cax = fig.add_subplot(gs_top[row, cb_col])
    ax.set_facecolor("black")

    patch = patches[idx]
    vmax  = float(np.nanpercentile(patch, 99.5))
    vmax  = max(round(vmax + 0.05, 2), 1.10)

    im = ax.imshow(patch, cmap=CMAP, vmin=0.0, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.axis("off")

    cb = fig.colorbar(im, cax=cax)
    cax.set_facecolor("black")
    cb.set_ticks([0.0, ICE_THR, vmax])
    cb.set_ticklabels(["0", "1", f"{vmax:.2f}"], color="white", fontsize=6)
    cb.outline.set_edgecolor("white")
    cb.ax.axhline(ICE_THR/vmax, color="white", lw=0.7, ls="--")
    plt.setp(cb.ax.yaxis.get_ticklines(), color="white")
    cax.text(0.5, -0.03, "CPR", transform=cax.transAxes,
             color="white", fontsize=6, ha="center", va="top")

    ax.text(0.03, 0.97, labels[idx],
            transform=ax.transAxes, color="white", fontsize=9,
            fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    ice_f = float(np.nanmean(patch > ICE_THR)) * 100
    ax.text(0.97, 0.03, f"CPR>1: {ice_f:.1f}%",
            transform=ax.transAxes, color="yellow", fontsize=5.5,
            va="bottom", ha="right",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    # --- Scatter panel with dual constraint ---
    axs = fig.add_subplot(gs_bot[row, col])
    sd  = scatter_data[idx]

    axs.set_facecolor("white")
    axs.axvspan(0.0, DOP_ICE_THR, ymin=ICE_THR / 2.0, ymax=1.0,
                color="red", alpha=0.08, zorder=0)

    not_ij = ~sd["ij"]
    if not_ij.sum() > 0:
        axs.scatter(sd["d"][not_ij], sd["c"][not_ij],
                    s=1.5, c="black", alpha=0.28, linewidths=0, rasterized=True)
    if sd["ij"].sum() > 0:
        axs.scatter(sd["d"][sd["ij"]], sd["c"][sd["ij"]],
                    s=3, c="red", alpha=0.70, linewidths=0, rasterized=True, zorder=4)

    axs.plot(sd["md"], sd["mc"], marker="*", color="blue",
             markersize=9, zorder=5)
    axs.axhline(ICE_THR,     color="gray",      lw=0.6, ls="--", alpha=0.6)
    axs.axvline(DOP_ICE_THR, color="steelblue", lw=0.6, ls=":",  alpha=0.65)

    axs.set_xlim(0.0, 1.0); axs.set_ylim(0.0, 2.0)
    axs.set_xlabel("DOP", fontsize=8)
    axs.set_ylabel("CPR", fontsize=8)
    axs.tick_params(labelsize=7)
    axs.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs.spines[["top", "right"]].set_visible(False)

    axs.text(0.97, 0.97, labels[idx],
             transform=axs.transAxes, ha="right", va="top",
             fontsize=9, fontweight="bold")

    axs.text(0.03, 0.04,
             f"CPR>1: {sd['ice']:.1f}%\n"
             f"joint: {sd['joint']:.2f}%",
             transform=axs.transAxes, fontsize=6, va="bottom", color="dimgray",
             linespacing=1.3)

# Section labels
ax_title_top.text(0.5, 0.55,
    "(a)  CPR — Ice-Candidate Regions",
    transform=ax_title_top.transAxes,
    ha="center", va="center", fontsize=11, fontweight="bold", color="black")
ax_title_bot.text(0.5, 0.90,
    "(b)  CPR vs DOP Scatter",
    transform=ax_title_bot.transAxes,
    ha="center", va="top", fontsize=11, fontweight="bold", color="black")

fig.suptitle(
    "Faustini Scene  |  Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06  |  L-band\n"
    f"Ice candidate: CPR > {ICE_THR:.1f}  &  DOP < {DOP_ICE_THR}  (red points + shaded quadrant)",
    fontsize=11, fontweight="bold", y=1.005,
)

fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
