"""
faustini_research_all.py
------------------------
Research-mode Faustini crater analysis: generates the same set of
diagnostic plots as the default-mode suite, but using CPR(mu_c) values
from Calculated_CPR_research.tif (log10-rescaled onto [0, 2]) paired
with SLC-derived DOP (full Stokes-Kennaugh, all 4 channels).

Ice-candidate criterion applied to every scatter:
    CPR(mu_c) > 1.0  AND  DOP < 0.13
Points meeting both criteria are plotted in red with a shaded quadrant.

Run from the project root:
    python cpr/faustini_research_all.py

Outputs -> cpr/faustini/outputs/previews/research/
    faustini_crater_cpr.png
    faustini_lat_strip.png
    faustini_combined.png
    faustini_hist_cpr.png
    faustini_hist_dop.png
    faustini_cpr_comparison.png
    faustini_histograms.png       (6-panel combined)
    faustini_scatter_combined.png
    faustini_scatter_all.png
    faustini_scatter_ice.png
    faustini_scatter_nonice.png
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
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter, zoom as nd_zoom
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CRATER_LAT  = -87.18
CRATER_DIAM = 42.48
CRATER_RAD  = CRATER_DIAM / 2.0
AZ_PX       = 9.4
RG_PX       = 25.0
SCENE_H     = 252825
HALF_LINES  = int(np.ceil(CRATER_RAD * 1000 / AZ_PX * 1.20))

MULTILOOK    = cfg.MULTILOOK_WINDOW
EPS          = 1e-10
ICE_CPR_THR  = 1.0      # upper half of log-rescaled [0,2] range
DOP_ICE_THR  = 0.13     # joint ice-candidate DOP threshold
DISPLAY_SZ   = 600
MAX_SCATTER  = 12000
MAX_HIST     = 500_000

CPR_RES_TIF = cfg.CPR_DIR / "Calculated_CPR_research.tif"
OUT_DIR     = cfg.PREV_DIR / "research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)

FOOTNOTE_L1 = (
    "Chandrayaan-2 DFSAR | Faustini Crater | 2021-05-06 | L-band  |  "
    f"Multilook {MULTILOOK[0]}x{MULTILOOK[1]}"
)
FOOTNOTE_L2 = (
    "CPR(mu_c) log10-rescaled onto [0,2]  --  NOT the literal formula value  |  "
    f"Ice candidate: CPR>1 & DOP<{DOP_ICE_THR}"
)
FOOTNOTE = FOOTNOTE_L1 + "\n" + FOOTNOTE_L2

# ---------------------------------------------------------------------------
# Shared helpers
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


def save_fig(fig, name, bg="white"):
    p = OUT_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=bg, pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {p.name}")


def resize_sq(patch, sz):
    h, w  = patch.shape
    nm    = np.isnan(patch)
    fill  = float(np.nanmean(patch)) if np.any(~nm) else 0.0
    res   = nd_zoom(np.where(nm, fill, patch), (sz / h, sz / w), order=1)
    nm_r  = nd_zoom(nm.astype(np.float32),     (sz / h, sz / w), order=1)
    res[nm_r > 0.4] = np.nan
    return res.astype(np.float32)


def subsample3(c, d, ij, n=MAX_SCATTER, seed=42):
    """Subsample (c, d, ij) arrays together, preserving joint mask."""
    rng_ = np.random.default_rng(seed)
    if len(c) > n:
        idx = rng_.choice(len(c), n, replace=False)
        return c[idx], d[idx], ij[idx]
    return c, d, ij.copy()


def strip_display_params(strip, r0_, r1_):
    sz    = DISPLAY_SZ
    disp  = resize_sq(strip, sz)
    scale = sz / strip.shape[0]
    lat_s = lat_all[r0_:r1_ + 1]
    lat_t = np.arange(np.ceil(lat_s.min() / 0.25) * 0.25,
                      np.floor(lat_s.max() / 0.25) * 0.25 + 0.001, 0.25)
    tr_d  = [int(np.argmin(np.abs(lat_s - lt)) * scale) for lt in lat_t]
    c_d   = int((centre_row - r0_) * scale)
    e_d   = [np.clip(int((centre_row - r0_ + s * CRATER_RAD * 1000 / AZ_PX) * scale),
                     0, sz - 1) for s in [-1, 1]]
    valid = disp[np.isfinite(disp)]
    vmax  = round(max(float(np.percentile(valid, 99.5)) + 0.05, 1.10), 2) if valid.size else 1.1
    return disp, vmax, lat_s, lat_t, tr_d, c_d, e_d


def draw_cpr_map(ax, disp, vmax, lat_t, tr_d, c_d, e_d, sz=DISPLAY_SZ):
    ax.set_facecolor("black")
    im = ax.imshow(disp, cmap=CMAP, vmin=0.0, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    for lt, tr in zip(lat_t, tr_d):
        col = "yellow" if abs(lt - CRATER_LAT) < 0.13 else "white"
        ax.axhline(tr, color=col, lw=0.7, ls="--", alpha=0.75)
        ax.text(-6, tr, f"{abs(lt):.2f}S", color=col,
                fontsize=7, va="center", ha="right")
    ax.axhline(c_d, color="yellow", lw=1.8, ls="-")
    for er in e_d:
        ax.axhline(er, color="cyan", lw=1.2, ls=":")
    ax.set_xlim(-1, sz); ax.set_ylim(sz, -1)
    ax.axis("off")
    return im


def cpr_colorbar(fig, ax, im, vmax):
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015)
    cb.set_ticks([0.0, 1.0, vmax])
    cb.set_ticklabels(["0", "1.0", f"{vmax:.2f}"], color="white", fontsize=9)
    cb.set_label("CPR(μ_c) [log10-rescaled]", color="white", fontsize=9)
    cb.outline.set_edgecolor("white")
    cb.ax.axhline(1.0 / vmax, color="white", lw=0.9, ls="--")
    plt.setp(cb.ax.yaxis.get_ticklines(), color="white")
    return cb


def draw_scatter(ax, cp_, dp_, ij_, mc_, md_, title_,
                 base_color="black", n_total=None, n_cpr1=None, n_joint=None,
                 fs=11):
    """
    Scatter plot with joint ice-candidate highlight.

    cp_, dp_, ij_ : subsampled arrays (ij_ = bool mask for CPR>1 & DOP<DOP_ICE_THR)
    mc_, md_      : mean CPR, mean DOP (from full, un-subsampled data)
    n_total/n_cpr1/n_joint : full-data counts for the stats annotation
    fs            : base font size (scale down to ~9 for compact panels)
    """
    ax.set_facecolor("white")

    # ── Ice-candidate quadrant shading ──────────────────────────────────
    ax.axvspan(0.0, DOP_ICE_THR, ymin=ICE_CPR_THR / 2.0, ymax=1.0,
               color="red", alpha=0.08, zorder=0)

    # ── Scatter points: non-ice (base_color) then ice-candidate (red) ───
    not_ij = ~ij_
    if not_ij.sum() > 0:
        ax.scatter(dp_[not_ij], cp_[not_ij],
                   s=2, c=base_color, alpha=0.28, linewidths=0, rasterized=True)
    if ij_.sum() > 0:
        ax.scatter(dp_[ij_], cp_[ij_],
                   s=4, c="red", alpha=0.70, linewidths=0, rasterized=True,
                   zorder=4, label=f"CPR>1 & DOP<{DOP_ICE_THR}")

    # ── Mean marker (blue star) ──────────────────────────────────────────
    ax.plot(md_, mc_, marker="*", color="blue", markersize=13, zorder=5,
            label=f"Mean ({md_:.3f}, {mc_:.3f})")

    # ── Reference lines ──────────────────────────────────────────────────
    ax.axhline(ICE_CPR_THR, color="gray",      lw=0.9, ls="--", alpha=0.65)
    ax.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlim(0, 1); ax.set_ylim(0, 2)
    ax.set_xlabel("DOP", fontsize=fs)
    ax.set_ylabel("CPR(μ_c)", fontsize=fs)
    ax.tick_params(labelsize=max(fs - 2, 7))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.spines[["top", "right"]].set_visible(False)

    # ── Legend (upper-left — sparse region for this CPR/DOP distribution) ──
    ax.legend(fontsize=max(fs - 3, 7), loc="upper left",
              framealpha=0.85, edgecolor="lightgray", handlelength=1.2)

    # ── Title ────────────────────────────────────────────────────────────
    ax.set_title(title_, fontsize=fs, fontweight="bold", pad=5)

    # ── Stats annotation (bottom-right) ──────────────────────────────────
    if n_total is not None and n_total > 0:
        pct_cpr1  = 100.0 * n_cpr1  / n_total
        pct_joint = 100.0 * n_joint / n_total
        ax.text(0.97, 0.02,
                f"CPR>1: {pct_cpr1:.1f}%\n"
                f"CPR>1 & DOP<{DOP_ICE_THR}: {pct_joint:.2f}%",
                transform=ax.transAxes, fontsize=max(fs - 4, 7),
                va="bottom", ha="right", color="dimgray",
                linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))


# ===========================================================================
# STEP 1 — geometry CSV -> per-row lat/lon
# ===========================================================================
print("Reading geometry CSV ...")
df       = pd.read_csv(GEOM_CSV)
lat_col  = [c for c in df.columns if "Latitude"    in c][0]
lon_col  = [c for c in df.columns if "Longitude"   in c][0]
slr_col  = [c for c in df.columns if "Slant_Range" in c][0]
slant    = df[slr_col].values
n_rng    = int(np.where(np.diff(slant) < -500)[0][0] + 1)
n_az     = len(df) // n_rng
lat_ties = df[lat_col].values[:n_az * n_rng].reshape(n_az, n_rng)[:, n_rng // 2]
lon_ties = df[lon_col].values[:n_az * n_rng].reshape(n_az, n_rng)[:, n_rng // 2]
az_px    = np.linspace(0, SCENE_H - 1, n_az)
lat_fn   = interp1d(az_px, lat_ties, kind="linear", fill_value="extrapolate")
lon_fn   = interp1d(az_px, lon_ties, kind="linear", fill_value="extrapolate")
lat_all  = lat_fn(np.arange(SCENE_H, dtype=np.float64))
lon_all  = lon_fn(np.arange(SCENE_H, dtype=np.float64))

centre_row = int(np.argmin(np.abs(lat_all - CRATER_LAT)))
r0 = max(0, centre_row - HALF_LINES)
r1 = min(SCENE_H - 1, centre_row + HALF_LINES)
print(f"  Crater window: [{r0:,}, {r1:,}]  centre lat={lat_all[centre_row]:.4f}")


# ===========================================================================
# STEP 2 — load research CPR (full scene then crop)
# ===========================================================================
print(f"Loading {CPR_RES_TIF.name} ...")
with rasterio.open(CPR_RES_TIF) as src:
    cpr_full = src.read(1).astype(np.float32)
cpr_full[cpr_full == cfg.NODATA] = np.nan

cpr_strip = cpr_full[r0:r1 + 1, :]

valid_scene      = cpr_full[np.isfinite(cpr_full) & (cpr_full >= 0)]
valid_crater_pre = cpr_strip[np.isfinite(cpr_strip) & (cpr_strip >= 0)]
rng_s = np.random.default_rng(0)
if len(valid_scene) > MAX_HIST:
    valid_scene = valid_scene[rng_s.choice(len(valid_scene), MAX_HIST, replace=False)]

n_valid       = int(np.isfinite(cpr_strip).sum())
pct_ice_strip = 100.0 * int((cpr_strip > ICE_CPR_THR).sum()) / n_valid if n_valid > 0 else 0.0
az_extent_km  = (r1 - r0 + 1) * AZ_PX / 1000.0
rg_extent_km  = 244 * RG_PX / 1000.0
print(f"  CPR strip: {cpr_strip.shape}  valid={n_valid:,}  CPR>1: {pct_ice_strip:.2f}%")


# ===========================================================================
# STEP 3 — DOP from SLC (all 4 channels), crater window
# ===========================================================================
def load_win(pol):
    with rasterio.open(cfg.SLI_PATHS[pol]) as src:
        win  = Window(0, r0, src.width, r1 - r0 + 1)
        real = src.read(1, window=win).astype(np.float32)
        imag = src.read(2, window=win).astype(np.float32)
    return real + 1j * imag

def ml(arr):
    az, rg = MULTILOOK
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)

print("Computing DOP from SLC (crater window) ...")
S_HH = load_win("HH"); S_VV = load_win("VV")
OC   = S_HH + S_VV;   diff = S_HH - S_VV;  del S_HH, S_VV
S_HV = load_win("HV"); S_VH = load_win("VH")
XP   = (S_HV + S_VH) * 0.5;               del S_HV, S_VH
SC      = np.empty_like(diff)
SC.real = diff.real - 2.0 * XP.imag
SC.imag = diff.imag + 2.0 * XP.real;      del diff, XP

P_SC = SC.real**2 + SC.imag**2
P_OC = OC.real**2 + OC.imag**2
CR   = SC.real*OC.real + SC.imag*OC.imag
CI   = SC.imag*OC.real - SC.real*OC.imag;  del SC, OC

ML_SC = ml(P_SC); ML_OC = ml(P_OC)
ML_CR = ml(CR);   ML_CI = ml(CI);          del P_SC, P_OC, CR, CI

A = ML_SC + ML_OC
B = ML_OC - ML_SC
dop_arr = np.sqrt(B**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A + EPS)
dop_arr = np.clip(dop_arr, 0.0, 1.0).astype(np.float32)
del ML_SC, ML_OC, ML_CR, ML_CI, A, B

# Pair CPR(mu_c) from TIF with DOP from SLC, pixel-by-pixel
cpr_flat = cpr_strip.ravel()
dop_flat = dop_arr.ravel()
mask     = np.isfinite(cpr_flat) & np.isfinite(dop_flat) & (cpr_flat >= 0)
cpr_1d   = cpr_flat[mask]
dop_1d   = dop_flat[mask]
del cpr_flat, dop_flat

# Partition into ice / non-ice
cpr_mask     = cpr_1d > ICE_CPR_THR
joint_mask   = cpr_mask & (dop_1d < DOP_ICE_THR)   # CPR>1 AND DOP<0.13
cpr_ice      = cpr_1d[cpr_mask];   dop_ice = dop_1d[cpr_mask]
cpr_non      = cpr_1d[~cpr_mask];  dop_non = dop_1d[~cpr_mask]

n_tot_all    = len(cpr_1d)
n_cpr1_all   = int(cpr_mask.sum())
n_joint_all  = int(joint_mask.sum())
ice_pct      = 100.0 * n_cpr1_all  / n_tot_all if n_tot_all > 0 else 0.0
joint_pct    = 100.0 * n_joint_all / n_tot_all if n_tot_all > 0 else 0.0
mean_cpr     = float(np.mean(cpr_1d));  mean_dop = float(np.mean(dop_1d))

# Joint masks for ice / non-ice sub-panels
ij_ice_full  = dop_ice < DOP_ICE_THR      # among CPR>1 pixels, which also have DOP<0.13
n_joint_ice  = int(ij_ice_full.sum())
ij_non_full  = np.zeros(len(cpr_non), dtype=bool)   # CPR<=1: joint criterion always fails

mc_ice = float(np.mean(cpr_ice)) if len(cpr_ice) > 0 else 0.0
md_ice = float(np.mean(dop_ice)) if len(dop_ice) > 0 else 0.0
mc_non = float(np.mean(cpr_non)) if len(cpr_non) > 0 else 0.0
md_non = float(np.mean(dop_non)) if len(dop_non) > 0 else 0.0

print(f"  Paired: n={n_tot_all:,}  CPR>1: {ice_pct:.2f}%  "
      f"CPR>1&DOP<{DOP_ICE_THR}: {joint_pct:.2f}%")

# Subsample for display (preserving joint mask)
cp_all, dp_all, ij_all = subsample3(cpr_1d, dop_1d, joint_mask, MAX_SCATTER, 42)
cp_ice, dp_ice, ij_ice = subsample3(cpr_ice, dop_ice, ij_ice_full, MAX_SCATTER // 2, 43)
cp_non, dp_non, ij_non = subsample3(cpr_non, dop_non, ij_non_full, MAX_SCATTER // 2, 44)

PANELS = [
    dict(cp=cp_all, dp=dp_all, ij=ij_all,
         mc=float(np.mean(cp_all)), md=float(np.mean(dp_all)),
         base_color="black",
         n_total=n_tot_all, n_cpr1=n_cpr1_all, n_joint=n_joint_all,
         title=f"All Crater Pixels  (n = {n_tot_all:,})"),
    dict(cp=cp_ice, dp=dp_ice, ij=ij_ice,
         mc=mc_ice, md=md_ice,
         base_color="#cc2200",
         n_total=len(cpr_ice), n_cpr1=len(cpr_ice), n_joint=n_joint_ice,
         title=f"CPR(μ_c) > 1  (n = {len(cpr_ice):,})"),
    dict(cp=cp_non, dp=dp_non, ij=ij_non,
         mc=mc_non, md=md_non,
         base_color="#004488",
         n_total=len(cpr_non), n_cpr1=0, n_joint=0,
         title=f"CPR(μ_c) ≤ 1  (n = {len(cpr_non):,})"),
]

MAP_LEGEND = [
    Line2D([0],[0], color="yellow", lw=1.8,
           label=f"Crater centre ({abs(CRATER_LAT):.2f}°S)"),
    Line2D([0],[0], color="cyan",   lw=1.2, ls=":",
           label=f"Crater edge (±{CRATER_RAD:.1f} km)"),
    Line2D([0],[0], color="white",  lw=0.8, ls="--",
           label="Lat gridline (0.25°)"),
]

BINS_CPR = np.linspace(0, 2.0, 100)
BINS_DOP = np.linspace(0, 1.0, 80)


# ===========================================================================
# PLOT 1 — faustini_crater_cpr.png
# ===========================================================================
print("\nGenerating plots ...")
disp, vmax, lat_s, lat_t, tr_d, c_d, e_d = strip_display_params(cpr_strip, r0, r1)

fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="black")
im = draw_cpr_map(ax, disp, vmax, lat_t, tr_d, c_d, e_d)
ax.text(DISPLAY_SZ / 2, -14,
        f"<-- {rg_extent_km:.1f} km (cross-track) -->",
        color="white", fontsize=8, ha="center", va="bottom")
cpr_colorbar(fig, ax, im, vmax)
ax.legend(handles=MAP_LEGEND, loc="lower right", fontsize=7.5,
          facecolor="black", edgecolor="white", labelcolor="white")
ax.set_title(
    f"Faustini Crater  |  CPR(μ_c) [log10-rescaled]  |  "
    f"Chandrayaan-2 DFSAR  |  2021-05-06\n"
    f"Centre {abs(CRATER_LAT):.2f}°S  |  Diam {CRATER_DIAM:.1f} km  |  "
    f"Along-track {az_extent_km:.1f} km  |  "
    f"CPR>1: {pct_ice_strip:.2f}%  |  "
    f"CPR>1 & DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
    color="white", fontsize=8.5, pad=10)
fig.savefig(OUT_DIR / "faustini_crater_cpr.png", dpi=200, bbox_inches="tight",
            facecolor="black", pad_inches=0.15)
plt.close(fig)
print("  Saved: faustini_crater_cpr.png")


# ===========================================================================
# PLOT 2 — faustini_lat_strip.png  (wide lat-range view)
# ===========================================================================
lat_lo = max(-89.5, float(lat_ties.min()))
lat_hi = min(-85.5, float(lat_ties.max()))
rows_m = np.where((lat_all >= lat_lo) & (lat_all <= lat_hi))[0]
rs0, rs1 = int(rows_m[0]), int(rows_m[-1])

with rasterio.open(CPR_RES_TIF) as src:
    strip_wide = src.read(
        1, window=Window(0, rs0, src.width, rs1 - rs0 + 1)).astype(np.float32)
strip_wide[strip_wide == cfg.NODATA] = np.nan

lat_s_w     = lat_all[rs0:rs1 + 1]
tick_lats   = np.arange(np.ceil(lat_lo * 2) / 2,
                        np.floor(lat_hi * 2) / 2 + 0.01, 0.5)
tick_rows_w = [int(np.argmin(np.abs(lat_s_w - tl))) for tl in tick_lats]
valid_w     = strip_wide[np.isfinite(strip_wide)]
vmax_w      = max(float(np.percentile(valid_w, 99)), 1.1) if valid_w.size else 1.1
pct_ice_w   = 100.0 * float(np.nanmean(strip_wide > ICE_CPR_THR))

fig, ax = plt.subplots(figsize=(4.0, 10.0), facecolor="black")
ax.set_facecolor("black")
im = ax.imshow(strip_wide, cmap=CMAP, vmin=0.0, vmax=vmax_w,
               aspect="auto", interpolation="nearest")
for tl, tr in zip(tick_lats, tick_rows_w):
    ax.axhline(tr, color="white", lw=0.6, ls="--", alpha=0.6)
    ax.text(-3, tr, f"{abs(tl):.1f}S",
            color="white", fontsize=7, va="center", ha="right")
ax.set_yticks([]); ax.set_xticks([]); ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
cb.set_ticks([0, 1.0, vmax_w])
cb.set_ticklabels(["0", "1.0", f"{vmax_w:.2f}"], color="white", fontsize=8)
cb.set_label("CPR(μ_c) [log10-rescaled]", color="white", fontsize=8)
cb.outline.set_edgecolor("white")
cb.ax.axhline(1.0 / vmax_w, color="white", lw=0.8, ls="--")
plt.setp(cb.ax.yaxis.get_ticklines(), color="white")
ax.set_title(
    f"Faustini  CPR(μ_c) [log10-rescaled]\n"
    f"lat [{lat_all[rs0]:.2f}°, {lat_all[rs1]:.2f}°]  |  "
    f"CPR>1: {pct_ice_w:.2f}%",
    color="white", fontsize=9, pad=6)
fig.savefig(OUT_DIR / "faustini_lat_strip.png", dpi=200, bbox_inches="tight",
            facecolor="black", pad_inches=0.15)
plt.close(fig)
print("  Saved: faustini_lat_strip.png")
del strip_wide


# ===========================================================================
# PLOT 3 — faustini_combined.png  (CPR map + CPR-vs-DOP scatter)
# ===========================================================================
SZ2    = 500
disp2  = resize_sq(cpr_strip, SZ2)
scale2 = SZ2 / cpr_strip.shape[0]
lat_s2 = lat_all[r0:r1 + 1]
lat_t2 = np.arange(np.ceil(lat_s2.min() / 0.25) * 0.25,
                   np.floor(lat_s2.max() / 0.25) * 0.25 + 0.001, 0.25)
tr_d2  = [int(np.argmin(np.abs(lat_s2 - lt)) * scale2) for lt in lat_t2]
c_d2   = int((centre_row - r0) * scale2)
e_d2   = [np.clip(int((centre_row - r0 + s * CRATER_RAD * 1000 / AZ_PX) * scale2),
                  0, SZ2 - 1) for s in [-1, 1]]
valid2 = disp2[np.isfinite(disp2)]
vmax2  = round(max(float(np.percentile(valid2, 99.5)) + 0.05, 1.10), 2) if valid2.size else 1.1

fig = plt.figure(figsize=(13, 7), facecolor="white")
ax_map = fig.add_axes([0.03, 0.08, 0.40, 0.84])
ax_sc  = fig.add_axes([0.52, 0.10, 0.45, 0.80])

ax_map.set_facecolor("black")
im2 = ax_map.imshow(disp2, cmap=CMAP, vmin=0.0, vmax=vmax2,
                    aspect="auto", interpolation="nearest")
for lt, tr in zip(lat_t2, tr_d2):
    col = "yellow" if abs(lt - CRATER_LAT) < 0.13 else "white"
    ax_map.axhline(tr, color=col, lw=0.7, ls="--", alpha=0.7)
    ax_map.text(-6, tr, f"{abs(lt):.2f}S",
                color=col, fontsize=7, va="center", ha="right")
ax_map.axhline(c_d2, color="yellow", lw=2.0, ls="-")
for er in e_d2:
    ax_map.axhline(er, color="cyan", lw=1.2, ls=":")
ax_map.set_xlim(-1, SZ2); ax_map.set_ylim(SZ2, -1); ax_map.axis("off")
cb2 = fig.colorbar(im2, ax=ax_map, fraction=0.042, pad=0.02)
cb2.set_ticks([0.0, 1.0, vmax2])
cb2.set_ticklabels(["0", "1.0", f"{vmax2:.2f}"], color="white", fontsize=8)
cb2.set_label("CPR(μ_c) [log10-rescaled]", color="white", fontsize=8)
cb2.outline.set_edgecolor("white")
cb2.ax.axhline(1.0 / vmax2, color="white", lw=0.9, ls="--")
plt.setp(cb2.ax.yaxis.get_ticklines(), color="white")
ax_map.legend(handles=[
    Line2D([0],[0], color="yellow", lw=2.0,
           label=f"Centre ({abs(CRATER_LAT):.2f}°S)"),
    Line2D([0],[0], color="cyan", lw=1.2, ls=":",
           label=f"Edge (±{CRATER_RAD:.1f} km)"),
], loc="lower right", fontsize=7,
   facecolor="black", edgecolor="white", labelcolor="white")
ax_map.set_title(
    f"Faustini — CPR(μ_c) [log-rescaled]\n"
    f"CPR>1: {pct_ice_strip:.2f}%  |  "
    f"CPR>1 & DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
    color="white", fontsize=8.5, pad=6,
    bbox=dict(facecolor="black", pad=4, edgecolor="none"))

draw_scatter(ax_sc, cp_all, dp_all, ij_all,
             float(np.mean(cp_all)), float(np.mean(dp_all)),
             f"Faustini — CPR(μ_c) vs DOP\nn = {n_tot_all:,} pixels",
             base_color="black",
             n_total=n_tot_all, n_cpr1=n_cpr1_all, n_joint=n_joint_all,
             fs=11)

fig.suptitle(
    "Faustini Crater  |  CPR(μ_c) [log10-rescaled]  |  "
    "Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=11, fontweight="bold", y=0.99)
save_fig(fig, "faustini_combined.png")


# ===========================================================================
# PLOTS 4–6 — histograms
# ===========================================================================
# ── 4. CPR(mu_c) histogram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
counts, _, _ = ax.hist(cpr_1d, bins=BINS_CPR, color="#2266cc", alpha=0.85,
                       density=True, label=f"All crater  n={n_tot_all:,}")
y_top = counts.max() * 1.18
ax.fill_betweenx([0, y_top], ICE_CPR_THR, 2.0, alpha=0.10, color="red", zorder=0)
ax.axvline(ICE_CPR_THR, color="red",    lw=1.8, ls="--",
           label=f"CPR = {ICE_CPR_THR:.1f}")
ax.axvline(mean_cpr,    color="orange", lw=1.8, ls="-",
           label=f"Mean = {mean_cpr:.3f}")
ax.set_ylim(0, y_top); ax.set_xlim(0, 2.0)
ax.set_xlabel("CPR(μ_c)  [log10-rescaled]", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=9, framealpha=0.75, loc="upper right")
ax.set_title(
    f"CPR(μ_c) Distribution — Faustini Crater  [log10-rescaled]\n"
    f"CPR>1 (upper half): {ice_pct:.2f}%  |  "
    f"CPR>1 & DOP<{DOP_ICE_THR}: {joint_pct:.2f}%  |  "
    f"n = {n_tot_all:,}",
    fontsize=10, fontweight="bold")
fig.text(0.5, -0.04, FOOTNOTE, ha="center", fontsize=7, color="gray",
         linespacing=1.5)
save_fig(fig, "faustini_hist_cpr.png")

# ── 5. DOP histogram ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.hist(dop_1d,  bins=BINS_DOP, color="#228833", alpha=0.80, density=True,
        label=f"All crater  n={n_tot_all:,}")
ax.hist(dop_ice, bins=BINS_DOP, color="red",   alpha=0.55, density=True,
        histtype="step", lw=2.0, label=f"CPR>1  n={len(dop_ice):,}")
ax.hist(dop_non, bins=BINS_DOP, color="navy",  alpha=0.40, density=True,
        histtype="step", lw=1.5, ls=":",
        label=f"CPR≤1  n={len(dop_non):,}")
ax.axvline(DOP_ICE_THR, color="steelblue", lw=1.5, ls=":",
           label=f"DOP = {DOP_ICE_THR}")
ax.axvline(mean_dop, color="orange", lw=1.8, ls="-",
           label=f"Mean (all) = {mean_dop:.3f}")
ax.set_xlabel("DOP", fontsize=12); ax.set_ylabel("Probability Density", fontsize=11)
ax.set_xlim(0, 1); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.75)
ax.set_title("DOP Distribution — Faustini Crater\n"
             "CPR(μ_c)>1 vs CPR(μ_c)≤1 comparison",
             fontsize=11, fontweight="bold")
fig.text(0.5, -0.04, FOOTNOTE, ha="center", fontsize=7, color="gray",
         linespacing=1.5)
save_fig(fig, "faustini_hist_dop.png")

# ── 6. CPR comparison: scene vs crater vs upper-half ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.hist(valid_scene,      bins=BINS_CPR, color="gray",    alpha=0.55, density=True,
        label=f"Whole scene  n={len(valid_scene):,} (subsample)")
ax.hist(valid_crater_pre, bins=BINS_CPR, color="#2266cc", alpha=0.65, density=True,
        label=f"Faustini crater  n={len(valid_crater_pre):,}")
ax.hist(cpr_ice,          bins=BINS_CPR, color="red",     alpha=0.80, density=True,
        label=f"Crater CPR(μ_c)>1  n={len(cpr_ice):,}")
ax.axvline(ICE_CPR_THR, color="black", lw=1.4, ls="--", alpha=0.75,
           label=f"CPR = {ICE_CPR_THR:.1f}")
ax.set_xlabel("CPR(μ_c)  [log10-rescaled]", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=11)
ax.set_xlim(0, 2.0); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.8)
ax.set_title("CPR(μ_c) — Scene vs Crater vs Upper-Half\n"
             "Faustini | Chandrayaan-2 DFSAR 2021-05-06",
             fontsize=11, fontweight="bold")
fig.text(0.5, -0.04, FOOTNOTE, ha="center", fontsize=7, color="gray",
         linespacing=1.5)
save_fig(fig, "faustini_cpr_comparison.png")
del valid_scene, valid_crater_pre


# ===========================================================================
# PLOTS 7–10 — scatter (combined + 3 individual)
# ===========================================================================
FOOTER_SC = FOOTNOTE_L1 + "\n" + FOOTNOTE_L2

# ── 7. Combined 3-panel scatter ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                         gridspec_kw={"wspace": 0.38}, facecolor="white")
for ax, p in zip(axes, PANELS):
    draw_scatter(ax, p["cp"], p["dp"], p["ij"],
                 p["mc"], p["md"], p["title"],
                 base_color=p["base_color"],
                 n_total=p["n_total"], n_cpr1=p["n_cpr1"], n_joint=p["n_joint"],
                 fs=11)
fig.suptitle(
    "CPR(μ_c) vs DOP — Faustini Crater  |  Chandrayaan-2 DFSAR  |  2021-05-06\n"
    f"Ice candidate: CPR>1 & DOP<{DOP_ICE_THR}  (red points + shaded quadrant)",
    fontsize=12, fontweight="bold", y=1.02)
fig.text(0.5, -0.04, FOOTER_SC, ha="center", fontsize=7.5, color="gray",
         linespacing=1.5)
save_fig(fig, "faustini_scatter_combined.png")

# ── 8–10. Individual scatter panels ──────────────────────────────────────
for p, fname in zip(PANELS, ["faustini_scatter_all.png",
                               "faustini_scatter_ice.png",
                               "faustini_scatter_nonice.png"]):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    draw_scatter(ax, p["cp"], p["dp"], p["ij"],
                 p["mc"], p["md"], p["title"],
                 base_color=p["base_color"],
                 n_total=p["n_total"], n_cpr1=p["n_cpr1"], n_joint=p["n_joint"],
                 fs=12)
    fig.text(0.5, -0.04, FOOTER_SC, ha="center", fontsize=7.5, color="gray",
             linespacing=1.5)
    save_fig(fig, fname)


# ===========================================================================
# PLOT 11 — faustini_histograms.png  (6-panel combined)
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white",
                         gridspec_kw={"hspace": 0.48, "wspace": 0.35})

# ── Row 0: A=CPR hist, B=DOP hist, C=comparison ──────────────────────────
ax = axes[0, 0]
counts, _, _ = ax.hist(cpr_1d, bins=BINS_CPR, color="#2266cc", alpha=0.85,
                       density=True, label=f"All  n={n_tot_all:,}")
y0 = counts.max() * 1.18
ax.fill_betweenx([0, y0], ICE_CPR_THR, 2.0, alpha=0.10, color="red", zorder=0)
ax.axvline(ICE_CPR_THR, color="red",    lw=1.8, ls="--",
           label=f"CPR={ICE_CPR_THR:.1f}")
ax.axvline(mean_cpr,    color="orange", lw=1.8, ls="-",
           label=f"Mean={mean_cpr:.3f}")
ax.set_ylim(0, y0); ax.set_xlim(0, 2.0)
ax.set_xlabel("CPR(μ_c)  [log10-rescaled]", fontsize=10)
ax.set_ylabel("Prob. Density", fontsize=10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.set_title(f"[A] CPR(μ_c) Histogram\nCPR>1: {ice_pct:.1f}%  |  "
             f"CPR>1&DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.spines[["top","right"]].set_visible(False)

ax = axes[0, 1]
ax.hist(dop_1d,  bins=BINS_DOP, color="#228833", alpha=0.80, density=True,
        label=f"All  n={n_tot_all:,}")
ax.hist(dop_ice, bins=BINS_DOP, color="red",  alpha=0.55, density=True,
        histtype="step", lw=2.0, label=f"CPR>1  n={len(dop_ice):,}")
ax.hist(dop_non, bins=BINS_DOP, color="navy", alpha=0.40, density=True,
        histtype="step", lw=1.5, ls=":", label=f"CPR≤1  n={len(dop_non):,}")
ax.axvline(DOP_ICE_THR, color="steelblue", lw=1.5, ls=":",
           label=f"DOP={DOP_ICE_THR}")
ax.axvline(mean_dop, color="orange", lw=1.8, ls="-",
           label=f"Mean={mean_dop:.3f}")
ax.set_xlabel("DOP", fontsize=10); ax.set_ylabel("Prob. Density", fontsize=10)
ax.set_xlim(0, 1)
ax.set_title("[B] DOP Histogram", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left"); ax.spines[["top","right"]].set_visible(False)

ax = axes[0, 2]
ax.hist(cpr_1d, bins=BINS_CPR, color="gray",    alpha=0.45, density=True,
        label="Crater (all)")
ax.hist(cpr_ice, bins=BINS_CPR, color="red",    alpha=0.80, density=True,
        label="CPR(μ_c)>1")
ax.hist(cpr_non, bins=BINS_CPR, color="#2266cc", alpha=0.55, density=True,
        label="CPR(μ_c)≤1")
ax.axvline(ICE_CPR_THR, color="black", lw=1.4, ls="--", alpha=0.75)
ax.set_xlabel("CPR(μ_c)  [log10-rescaled]", fontsize=10)
ax.set_ylabel("Prob. Density", fontsize=10)
ax.set_xlim(0, 2.0); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.set_title("[C] CPR(μ_c) — Upper / Lower Half", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

# ── Row 1: D/E/F scatter panels ───────────────────────────────────────────
labels_6 = ["[D] All Crater Pixels",
             "[E] CPR(μ_c) > 1",
             "[F] CPR(μ_c) ≤ 1"]
for col_idx, (p, lab) in enumerate(zip(PANELS, labels_6)):
    ax = axes[1, col_idx]
    draw_scatter(ax, p["cp"], p["dp"], p["ij"],
                 p["mc"], p["md"],
                 f"{lab}  (n = {len(p['cp']):,})",
                 base_color=p["base_color"],
                 n_total=p["n_total"], n_cpr1=p["n_cpr1"], n_joint=p["n_joint"],
                 fs=10)

fig.suptitle(
    "Faustini Crater — CPR(μ_c) [log10-rescaled] + DOP\n"
    "Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06  |  L-band  |  "
    f"Ice candidate: CPR>1 & DOP<{DOP_ICE_THR}",
    fontsize=12, fontweight="bold", y=1.01)
fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color="gray",
         linespacing=1.5)
save_fig(fig, "faustini_histograms.png")

print(f"\nAll 11 research-mode Faustini plots saved to:\n  {OUT_DIR}")
