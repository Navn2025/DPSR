"""
faustini_histograms.py
----------------------
6-panel comprehensive figure for Faustini crater.

Row 1:
  [A] CPR histogram (crater)
  [B] DOP histogram (crater)
  [C] CPR distribution comparison: whole scene vs crater vs ice candidates

Row 2:
  [D] CPR vs DOP scatter — all crater pixels
  [E] CPR vs DOP scatter — ice candidates only  (CPR > 1.0)
  [F] CPR vs DOP scatter — non-ice only          (CPR <= 1.0)

Ice-candidate criterion: CPR > 1  AND  DOP < 0.13
Red quadrant + red points mark pixels meeting both criteria.

Output: cpr/faustini/outputs/previews/faustini_histograms.png
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
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CRATER_LAT  = -87.18
CRATER_DIAM = 42.48
CRATER_RAD  = CRATER_DIAM / 2.0
AZ_PX       = 9.4
SCENE_H     = 252825
HALF_LINES  = int(np.ceil(CRATER_RAD * 1000 / AZ_PX * 1.20))
MULTILOOK   = cfg.MULTILOOK_WINDOW
EPS         = 1e-10
ICE_THR     = 1.0
DOP_ICE_THR = 0.13
MAX_SCATTER = 12000
MAX_HIST    = 500_000

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)
cfg.PREV_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — crater row window from geometry CSV
# ---------------------------------------------------------------------------
print("Reading geometry CSV ...")
df      = pd.read_csv(GEOM_CSV)
lat_col = [c for c in df.columns if "Latitude"    in c][0]
slr_col = [c for c in df.columns if "Slant_Range" in c][0]
slant   = df[slr_col].values
n_rng   = int(np.where(np.diff(slant) < -500)[0][0] + 1)
n_az    = len(df) // n_rng
lat_ties = df[lat_col].values[:n_az*n_rng].reshape(n_az, n_rng)[:, n_rng//2]
az_px    = np.linspace(0, SCENE_H - 1, n_az)
lat_fn   = interp1d(az_px, lat_ties, kind="linear", fill_value="extrapolate")
lat_all  = lat_fn(np.arange(SCENE_H, dtype=np.float64))

centre_row = int(np.argmin(np.abs(lat_all - CRATER_LAT)))
r0 = max(0, centre_row - HALF_LINES)
r1 = min(SCENE_H, centre_row + HALF_LINES)
print(f"  Crater window: rows [{r0:,}, {r1:,}]  centre lat={lat_all[centre_row]:.4f}")

# ---------------------------------------------------------------------------
# Step 2 — load pre-computed CPR for scene comparison histogram
# ---------------------------------------------------------------------------
print("Loading full-scene CPR (for comparison histogram) ...")
with rasterio.open(cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME) as src:
    cpr_scene_full = src.read(1).astype(np.float32)
cpr_scene_full[cpr_scene_full == cfg.NODATA] = np.nan
valid_scene     = cpr_scene_full[np.isfinite(cpr_scene_full) & (cpr_scene_full > 0) & (cpr_scene_full <= 2.5)]
cpr_crater_pre  = cpr_scene_full[r0:r1, :]
valid_crater_pre = cpr_crater_pre[np.isfinite(cpr_crater_pre) & (cpr_crater_pre > 0) & (cpr_crater_pre <= 2.5)]
del cpr_scene_full, cpr_crater_pre

rng_s = np.random.default_rng(0)
if len(valid_scene) > MAX_HIST:
    valid_scene = valid_scene[rng_s.choice(len(valid_scene), MAX_HIST, replace=False)]
print(f"  Scene CPR n={len(valid_scene):,}  Crater CPR n={len(valid_crater_pre):,}")

# ---------------------------------------------------------------------------
# Step 3 — compute CPR & DOP from SLC for crater
# ---------------------------------------------------------------------------
def load_win(pol):
    with rasterio.open(cfg.SLI_PATHS[pol]) as src:
        win  = Window(0, r0, src.width, r1 - r0)
        real = src.read(1, window=win).astype(np.float32)
        imag = src.read(2, window=win).astype(np.float32)
    return real + 1j * imag

def ml(arr):
    az, rg = MULTILOOK
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)

print("Computing CPR & DOP from SLC (crater window) ...")
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

cpr_arr = ML_SC / (ML_OC + EPS)
A = ML_SC + ML_OC
B = ML_OC - ML_SC
dop_arr = np.sqrt(B**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A + EPS)
dop_arr = np.clip(dop_arr, 0.0, 1.0)

mask    = (ML_OC > EPS) & (cpr_arr > 0) & (cpr_arr <= 2.5) & np.isfinite(dop_arr)
cpr_1d  = cpr_arr[mask].ravel()
dop_1d  = dop_arr[mask].ravel()
del cpr_arr, dop_arr, ML_SC, ML_OC, ML_CR, ML_CI, A, B

cpr_mask    = cpr_1d > ICE_THR
joint_mask  = cpr_mask & (dop_1d < DOP_ICE_THR)
cpr_ice     = cpr_1d[cpr_mask];    dop_ice    = dop_1d[cpr_mask]
cpr_nonice  = cpr_1d[~cpr_mask];   dop_nonice = dop_1d[~cpr_mask]

n_tot_all   = len(cpr_1d)
n_cpr1_all  = int(cpr_mask.sum())
n_joint_all = int(joint_mask.sum())
ice_pct     = 100.0 * n_cpr1_all  / n_tot_all if n_tot_all > 0 else 0.0
joint_pct   = 100.0 * n_joint_all / n_tot_all if n_tot_all > 0 else 0.0

ij_ice_full = dop_ice < DOP_ICE_THR
n_joint_ice = int(ij_ice_full.sum())
ij_non_full = np.zeros(len(cpr_nonice), dtype=bool)

mean_cpr = float(np.mean(cpr_1d)); mean_dop = float(np.mean(dop_1d))
print(f"  All:   n={n_tot_all:,}  CPR={mean_cpr:.3f}  DOP={mean_dop:.3f}  ice={ice_pct:.2f}%  "
      f"joint={joint_pct:.2f}%")
print(f"  Ice:   n={len(cpr_ice):,}  CPR={np.mean(cpr_ice):.3f}  DOP={np.mean(dop_ice):.3f}")
print(f"  Non-ice: n={len(cpr_nonice):,}  CPR={np.mean(cpr_nonice):.3f}  DOP={np.mean(dop_nonice):.3f}")

def subsample3(c, d, ij, n=MAX_SCATTER, seed=42):
    rng_ = np.random.default_rng(seed)
    if len(c) > n:
        idx = rng_.choice(len(c), n, replace=False)
        return c[idx], d[idx], ij[idx]
    return c, d, ij.copy()

cp_all, dp_all, ij_all = subsample3(cpr_1d,    dop_1d,    joint_mask,  MAX_SCATTER,   42)
cp_ice, dp_ice, ij_ice = subsample3(cpr_ice,   dop_ice,   ij_ice_full, MAX_SCATTER//2, 43)
cp_non, dp_non, ij_non = subsample3(cpr_nonice, dop_nonice, ij_non_full, MAX_SCATTER//2, 44)

# ---------------------------------------------------------------------------
# Helper: scatter with dual constraint
# ---------------------------------------------------------------------------
BINS_CPR = np.linspace(0, 2.5, 120)
BINS_DOP = np.linspace(0, 1.0,  80)
SUBTITLE = (f"Chandrayaan-2 DFSAR | Faustini Crater | 2021-05-06 | L-band\n"
            f"Centre 87.18S, Diam {CRATER_DIAM:.1f} km | "
            f"Multilook {MULTILOOK[0]}x{MULTILOOK[1]}\n"
            f"Ice candidate: CPR > {ICE_THR:.1f}  &  DOP < {DOP_ICE_THR}")

def save(fig_, name):
    p = cfg.PREV_DIR / name
    fig_.savefig(p, dpi=200, bbox_inches="tight",
                 facecolor="white", pad_inches=0.18)
    plt.close(fig_)
    print(f"  Saved: {p.name}")


def scatter_fig(cp_, dp_, ij_, mc_, md_,
                n_total, n_cpr1, n_joint,
                title_, dot_color="black", fs=12):
    fig_, ax_ = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    ax_.set_facecolor("white")

    # Ice-candidate quadrant shading
    ax_.axvspan(0.0, DOP_ICE_THR, ymin=ICE_THR / 2.0, ymax=1.0,
                color="red", alpha=0.08, zorder=0)

    not_ij = ~ij_
    if not_ij.sum() > 0:
        ax_.scatter(dp_[not_ij], cp_[not_ij],
                    s=2, c=dot_color, alpha=0.28, linewidths=0, rasterized=True)
    if ij_.sum() > 0:
        ax_.scatter(dp_[ij_], cp_[ij_],
                    s=4, c="red", alpha=0.70, linewidths=0, rasterized=True,
                    zorder=4, label=f"CPR>1 & DOP<{DOP_ICE_THR}")

    ax_.plot(md_, mc_, marker="*", color="blue", markersize=14, zorder=5,
             label=f"Mean ({md_:.3f}, {mc_:.3f})")
    ax_.axhline(ICE_THR,     color="gray",      lw=0.9, ls="--", alpha=0.65)
    ax_.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)

    ax_.set_xlim(0, 1.0); ax_.set_ylim(0, 2.0)
    ax_.set_xlabel("DOP", fontsize=fs); ax_.set_ylabel("CPR", fontsize=fs)
    ax_.tick_params(labelsize=fs - 2)
    ax_.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_.spines[["top", "right"]].set_visible(False)
    ax_.legend(fontsize=fs - 3, loc="upper left", framealpha=0.85,
               edgecolor="lightgray", handlelength=1.2)
    ax_.set_title(title_, fontsize=fs, fontweight="bold", pad=5)

    if n_total > 0:
        pct_cpr1  = 100.0 * n_cpr1  / n_total
        pct_joint = 100.0 * n_joint / n_total
        ax_.text(0.97, 0.02,
                 f"CPR>1: {pct_cpr1:.1f}%\n"
                 f"CPR>1 & DOP<{DOP_ICE_THR}: {pct_joint:.2f}%",
                 transform=ax_.transAxes, fontsize=fs - 4,
                 va="bottom", ha="right", color="dimgray", linespacing=1.4,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    fig_.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=7.5, color="gray",
              linespacing=1.5)
    return fig_


# ---------------------------------------------------------------------------
# Step 4 — save 6 separate figures
# ---------------------------------------------------------------------------
print("Saving separate figures ...")

# ── 1. CPR histogram ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
counts, _, _ = ax.hist(cpr_1d, bins=BINS_CPR, color="#2266cc", alpha=0.85,
                        density=True, label=f"All crater  n={n_tot_all:,}")
y_top = counts.max() * 1.18
ax.fill_betweenx([0, y_top], ICE_THR, 2.5, alpha=0.10, color="red", zorder=0)
ax.axvline(ICE_THR,  color="red",    lw=1.8, ls="--", label="CPR = 1.0 (ice threshold)")
ax.axvline(mean_cpr, color="orange", lw=1.8, ls="-",  label=f"Mean = {mean_cpr:.3f}")
ax.set_ylim(0, y_top); ax.set_xlabel("CPR", fontsize=13)
ax.set_ylabel("Probability Density", fontsize=12); ax.set_xlim(0, 2.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, framealpha=0.7)
ax.set_title(f"CPR Distribution — Faustini Crater\n"
             f"Ice candidates (CPR>1): {ice_pct:.2f}%  |  "
             f"Joint (CPR>1&DOP<{DOP_ICE_THR}): {joint_pct:.2f}%  |  "
             f"n = {n_tot_all:,}",
             fontsize=10, fontweight="bold")
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
save(fig, "faustini_hist_cpr.png")

# ── 2. DOP histogram ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.hist(dop_1d,  bins=BINS_DOP, color="#228833", alpha=0.80, density=True,
        label=f"All crater  n={n_tot_all:,}")
ax.hist(dop_ice, bins=BINS_DOP, color="red", alpha=0.55, density=True,
        histtype="step", lw=2.0, label=f"Ice candidates (CPR>1)  n={len(dop_ice):,}")
ax.hist(dop_nonice, bins=BINS_DOP, color="navy", alpha=0.40, density=True,
        histtype="step", lw=1.5, ls=":",
        label=f"Non-ice (CPR<=1)  n={len(dop_nonice):,}")
ax.axvline(DOP_ICE_THR, color="steelblue", lw=1.5, ls=":", label=f"DOP = {DOP_ICE_THR}")
ax.axvline(mean_dop, color="orange", lw=1.8, ls="-", label=f"Mean (all) = {mean_dop:.3f}")
ax.set_xlabel("DOP", fontsize=13); ax.set_ylabel("Probability Density", fontsize=12)
ax.set_xlim(0, 1.0); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.7)
ax.set_title("DOP Distribution — Faustini Crater\n"
             "Ice candidates vs Non-ice comparison",
             fontsize=11, fontweight="bold")
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
save(fig, "faustini_hist_dop.png")

# ── 3. CPR comparison: whole scene vs crater vs ice ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.hist(valid_scene,     bins=BINS_CPR, color="gray",    alpha=0.55, density=True,
        label=f"Whole Faustini scene  n={len(valid_scene):,} (subsample)")
ax.hist(valid_crater_pre, bins=BINS_CPR, color="#2266cc", alpha=0.65, density=True,
        label=f"Faustini crater  n={len(valid_crater_pre):,}")
ax.hist(cpr_ice,          bins=BINS_CPR, color="red",     alpha=0.80, density=True,
        label=f"Crater ice candidates  n={len(cpr_ice):,}")
ax.axvline(ICE_THR, color="black", lw=1.4, ls="--", alpha=0.75, label="CPR = 1.0")
ax.set_xlabel("CPR", fontsize=13); ax.set_ylabel("Probability Density", fontsize=12)
ax.set_xlim(0, 2.5); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.8)
ax.set_title("CPR Comparison — Whole Scene vs Crater vs Ice Candidates\n"
             "Faustini | Chandrayaan-2 DFSAR 2021-05-06",
             fontsize=11, fontweight="bold")
fig.text(0.5, -0.04, SUBTITLE, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
save(fig, "faustini_cpr_comparison.png")

# ── 4–6. Scatter figures ──────────────────────────────────────────────────────
fig = scatter_fig(
    cp_all, dp_all, ij_all,
    float(np.mean(cp_all)), float(np.mean(dp_all)),
    n_tot_all, n_cpr1_all, n_joint_all,
    f"CPR vs DOP — All Faustini Crater Pixels\nn = {n_tot_all:,}",
    dot_color="black", fs=12,
)
save(fig, "faustini_scatter_all.png")

fig = scatter_fig(
    cp_ice, dp_ice, ij_ice,
    float(np.mean(cpr_ice)), float(np.mean(dop_ice)),
    len(cpr_ice), len(cpr_ice), n_joint_ice,
    f"CPR vs DOP — Ice Candidates Only (CPR > 1)\nn = {len(cpr_ice):,}",
    dot_color="#cc2200", fs=12,
)
save(fig, "faustini_scatter_ice.png")

fig = scatter_fig(
    cp_non, dp_non, ij_non,
    float(np.mean(cpr_nonice)), float(np.mean(dop_nonice)),
    len(cpr_nonice), 0, 0,
    f"CPR vs DOP — Non-Ice Pixels (CPR ≤ 1)\nn = {len(cpr_nonice):,}",
    dot_color="#004488", fs=12,
)
save(fig, "faustini_scatter_nonice.png")

# ---------------------------------------------------------------------------
# 6-panel combined figure
# ---------------------------------------------------------------------------
print("Saving 6-panel combined figure ...")

def draw_panel_scatter(ax, cp_, dp_, ij_, mc_, md_,
                       n_total, n_cpr1, n_joint,
                       title_, dot_color="black", fs=10):
    ax.set_facecolor("white")
    ax.axvspan(0.0, DOP_ICE_THR, ymin=ICE_THR / 2.0, ymax=1.0,
               color="red", alpha=0.08, zorder=0)
    not_ij = ~ij_
    if not_ij.sum() > 0:
        ax.scatter(dp_[not_ij], cp_[not_ij],
                   s=2, c=dot_color, alpha=0.28, linewidths=0, rasterized=True)
    if ij_.sum() > 0:
        ax.scatter(dp_[ij_], cp_[ij_],
                   s=4, c="red", alpha=0.70, linewidths=0, rasterized=True,
                   zorder=4, label=f"CPR>1 & DOP<{DOP_ICE_THR}")
    ax.plot(md_, mc_, marker="*", color="blue", markersize=11, zorder=5,
            label=f"Mean ({md_:.3f}, {mc_:.3f})")
    ax.axhline(ICE_THR,     color="gray",      lw=0.9, ls="--", alpha=0.65)
    ax.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 2.0)
    ax.set_xlabel("DOP", fontsize=fs); ax.set_ylabel("CPR", fontsize=fs)
    ax.tick_params(labelsize=max(fs - 2, 6))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=max(fs - 3, 6), loc="upper left",
              framealpha=0.85, edgecolor="lightgray", handlelength=1.2)
    ax.set_title(title_, fontsize=fs, fontweight="bold", pad=4)
    if n_total > 0:
        pct_cpr1  = 100.0 * n_cpr1  / n_total
        pct_joint = 100.0 * n_joint / n_total
        ax.text(0.97, 0.02,
                f"CPR>1: {pct_cpr1:.1f}%\n"
                f"CPR>1&DOP<{DOP_ICE_THR}: {pct_joint:.2f}%",
                transform=ax.transAxes, fontsize=max(fs - 4, 6),
                va="bottom", ha="right", color="dimgray", linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))


fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white",
                         gridspec_kw={"hspace": 0.50, "wspace": 0.35})

# ── Row 0: histograms ────────────────────────────────────────────────────────
ax = axes[0, 0]
counts, _, _ = ax.hist(cpr_1d, bins=BINS_CPR, color="#2266cc", alpha=0.85,
                        density=True, label=f"All  n={n_tot_all:,}")
y0 = counts.max() * 1.18
ax.fill_betweenx([0, y0], ICE_THR, 2.5, alpha=0.10, color="red", zorder=0)
ax.axvline(ICE_THR,  color="red",    lw=1.8, ls="--", label=f"CPR={ICE_THR:.1f}")
ax.axvline(mean_cpr, color="orange", lw=1.8, ls="-",  label=f"Mean={mean_cpr:.3f}")
ax.set_ylim(0, y0); ax.set_xlim(0, 2.5)
ax.set_xlabel("CPR", fontsize=10); ax.set_ylabel("Prob. Density", fontsize=10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.set_title(f"[A] CPR Histogram\nCPR>1: {ice_pct:.1f}%  |  "
             f"CPR>1&DOP<{DOP_ICE_THR}: {joint_pct:.2f}%",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)

ax = axes[0, 1]
ax.hist(dop_1d,  bins=BINS_DOP, color="#228833", alpha=0.80, density=True,
        label=f"All  n={n_tot_all:,}")
ax.hist(dop_ice, bins=BINS_DOP, color="red",  alpha=0.55, density=True,
        histtype="step", lw=2.0, label=f"CPR>1  n={len(dop_ice):,}")
ax.hist(dop_nonice, bins=BINS_DOP, color="navy", alpha=0.40, density=True,
        histtype="step", lw=1.5, ls=":", label=f"CPR≤1  n={len(dop_nonice):,}")
ax.axvline(DOP_ICE_THR, color="steelblue", lw=1.5, ls=":", label=f"DOP={DOP_ICE_THR}")
ax.axvline(mean_dop, color="orange", lw=1.8, ls="-", label=f"Mean={mean_dop:.3f}")
ax.set_xlabel("DOP", fontsize=10); ax.set_ylabel("Prob. Density", fontsize=10)
ax.set_xlim(0, 1.0)
ax.set_title("[B] DOP Histogram", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

ax = axes[0, 2]
ax.hist(cpr_1d,  bins=BINS_CPR, color="gray",    alpha=0.45, density=True,
        label="Crater (all)")
ax.hist(cpr_ice, bins=BINS_CPR, color="red",    alpha=0.80, density=True,
        label="CPR>1")
ax.hist(cpr_nonice, bins=BINS_CPR, color="#2266cc", alpha=0.55, density=True,
        label="CPR≤1")
ax.axvline(ICE_THR, color="black", lw=1.4, ls="--", alpha=0.75)
ax.set_xlabel("CPR", fontsize=10); ax.set_ylabel("Prob. Density", fontsize=10)
ax.set_xlim(0, 2.5); ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.set_title("[C] CPR — Upper / Lower Half", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

# ── Row 1: scatter panels ─────────────────────────────────────────────────────
panels_6 = [
    dict(cp=cp_all, dp=dp_all, ij=ij_all,
         mc=float(np.mean(cp_all)), md=float(np.mean(dp_all)),
         n_total=n_tot_all, n_cpr1=n_cpr1_all, n_joint=n_joint_all,
         dot_color="black",    label="[D] All Crater Pixels"),
    dict(cp=cp_ice, dp=dp_ice, ij=ij_ice,
         mc=float(np.mean(cpr_ice)), md=float(np.mean(dop_ice)),
         n_total=len(cpr_ice), n_cpr1=len(cpr_ice), n_joint=n_joint_ice,
         dot_color="#cc2200",  label="[E] CPR > 1"),
    dict(cp=cp_non, dp=dp_non, ij=ij_non,
         mc=float(np.mean(cpr_nonice)), md=float(np.mean(dop_nonice)),
         n_total=len(cpr_nonice), n_cpr1=0, n_joint=0,
         dot_color="#004488",  label="[F] CPR ≤ 1"),
]

for col_idx, p6 in enumerate(panels_6):
    draw_panel_scatter(
        axes[1, col_idx],
        p6["cp"], p6["dp"], p6["ij"],
        p6["mc"], p6["md"],
        p6["n_total"], p6["n_cpr1"], p6["n_joint"],
        f"{p6['label']}  (n = {len(p6['cp']):,})",
        dot_color=p6["dot_color"], fs=10,
    )

fig.suptitle(
    "Faustini Crater — CPR + DOP  |  Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06\n"
    f"Ice candidate: CPR > {ICE_THR:.1f}  &  DOP < {DOP_ICE_THR}",
    fontsize=12, fontweight="bold", y=1.01)
fig.text(0.5, -0.02, SUBTITLE, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
p_ = cfg.PREV_DIR / "faustini_histograms.png"
fig.savefig(p_, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
plt.close(fig)
print(f"  Saved: {p_.name}")

print("\nAll 7 figures saved to:", cfg.PREV_DIR)
