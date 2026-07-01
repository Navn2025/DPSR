"""
faustini_scatter3.py
--------------------
Produces 4 output files for the Faustini crater CPR vs DOP scatter:
  faustini_scatter_combined.png  — all 3 panels side-by-side (1 image)
  faustini_scatter_all.png       — all crater pixels
  faustini_scatter_ice.png       — ice candidates (CPR > 1)
  faustini_scatter_nonice.png    — non-ice (CPR <= 1)

Ice-candidate criterion: CPR > 1  AND  DOP < 0.13
Red quadrant + red points mark pixels meeting both criteria.
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

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)
cfg.PREV_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — crater window
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
print(f"  Window: [{r0:,}, {r1:,}]")

# ---------------------------------------------------------------------------
# Step 2 — compute CPR & DOP
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

print("Computing CPR & DOP ...")
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
A = ML_SC + ML_OC; B = ML_OC - ML_SC
dop_arr = np.sqrt(B**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A + EPS)
dop_arr = np.clip(dop_arr, 0.0, 1.0)
del ML_SC, ML_OC, ML_CR, ML_CI, A, B

mask   = (cpr_arr > 0) & (cpr_arr <= 2.5) & np.isfinite(dop_arr)
cpr_1d = cpr_arr[mask].ravel()
dop_1d = dop_arr[mask].ravel()
del cpr_arr, dop_arr

# Partition: CPR-only ice / non-ice / joint
cpr_mask    = cpr_1d > ICE_THR
joint_mask  = cpr_mask & (dop_1d < DOP_ICE_THR)
cpr_ice     = cpr_1d[cpr_mask];   dop_ice = dop_1d[cpr_mask]
cpr_non     = cpr_1d[~cpr_mask];  dop_non = dop_1d[~cpr_mask]

n_tot_all   = len(cpr_1d)
n_cpr1_all  = int(cpr_mask.sum())
n_joint_all = int(joint_mask.sum())
ice_pct     = 100.0 * n_cpr1_all  / n_tot_all if n_tot_all > 0 else 0.0
joint_pct   = 100.0 * n_joint_all / n_tot_all if n_tot_all > 0 else 0.0

ij_ice_full = dop_ice < DOP_ICE_THR   # among CPR>1 pixels
n_joint_ice = int(ij_ice_full.sum())
ij_non_full = np.zeros(len(cpr_non), dtype=bool)

print(f"  All n={n_tot_all:,}  CPR>1: {ice_pct:.2f}%  "
      f"CPR>1&DOP<{DOP_ICE_THR}: {joint_pct:.2f}%  |  "
      f"Ice n={len(cpr_ice):,}  Non-ice n={len(cpr_non):,}")

def subsample3(c, d, ij, n=MAX_SCATTER, seed=42):
    rng_ = np.random.default_rng(seed)
    if len(c) > n:
        idx = rng_.choice(len(c), n, replace=False)
        return c[idx], d[idx], ij[idx]
    return c, d, ij.copy()

cp_all, dp_all, ij_all = subsample3(cpr_1d, dop_1d, joint_mask, MAX_SCATTER, 42)
cp_ice, dp_ice, ij_ice = subsample3(cpr_ice, dop_ice, ij_ice_full, MAX_SCATTER//2, 43)
cp_non, dp_non, ij_non = subsample3(cpr_non, dop_non, ij_non_full, MAX_SCATTER//2, 44)

FOOTER = (f"Chandrayaan-2 DFSAR | Faustini Crater | 2021-05-06 | L-band | "
          f"Multilook {MULTILOOK[0]}x{MULTILOOK[1]}\n"
          f"Ice candidate: CPR > {ICE_THR:.1f}  &  DOP < {DOP_ICE_THR}")

# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------
PANELS = [
    dict(cp=cp_all, dp=dp_all, ij=ij_all,
         mc=float(np.mean(cp_all)), md=float(np.mean(dp_all)),
         base_color="black",
         n_total=n_tot_all, n_cpr1=n_cpr1_all, n_joint=n_joint_all,
         title=f"All Crater Pixels  (n = {n_tot_all:,})"),
    dict(cp=cp_ice, dp=dp_ice, ij=ij_ice,
         mc=float(np.mean(cpr_ice)), md=float(np.mean(dop_ice)),
         base_color="#cc2200",
         n_total=len(cpr_ice), n_cpr1=len(cpr_ice), n_joint=n_joint_ice,
         title=f"Ice Candidates — CPR > 1  (n = {len(cpr_ice):,})"),
    dict(cp=cp_non, dp=dp_non, ij=ij_non,
         mc=float(np.mean(cpr_non)), md=float(np.mean(dop_non)),
         base_color="#004488",
         n_total=len(cpr_non), n_cpr1=0, n_joint=0,
         title=f"Non-Ice — CPR ≤ 1  (n = {len(cpr_non):,})"),
]

# ---------------------------------------------------------------------------
# Scatter helper
# ---------------------------------------------------------------------------
def draw_scatter(ax, p, fs=11):
    cp_, dp_, ij_ = p["cp"], p["dp"], p["ij"]
    mc_, md_      = p["mc"], p["md"]
    n_total       = p["n_total"]
    n_cpr1        = p["n_cpr1"]
    n_joint       = p["n_joint"]

    ax.set_facecolor("white")

    # Ice-candidate quadrant shading
    ax.axvspan(0.0, DOP_ICE_THR, ymin=ICE_THR / 2.0, ymax=1.0,
               color="red", alpha=0.08, zorder=0)

    # Non-ice (base colour) then ice-candidate (red)
    not_ij = ~ij_
    if not_ij.sum() > 0:
        ax.scatter(dp_[not_ij], cp_[not_ij],
                   s=2, c=p["base_color"], alpha=0.28,
                   linewidths=0, rasterized=True)
    if ij_.sum() > 0:
        ax.scatter(dp_[ij_], cp_[ij_],
                   s=4, c="red", alpha=0.70,
                   linewidths=0, rasterized=True, zorder=4,
                   label=f"CPR>1 & DOP<{DOP_ICE_THR}")

    # Blue mean star
    ax.plot(md_, mc_, marker="*", color="blue", markersize=13, zorder=5,
            label=f"Mean ({md_:.3f}, {mc_:.3f})")

    # Reference lines
    ax.axhline(ICE_THR,     color="gray",      lw=0.9, ls="--", alpha=0.65)
    ax.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 2.0)
    ax.set_xlabel("DOP", fontsize=fs)
    ax.set_ylabel("CPR", fontsize=fs)
    ax.tick_params(labelsize=max(fs - 2, 7))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=max(fs - 3, 7), loc="upper left",
              framealpha=0.85, edgecolor="lightgray", handlelength=1.2)
    ax.set_title(p["title"], fontsize=fs, fontweight="bold", pad=5)

    if n_total > 0:
        pct_cpr1  = 100.0 * n_cpr1  / n_total
        pct_joint = 100.0 * n_joint / n_total
        ax.text(0.97, 0.02,
                f"CPR>1: {pct_cpr1:.1f}%\n"
                f"CPR>1 & DOP<{DOP_ICE_THR}: {pct_joint:.2f}%",
                transform=ax.transAxes, fontsize=max(fs - 4, 7),
                va="bottom", ha="right", color="dimgray", linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

# ---------------------------------------------------------------------------
# Combined figure (all 3 side by side)
# ---------------------------------------------------------------------------
print("Saving combined figure ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                         gridspec_kw={"wspace": 0.38},
                         facecolor="white")
for ax, p in zip(axes, PANELS):
    draw_scatter(ax, p, fs=11)

fig.suptitle(
    "CPR vs DOP — Faustini Crater  |  Chandrayaan-2 DFSAR  |  2021-05-06\n"
    f"Ice candidate: CPR > {ICE_THR:.1f}  &  DOP < {DOP_ICE_THR}  (red points + shaded quadrant)",
    fontsize=12, fontweight="bold", y=1.02)
fig.text(0.5, -0.04, FOOTER, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
out = cfg.PREV_DIR / "faustini_scatter_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
plt.close(fig)
print(f"  Saved: {out.name}")

# ---------------------------------------------------------------------------
# 3 individual figures
# ---------------------------------------------------------------------------
fnames = ["faustini_scatter_all.png",
          "faustini_scatter_ice.png",
          "faustini_scatter_nonice.png"]

for p, fname in zip(PANELS, fnames):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    draw_scatter(ax, p, fs=12)
    fig.text(0.5, -0.04, FOOTER, ha="center", fontsize=7.5, color="gray", linespacing=1.5)
    out = cfg.PREV_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print(f"  Saved: {fname}")

print(f"\nAll 4 files saved to: {cfg.PREV_DIR}")
