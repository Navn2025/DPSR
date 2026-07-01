"""
cpr_dop_scatter.py
------------------
Compute CPR vs DOP scatter plots for the 9 ice-candidate regions
found by ice_zoom.py, styled after Fig. 4 of O'Brien & Byrne (2022).

DOP (Degree of Polarization) is computed from the circular-basis
Stokes parameters of the backscattered wave:

  S_RR = SC_field / 2   (same-sense circular)
  S_RL = OC_field / 2   (opposite-sense circular)

  g0 = <|S_RR|^2> + <|S_RL|^2>     = (ML_SC + ML_OC) / 4
  g1 = <|S_RL|^2> - <|S_RR|^2>     = (ML_OC - ML_SC) / 4
  g2 = 2 Re(<S_RR S_RL*>)           = Re(ML_cross) / 2
  g3 = -2 Im(<S_RR S_RL*>)          = -Im(ML_cross) / 2

  DOP = sqrt(g1^2 + g2^2 + g3^2) / g0
      = sqrt((ML_OC-ML_SC)^2 + 4*Re(ML_cross)^2 + 4*Im(ML_cross)^2)
        / (ML_SC + ML_OC)

  CPR = ML_SC / ML_OC

Usage:
    python cpr/cpr_dop_scatter.py
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
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AZ_PX          = 9.4
MULTILOOK      = cfg.MULTILOOK_WINDOW        # (19, 3)
EPS            = 1e-10
MAX_SCATTER_PX = 8000    # max dots per panel (random subsample)
RNG_SEED       = 42

# Ice-candidate criterion: high CPR (volume/subsurface scattering) AND low
# DOP (depolarized return) together are a stronger joint indicator than
# CPR>1 alone -- a rough but non-depolarizing scatterer can still have
# CPR>1 without being an ice candidate, whereas requiring DOP<0.13 as well
# selects for genuinely incoherent, strongly depolarizing volume scattering
# consistent with buried/coherent-backscatter ice deposits.
CPR_ICE_THRESH = 1.0
DOP_ICE_THRESH = 0.13

# Ice-candidate cluster centres (azimuth line) from ice_zoom.py
CLUSTER_CENTRES = [29248, 30174, 36791, 37708, 38471, 39170, 40396, 71492, 219456]
CLUSTER_LABELS  = [f"R{i+1}" for i in range(len(CLUSTER_CENTRES))]

# Patch half-size (same as ice_zoom.py: ~6.1 km)
PATCH_AZ   = int(round(244 * 25.0 / AZ_PX))   # 649

SCENE_H    = 252825

OUT_DIR    = cfg.PREV_DIR / "cpr_dop"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / "cpr_dop_scatter.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_window(pol: str, r0: int, r1: int) -> np.ndarray:
    """Load SLC complex window -> complex64 array."""
    path = cfg.SLI_PATHS[pol]
    with rasterio.open(path) as src:
        win  = Window(0, r0, src.width, r1 - r0)
        real = src.read(1, window=win).astype(np.float32)
        imag = src.read(2, window=win).astype(np.float32)
    return real + 1j * imag


def multilook(arr: np.ndarray) -> np.ndarray:
    az, rg = MULTILOOK
    return uniform_filter(arr.astype(np.float64), size=(az, rg)).astype(np.float32)


def compute_cpr_dop(r0: int, r1: int):
    """Return (cpr_1d, dop_1d) flat arrays for valid pixels in [r0, r1)."""
    # --- load HH, VV ---
    S_HH = load_window("HH", r0, r1)
    S_VV = load_window("VV", r0, r1)
    OC   = S_HH + S_VV
    diff = S_HH - S_VV
    del S_HH, S_VV

    # --- load HV, VH ---
    S_HV = load_window("HV", r0, r1)
    S_VH = load_window("VH", r0, r1)
    S_XP = (S_HV + S_VH) * 0.5
    del S_HV, S_VH

    # --- SC field ---
    SC      = np.empty_like(diff)
    SC.real = diff.real - 2.0 * S_XP.imag
    SC.imag = diff.imag + 2.0 * S_XP.real
    del diff, S_XP

    # --- powers and cross-correlation ---
    P_SC       = SC.real**2 + SC.imag**2
    P_OC       = OC.real**2 + OC.imag**2
    # SC * conj(OC)
    cross_real = SC.real * OC.real + SC.imag * OC.imag
    cross_imag = SC.imag * OC.real - SC.real * OC.imag
    del SC, OC

    # --- multilook ---
    ML_SC = multilook(P_SC);   del P_SC
    ML_OC = multilook(P_OC);   del P_OC
    ML_CR = multilook(cross_real); del cross_real
    ML_CI = multilook(cross_imag); del cross_imag

    # --- CPR ---
    cpr = ML_SC / (ML_OC + EPS)

    # --- DOP ---
    A   = ML_SC + ML_OC
    B   = ML_OC - ML_SC
    dop = np.sqrt(B**2 + 4.0 * ML_CR**2 + 4.0 * ML_CI**2) / (A + EPS)
    dop = np.clip(dop, 0.0, 1.0)

    # --- mask: physical CPR and DOP ---
    mask = (ML_OC > EPS) & (cpr > 0) & (cpr <= 2.5) & (dop >= 0) & (dop <= 1.0)

    return cpr[mask].ravel(), dop[mask].ravel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)

print(f"Computing CPR & DOP for {len(CLUSTER_CENTRES)} regions ...")

results = []
for label, az_centre in zip(CLUSTER_LABELS, CLUSTER_CENTRES):
    half = PATCH_AZ // 2
    r0   = max(0, az_centre - half)
    r1   = min(SCENE_H, az_centre + half)
    print(f"  {label}  az={az_centre:,}  rows [{r0:,}, {r1:,}] ...", end=" ", flush=True)

    cpr_1d, dop_1d = compute_cpr_dop(r0, r1)

    mean_cpr   = float(np.mean(cpr_1d))
    mean_dop   = float(np.mean(dop_1d))
    n_ice      = int((cpr_1d > CPR_ICE_THRESH).sum())
    ice_mask   = (cpr_1d > CPR_ICE_THRESH) & (dop_1d < DOP_ICE_THRESH)
    n_ice_comb = int(ice_mask.sum())
    print(
        f"n={len(cpr_1d):,}  mean_CPR={mean_cpr:.3f}  mean_DOP={mean_dop:.3f}  "
        f"CPR>1={100*n_ice/len(cpr_1d):.1f}%  "
        f"CPR>1 & DOP<{DOP_ICE_THRESH}={100*n_ice_comb/len(cpr_1d):.2f}%"
    )

    # subsample for display (keep ice-candidate points preferentially so
    # the joint criterion stays visible even after subsampling)
    if len(cpr_1d) > MAX_SCATTER_PX:
        idx    = rng.choice(len(cpr_1d), MAX_SCATTER_PX, replace=False)
        c_plot = cpr_1d[idx]
        d_plot = dop_1d[idx]
        ice_plot = ice_mask[idx]
    else:
        c_plot, d_plot, ice_plot = cpr_1d, dop_1d, ice_mask

    results.append(dict(
        label=label, az=az_centre,
        cpr_plot=c_plot, dop_plot=d_plot, ice_plot=ice_plot,
        mean_cpr=mean_cpr, mean_dop=mean_dop,
        n_ice=n_ice, n_ice_comb=n_ice_comb, n_total=len(cpr_1d),
    ))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
ncols = 3
nrows = int(np.ceil(len(results) / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    sharex=False, sharey=False,
)
axes = np.array(axes).ravel()

for idx, (res, ax) in enumerate(zip(results, axes)):
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Ice-candidate quadrant: DOP < DOP_ICE_THRESH & CPR > CPR_ICE_THRESH
    ax.axvspan(0.0, DOP_ICE_THRESH, ymin=(CPR_ICE_THRESH / 2.0), ymax=1.0,
               color="red", alpha=0.08, zorder=0)

    not_ice = ~res["ice_plot"]
    ax.scatter(
        res["dop_plot"][not_ice], res["cpr_plot"][not_ice],
        s=2, c="black", alpha=0.35, linewidths=0, rasterized=True,
    )
    ax.scatter(
        res["dop_plot"][res["ice_plot"]], res["cpr_plot"][res["ice_plot"]],
        s=4, c="red", alpha=0.7, linewidths=0, rasterized=True, zorder=4,
        label="Ice candidate",
    )

    # Blue star at mean (red is reserved for ice-candidate points above)
    ax.plot(res["mean_dop"], res["mean_cpr"],
            marker="*", color="blue", markersize=12, zorder=5,
            label=f"Mean ({res['mean_dop']:.2f}, {res['mean_cpr']:.2f})")

    # CPR=1.0 and DOP threshold reference lines
    ax.axhline(CPR_ICE_THRESH, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax.axvline(DOP_ICE_THRESH, color="gray", lw=0.7, ls="--", alpha=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel("DOP", fontsize=10)
    ax.set_ylabel("CPR", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.spines[["top", "right"]].set_visible(False)

    # Panel label (upper right, inside)
    ax.text(0.97, 0.97, res["label"],
            transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold")

    # Ice % annotation
    ice_pct      = 100.0 * res["n_ice"] / res["n_total"]
    ice_comb_pct = 100.0 * res["n_ice_comb"] / res["n_total"]
    ax.text(0.03, 0.04,
            f"CPR>1: {ice_pct:.1f}%\n"
            f"CPR>1 & DOP<{DOP_ICE_THRESH}: {ice_comb_pct:.2f}%\n"
            f"az={res['az']*AZ_PX/1000:.0f} km",
            transform=ax.transAxes, fontsize=6.5, va="bottom", color="dimgray")

# hide unused axes
for ax in axes[len(results):]:
    ax.set_visible(False)

# One shared legend for the whole figure (avoids per-panel clutter)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.045),
           ncol=3, fontsize=9, frameon=False)

fig.suptitle(
    "CPR vs DOP — Faustini Scene Ice-Candidate Regions\n"
    "Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06  |  L-band\n"
    f"Ice candidate = CPR > {CPR_ICE_THRESH:.1f}  &  DOP < {DOP_ICE_THRESH}",
    fontsize=11, fontweight="bold", y=1.09,
)
plt.tight_layout(pad=0.8)

fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
