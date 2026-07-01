"""
faustini_zoom.py
----------------
Comprehensive polarimetric feature extraction for the inner-crater
sub-region of Faustini (centre ±700 azimuth lines ≈ 13.2 km × 6.1 km).

Both default CPR (Putrevu formula) and research CPR (mu_c log-rescaled)
are analysed alongside 7 additional polarimetric features.

9 features total:
  CPR(def)   ML_SC / ML_OC
  CPR(res)   log10-rescaled mu_c (from Calculated_CPR_research.tif)
  DOP        Stokes-Kennaugh degree of polarization
  Span_dB    10·log10(P_HH + 2·P_HV + P_VV)
  copol_dB   10·log10(P_HH / P_VV) co-pol imbalance
  xpol_frac  P_HV / (P_HH + P_VV)
  H          Cloude-Pottier entropy [0,1]
  A          Cloude-Pottier anisotropy [0,1]
  alpha      Cloude-Pottier mean alpha angle [0°,90°]

Ice criterion (applied to all scatter plots): CPR > 1  AND  DOP < 0.13

Outputs → cpr/faustini/outputs/previews/zoom/
  zoom_feature_maps.png     3×3 spatial feature maps
  zoom_histograms.png       2×3 feature histograms
  zoom_scatter_default.png  default CPR vs DOP (3 panels)
  zoom_scatter_research.png research CPR vs DOP (3 panels)
  zoom_compare.png          CPR_def map | def-vs-res scatter | CPR_res map
  zoom_halpha.png           Cloude-Pottier H-α plane
  zoom_polsar.png           PolSAR decomp maps
  zoom_polsar_hist.png      PolSAR decomp histograms

Usage:
    python cpr/faustini_zoom.py
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

# ===========================================================================
# CONSTANTS
# ===========================================================================
CRATER_LAT   = -87.18
SCENE_H      = 252825
AZ_PX        = 9.4
RG_PX        = 25.0
ML_AZ, ML_RG = cfg.MULTILOOK_WINDOW
EPS          = 1e-10
ZOOM_HALF    = 700
DISPLAY_SZ   = 512
MAX_SCATTER  = 15_000
ICE_THR      = 1.0
DOP_ICE_THR  = 0.13
RG_W         = 244

CPR_DEF_TIF  = cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME
CPR_RES_TIF  = cfg.CPR_DIR / "Calculated_CPR_research.tif"

OUT_DIR = cfg.PREV_DIR / "zoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEOM_CSV = (
    cfg.BASE_DIR / "faustini" / "geometry" / "calibrated" / "20210506"
    / "ch2_sar_ncxl_20210506t022608652_g_sli_xx_fp_xx_d18.csv"
)

# ===========================================================================
# HELPERS
# ===========================================================================
def make_cpr_cmap():
    nodes = [(0,"#000000"),(0.10,"#003300"),(0.30,"#00cc00"),
             (0.52,"#0000ff"),(0.74,"#ff00ff"),(0.90,"#ff5566"),(1,"#ffffff")]
    c = mcolors.LinearSegmentedColormap.from_list("cpr_paper", nodes)
    c.set_bad("black")
    return c
CPR_CMAP = make_cpr_cmap()


def resize_sq(patch, sz=DISPLAY_SZ):
    h, w = patch.shape
    nm   = np.isnan(patch)
    fill = float(np.nanmean(patch)) if np.any(~nm) else 0.0
    r    = nd_zoom(np.where(nm, fill, patch), (sz/h, sz/w), order=1)
    nm_r = nd_zoom(nm.astype(np.float32), (sz/h, sz/w), order=1)
    r[nm_r > 0.4] = np.nan
    return r.astype(np.float32)


def save_fig(fig, name, dpi=200):
    p = OUT_DIR / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {p.name}")


def subsample(c, d, ij, n=MAX_SCATTER, seed=42):
    rng_ = np.random.default_rng(seed)
    if len(c) > n:
        idx = rng_.choice(len(c), n, replace=False)
        return c[idx], d[idx], ij[idx]
    return c, d, ij.copy()


def map_panel(ax, arr2d, title, cmap, vmin=None, vmax=None, bad="black"):
    disp  = resize_sq(arr2d)
    valid = disp[np.isfinite(disp)]
    if vmin is None: vmin = float(np.percentile(valid, 2))  if valid.size else 0.0
    if vmax is None: vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
    ax.set_facecolor(bad)
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    return im


def draw_scatter(ax, cp_, dp_, ij_, mc_, md_, title_,
                 base_color="black", n_total=None, n_cpr1=None, n_joint=None,
                 fs=11, ylab="CPR", ylim=(0.0, 2.5), ythr=ICE_THR):
    ax.set_facecolor("white")
    y0, y1 = ylim
    ymin_f = max(0.0, (ythr - y0) / (y1 - y0)) if (y1 - y0) > 0 else 0.5
    ax.axvspan(0.0, DOP_ICE_THR, ymin=ymin_f, ymax=1.0,
               color="red", alpha=0.08, zorder=0)
    not_ij = ~ij_
    if not_ij.sum() > 0:
        ax.scatter(dp_[not_ij], cp_[not_ij],
                   s=2, c=base_color, alpha=0.25, linewidths=0, rasterized=True)
    if ij_.sum() > 0:
        ax.scatter(dp_[ij_], cp_[ij_],
                   s=4, c="red", alpha=0.70, linewidths=0, rasterized=True,
                   zorder=4, label=f"CPR>1 & DOP<{DOP_ICE_THR}")
    ax.plot(md_, mc_, marker="*", color="blue", markersize=13, zorder=5,
            label=f"Mean ({md_:.3f}, {mc_:.3f})")
    ax.axhline(ythr,        color="gray",      lw=0.9, ls="--", alpha=0.65)
    ax.axvline(DOP_ICE_THR, color="steelblue", lw=0.9, ls=":",  alpha=0.70)
    ax.set_xlim(0, 1); ax.set_ylim(y0, y1)
    ax.set_xlabel("DOP", fontsize=fs); ax.set_ylabel(ylab, fontsize=fs)
    ax.tick_params(labelsize=max(fs-2, 7))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=max(fs-3, 7), loc="upper left",
              framealpha=0.85, edgecolor="lightgray", handlelength=1.2)
    ax.set_title(title_, fontsize=fs, fontweight="bold", pad=5)
    if n_total is not None and n_total > 0:
        ax.text(0.97, 0.02,
                f"CPR>1: {100*n_cpr1/n_total:.1f}%\n"
                f"& DOP<{DOP_ICE_THR}: {100*n_joint/n_total:.2f}%",
                transform=ax.transAxes, fontsize=max(fs-4, 7),
                va="bottom", ha="right", color="dimgray", linespacing=1.4,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))


def ml2d(x):
    return uniform_filter(x.astype(np.float64), size=(ML_AZ, ML_RG)).astype(np.float32)


# ===========================================================================
# SECTION 1 — geometry → zoom window
# ===========================================================================
print("="*60)
print("Reading geometry CSV ...")
df      = pd.read_csv(GEOM_CSV)
lat_col = [c for c in df.columns if "Latitude"    in c][0]
slr_col = [c for c in df.columns if "Slant_Range" in c][0]
slant   = df[slr_col].values
n_rng   = int(np.where(np.diff(slant) < -500)[0][0] + 1)
n_az    = len(df) // n_rng
lat_ties = df[lat_col].values[:n_az*n_rng].reshape(n_az, n_rng)[:, n_rng//2]
lat_fn   = interp1d(np.linspace(0, SCENE_H-1, n_az), lat_ties,
                    kind="linear", fill_value="extrapolate")
lat_all  = lat_fn(np.arange(SCENE_H, dtype=np.float64))

centre_row = int(np.argmin(np.abs(lat_all - CRATER_LAT)))
zr0   = max(0,         centre_row - ZOOM_HALF)
zr1   = min(SCENE_H-1, centre_row + ZOOM_HALF)
ZH    = zr1 - zr0 + 1
az_km = ZH   * AZ_PX / 1000.0
rg_km = RG_W * RG_PX / 1000.0
print(f"  Zoom: rows [{zr0:,}, {zr1:,}]  {ZH} lines  "
      f"{az_km:.1f} km x {rg_km:.1f} km")

FOOTNOTE = (
    f"Chandrayaan-2 DFSAR  |  Faustini inner-crater zoom "
    f"({az_km:.1f}x{rg_km:.1f} km)  |  2021-05-06  |  L-band  |  "
    f"Multilook {ML_AZ}x{ML_RG}\n"
    f"Ice candidate: CPR > 1  &  DOP < {DOP_ICE_THR}"
)

def add_footnote(fig):
    fig.text(0.5, -0.03, FOOTNOTE, ha="center", fontsize=7.5,
             color="gray", linespacing=1.5)


# ===========================================================================
# SECTION 2 — Load SLC and compute all polarimetric features
# ===========================================================================
def slc_win(pol):
    with rasterio.open(cfg.SLI_PATHS[pol]) as src:
        w  = Window(0, zr0, src.width, ZH)
        re = src.read(1, window=w).astype(np.float32)
        im = src.read(2, window=w).astype(np.float32)
    return re + 1j * im

print("\nLoading SLC data ...")
S_HH = slc_win("HH")
S_VV = slc_win("VV")
OC   = S_HH + S_VV
diff = S_HH - S_VV

# Individual HH, VV powers for span/copol (recover from OC/diff)
SHH_re = (OC.real + diff.real) * 0.5
SHH_im = (OC.imag + diff.imag) * 0.5
SVV_re = (OC.real - diff.real) * 0.5
SVV_im = (OC.imag - diff.imag) * 0.5
P_HH = SHH_re**2 + SHH_im**2
P_VV = SVV_re**2 + SVV_im**2
del SHH_re, SHH_im, SVV_re, SVV_im, S_HH, S_VV

S_HV = slc_win("HV")
S_VH = slc_win("VH")
XP   = (S_HV + S_VH) * 0.5
del S_HV, S_VH

SC_re = diff.real - 2.0*XP.imag
SC_im = diff.imag + 2.0*XP.real

print("  Computing single-look products ...")
P_HV = XP.real**2  + XP.imag**2
P_SC = SC_re**2    + SC_im**2
P_OC = OC.real**2  + OC.imag**2
P_df = diff.real**2 + diff.imag**2

# DOP cross-corr
CR = SC_re*OC.real + SC_im*OC.imag
CI = SC_im*OC.real - SC_re*OC.imag

# T3 Pauli cross-corr (k1=OC/rt2, k2=diff/rt2, k3=rt2*XP)
c12r = OC.real*diff.real + OC.imag*diff.imag
c12i = OC.imag*diff.real - OC.real*diff.imag
c13r = OC.real*XP.real   + OC.imag*XP.imag
c13i = OC.imag*XP.real   - OC.real*XP.imag
c23r = diff.real*XP.real  + diff.imag*XP.imag
c23i = diff.imag*XP.real  - diff.real*XP.imag

del OC, diff, XP, SC_re, SC_im

print("  Multiloking ...")
ML_HH   = ml2d(P_HH)
ML_VV   = ml2d(P_VV)
ML_HV   = ml2d(P_HV)
ML_SC   = ml2d(P_SC)
ML_OC   = ml2d(P_OC)
ML_CR   = ml2d(CR)
ML_CI   = ml2d(CI)
ML_T22  = ml2d(P_df) / 2.0
ML_T12r = ml2d(c12r) / 2.0;  ML_T12i = ml2d(c12i) / 2.0
ML_T13r = ml2d(c13r);         ML_T13i = ml2d(c13i)
ML_T23r = ml2d(c23r);         ML_T23i = ml2d(c23i)
del P_HH, P_VV, P_HV, P_SC, P_OC, P_df, CR, CI
del c12r, c12i, c13r, c13i, c23r, c23i

# T11 — before deleting ML_OC
ML_T11 = (ML_OC / 2.0).astype(np.float32)

# CPR (default from SLC)
cpr_slc = (ML_SC / (ML_OC + EPS)).astype(np.float32)

# DOP
A_dop   = ML_SC + ML_OC
B_dop   = ML_OC - ML_SC
dop_slc = np.sqrt(B_dop**2 + 4*ML_CR**2 + 4*ML_CI**2) / (A_dop + EPS)
dop_slc = np.clip(dop_slc, 0.0, 1.0).astype(np.float32)

valid = (ML_OC > EPS) & np.isfinite(cpr_slc) & (cpr_slc > 0) & (cpr_slc < 10)
cpr_slc[~valid] = np.nan
dop_slc[~valid] = np.nan

del ML_SC, ML_CR, ML_CI, A_dop, B_dop, ML_OC

# T33 — before deleting ML_HV
ML_T33 = (2.0 * ML_HV).astype(np.float32)

# Span, co-pol ratio, cross-pol fraction
span_db   = np.where(valid,
                     10*np.log10(np.maximum(ML_HH + 2*ML_HV + ML_VV, EPS)),
                     np.nan).astype(np.float32)
copol_db  = np.where(valid,
                     10*np.log10(ML_HH / (ML_VV + EPS)),
                     np.nan).astype(np.float32)
xpol_frac = np.where(valid,
                     ML_HV / (ML_HH + ML_VV + EPS),
                     np.nan).astype(np.float32)
del ML_HH, ML_VV, ML_HV

# ===========================================================================
# SECTION 3 — Cloude-Pottier T3 eigendecomposition
# ===========================================================================
N = ZH * RG_W
print(f"\nBuilding T3 ({N:,} matrices) ...")
T3 = np.zeros((N, 3, 3), dtype=np.complex128)
T3[:, 0, 0] = ML_T11.ravel().astype(np.float64)
T3[:, 1, 1] = ML_T22.ravel().astype(np.float64)
T3[:, 2, 2] = ML_T33.ravel().astype(np.float64)
T3[:, 0, 1] = (ML_T12r + 1j*ML_T12i).ravel()
T3[:, 0, 2] = (ML_T13r + 1j*ML_T13i).ravel()
T3[:, 1, 2] = (ML_T23r + 1j*ML_T23i).ravel()
T3[:, 1, 0] = T3[:, 0, 1].conj()
T3[:, 2, 0] = T3[:, 0, 2].conj()
T3[:, 2, 1] = T3[:, 1, 2].conj()
del ML_T11, ML_T22, ML_T33, ML_T12r, ML_T12i, ML_T13r, ML_T13i, ML_T23r, ML_T23i

print("  Running eigh ...")
eigvals, eigvecs = np.linalg.eigh(T3)   # ascending, real eigenvalues
del T3

trace = eigvals.sum(axis=1, keepdims=True)
trace = np.where(trace > EPS, trace, EPS)
p = np.clip(eigvals / trace, 1e-12, 1.0)

# H = -sum p_i * log3(p_i)
H_flat = -np.sum(p * np.log(p) / np.log(3), axis=1)

# A = (p2 - p3) / (p2 + p3)  [ascending: idx1=lambda2, idx0=lambda3]
denom_a = p[:, 1] + p[:, 0]
A_flat  = np.where(denom_a > EPS, (p[:, 1] - p[:, 0]) / denom_a, 0.0)

# alpha = sum p_i * arccos(|v_i[0]|) degrees
alpha_flat = np.sum(
    p * np.degrees(np.arccos(np.clip(np.abs(eigvecs[:, 0, :]), 0.0, 1.0))),
    axis=1
)
del eigvals, eigvecs, p, denom_a

H_map     = np.where(valid.ravel(), H_flat,     np.nan).reshape(ZH, RG_W).astype(np.float32)
A_map     = np.where(valid.ravel(), A_flat,     np.nan).reshape(ZH, RG_W).astype(np.float32)
alpha_map = np.where(valid.ravel(), alpha_flat, np.nan).reshape(ZH, RG_W).astype(np.float32)
del H_flat, A_flat, alpha_flat

print(f"  H in [{np.nanmin(H_map):.3f}, {np.nanmax(H_map):.3f}]")
print(f"  A in [{np.nanmin(A_map):.3f}, {np.nanmax(A_map):.3f}]")
print(f"  alpha in [{np.nanmin(alpha_map):.1f}, {np.nanmax(alpha_map):.1f}] deg")

# ===========================================================================
# SECTION 4 — Load precomputed CPR TIFs
# ===========================================================================
def tif_win(path):
    with rasterio.open(path) as src:
        arr = src.read(1, window=Window(0, zr0, src.width, ZH)).astype(np.float32)
        nd  = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr

print("\nLoading CPR TIFs ...")
cpr_def_map = tif_win(CPR_DEF_TIF)
cpr_res_map = tif_win(CPR_RES_TIF)
cpr_def_map[(cpr_def_map <= 0) | (cpr_def_map > 10)] = np.nan
cpr_res_map[~valid] = np.nan

# ===========================================================================
# SECTION 5 — Build flat arrays for scatter / histogram
# ===========================================================================
vmask_def = valid & np.isfinite(cpr_def_map)
vmask_res = valid & np.isfinite(cpr_res_map)

# Default CPR scatter
cpr_d1 = cpr_def_map[vmask_def].ravel()
dop_d1 = dop_slc[vmask_def].ravel()
ij_d1  = (cpr_d1 > ICE_THR) & (dop_d1 < DOP_ICE_THR)
cm_d   = cpr_d1 > ICE_THR

# Research CPR scatter
cpr_r1 = cpr_res_map[vmask_res].ravel()
dop_r1 = dop_slc[vmask_res].ravel()
ij_r1  = (cpr_r1 > ICE_THR) & (dop_r1 < DOP_ICE_THR)
cm_r   = cpr_r1 > ICE_THR

def s_stats(c, d, ij, cm):
    n = len(c)
    return dict(n_tot=n, n_cpr1=int(cm.sum()), n_joint=int(ij.sum()),
                mc=float(np.mean(c)), md=float(np.mean(d)))

sd  = s_stats(cpr_d1, dop_d1, ij_d1, cm_d)
sr  = s_stats(cpr_r1, dop_r1, ij_r1, cm_r)

print(f"Default  n={sd['n_tot']:,}  CPR>1={100*sd['n_cpr1']/sd['n_tot']:.2f}%  "
      f"joint={100*sd['n_joint']/sd['n_tot']:.3f}%")
print(f"Research n={sr['n_tot']:,}  CPR>1={100*sr['n_cpr1']/sr['n_tot']:.2f}%  "
      f"joint={100*sr['n_joint']/sr['n_tot']:.3f}%")

# Partition for 3-panel scatter (all/ice/non-ice)
ice_d = cpr_d1[cm_d]; dop_ice_d = dop_d1[cm_d]
non_d = cpr_d1[~cm_d]; dop_non_d = dop_d1[~cm_d]
ij_ice_d_full = dop_ice_d < DOP_ICE_THR

ice_r = cpr_r1[cm_r]; dop_ice_r = dop_r1[cm_r]
non_r = cpr_r1[~cm_r]; dop_non_r = dop_r1[~cm_r]
ij_ice_r_full = dop_ice_r < DOP_ICE_THR

cp_a_d, dp_a_d, ij_a_d = subsample(cpr_d1, dop_d1, ij_d1, MAX_SCATTER, 10)
cp_i_d, dp_i_d, ij_i_d = subsample(ice_d, dop_ice_d, ij_ice_d_full, MAX_SCATTER//2, 11)
cp_n_d, dp_n_d, ij_n_d = subsample(non_d, dop_non_d,
                                    np.zeros(len(non_d), bool), MAX_SCATTER//2, 12)

cp_a_r, dp_a_r, ij_a_r = subsample(cpr_r1, dop_r1, ij_r1, MAX_SCATTER, 20)
cp_i_r, dp_i_r, ij_i_r = subsample(ice_r, dop_ice_r, ij_ice_r_full, MAX_SCATTER//2, 21)
cp_n_r, dp_n_r, ij_n_r = subsample(non_r, dop_non_r,
                                    np.zeros(len(non_r), bool), MAX_SCATTER//2, 22)


# ===========================================================================
# FIGURE 1 — zoom_feature_maps.png
# ===========================================================================
print("\n[1] zoom_feature_maps.png ...")
FEATS = [
    (cpr_def_map, "CPR (Default)",       CPR_CMAP,  0.0,  2.5, "black"),
    (cpr_res_map, "CPR (Research mu_c)", CPR_CMAP,  0.0,  2.0, "black"),
    (dop_slc,     "DOP",                 "plasma",   0.0,  1.0, "gray"),
    (span_db,     "Span [dB]",           "gray",     None, None, "gray"),
    (copol_db,    "HH/VV [dB]",          "RdBu_r",  None, None, "gray"),
    (xpol_frac,   "HV fraction",         "hot_r",    0.0,  0.4, "gray"),
    (H_map,       "Entropy H",           "viridis",  0.0,  1.0, "gray"),
    (A_map,       "Anisotropy A",        "viridis",  0.0,  1.0, "gray"),
    (alpha_map,   "Alpha [deg]",         "coolwarm", 0.0, 90.0, "gray"),
]

fig, axes = plt.subplots(3, 3, figsize=(14, 13), facecolor="white",
                         gridspec_kw={"hspace": 0.08, "wspace": 0.05})
for ax, (arr, title, cmap, vmin, vmax, bad) in zip(axes.ravel(), FEATS):
    im = map_panel(ax, arr, title, cmap, vmin, vmax, bad)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
fig.suptitle(
    "Faustini Inner-Crater — Polarimetric Feature Maps\n"
    f"Zoom {az_km:.1f}x{rg_km:.1f} km  |  Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.005)
add_footnote(fig)
save_fig(fig, "zoom_feature_maps.png")


# ===========================================================================
# FIGURE 2 — zoom_histograms.png
# ===========================================================================
print("[2] zoom_histograms.png ...")
HIST_FEATS = [
    (cpr_d1,              "CPR (Default)",      "#2266aa", (0,   2.5), ICE_THR,      None),
    (cpr_r1,              "CPR (Research mu_c)","#bb4400", (0,   2.0), ICE_THR,      None),
    (dop_slc[valid],      "DOP",                "#226644", (0,   1.0), None, DOP_ICE_THR),
    (H_map[valid],        "Entropy H",          "#664488", (0,   1.0), None,         None),
    (A_map[valid],        "Anisotropy A",       "#885522", (0,   1.0), None,         None),
    (alpha_map[valid],    "Alpha [deg]",        "#336688", (0,  90.0), None,         None),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white",
                         gridspec_kw={"hspace": 0.42, "wspace": 0.35})
for ax, (data, label, color, xlim, vthr, hthr) in zip(axes.ravel(), HIST_FEATS):
    finite = data[np.isfinite(data)]
    ax.hist(finite, bins=100, color=color, alpha=0.75, edgecolor="none",
            range=xlim, density=True)
    med = float(np.median(finite)); mn = float(np.mean(finite))
    ax.axvline(med, color="black",  lw=1.2, ls="--", label=f"Med {med:.3f}")
    ax.axvline(mn,  color="orange", lw=1.2, ls="-",  label=f"Mn  {mn:.3f}")
    if vthr is not None:
        ax.axvline(vthr, color="red",       lw=1.0, ls=":", label=f"CPR={vthr}")
    if hthr is not None:
        ax.axvline(hthr, color="steelblue", lw=1.0, ls=":", label=f"DOP={hthr}")
    ax.set_xlabel(label, fontsize=10); ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(xlim); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{label} Histogram", fontsize=10, fontweight="bold")
fig.suptitle(
    "Faustini Inner-Crater — Feature Histograms\n"
    "Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_histograms.png")


# ===========================================================================
# FIGURE 3 — zoom_scatter_default.png
# ===========================================================================
print("[3] zoom_scatter_default.png ...")
n_ice_d = sd["n_cpr1"]; n_tot_d = sd["n_tot"]
DPANELS = [
    dict(cp=cp_a_d, dp=dp_a_d, ij=ij_a_d,
         mc=sd["mc"], md=sd["md"], base_color="black",
         n_total=n_tot_d, n_cpr1=n_ice_d, n_joint=sd["n_joint"],
         title=f"All Inner-Crater  (n={n_tot_d:,})"),
    dict(cp=cp_i_d, dp=dp_i_d, ij=ij_i_d,
         mc=float(np.mean(ice_d)), md=float(np.mean(dop_ice_d)),
         base_color="#cc2200",
         n_total=n_ice_d, n_cpr1=n_ice_d, n_joint=int(ij_ice_d_full.sum()),
         title=f"CPR(def) > 1  (n={n_ice_d:,})"),
    dict(cp=cp_n_d, dp=dp_n_d, ij=ij_n_d,
         mc=float(np.mean(non_d)), md=float(np.mean(dop_non_d)),
         base_color="#004488",
         n_total=n_tot_d-n_ice_d, n_cpr1=0, n_joint=0,
         title=f"CPR(def) <= 1  (n={n_tot_d-n_ice_d:,})"),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                         gridspec_kw={"wspace": 0.38}, facecolor="white")
for ax, p in zip(axes, DPANELS):
    draw_scatter(ax, p["cp"], p["dp"], p["ij"], p["mc"], p["md"], p["title"],
                 base_color=p["base_color"],
                 n_total=p["n_total"], n_cpr1=p["n_cpr1"], n_joint=p["n_joint"],
                 fs=11, ylab="CPR (Default)", ylim=(0, 2.5))
fig.suptitle(
    "CPR (Default) vs DOP — Faustini Inner-Crater  |  Chandrayaan-2 DFSAR\n"
    f"Ice candidate: CPR > {ICE_THR}  &  DOP < {DOP_ICE_THR}",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_scatter_default.png")


# ===========================================================================
# FIGURE 4 — zoom_scatter_research.png
# ===========================================================================
print("[4] zoom_scatter_research.png ...")
n_ice_r = sr["n_cpr1"]; n_tot_r = sr["n_tot"]
RPANELS = [
    dict(cp=cp_a_r, dp=dp_a_r, ij=ij_a_r,
         mc=sr["mc"], md=sr["md"], base_color="black",
         n_total=n_tot_r, n_cpr1=n_ice_r, n_joint=sr["n_joint"],
         title=f"All Inner-Crater  (n={n_tot_r:,})"),
    dict(cp=cp_i_r, dp=dp_i_r, ij=ij_i_r,
         mc=float(np.mean(ice_r)), md=float(np.mean(dop_ice_r)),
         base_color="#cc2200",
         n_total=n_ice_r, n_cpr1=n_ice_r, n_joint=int(ij_ice_r_full.sum()),
         title=f"CPR(res) > 1  (n={n_ice_r:,})"),
    dict(cp=cp_n_r, dp=dp_n_r, ij=ij_n_r,
         mc=float(np.mean(non_r)), md=float(np.mean(dop_non_r)),
         base_color="#004488",
         n_total=n_tot_r-n_ice_r, n_cpr1=0, n_joint=0,
         title=f"CPR(res) <= 1  (n={n_tot_r-n_ice_r:,})"),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                         gridspec_kw={"wspace": 0.38}, facecolor="white")
for ax, p in zip(axes, RPANELS):
    draw_scatter(ax, p["cp"], p["dp"], p["ij"], p["mc"], p["md"], p["title"],
                 base_color=p["base_color"],
                 n_total=p["n_total"], n_cpr1=p["n_cpr1"], n_joint=p["n_joint"],
                 fs=11, ylab="CPR(mu_c) [log-rescaled]", ylim=(0, 2.0))
fig.suptitle(
    "CPR (Research mu_c) vs DOP — Faustini Inner-Crater  |  Chandrayaan-2 DFSAR\n"
    f"Ice candidate: CPR(res) > {ICE_THR}  &  DOP < {DOP_ICE_THR}",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_scatter_research.png")


# ===========================================================================
# FIGURE 5 — zoom_compare.png
# ===========================================================================
print("[5] zoom_compare.png ...")
both_valid = vmask_def & vmask_res
cd_b = cpr_def_map[both_valid].ravel()
cr_b = cpr_res_map[both_valid].ravel()
dp_b = dop_slc[both_valid].ravel()
rng_ = np.random.default_rng(99)
if len(cd_b) > MAX_SCATTER:
    idx_ = rng_.choice(len(cd_b), MAX_SCATTER, replace=False)
    cd_b, cr_b, dp_b = cd_b[idx_], cr_b[idx_], dp_b[idx_]

fig = plt.figure(figsize=(18, 6.5), facecolor="white")
gs  = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1.4, 1], wspace=0.08)

ax_def = fig.add_subplot(gs[0])
cax_d  = fig.add_subplot(gs[1])
ax_def.set_facecolor("black")
im_d   = ax_def.imshow(resize_sq(cpr_def_map), cmap=CPR_CMAP, vmin=0, vmax=2.5,
                        aspect="auto", interpolation="nearest")
ax_def.set_xticks([]); ax_def.set_yticks([])
ax_def.set_title("CPR (Default)", fontsize=11, fontweight="bold", pad=5)
fig.colorbar(im_d, cax=cax_d).ax.tick_params(labelsize=8)

ax_sc = fig.add_subplot(gs[2])
sc = ax_sc.scatter(cd_b, cr_b, c=dp_b, cmap="plasma_r", vmin=0, vmax=1,
                   s=3, alpha=0.45, linewidths=0, rasterized=True)
cb_sc = fig.colorbar(sc, ax=ax_sc, fraction=0.038, pad=0.02)
cb_sc.set_label("DOP", fontsize=10); cb_sc.ax.tick_params(labelsize=8)
ax_sc.axhline(ICE_THR, color="gray", lw=0.9, ls="--", alpha=0.7)
ax_sc.axvline(ICE_THR, color="gray", lw=0.9, ls="--", alpha=0.7)
ax_sc.set_xlim(0, 2.5); ax_sc.set_ylim(0, 2.0)
ax_sc.set_xlabel("CPR (Default)", fontsize=11)
ax_sc.set_ylabel("CPR (Research mu_c)", fontsize=11)
ax_sc.tick_params(labelsize=9)
ax_sc.spines[["top", "right"]].set_visible(False)
ax_sc.set_title("CPR Comparison (colored by DOP)", fontsize=11, fontweight="bold")
if len(cd_b) > 1:
    r = float(np.corrcoef(cd_b, cr_b)[0, 1])
    ax_sc.text(0.03, 0.97, f"r = {r:.3f}", transform=ax_sc.transAxes,
               fontsize=9, va="top", color="dimgray")

ax_res = fig.add_subplot(gs[3])
ax_res.set_facecolor("black")
im_r   = ax_res.imshow(resize_sq(cpr_res_map), cmap=CPR_CMAP, vmin=0, vmax=2.0,
                        aspect="auto", interpolation="nearest")
ax_res.set_xticks([]); ax_res.set_yticks([])
ax_res.set_title("CPR (Research mu_c)", fontsize=11, fontweight="bold", pad=5)
fig.colorbar(im_r, ax=ax_res, fraction=0.046, pad=0.02).ax.tick_params(labelsize=8)

fig.suptitle(
    "Faustini Inner-Crater — CPR Comparison  |  Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_compare.png")


# ===========================================================================
# FIGURE 6 — zoom_halpha.png
# ===========================================================================
print("[6] zoom_halpha.png ...")
H_v    = H_map[valid].ravel()
A_v    = A_map[valid].ravel()
al_v   = alpha_map[valid].ravel()
fin_ha = np.isfinite(H_v) & np.isfinite(al_v)
Hv     = H_v[fin_ha]; alv = al_v[fin_ha]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor="white",
                         gridspec_kw={"wspace": 0.35})

ax1 = axes[0]
ax1.set_facecolor("black")
hb = ax1.hexbin(Hv, alv, gridsize=80, cmap="hot_r", mincnt=1,
                bins="log", extent=(0, 1, 0, 90))
fig.colorbar(hb, ax=ax1, label="log10(count)")
ax1.axvline(0.5,  color="white", lw=0.9, ls="--", alpha=0.85)
ax1.axvline(0.9,  color="white", lw=0.9, ls="--", alpha=0.85)
ax1.axhline(42.5, color="white", lw=0.9, ls=":",  alpha=0.85)
ax1.axhline(47.5, color="white", lw=0.9, ls=":",  alpha=0.85)

zone_labels = {
    (0.15, 70): "Z1: Surface\nscatter",
    (0.70, 75): "Z2: Dipole",
    (0.95, 75): "Z3: Random\n/ice",
    (0.15, 22): "Z4: Low H\nvegetation",
    (0.70, 22): "Z5: Mixed",
    (0.95, 22): "Z6: High H",
    (0.25,  8): "Z7: Bragg",
    (0.70,  8): "Z8: Mix",
    (0.95,  8): "Z9: Noise",
}
for (hx, ax_), lab in zone_labels.items():
    ax1.text(hx, ax_, lab, fontsize=5.5, ha="center", color="yellow",
             va="center", alpha=0.9,
             bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"))

ax1.set_xlim(0, 1); ax1.set_ylim(0, 90)
ax1.set_xlabel("Entropy H",     fontsize=11)
ax1.set_ylabel("Alpha [deg]",   fontsize=11)
ax1.set_title("H-alpha Plane (log density)", fontsize=11, fontweight="bold")
ax1.tick_params(labelsize=9)

ax2 = axes[1]
fin_all = np.isfinite(H_v) & np.isfinite(al_v) & np.isfinite(A_v)
Hv2 = H_v[fin_all]; alv2 = al_v[fin_all]; Av2 = A_v[fin_all]
if len(Hv2) > MAX_SCATTER:
    idx2 = np.random.default_rng(77).choice(len(Hv2), MAX_SCATTER, replace=False)
    Hv2, alv2, Av2 = Hv2[idx2], alv2[idx2], Av2[idx2]
sc2 = ax2.scatter(Hv2, alv2, c=Av2, cmap="viridis", vmin=0, vmax=1,
                  s=2, alpha=0.4, linewidths=0, rasterized=True)
cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.02)
cb2.set_label("Anisotropy A", fontsize=10); cb2.ax.tick_params(labelsize=8)
ax2.axvline(0.5,  color="gray", lw=0.8, ls="--", alpha=0.7)
ax2.axvline(0.9,  color="gray", lw=0.8, ls="--", alpha=0.7)
ax2.axhline(42.5, color="gray", lw=0.8, ls=":",  alpha=0.7)
ax2.axhline(47.5, color="gray", lw=0.8, ls=":",  alpha=0.7)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 90)
ax2.set_xlabel("Entropy H",   fontsize=11)
ax2.set_ylabel("Alpha [deg]", fontsize=11)
ax2.set_title("H-alpha plane (colored by Anisotropy A)", fontsize=11, fontweight="bold")
ax2.tick_params(labelsize=9)
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Faustini Inner-Crater — Cloude-Pottier H-alpha Decomposition\n"
    "Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_halpha.png")


# ===========================================================================
# FIGURE 7 — zoom_polsar.png  (PolSAR feature maps)
# ===========================================================================
print("[7] zoom_polsar.png ...")
POLSAR_MAPS = [
    (span_db,   "Span [dB]",    "gray",    None,  None,  "gray"),
    (copol_db,  "HH/VV [dB]",  "RdBu_r",  None,  None,  "gray"),
    (xpol_frac, "HV fraction",  "hot_r",   0.0,   0.4,   "gray"),
    (H_map,     "Entropy H",    "viridis", 0.0,   1.0,   "gray"),
    (A_map,     "Anisotropy A", "viridis", 0.0,   1.0,   "gray"),
    (alpha_map, "Alpha [deg]",  "coolwarm",0.0,  90.0,   "gray"),
]
fig, axes = plt.subplots(2, 3, figsize=(14, 10), facecolor="white",
                         gridspec_kw={"hspace": 0.08, "wspace": 0.05})
for ax, (arr, title, cmap, vmin, vmax, bad) in zip(axes.ravel(), POLSAR_MAPS):
    im = map_panel(ax, arr, title, cmap, vmin, vmax, bad)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
fig.suptitle(
    "Faustini Inner-Crater — PolSAR Decomposition Maps\n"
    "Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.005)
add_footnote(fig)
save_fig(fig, "zoom_polsar.png")

# ── zoom_polsar_hist.png ──────────────────────────────────────────────────
print("[8] zoom_polsar_hist.png ...")
POLSAR_HISTS = [
    (span_db[valid],   "Span [dB]",    "#555555"),
    (copol_db[valid],  "HH/VV [dB]",  "#993311"),
    (xpol_frac[valid], "HV fraction", "#cc7700"),
    (H_map[valid],     "Entropy H",   "#664488"),
    (A_map[valid],     "Anisotropy A","#885522"),
    (alpha_map[valid], "Alpha [deg]", "#336688"),
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white",
                         gridspec_kw={"hspace": 0.42, "wspace": 0.32})
for ax, (data, label, color) in zip(axes.ravel(), POLSAR_HISTS):
    finite = data[np.isfinite(data)]
    p2, p98 = np.percentile(finite, 2), np.percentile(finite, 98)
    ax.hist(finite, bins=100, color=color, alpha=0.75, edgecolor="none",
            range=(p2, p98), density=True)
    med = float(np.median(finite))
    ax.axvline(med, color="black", lw=1.2, ls="--", label=f"Med {med:.3f}")
    ax.set_xlabel(label, fontsize=10); ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{label} Distribution", fontsize=10, fontweight="bold")
fig.suptitle(
    "Faustini Inner-Crater — PolSAR Feature Histograms\n"
    "Chandrayaan-2 DFSAR  |  2021-05-06",
    fontsize=12, fontweight="bold", y=1.02)
add_footnote(fig)
save_fig(fig, "zoom_polsar_hist.png")

print("\n" + "="*60)
print(f"All outputs saved to: {OUT_DIR}")
print("="*60)
