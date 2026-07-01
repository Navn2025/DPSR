"""
ice_zoom.py
-----------
Zoomed CPR panels for ice-candidate clusters in the Faustini scene,
styled after Fig. 3 of Putrevu et al. (2023).

Each patch is ~6.1 x 6.1 km in ground distance (PATCH_AZ = 649 lines @ 9.4 m,
SWATH = 244 px @ 25 m).  Patches are block-averaged to 244x244 for a
visually-square display, matching the paper style.

Usage:
    python cpr/ice_zoom.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter, zoom
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR  = cfg.PREV_DIR / "ice_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ice_candidate_panels.png"

# ---------------------------------------------------------------------------
# Scene / patch parameters
# ---------------------------------------------------------------------------
ICE_THR        = 1.0     # CPR > this -> ice candidate
N_PANELS       = 9       # 3 x 3 grid
RG_PX          = 25.0    # m per range pixel
AZ_PX          = 9.4     # m per azimuth pixel
SWATH_W        = 244     # range pixels

# Ground-square patch: same km in az as in rg
PATCH_AZ       = int(round(SWATH_W * RG_PX / AZ_PX))   # 649 lines ~ 6.1 km
DISPLAY_SZ     = SWATH_W                                 # pixels after resize (244x244)
MIN_DIST       = PATCH_AZ + 50
DENSITY_SMOOTH = 150

# ---------------------------------------------------------------------------
# Colormap: black -> dark green -> bright green -> blue -> magenta -> white
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
# Data helpers
# ---------------------------------------------------------------------------

def load_cpr():
    path = cfg.CPR_DIR / cfg.CPR_OUTPUT_NAME
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr[arr == cfg.NODATA] = np.nan
    return arr


def find_centres(cpr):
    """Azimuth-line indices of the N densest ice-candidate clusters."""
    ice  = (cpr > ICE_THR) & np.isfinite(cpr)
    dens = uniform_filter(ice.astype(np.float32),
                          size=(DENSITY_SMOOTH, 1)).mean(axis=1)
    peaks, _ = find_peaks(dens, height=0.003, distance=MIN_DIST)

    if len(peaks) == 0:
        half = PATCH_AZ // 2
        return list(np.linspace(half, cpr.shape[0] - half, N_PANELS, dtype=int))

    order = np.argsort(-dens[peaks])
    return sorted(peaks[order[:N_PANELS]].tolist())


def extract_patch(cpr, az):
    half = PATCH_AZ // 2
    r0   = max(0, az - half)
    r1   = min(cpr.shape[0], r0 + PATCH_AZ)
    return cpr[r0:r1, :]


def resize_square(patch, target=DISPLAY_SZ):
    """Block-average (649,244) -> (244,244) for square display."""
    h, w = patch.shape
    if h == target and w == target:
        return patch
    nan_mask = np.isnan(patch)
    fill     = float(np.nanmean(patch)) if np.any(~nan_mask) else 0.0
    filled   = np.where(nan_mask, fill, patch)
    scale_h  = target / h
    scale_w  = target / w
    resized  = zoom(filled,   (scale_h, scale_w), order=1)
    nan_res  = zoom(nan_mask.astype(np.float32), (scale_h, scale_w), order=1)
    resized[nan_res > 0.4] = np.nan
    return resized.astype(np.float32)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(patches, centres):
    ncols = 3
    nrows = int(np.ceil(len(patches) / ncols))

    # GridSpec: each image col is wide, each colorbar col is narrow
    # pattern: [img, cb, gap, img, cb, gap, img, cb]
    col_widths = []
    for _ in range(ncols):
        col_widths += [1.0, 0.06, 0.04]   # image, colorbar, gap
    col_widths = col_widths[:-1]           # drop trailing gap

    fig = plt.figure(
        figsize=(5.2 * ncols, 5.0 * nrows + 0.5),
        facecolor="black",
    )
    gs = gridspec.GridSpec(
        nrows, len(col_widths),
        figure=fig,
        width_ratios=col_widths,
        hspace=0.06,
        wspace=0.0,
        top=0.95,
        bottom=0.03,
        left=0.01,
        right=0.99,
    )

    for idx, (patch, az) in enumerate(zip(patches, centres)):
        row = idx // ncols
        col = idx % ncols
        img_col = col * 3          # image column index in gs
        cb_col  = img_col + 1     # colorbar column index

        ax  = fig.add_subplot(gs[row, img_col])
        cax = fig.add_subplot(gs[row, cb_col])

        # Resize to square for display
        disp = resize_square(patch)

        p99  = float(np.nanpercentile(disp, 99.5))
        vmax = max(round(p99 + 0.05, 2), 1.10)

        im = ax.imshow(
            disp,
            cmap=CMAP,
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_facecolor("black")
        ax.axis("off")

        # --- Colorbar ---
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cax.set_facecolor("black")

        # Ticks: top = vmax, middle = 1.0, bottom = 0
        cb.set_ticks([0.0, ICE_THR, vmax])
        cb.set_ticklabels([" 0", " 1.0", f" {vmax:.2f}"],
                          color="white", fontsize=7)
        cb.outline.set_edgecolor("white")
        plt.setp(cb.ax.yaxis.get_ticklines(), color="white")
        cb.ax.tick_params(direction="in", color="white", width=0.8)
        cb.ax.set_facecolor("black")

        # Ice-threshold dashed line on colorbar
        cb.ax.axhline(ICE_THR / vmax, color="white", lw=0.8,
                      ls="--", xmin=0.1, xmax=0.9)

        # "CPR" label at bottom of colorbar
        cax.text(0.5, -0.04, "CPR", transform=cax.transAxes,
                 color="white", fontsize=7, ha="center", va="top")

        # --- Panel annotations ---
        ax.text(0.03, 0.97, f"R{idx+1}",
                transform=ax.transAxes,
                color="white", fontsize=11, fontweight="bold", va="top",
                bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))

        ice_frac = float(np.nanmean(patch > ICE_THR)) * 100.0
        az_km    = az * AZ_PX / 1000.0
        ax.text(0.97, 0.03,
                f"CPR>1: {ice_frac:.1f}%\naz={az_km:.0f} km",
                transform=ax.transAxes,
                color="yellow", fontsize=7, va="bottom", ha="right",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    fig.suptitle(
        "Faustini Crater  -  CPR Ice-Candidate Regions  (CPR > 1.0)\n"
        "Chandrayaan-2 DFSAR Full-Pol SLI  |  2021-05-06  |  L-band",
        color="white", fontsize=12, fontweight="bold", y=0.99,
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading CPR ...")
    cpr = load_cpr()
    H, W = cpr.shape
    n_ice = int((cpr > ICE_THR).sum())
    print(f"  Shape  : {W} range x {H} azimuth")
    print(f"  Ice px : {n_ice:,}  ({100*n_ice/(H*W):.2f}%)")
    print(f"  Patch  : {W} x {PATCH_AZ} px  ->  displayed as {DISPLAY_SZ}x{DISPLAY_SZ}")

    print("Finding ice-candidate cluster centres ...")
    centres = find_centres(cpr)
    print(f"  Centres (az line): {centres}")

    print("Extracting patches and plotting ...")
    patches = [extract_patch(cpr, az) for az in centres]
    fig     = make_figure(patches, centres)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor="black", pad_inches=0.1)
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
