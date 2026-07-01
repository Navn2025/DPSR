"""
visualizer.py
=============
STEP 5 + STEP 10 of the Diviner integration pipeline.

STEP  5 — Quick-look preview PNGs for the three Diviner layers
           (Tmean, ZIT, Pump) using a 2–98 percentile stretch.
           Saved to outputs/previews/.

STEP 10 — Publication-quality maps for all nine feature bands and the
           Ice Confidence layer; per-feature histograms; a Pearson
           correlation matrix heat-map; five physically motivated
           scatter plots.
           Saved to outputs/diviner/.

No existing file is ever overwritten; files that already exist are
silently skipped so the pipeline is safely re-entrant.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for any environment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = logging.getLogger("diviner_pipeline.visualizer")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _valid_pixels(arr: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Return finite, non-nodata values as a flat 1-D float64 array."""
    return arr[(arr != nodata) & np.isfinite(arr)].ravel().astype(np.float64)


def _pct_stretch(
    arr: np.ndarray,
    lo:  int = 2,
    hi:  int = 98,
    nodata: float = -9999.0,
) -> Tuple[float, float]:
    """Return (vmin, vmax) from percentile stretch of valid pixels."""
    v = _valid_pixels(arr, nodata)
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _display_arr(arr: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Return float64 copy with nodata/NaN/Inf replaced by np.nan."""
    out = arr.astype(np.float64)
    out[(out == nodata) | ~np.isfinite(out)] = np.nan
    return out


_TARGET_MAX_PX = 2048   # maximum pixels per dimension for any rendered image


def _save_map(
    arr:       np.ndarray,
    out_path:  Path,
    title:     str,
    cbar_label: str,
    cmap:      str,
    nodata:    float = -9999.0,
    lo:        int   = 2,
    hi:        int   = 98,
    dpi:       int   = 150,
) -> None:
    """
    Render a single 2-D array as a colour-mapped image and save to disk.
    Silently skips if *out_path* already exists.

    Arrays larger than _TARGET_MAX_PX in either dimension are uniformly
    subsampled before rendering so matplotlib never works on >4M pixels.
    Statistics (vmin/vmax) are computed from the full array first.
    """
    if out_path.exists():
        log.info(f"  [skip] {out_path.name} already exists.")
        return

    # Compute stretch from the full array before downsampling
    vmin, vmax = _pct_stretch(arr, lo, hi, nodata)
    if vmin == vmax:
        vmax = vmin + 1.0   # avoid a flat colour map for constant arrays

    # Uniform spatial downsampling so matplotlib never processes >4M pixels
    ds = max(1, max(arr.shape[0], arr.shape[1]) // _TARGET_MAX_PX)
    disp_arr = arr[::ds, ::ds]
    disp = _display_arr(disp_arr, nodata)

    h, w = disp.shape
    fig_w = max(6.0, min(14.0, w / 250.0 * 6.0))
    fig_h = max(4.0, min(20.0, fig_w * (h / max(w, 1))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ds_note = f" (1:{ds} subsample)" if ds > 1 else ""
    ax.set_title(f"{title}{ds_note}", fontsize=9, pad=6)
    ax.set_xlabel("X (pixels)", fontsize=7)
    ax.set_ylabel("Y (pixels)", fontsize=7)
    ax.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved map: {out_path.name}  ({w}×{h} px, ds=1:{ds})")


# ---------------------------------------------------------------------------
# STEP 5 — Quick-look previews for the three Diviner bands
# ---------------------------------------------------------------------------

def save_diviner_previews(
    tmean:   np.ndarray,
    zit:     np.ndarray,
    pump:    np.ndarray,
    out_dir: Path,
    nodata:  float = -9999.0,
) -> None:
    """Save three quick-look PNG previews (Tmean, ZIT, Pump)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_map(
        tmean,
        out_dir / "Tmean_preview.png",
        title="Diviner — Mean Surface Temperature (Tmean)\n2–98 percentile stretch",
        cbar_label="Temperature (K)",
        cmap="RdYlBu_r",
        nodata=nodata,
    )
    _save_map(
        zit,
        out_dir / "ZIT_preview.png",
        title="Diviner — Zero-Incidence Temperature (ZIT)\n2–98 percentile stretch",
        cbar_label="Temperature (K)",
        cmap="plasma",
        nodata=nodata,
    )
    _save_map(
        pump,
        out_dir / "Pump_preview.png",
        title="Diviner — Volatile Pump Parameter\n2–98 percentile stretch",
        cbar_label="Pump (dimensionless)",
        cmap="YlOrRd",
        nodata=nodata,
    )


# ---------------------------------------------------------------------------
# STEP 10 — Publication-quality feature maps
# ---------------------------------------------------------------------------

_UNITS: Dict[str, str] = {
    "DEM":           "Elevation (m)",
    "Slope":         "Slope (°)",
    "PSR":           "PSR mask (0 = lit, 1 = shadow)",
    "DPSR":          "DPSR mask (0 = lit, 1 = doubly shadowed)",
    "CPR":           "Circular Polarisation Ratio",
    "DOP":           "Degree of Polarisation (0–1)",
    "Tmean":         "Mean Temperature (K)",
    "ZIT":           "Zero-Incidence Temperature (K)",
    "Pump":          "Pump parameter",
    "IceConfidence": "Ice Confidence Score (0–1)",
}


def save_feature_maps(
    bands:   Dict[str, np.ndarray],
    cmaps:   Dict[str, str],
    out_dir: Path,
    nodata:  float = -9999.0,
) -> None:
    """Save one publication-quality PNG per feature band."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in bands.items():
        cmap = cmaps.get(name, "viridis")
        unit = _UNITS.get(name, name)
        _save_map(
            arr,
            out_dir / f"{name}_map.png",
            title=f"{name} — Lunar South Pole (> 80°S)\n2–98 percentile stretch",
            cbar_label=unit,
            cmap=cmap,
            nodata=nodata,
        )


# ---------------------------------------------------------------------------
# STEP 10 — Per-feature histograms
# ---------------------------------------------------------------------------

def save_feature_histograms(
    bands:   Dict[str, np.ndarray],
    out_dir: Path,
    nodata:  float = -9999.0,
    n_bins:  int   = 128,
) -> None:
    """Save one histogram PNG per feature band."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _MAX_HIST_PX = 5_000_000  # cap for histogram / stats computation

    for name, arr in bands.items():
        out_path = out_dir / f"{name}_histogram.png"
        if out_path.exists():
            log.info(f"  [skip] {out_path.name} already exists.")
            continue

        v = _valid_pixels(arr, nodata)
        if v.size == 0:
            log.warning(f"  {name}: no valid pixels — histogram skipped.")
            continue

        # Subsample for large rasters to avoid OOM in median/mean computation
        if v.size > _MAX_HIST_PX:
            v = np.random.default_rng(seed=0).choice(v, _MAX_HIST_PX, replace=False)

        med  = float(np.median(v))
        mean = float(v.mean())

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(v, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(med,  color="red",    lw=1.5, label=f"Median = {med:.4e}")
        ax.axvline(mean, color="orange", lw=1.2, ls="--", label=f"Mean = {mean:.4e}")

        ax.set_xlabel(_UNITS.get(name, name), fontsize=10)
        ax.set_ylabel("Pixel count", fontsize=10)
        ax.set_title(
            f"{name} — Value Histogram  ({v.size:,} valid pixels)", fontsize=10
        )
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved histogram: {out_path.name}")


# ---------------------------------------------------------------------------
# STEP 10 — Correlation matrix
# ---------------------------------------------------------------------------

def save_correlation_matrix(
    bands:   Dict[str, np.ndarray],
    out_dir: Path,
    nodata:  float = -9999.0,
) -> None:
    """
    Compute and save a Pearson correlation matrix heat-map.

    Only pixels that are valid in EVERY band contribute to the
    correlation so every pair shares the same population.
    """
    out_path = out_dir / "Correlation_Matrix.png"
    if out_path.exists():
        log.info(f"  [skip] {out_path.name} already exists.")
        return

    names = list(bands.keys())
    n     = len(names)
    arrs  = [bands[k].ravel().astype(np.float64) for k in names]
    valid = [np.isfinite(a) & (a != nodata) for a in arrs]

    # Pairwise Pearson r — each pair uses its own co-located valid mask
    # so that all-nodata bands (e.g. DOP) only blank their own row/col.
    mat = np.full((n, n), np.nan)
    n_pairs: Dict[tuple, int] = {}
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            pair_mask = valid[i] & valid[j]
            k = int(pair_mask.sum())
            n_pairs[(i, j)] = k
            if k >= 10:
                xi = arrs[i][pair_mask]
                xj = arrs[j][pair_mask]
                if xi.std() > 0 and xj.std() > 0:
                    mat[i, j] = mat[j, i] = float(np.corrcoef(xi, xj)[0, 1])

    n_valid_bands = int(sum(v.any() for v in valid))
    if n_valid_bands < 2:
        log.warning("  Correlation matrix: fewer than 2 bands have valid pixels — skipped.")
        return

    # Mask NaN cells for imshow (greyed out)
    mat_plot = np.where(np.isnan(mat), 0.0, mat)
    nan_mask = np.isnan(mat)

    fig, ax = plt.subplots(figsize=(n * 1.0 + 1.5, n * 0.9 + 1.2))
    im = ax.imshow(mat_plot, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    if nan_mask.any():
        ax.imshow(np.where(nan_mask, 0.5, np.nan),
                  vmin=0, vmax=1, cmap="Greys", aspect="auto", alpha=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=5.5, color="black")
            else:
                colour = "white" if abs(mat[i, j]) > 0.65 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color=colour)

    ax.set_title(
        f"Feature Correlation Matrix — Pearson r (pairwise valid pixels)",
        fontsize=9,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# STEP 10 — Scatter plots
# ---------------------------------------------------------------------------

def save_scatter_plots(
    bands:    Dict[str, np.ndarray],
    out_dir:  Path,
    nodata:   float = -9999.0,
    max_pts:  int   = 60_000,
) -> None:
    """
    Save five scatter plots motivated by ice-detection physics.

    Pairs
    -----
    CPR   vs DOP   — high CPR + low DOP → volume / double-bounce ice signal
    CPR   vs Tmean — high CPR expected in cold PSR zones (correlation check)
    CPR   vs ZIT   — high CPR + low ZIT → cold-trapped ice hypothesis
    DOP   vs Tmean — low DOP expected in permanently shadowed cold regions
    Tmean vs Pump  — cold surfaces → more efficient volatile pumping
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("CPR",   "DOP",   "CPR vs DOP",              "steelblue"),
        ("CPR",   "Tmean", "CPR vs Mean Temperature",  "darkorange"),
        ("CPR",   "ZIT",   "CPR vs ZIT",               "purple"),
        ("DOP",   "Tmean", "DOP vs Mean Temperature",  "green"),
        ("Tmean", "Pump",  "Temperature vs Pump",      "crimson"),
    ]

    rng = np.random.default_rng(seed=42)   # reproducible subsampling

    for x_name, y_name, title, colour in pairs:
        out_path = out_dir / f"scatter_{x_name}_vs_{y_name}.png"
        if out_path.exists():
            log.info(f"  [skip] {out_path.name} already exists.")
            continue

        x_arr = bands.get(x_name)
        y_arr = bands.get(y_name)
        if x_arr is None or y_arr is None:
            log.warning(f"  {title}: band missing — scatter skipped.")
            continue

        x_flat = x_arr.ravel().astype(np.float64)
        y_flat = y_arr.ravel().astype(np.float64)

        valid = (
            np.isfinite(x_flat) & (x_flat != nodata) &
            np.isfinite(y_flat) & (y_flat != nodata)
        )
        x_v = x_flat[valid]
        y_v = y_flat[valid]

        if x_v.size == 0:
            log.warning(f"  {title}: no co-located valid pixels — skipped.")
            continue

        # Subsample for plotting speed if the scene is large
        if x_v.size > max_pts:
            idx = rng.choice(x_v.size, max_pts, replace=False)
            x_v = x_v[idx]
            y_v = y_v[idx]

        r = float(np.corrcoef(x_v, y_v)[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_v, y_v, s=0.9, alpha=0.25, c=colour, rasterized=True)
        ax.set_xlabel(_UNITS.get(x_name, x_name), fontsize=10)
        ax.set_ylabel(_UNITS.get(y_name, y_name), fontsize=10)
        ax.set_title(
            f"{title}\nPearson r = {r:.3f}   n = {x_v.size:,}",
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved scatter: {out_path.name}")


# ---------------------------------------------------------------------------
# Ice Confidence PNG (standalone helper called from STEP 9)
# ---------------------------------------------------------------------------

def save_ice_confidence_png(
    ice_map:  np.ndarray,
    out_path: Path,
    nodata:   float = -9999.0,
) -> None:
    """Save the Ice Confidence Map as a PNG with a YlGnBu colour map."""
    _save_map(
        ice_map,
        out_path,
        title=(
            "Lunar South Pole — Physics-Based Ice Confidence Score\n"
            "(0 = no ice evidence  →  1 = strong multi-indicator evidence)"
        ),
        cbar_label="Ice Confidence Score (0–1)",
        cmap="YlGnBu",
        nodata=nodata,
        lo=0,
        hi=99,
    )
