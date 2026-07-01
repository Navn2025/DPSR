"""
plots.py
========
Publication-quality figures for the CPR validation.
All figures are saved to the outputs/figures/ directory.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

log = logging.getLogger("validation.plots")

DPI = 150
CMAP_CPR  = "viridis"
CMAP_DIFF = "RdBu_r"
CMAP_ABS  = "hot_r"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_lim(arr: np.ndarray, lo: float = 2, hi: float = 98):
    v = arr[np.isfinite(arr) & (arr > 0)]
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _downsample_2d(arr: np.ndarray, max_px: int = 1024) -> np.ndarray:
    """Block-mean downsample to at most max_px × max_px."""
    h, w = arr.shape
    step = max(1, max(h, w) // max_px)
    if step == 1:
        return arr
    h2 = (h // step) * step
    w2 = (w // step) * step
    blk = arr[:h2, :w2].reshape(h2 // step, step, w2 // step, step)
    with np.errstate(all="ignore"):
        return np.nanmean(blk, axis=(1, 3))


def _save(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 1: Side-by-side CPR maps
# ---------------------------------------------------------------------------

def plot_side_by_side(
    calc:    np.ndarray,
    offic:   np.ndarray,
    mask:    np.ndarray,
    out_dir: Path,
):
    """Two-panel: calculated (left) vs official (right) CPR maps."""
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_d  = _downsample_2d(np.where(mask, calc,  np.nan))
    offic_d = _downsample_2d(np.where(mask, offic, np.nan))

    vmin, vmax = 0.0, 2.0   # physical CPR range for display

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, arr, title in zip(
        axes,
        [calc_d, offic_d],
        ["Calculated CPR\n(circular basis, this study)",
         "Official DFSAR CPR\n(Putrevu et al. 2023)"],
    ):
        im = ax.imshow(arr, cmap=CMAP_CPR, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            "CPR (dimensionless)", fontsize=8
        )
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("Easting [pixels]", fontsize=8)
        ax.set_ylabel("Northing [pixels]", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("CPR Map Comparison — Moon South Pole Stereographic", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir / "validation_maps.png")


# ---------------------------------------------------------------------------
# Figure 2: Scatter plot with 1:1 line
# ---------------------------------------------------------------------------

def plot_scatter(
    calc_1d: np.ndarray,
    offic_1d: np.ndarray,
    metrics:  Dict,
    out_dir:  Path,
    max_pts:  int = 200_000,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(calc_1d) > max_pts:
        idx = np.random.default_rng(0).choice(len(calc_1d), max_pts, replace=False)
        a, b = calc_1d[idx], offic_1d[idx]
    else:
        a, b = calc_1d, offic_1d

    lim = min(float(max(a.max(), b.max())) * 1.05, 3.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(b, a, s=0.4, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot([0, lim], [0, lim], "r-", lw=1.5, label="1:1 line", zorder=5)

    r  = metrics.get("pearson_r", float("nan"))
    r2 = metrics.get("r2", float("nan"))
    rm = metrics.get("rmse", float("nan"))
    bi = metrics.get("bias", float("nan"))

    ax.set_xlabel("Official CPR (Putrevu et al. 2023)", fontsize=12)
    ax.set_ylabel("Calculated CPR (this study)", fontsize=12)
    ax.set_title(
        f"CPR Scatter Plot\n"
        f"r = {r:.4f}   R² = {r2:.4f}   RMSE = {rm:.4f}   Bias = {bi:+.4f}\n"
        f"n = {len(calc_1d):,} pixels",
        fontsize=10,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "validation_scatter.png")


# ---------------------------------------------------------------------------
# Figure 3: Difference map
# ---------------------------------------------------------------------------

def plot_difference_map(
    diff:    np.ndarray,
    mask:    np.ndarray,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    d_ds = _downsample_2d(np.where(mask, diff, np.nan))
    abs_max = float(np.nanpercentile(np.abs(d_ds[np.isfinite(d_ds)]), 95))
    abs_max = max(abs_max, 0.05)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(d_ds, cmap=CMAP_DIFF, vmin=-abs_max, vmax=abs_max,
                   aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Calculated - Official CPR", fontsize=9
    )
    ax.set_title(
        f"CPR Difference Map  (Calculated - Official)\n"
        f"Symmetric 5-95th pct range: ±{abs_max:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Easting [pixels]", fontsize=8)
    ax.set_ylabel("Northing [pixels]", fontsize=8)
    _save(fig, out_dir / "difference_map.png")


# ---------------------------------------------------------------------------
# Figure 4: Absolute difference map
# ---------------------------------------------------------------------------

def plot_absolute_difference(
    abs_diff: np.ndarray,
    mask:     np.ndarray,
    out_dir:  Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    d_ds = _downsample_2d(np.where(mask, abs_diff, np.nan))
    vmax = float(np.nanpercentile(d_ds[np.isfinite(d_ds)], 95))
    vmax = max(vmax, 0.05)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(d_ds, cmap=CMAP_ABS, vmin=0, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "|Calculated - Official| CPR", fontsize=9
    )
    ax.set_title(
        f"Absolute CPR Difference\n(95th pct = {vmax:.3f})",
        fontsize=10,
    )
    ax.set_xlabel("Easting [pixels]", fontsize=8)
    ax.set_ylabel("Northing [pixels]", fontsize=8)
    _save(fig, out_dir / "absolute_difference.png")


# ---------------------------------------------------------------------------
# Figure 5: Histogram comparison
# ---------------------------------------------------------------------------

def plot_histogram_comparison(
    calc_1d:  np.ndarray,
    offic_1d: np.ndarray,
    metrics:  Dict,
    out_dir:  Path,
    bins: int = 200,
    xlim: Tuple[float, float] = (0.0, 2.5),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(calc_1d,  bins=bins, range=xlim, density=True,
            alpha=0.6, color="steelblue", label="Calculated CPR (this study)")
    ax.hist(offic_1d, bins=bins, range=xlim, density=True,
            alpha=0.5, color="tomato",    label="Official CPR (Putrevu 2023)")

    ax.axvline(1.0, color="green", lw=1.0, ls="--", label="CPR = 1.0")
    ax.axvline(float(np.median(calc_1d)),  color="steelblue", lw=1.5,
               label=f"Calc median = {np.median(calc_1d):.3f}")
    ax.axvline(float(np.median(offic_1d)), color="tomato",    lw=1.5,
               label=f"Official median = {np.median(offic_1d):.3f}")

    hi = metrics.get("hist_intersection", float("nan"))
    mi = metrics.get("mutual_information", float("nan"))

    ax.set_xlabel("CPR (dimensionless)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(
        f"CPR Histogram Comparison  (overlap pixels, n={len(calc_1d):,})\n"
        f"Histogram intersection = {hi:.4f}   Mutual information = {mi:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "histogram_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6: Q-Q plot
# ---------------------------------------------------------------------------

def plot_qq(
    calc_1d:  np.ndarray,
    offic_1d: np.ndarray,
    out_dir:  Path,
    n_quantiles: int = 1000,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pcts = np.linspace(0.5, 99.5, n_quantiles)
    q_c = np.percentile(calc_1d,  pcts)
    q_o = np.percentile(offic_1d, pcts)

    lim = min(float(max(q_c.max(), q_o.max())) * 1.05, 3.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_o, q_c, s=3, alpha=0.6, color="steelblue", rasterized=True)
    ax.plot([0, lim], [0, lim], "r-", lw=1.5, label="1:1 line")

    ax.set_xlabel("Official CPR quantiles", fontsize=12)
    ax.set_ylabel("Calculated CPR quantiles", fontsize=12)
    ax.set_title(
        f"Q-Q Plot: Calculated vs Official CPR\n"
        f"({n_quantiles} quantiles from 0.5th to 99.5th percentile)",
        fontsize=10,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "qq_plot.png")


# ---------------------------------------------------------------------------
# Figure 7: Box plot
# ---------------------------------------------------------------------------

def plot_boxplot(
    calc_1d:  np.ndarray,
    offic_1d: np.ndarray,
    out_dir:  Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    bp = ax.boxplot(
        [calc_1d[calc_1d <= 3.0], offic_1d[offic_1d <= 3.0]],
        patch_artist=True,
        notch=True,
        vert=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Calculated CPR\n(this study)", "Official CPR\n(Putrevu 2023)"])
    for patch, color in zip(bp["boxes"], ["steelblue", "tomato"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(1.0, color="green", ls="--", lw=1.0, label="CPR = 1.0")
    ax.set_ylabel("CPR (dimensionless)", fontsize=12)
    ax.set_title(
        f"CPR Distribution Box Plot\n"
        f"(values ≤ 3.0 shown;  n={len(calc_1d):,} overlap pixels)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / "boxplot.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(
    calc_2d:  np.ndarray,
    offic_2d: np.ndarray,
    metrics:  Dict,
    out_dir:  Path,
):
    """Call all individual plot functions."""
    mask     = metrics["overlap_mask"]
    calc_1d  = metrics["calc_1d"]
    offic_1d = metrics["offic_1d"]
    diff     = metrics["diff_arr"]
    abs_diff = metrics["abs_diff_arr"]

    log.info("  Generating side-by-side CPR maps ...")
    plot_side_by_side(calc_2d, offic_2d, mask, out_dir)

    log.info("  Generating scatter plot ...")
    plot_scatter(calc_1d, offic_1d, metrics, out_dir)

    log.info("  Generating difference map ...")
    plot_difference_map(diff, mask, out_dir)

    log.info("  Generating absolute difference map ...")
    plot_absolute_difference(abs_diff, mask, out_dir)

    log.info("  Generating histogram comparison ...")
    plot_histogram_comparison(calc_1d, offic_1d, metrics, out_dir)

    log.info("  Generating Q-Q plot ...")
    plot_qq(calc_1d, offic_1d, out_dir)

    log.info("  Generating box plot ...")
    plot_boxplot(calc_1d, offic_1d, out_dir)
