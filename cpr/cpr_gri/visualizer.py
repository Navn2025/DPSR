"""
visualizer.py
=============
STEP 7 -- Publication-quality preview images: HH/HV/VH/VV backscatter
(dB) and CPR, each with a 2nd-98th percentile stretch, plus a CPR
histogram.

Also renders the STEP 9 comparison figures (scatter plot, difference
map, histogram overlap) against the official DFSAR CPR mosaic.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = logging.getLogger("cpr_gri_pipeline.visualizer")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_clip(arr: np.ndarray, lo: int, hi: int) -> Tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def _save_image(arr, out_path, title, unit, cmap, lo=2, hi=98) -> None:
    disp = arr.copy().astype(np.float64)
    vmin, vmax = _pct_clip(disp, lo, hi)

    h, w = disp.shape
    fig_w = max(3.0, w / 207 * 4)
    fig_h = min(max(10.0, fig_w * (h / w) * 0.15), 20.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("Range (samples)", fontsize=7)
    ax.set_ylabel("Azimuth (lines)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path.name}  [{out_path}]")


# ---------------------------------------------------------------------------
# STEP 7 -- Power / CPR previews
# ---------------------------------------------------------------------------

def save_power_previews(powers: dict, preview_dir: Path) -> None:
    """powers: dict mapping "HH"/"HV"/"VH"/"VV" -> linear power array."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for pol, power in powers.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            db = 10.0 * np.log10(np.where(power > 0, power, np.nan))
        _save_image(
            db, preview_dir / f"{pol}.png",
            title=f"{pol} Backscatter (calibrated, multilooked)\n(2-98 pct stretch)",
            unit="sigma0 (dB)", cmap="gray",
        )


def save_cpr_preview(
    cpr: np.ndarray, preview_dir: Path, nodata: float,
    title: str = "CPR_GRI = 1 + 4*P_XP / (P_HH+P_VV)\n(reflection-symmetry approximation, 2-98 pct stretch)",
) -> None:
    disp = np.where(cpr == nodata, np.nan, cpr)
    _save_image(
        disp, preview_dir / "CPR.png",
        title=title,
        unit="CPR (dimensionless)", cmap="viridis",
    )


def save_cpr_histogram(cpr: np.ndarray, preview_dir: Path, nodata: float, n_bins: int = 300) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    valid = cpr[(cpr != nodata) & np.isfinite(cpr)]
    if valid.size == 0:
        log.warning("No valid CPR pixels -- histogram skipped.")
        return
    p2, p98 = np.percentile(valid, [2, 98])
    clipped = valid[(valid >= p2) & (valid <= p98)]
    med, mean_val = float(np.median(valid)), float(valid.mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(clipped, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med, color="red", lw=1.5, label=f"Median = {med:.3f}")
    ax.axvline(mean_val, color="orange", lw=1.2, ls="--", label=f"Mean = {mean_val:.3f}")
    ax.axvline(1.0, color="lime", lw=1.0, ls=":", label="CPR = 1.0 (reference)")
    ax.set_xlabel("CPR", fontsize=11)
    ax.set_ylabel("Pixel count", fontsize=11)
    ax.set_title(f"CPR_GRI Histogram (2nd-98th pct: {p2:.2f}-{p98:.2f})\nValid pixels: {valid.size:,}", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = preview_dir / "Histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved histogram: {out.name}")


# ---------------------------------------------------------------------------
# STEP 9 -- Comparison figures
# ---------------------------------------------------------------------------

def save_scatter(ours: np.ndarray, ref: np.ndarray, out_dir: Path, stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(ref, ours, gridsize=80, cmap="inferno", bins="log", mincnt=1)
    lim = float(max(np.percentile(ref, 99.5), np.percentile(ours, 99.5)))
    ax.plot([0, lim], [0, lim], "c--", lw=1.2, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Official DFSAR CPR mosaic")
    ax.set_ylabel("CPR_GRI (this pipeline)")
    ax.set_title(
        f"CPR_GRI vs Official Mosaic (n={stats['n']:,})\n"
        f"Pearson r={stats['pearson_r']:.3f}  Spearman rho={stats['spearman_r']:.3f}  "
        f"RMSE={stats['rmse']:.3f}  Bias={stats['bias']:+.3f}",
        fontsize=10,
    )
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "Scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out.name}")


def save_difference_map(diff_2d: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_image(
        diff_2d, out_dir / "Difference_map.png",
        title="CPR_GRI - Official CPR mosaic (resampled to GRI grid)",
        unit="CPR difference", cmap="RdBu_r", lo=1, hi=99,
    )


def save_histogram_overlap(ours: np.ndarray, ref: np.ndarray, out_dir: Path, stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hi = float(np.percentile(np.concatenate([ours, ref]), 99))
    bins = np.linspace(0.0, max(hi, 1e-6), 100)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ours, bins=bins, alpha=0.55, label="CPR_GRI (ours)", color="steelblue", density=True)
    ax.hist(ref, bins=bins, alpha=0.55, label="Official mosaic", color="darkorange", density=True)
    ax.set_xlabel("CPR")
    ax.set_ylabel("Density")
    ax.set_title(f"Histogram overlap = {stats['hist_overlap']:.3f}")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "Histogram_overlap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out.name}")
