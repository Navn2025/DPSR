"""
visualizer.py
=============
STEP 10 -- Publication-quality preview images: HH/HV/VH/VV power and DOP,
each with a 2nd-98th percentile stretch, plus a DOP histogram.

Mirrors the CPR pipeline's visualisation conventions (cpr/visualizer.py)
so CPR and DOP previews are visually consistent for side-by-side review.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = logging.getLogger("dop_pipeline.visualizer")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_clip(
    arr: np.ndarray,
    lo: int = 2,
    hi: int = 98,
    nodata: float = -9999.0,
) -> Tuple[float, float]:
    """Return (vmin, vmax) from percentile stretch of valid pixels."""
    valid = arr[(arr != nodata) & np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def _save_image(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    unit: str,
    cmap: str,
    lo: int = 2,
    hi: int = 98,
    nodata: float = -9999.0,
    downsample: int = 100,
) -> None:
    """Render a single 2-D array and save to disk."""
    disp = arr.copy().astype(np.float64)
    disp[(disp == nodata) | ~np.isfinite(disp)] = np.nan

    vmin, vmax = _pct_clip(arr, lo, hi, nodata)

    h, w = disp.shape
    fig_w = max(3.0, w / 244 * 4)
    fig_h = max(12.0, fig_w * (h / w) * 0.05)
    fig_h = min(fig_h, 20.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        disp,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("Range (samples)", fontsize=7)
    ax.set_ylabel(f"Azimuth (lines, 1:{downsample} downsampled)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path.name}  [{out_path}]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_power_previews(
    powers: dict,
    preview_dir: Path,
    downsample: int = 100,
    nodata: float = 0.0,
) -> None:
    """
    Save one PNG per polarisation power channel.

    Parameters
    ----------
    powers : dict mapping "HH"/"HV"/"VH"/"VV" -> multilooked power array
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    for pol, power in powers.items():
        ds = power[::downsample, :]
        _save_image(
            ds,
            preview_dir / f"{pol}.png",
            title=f"{pol} Power (multilooked)\n(1:{downsample} azimuth downsample, 2-98 pct stretch)",
            unit="Power (linear DN^2)",
            cmap="gray",
            nodata=nodata,
            downsample=downsample,
        )


def save_dop_preview(
    dop: np.ndarray,
    preview_dir: Path,
    downsample: int = 100,
    nodata: float = -9999.0,
) -> None:
    """Save downsampled DOP.png."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    ds = dop[::downsample, :]

    disp = ds.copy().astype(np.float64)
    disp[(disp == nodata) | ~np.isfinite(disp)] = np.nan
    valid = disp[np.isfinite(disp)]
    vmin, vmax = (float(np.percentile(valid, 2)), float(np.percentile(valid, 98))) if valid.size else (0.0, 1.0)

    h, w = disp.shape
    fig_w = max(3.0, w / 244 * 4)
    fig_h = max(12.0, fig_w * (h / w) * 0.05)
    fig_h = min(fig_h, 20.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(disp, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("DOP (dimensionless, 0-1)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(
        f"Degree of Polarization (DOP)\n"
        f"= sqrt(S1^2+S2^2+S3^2) / S0\n"
        f"(1:{downsample} azimuth downsample, 2-98 pct stretch)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Range (samples)", fontsize=7)
    ax.set_ylabel(f"Azimuth (lines, 1:{downsample} downsampled)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    out = preview_dir / "DOP.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out.name}  [{out}]")


def save_dop_histogram(
    dop: np.ndarray,
    preview_dir: Path,
    nodata: float = -9999.0,
    n_bins: int = 300,
) -> None:
    """Save a histogram of DOP values, marking median and mean."""
    preview_dir.mkdir(parents=True, exist_ok=True)

    valid = dop[(dop != nodata) & np.isfinite(dop)]
    if valid.size == 0:
        log.warning("No valid DOP pixels -- histogram skipped.")
        return

    med      = float(np.median(valid))
    mean_val = float(valid.mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(valid, bins=n_bins, range=(0.0, 1.0), color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med,      color="red",    lw=1.5, label=f"Median = {med:.3f}")
    ax.axvline(mean_val, color="orange", lw=1.2, ls="--", label=f"Mean = {mean_val:.3f}")

    ax.set_xlabel("DOP", fontsize=11)
    ax.set_ylabel("Pixel count", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"DOP Histogram\nTotal valid pixels: {valid.size:,}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    out = preview_dir / "Histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved histogram: {out.name}")
