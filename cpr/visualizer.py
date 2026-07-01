"""
visualizer.py
=============
Generate and save preview images of HH power, VV power, and CPR.

All images:
    - use aspect='auto' to handle the 244 W x 272 631 H swath correctly
    - apply 2nd-98th percentile stretching (not raw min/max)
    - are saved as PNG files at 150 dpi
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = logging.getLogger("cpr_pipeline.visualizer")


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
) -> None:
    """Render a single 2-D array and save to disk."""
    disp = arr.copy().astype(np.float64)
    disp[(disp == nodata) | ~np.isfinite(disp)] = np.nan

    vmin, vmax = _pct_clip(arr, lo, hi, nodata)

    h, w = disp.shape
    # Scale figure width so the narrow swath is always visible
    fig_w = max(3.0, w / 244 * 4)
    fig_h = max(12.0, fig_w * (h / w) * 0.05)   # cap height for very tall images
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
    ax.set_ylabel(f"Azimuth (lines, 1:{100} downsampled)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path.name}  [{out_path}]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_power_preview(
    power_a: np.ndarray,
    power_b: np.ndarray,
    preview_dir: Path,
    downsample: int = 100,
    nodata: float = -9999.0,
    labels: tuple = ("SC_power", "OC_power"),
) -> None:
    """
    Save downsampled power preview PNGs.

    Parameters
    ----------
    power_a, power_b : full-resolution power arrays
    preview_dir      : output directory
    downsample       : take every N-th line (azimuth axis)
    labels           : (label_a, label_b) used for filenames and titles
    """
    preview_dir.mkdir(parents=True, exist_ok=True)

    label_a, label_b = labels
    ds_a = power_a[::downsample, :]
    ds_b = power_b[::downsample, :]

    _save_image(
        ds_a,
        preview_dir / f"{label_a}.png",
        title=f"{label_a} (single-look power)\n(1:{downsample} azimuth downsample)",
        unit="Power (linear DN^2)",
        cmap="gray",
        nodata=nodata,
    )
    _save_image(
        ds_b,
        preview_dir / f"{label_b}.png",
        title=f"{label_b} (single-look power)\n(1:{downsample} azimuth downsample)",
        unit="Power (linear DN^2)",
        cmap="gray",
        nodata=nodata,
    )


def save_cpr_preview(
    cpr: np.ndarray,
    preview_dir: Path,
    downsample: int = 100,
    nodata: float = -9999.0,
    title: str = None,
) -> None:
    """Save downsampled CPR.png."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    cpr_ds = cpr[::downsample, :]

    if title is None:
        title = (
            f"Circular Polarization Ratio (CPR)\n"
            f"= mean(|S_HH-S_VV+2j*S_HV|^2) / mean(|S_HH+S_VV|^2)\n"
            f"(1:{downsample} azimuth downsample, 2-98 pct stretch)"
        )

    _save_image(
        cpr_ds,
        preview_dir / "CPR.png",
        title=title,
        unit="CPR (dimensionless)",
        cmap="viridis",
        nodata=nodata,
    )


def save_cpr_histogram(
    cpr: np.ndarray,
    preview_dir: Path,
    nodata: float = -9999.0,
    n_bins: int = 300,
) -> None:
    """
    Save a histogram of CPR values (clipped to 2nd-98th percentile).
    Also marks median and the CPR = 1 reference line.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)

    valid = cpr[(cpr != nodata) & np.isfinite(cpr)]
    if valid.size == 0:
        log.warning("No valid CPR pixels -- histogram skipped.")
        return

    p2, p98   = np.percentile(valid, [2, 98])
    clipped   = valid[(valid >= p2) & (valid <= p98)]
    med       = float(np.median(valid))
    mean_val  = float(valid.mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(clipped, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med,     color="red",    lw=1.5, label=f"Median = {med:.3f}")
    ax.axvline(mean_val,color="orange", lw=1.2, ls="--",
               label=f"Mean   = {mean_val:.3f}")
    ax.axvline(1.0,     color="lime",   lw=1.0, ls=":",
               label="CPR = 1.0 (reference)")

    ax.set_xlabel("CPR", fontsize=11)
    ax.set_ylabel("Pixel count", fontsize=11)
    ax.set_title(
        f"CPR Histogram (2nd-98th percentile range: {p2:.2f} -- {p98:.2f})\n"
        f"Total valid pixels: {valid.size:,}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    out = preview_dir / "CPR_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved histogram: {out.name}")
