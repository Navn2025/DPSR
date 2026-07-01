"""
visualizer.py
=============
Generate PNG visualisations for every processed layer.

All rendering happens at runtime when the Python program is executed.
No images are embedded in this file.

Outputs
  outputs/previews/<BandName>.png   — one per layer
  outputs/previews/overview_all_bands.png — grid of all bands
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive: writes PNGs without opening windows
import matplotlib.pyplot as plt

from config import (
    BAND_CMAPS, DEFAULT_CMAP,
    FIGURE_DPI, FIGURE_SIZE,
    PREVIEWS_DIR,
)
from utils import get_logger

log = get_logger("visualizer")


# -- Single-band preview -------------------------------------------------------

def save_band_preview(
    array:   Optional[np.ndarray],
    name:    str,
    out_dir: Path = PREVIEWS_DIR,
) -> Path:
    """
    Render *array* as a grey-scale or colour PNG with a colorbar.

    Parameters
    ----------
    array   : 2-D float32 (may contain NaN)
    name    : layer name used for title and output filename
    out_dir : destination directory

    Returns the path of the saved PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"

    cmap = BAND_CMAPS.get(name, DEFAULT_CMAP)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if array is None or not np.any(np.isfinite(array)):
        # No data — show a placeholder
        ax.text(
            0.5, 0.5, f"No Data Available\n({name})",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=14, color="red",
        )
        ax.set_facecolor("#111111")
        im = None
    else:
        masked = np.ma.array(array, mask=~np.isfinite(array))
        im = ax.imshow(masked, cmap=cmap, interpolation="nearest")

    ax.set_title(
        f"Chandrayaan-2  |  Lunar South Pole  |  {name}",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Column (pixels)", fontsize=9)
    ax.set_ylabel("Row (pixels)",    fontsize=9)
    ax.tick_params(labelsize=8)

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalised Value [0–1]", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Preview saved -> {out_path.name}")
    return out_path


# -- All-bands grid ------------------------------------------------------------

def save_composite_overview(
    bands:   dict[str, Optional[np.ndarray]],
    out_dir: Path = PREVIEWS_DIR,
) -> Path:
    """
    Build a grid overview by reading the already-saved individual PNGs.

    This avoids holding all band arrays in RAM simultaneously.
    If a PNG for a band doesn't exist yet it is skipped gracefully.
    """
    import matplotlib.image as mpimg

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overview_all_bands.png"

    # Collect names that have a saved PNG
    names = [n for n in bands.keys() if (out_dir / f"{n}.png").exists()]
    if not names:
        log.warning("No individual PNGs found for composite overview.")
        return out_path

    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 4.2),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    for i, name in enumerate(names):
        ax      = axes_flat[i]
        png     = out_dir / f"{name}.png"
        try:
            img = mpimg.imread(str(png))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Error", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.set_title(name, fontsize=11, pad=4)
        ax.axis("off")

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "CH-2 DFSAR  --  Lunar South Pole Ice Detection  |  Feature Stack Overview",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Overview saved -> {out_path.name}")
    return out_path


# -- Batch entry point ---------------------------------------------------------

def save_all_previews(
    bands:   dict[str, Optional[np.ndarray]],
    out_dir: Path = PREVIEWS_DIR,
) -> None:
    """Save individual PNG for every band, then the composite overview."""
    log.info(f"Generating {len(bands)} preview(s)…")
    for name, array in bands.items():
        try:
            save_band_preview(array, name, out_dir)
        except Exception as exc:
            log.error(f"Preview failed [{name}]: {exc}")

    try:
        save_composite_overview(bands, out_dir)
    except Exception as exc:
        log.error(f"Composite overview failed: {exc}")
