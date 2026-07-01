"""
Chandrayaan-2 DFSAR Level-2 GeoTIFF Processor
===============================================
Reads all GeoTIFF files from the DFSAR data directory, prints metadata,
normalises Band-1, and saves visualisations to DFSAR_Images/.

Designed to be extended for:
  - Circular Polarisation Ratio (CPR)
  - Degree of Polarisation (DOP)
  - Radar Backscatter Analysis
  - DEM / PSR / DPSR Overlay
  - Ice Probability Mapping
  - Landing Site Selection
  - Rover Path Planning
"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from rasterio.errors import NotGeoreferencedWarning

# ── suppress rasterio CRS warnings for files that lack a projection ──────────
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
matplotlib.use("TkAgg")   # change to "Agg" to suppress display windows


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RasterMetadata:
    """Holds metadata extracted from a single GeoTIFF."""
    filename:   str
    crs:        str
    width:      int
    height:     int
    band_count: int
    dtype:      str
    res_x:      float
    res_y:      float
    bounds:     tuple


@dataclass
class ProcessingResult:
    """Tracks the outcome of processing one file."""
    path:    Path
    success: bool
    message: str = ""


@dataclass
class SessionSummary:
    """Accumulates results across the whole run."""
    total:   int = 0
    success: int = 0
    failed:  int = 0
    results: list[ProcessingResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_tif_files(data_dir: Path) -> list[Path]:
    """Return a sorted list of every .tif / .tiff in *data_dir*."""
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(data_dir.glob(pattern))
    return sorted(set(found))


def read_metadata(src: rasterio.DatasetReader, filepath: Path) -> RasterMetadata:
    """Extract metadata from an open rasterio dataset."""
    crs_str = str(src.crs) if src.crs else "Unknown / Not georeferenced"
    res_x, res_y = src.res if src.res else (float("nan"), float("nan"))
    return RasterMetadata(
        filename=filepath.name,
        crs=crs_str,
        width=src.width,
        height=src.height,
        band_count=src.count,
        dtype=src.dtypes[0],
        res_x=res_x,
        res_y=res_y,
        bounds=tuple(src.bounds),
    )


def print_metadata(meta: RasterMetadata) -> None:
    """Pretty-print raster metadata to stdout."""
    sep = "─" * 60
    print(sep)
    print(f"  File      : {meta.filename}")
    print(f"  CRS       : {meta.crs}")
    print(f"  Size      : {meta.width} × {meta.height} px")
    print(f"  Bands     : {meta.band_count}")
    print(f"  Dtype     : {meta.dtype}")
    print(f"  Resolution: {meta.res_x:.6f} × {meta.res_y:.6f} deg/px")
    print(f"  Bounds    : {meta.bounds}")
    print(sep)


def read_band1(
    src: rasterio.DatasetReader,
) -> tuple[np.ndarray, Optional[float]]:
    """Read Band-1 and return (data_array, nodata_value)."""
    data = src.read(1).astype(np.float64)
    nodata = src.nodata
    return data, nodata


# ─────────────────────────────────────────────────────────────────────────────
# Processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def mask_nodata(data: np.ndarray, nodata: Optional[float]) -> np.ma.MaskedArray:
    """Mask NoData pixels; also mask NaN and ±Inf."""
    mask = ~np.isfinite(data)
    if nodata is not None:
        if np.isnan(nodata):
            mask |= np.isnan(data)
        else:
            mask |= (data == nodata)
    return np.ma.array(data, mask=mask)


def normalise(masked: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """
    Normalise to [0, 1] using 2nd–98th percentile stretch so extreme outliers
    do not collapse the dynamic range.
    """
    valid = masked.compressed()
    if valid.size == 0:
        return masked
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    if hi == lo:
        return np.ma.array(np.zeros_like(masked.data), mask=masked.mask)
    clipped = np.ma.clip(masked, lo, hi)
    return (clipped - lo) / (hi - lo)


def derive_product_label(filepath: Path) -> str:
    """
    Infer a human-readable label from the DFSAR filename convention.

    Expected token layout (underscore-split):
      ch2_sar_<mode>_<timestamp>_d_<product>_<aux1>_cp_<pol>_<version>.tif
    Falls back to the stem if the name is non-standard.
    """
    parts = filepath.stem.split("_")
    try:
        # indices: 0=ch2, 1=sar, 2=mode, 3=ts, 4=d, 5=product, 6=aux, 7=cp, 8=pol, 9=ver
        mode    = parts[2].upper()           # e.g. NCXL
        product = parts[5].upper()           # e.g. GRI / SLI / SRI
        aux     = parts[6].upper()           # e.g. IN / XX / MA
        pol     = parts[8].upper()           # e.g. LH / LV / XX
        return f"{mode}  {product}  aux={aux}  pol={pol}"
    except IndexError:
        return filepath.stem


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_and_save(
    normalised: np.ma.MaskedArray,
    title:      str,
    out_path:   Path,
    *,
    cmap:       str  = "gray",
    figsize:    tuple = (8, 8),
    show:       bool  = True,
) -> None:
    """
    Render *normalised* data with a colourbar and save to *out_path* as PNG.
    If *show* is True, display the figure in a window as well.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(normalised, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Intensity (2–98 percentile stretch)", fontsize=9)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Column (pixels)", fontsize=9)
    ax.set_ylabel("Row (pixels)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    Saved → {out_path.name}")

    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_single_file(
    filepath: Path,
    out_dir:  Path,
    *,
    show:     bool = True,
) -> ProcessingResult:
    """
    Full processing pipeline for one GeoTIFF:
      read → metadata → band-1 → mask → normalise → visualise → save PNG.

    Returns a ProcessingResult (success=True/False).
    """
    print(f"\n[>] Processing: {filepath.name}")
    try:
        with rasterio.open(filepath) as src:
            meta = read_metadata(src, filepath)
            print_metadata(meta)

            data, nodata = read_band1(src)

        masked     = mask_nodata(data, nodata)
        normalised = normalise(masked)

        label    = derive_product_label(filepath)
        title    = f"CH-2 DFSAR  |  {label}\n{filepath.name}"
        out_path = out_dir / (filepath.stem + ".png")

        visualise_and_save(normalised, title, out_path, show=show)

        print(f"    [OK] Done.")
        return ProcessingResult(path=filepath, success=True)

    except rasterio.errors.RasterioIOError as exc:
        msg = f"Rasterio IO error — file may be corrupted: {exc}"
        print(f"    [SKIP] {msg}")
        return ProcessingResult(path=filepath, success=False, message=msg)

    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error: {type(exc).__name__}: {exc}"
        print(f"    [SKIP] {msg}")
        return ProcessingResult(path=filepath, success=False, message=msg)


# ─────────────────────────────────────────────────────────────────────────────
# Session runner
# ─────────────────────────────────────────────────────────────────────────────

def run_session(
    data_dir: Path,
    out_dir:  Path,
    *,
    show:     bool = True,
) -> SessionSummary:
    """
    Discover all TIFFs in *data_dir*, process each, and return a summary.
    """
    tif_files = discover_tif_files(data_dir)
    summary = SessionSummary(total=len(tif_files))

    if not tif_files:
        print(f"[!] No GeoTIFF files found in: {data_dir}")
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═' * 60}")
    print(f"  CH-2 DFSAR Processor")
    print(f"  Input  : {data_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Files  : {summary.total} GeoTIFF(s) found")
    print(f"{'═' * 60}")

    for idx, filepath in enumerate(tif_files, start=1):
        print(f"\n[{idx}/{summary.total}]", end="")
        result = process_single_file(filepath, out_dir, show=show)
        summary.results.append(result)
        if result.success:
            summary.success += 1
        else:
            summary.failed += 1

    return summary


def print_summary(summary: SessionSummary, out_dir: Path) -> None:
    """Print the final processing summary."""
    print(f"\n{'═' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Total files found  : {summary.total}")
    print(f"  Successfully done  : {summary.success}")
    print(f"  Failed / skipped   : {summary.failed}")
    print(f"  Output directory   : {out_dir}")

    if summary.failed:
        print("\n  Failed files:")
        for r in summary.results:
            if not r.success:
                print(f"    • {r.path.name}  —  {r.message}")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Extension stubs  (implement these in future sprints)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cpr(lh_path: Path, lv_path: Path) -> np.ndarray:
    """
    Circular Polarisation Ratio  CPR = σ°_LH / σ°_LV.

    High CPR (> 1) indicates volume scattering consistent with ice deposits.
    """
    raise NotImplementedError("CPR computation — TODO")


def compute_dop(
    lh_path: Path, lv_path: Path, hh_path: Optional[Path] = None
) -> np.ndarray:
    """
    Degree of Polarisation  DOP ∈ [0, 1].

    Fully polarised returns → DOP ≈ 1; depolarising (rough/subsurface) → DOP < 1.
    """
    raise NotImplementedError("DOP computation — TODO")


def compute_backscatter_db(sigma0_linear: np.ndarray) -> np.ndarray:
    """Convert linear σ° to dB:  σ°_dB = 10 · log10(σ°_linear)."""
    raise NotImplementedError("Backscatter dB conversion — TODO")


def overlay_dem(sar_data: np.ndarray, dem_path: Path) -> np.ndarray:
    """Co-register and overlay DEM slope/aspect on SAR data."""
    raise NotImplementedError("DEM overlay — TODO")


def overlay_psr(sar_data: np.ndarray, psr_path: Path) -> np.ndarray:
    """Mask permanently shadowed regions using a PSR map."""
    raise NotImplementedError("PSR overlay — TODO")


def overlay_dpsr(sar_data: np.ndarray, dpsr_path: Path) -> np.ndarray:
    """Apply DPSR (double-bounce PSR) detection mask to SAR data."""
    raise NotImplementedError("DPSR overlay — TODO")


def ice_probability_map(
    cpr: np.ndarray, dop: np.ndarray, psr_mask: np.ndarray
) -> np.ndarray:
    """
    Combine CPR, DOP, and PSR to estimate per-pixel ice probability.

    Simple heuristic:  P_ice = CPR_norm * (1 - DOP) * PSR_mask
    Replace with a trained classifier (RF / SVM) in future iterations.
    """
    raise NotImplementedError("Ice probability mapping — TODO")


def select_landing_sites(
    ice_prob: np.ndarray,
    slope: np.ndarray,
    *,
    min_ice_prob: float = 0.6,
    max_slope_deg: float = 15.0,
) -> np.ndarray:
    """
    Return a boolean mask of candidate landing sites:
      safe slope  AND  high ice probability.
    """
    raise NotImplementedError("Landing site selection — TODO")


def plan_rover_path(
    start: tuple[int, int],
    goal:  tuple[int, int],
    slope: np.ndarray,
    *,
    max_slope_deg: float = 20.0,
) -> list[tuple[int, int]]:
    """A* path planner respecting terrain slope constraints."""
    raise NotImplementedError("Rover path planning — TODO")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── directory layout ──────────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent   # …/DFSAR/
    data_dir   = script_dir / "data"               # …/DFSAR/data/
    out_dir    = script_dir / "DFSAR_Images"       # …/DFSAR/DFSAR_Images/

    # Set show=False to suppress pop-up windows (headless / CI environments).
    show_plots = True

    summary = run_session(data_dir, out_dir, show=show_plots)
    print_summary(summary, out_dir)


if __name__ == "__main__":
    main()
