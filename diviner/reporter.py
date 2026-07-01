"""
reporter.py
===========
STEP 11 + STEP 12 of the Diviner integration pipeline.

STEP 11 — Write a CSV table with full statistics (mean, median, min, max,
           std, percentiles) for every feature band.

STEP 12 — Build a multi-page PDF report using matplotlib's PdfPages
           backend (no reportlab dependency).  The PDF summarises:
             • Input datasets
             • Processing workflow
             • Per-band statistics
             • Correlation analysis (embedded image)
             • Feature stack band catalogue
             • Ice Confidence methodology and weights
             • All generated output files

No existing file is overwritten.
"""

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

log = logging.getLogger("diviner_pipeline.reporter")


# ---------------------------------------------------------------------------
# STEP 11 — CSV statistics report
# ---------------------------------------------------------------------------

def write_statistics_csv(
    bands:             Dict[str, np.ndarray],
    out_path:          Path,
    nodata:            float     = -9999.0,
    percentile_levels: List[int] = None,
) -> None:
    """
    Write one row per feature band to a CSV statistics file.

    Columns: Band, Mean, Median, Min, Max, Std, P1…P99,
             Valid_Pixels, Total_Pixels, Valid_Pct.
    """
    if out_path.exists():
        log.info(f"  [skip] {out_path.name} already exists.")
        return

    if percentile_levels is None:
        percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    pct_headers = [f"P{p}" for p in percentile_levels]
    header = (
        ["Band", "Mean", "Median", "Min", "Max", "Std"]
        + pct_headers
        + ["Valid_Pixels", "Total_Pixels", "Valid_Pct"]
    )

    _MAX_STAT_PX = 5_000_000   # sample cap for large arrays

    rows = []
    for name, arr in bands.items():
        v_full = arr[(arr != nodata) & np.isfinite(arr)].ravel()
        total  = int(arr.size)

        if v_full.size == 0:
            row = (
                [name]
                + ["nan"] * 5
                + ["nan"] * len(percentile_levels)
                + [0, total, "0.00"]
            )
        else:
            # True min/max from full population; sample for distribution stats
            v_min = float(v_full.min())
            v_max = float(v_full.max())
            v = v_full
            if v_full.size > _MAX_STAT_PX:
                v = np.random.default_rng(seed=0).choice(v_full, _MAX_STAT_PX, replace=False)
            pcts = [f"{float(np.percentile(v, p)):.6e}" for p in percentile_levels]
            row = (
                [
                    name,
                    f"{v.mean():.6e}",
                    f"{float(np.median(v)):.6e}",
                    f"{v_min:.6e}",
                    f"{v_max:.6e}",
                    f"{v.std(dtype=np.float64):.6e}",
                ]
                + pcts
                + [int(v_full.size), total, f"{100 * v_full.size / total:.2f}"]
            )
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    log.info(f"  Saved CSV: {out_path}  ({out_path.stat().st_size / 1024:.0f} kB)")


# ---------------------------------------------------------------------------
# STEP 12 — PDF report helpers
# ---------------------------------------------------------------------------

def _text_page(pdf: PdfPages, title: str, body_lines: List[str]) -> None:
    """Add an A4 monospace-text page to the PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    ax  = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")

    body = "\n".join(body_lines)
    full = f"{title}\n{'━' * 68}\n\n{body}"

    ax.text(
        0.0, 1.0, full,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7.2,
        family="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_page(pdf: PdfPages, img_path: Path, caption: str) -> None:
    """Add an image page to the PDF; silently skip if the file is missing."""
    if not img_path.exists():
        log.debug(f"  PDF: missing image {img_path.name} — skipped.")
        return

    try:
        img = plt.imread(str(img_path))
    except Exception as exc:
        log.warning(f"  PDF: could not read {img_path.name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    ax.set_title(caption, fontsize=8, pad=4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# STEP 12 — Main PDF builder
# ---------------------------------------------------------------------------

def write_pdf_report(
    out_path:     Path,
    bands:        Dict[str, np.ndarray],
    input_paths:  Dict[str, Path],
    aligned_dir:  Path,
    preview_dir:  Path,
    out_dir:      Path,
    weights:      Dict[str, float],
    nodata:       float = -9999.0,
) -> None:
    """
    Build a multi-page PDF report summarising the entire Diviner pipeline.

    Parameters
    ----------
    out_path    : destination PDF file (skipped if already exists).
    bands       : dict of all feature arrays (for statistics).
    input_paths : dict mapping label → Path for the input file table.
    aligned_dir : directory containing aligned GeoTIFFs.
    preview_dir : directory containing preview PNGs.
    out_dir     : main output directory (outputs/diviner/).
    weights     : ice confidence weights dict.
    nodata      : nodata sentinel.
    """
    if out_path.exists():
        log.info(f"  [skip] {out_path.name} already exists.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    with PdfPages(out_path) as pdf:

        # ── Cover page ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        ax  = fig.add_axes([0.10, 0.20, 0.80, 0.60])
        ax.axis("off")
        ax.text(0.5, 0.88,
                "ISRO Hackathon — Lunar South Pole Ice Detection",
                ha="center", fontsize=15, weight="bold")
        ax.text(0.5, 0.76,
                "Diviner Thermal Integration Pipeline Report",
                ha="center", fontsize=12)
        ax.text(0.5, 0.64,
                "Chandrayaan-2 DFSAR + LRO Diviner",
                ha="center", fontsize=10, color="#444444")
        ax.text(0.5, 0.52,
                f"Generated: {today}",
                ha="center", fontsize=9, color="gray")
        ax.text(0.5, 0.42,
                "Module: diviner/  |  Version: 1.0.0",
                ha="center", fontsize=8, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Section 1: Input datasets ─────────────────────────────────────
        lines = ["Input files used by this pipeline run:\n"]
        for label, path in input_paths.items():
            exists  = "[OK]" if path.exists() else "[MISSING]"
            size_s  = (
                f"{path.stat().st_size / 1024**2:.1f} MB"
                if path.exists() else "—"
            )
            lines.append(f"  {exists} [{label:<16}]  {size_s}")
            lines.append(f"           {path}")
            lines.append("")
        _text_page(pdf, "Section 1 — Input Datasets", lines)

        # ── Section 2: Processing workflow ────────────────────────────────
        workflow = [
            "Step  1  Load all input datasets and print full metadata",
            "          (CRS, resolution, bounds, dtype, nodata, statistics)",
            "",
            "Step  2  Convert GMT .grd files (Tmean, Pump) to float32 GeoTIFF",
            "          Originals are never modified.  Output: outputs/diviner/",
            "",
            "Step  3  Reproject every dataset into the reference CRS",
            "          (default reference: results/DPSR.tif — LOLA 20m polar grid)",
            "",
            "Step  4  Resample all bands to the identical pixel grid",
            "          (same width, height, transform, CRS as reference)",
            "",
            "Step  5  Generate quick-look preview PNGs",
            "          Tmean_preview.png | ZIT_preview.png | Pump_preview.png",
            "          → outputs/previews/",
            "",
            "Step  6  Compute descriptive statistics for every feature band",
            "          (min, max, mean, median, std, percentiles, histogram)",
            "",
            "Step  7  Build 9-band Feature Stack → Feature_Stack.tif",
            "          Band order: DEM | Slope | PSR | DPSR | CPR | DOP |",
            "                      Tmean | ZIT | Pump",
            "",
            "Step  8  Compute physics-based Ice Confidence Score",
            "          Weighted sum of 8 normalised indicators (see Section 6)",
            "",
            "Step  9  Save Ice_Confidence_Map.tif and Ice_Confidence_Map.png",
            "",
            "Step 10  Generate publication-quality visualisations:",
            "          • Feature maps (one PNG per band)",
            "          • Per-feature histograms",
            "          • Pearson correlation matrix heat-map",
            "          • Five scatter plots",
            "",
            "Step 11  Write statistics_report.csv",
            "",
            "Step 12  Generate this PDF report",
        ]
        _text_page(pdf, "Section 2 — Processing Workflow", workflow)

        # ── Section 3: Statistics summary ─────────────────────────────────
        col_w = 12
        hdr   = (f"{'Band':<12}"
                 f"{'Mean':>{col_w}}"
                 f"{'Median':>{col_w}}"
                 f"{'Min':>{col_w}}"
                 f"{'Max':>{col_w}}"
                 f"{'Std':>{col_w}}"
                 f"{'Valid%':>8}")
        lines = ["Descriptive statistics for all feature bands:\n", hdr, "─" * 76]

        _MAX_STAT_PX = 5_000_000
        for name, arr in bands.items():
            v_full = arr[(arr != nodata) & np.isfinite(arr)].ravel()
            total  = arr.size
            if v_full.size > 0:
                v_min = float(v_full.min())
                v_max = float(v_full.max())
                v = v_full if v_full.size <= _MAX_STAT_PX else \
                    np.random.default_rng(seed=0).choice(v_full, _MAX_STAT_PX, replace=False)
                lines.append(
                    f"{name:<12}"
                    f"{v.mean():>{col_w}.4e}"
                    f"{float(np.median(v)):>{col_w}.4e}"
                    f"{v_min:>{col_w}.4e}"
                    f"{v_max:>{col_w}.4e}"
                    f"{v.std(dtype=np.float64):>{col_w}.4e}"
                    f"{100*v_full.size/total:>7.2f}%"
                )
            else:
                lines.append(f"{name:<12}  (no valid pixels)")
        _text_page(pdf, "Section 3 — Statistics Summary", lines)

        # ── Section 4: Correlation matrix ─────────────────────────────────
        corr_img = out_dir / "Correlation_Matrix.png"
        _image_page(pdf, corr_img, "Section 4 — Feature Correlation Matrix (Pearson r)")

        # ── Section 5: Feature stack band catalogue ────────────────────────
        band_catalogue = [
            ("DEM",   "LOLA 20 m digital elevation model (metres)"),
            ("Slope", "Terrain slope derived from DEM (degrees)"),
            ("PSR",   "Permanently Shadowed Region mask (0 = lit, 1 = shadow)"),
            ("DPSR",  "Doubly PSR mask (0 = lit, 1 = doubly shadowed)"),
            ("CPR",   "Circular Polarisation Ratio — DFSAR official mosaic"),
            ("DOP",   "Degree of Polarisation — DFSAR full-pol SLI"),
            ("Tmean", "LRO Diviner mean surface temperature (K)"),
            ("ZIT",   "LRO Diviner zero-incidence temperature (K)"),
            ("Pump",  "LRO Diviner volatile pump parameter"),
        ]
        lines = [
            "9-band Feature Stack  (Feature_Stack.tif)\n",
            f"{'Band':>2}  {'Name':<8}  Description",
            "─" * 64,
        ]
        for i, (name, desc) in enumerate(band_catalogue, 1):
            lines.append(f"  {i:>2}  {name:<8}  {desc}")
        _text_page(pdf, "Section 5 — Feature Stack Band Catalogue", lines)

        # ── Section 6: Ice confidence methodology ─────────────────────────
        w_total = sum(weights.values())
        meth = [
            "Physics-Based Ice Confidence Score — Methodology\n",
            "Formula: score = Σ(norm_i × w_i) / Σ(w_i)  over available bands",
            "         norm_i ∈ [0,1];  score ∈ [0,1]\n",
            f"{'Indicator':<10}  {'Weight':>7}  {'Ice sense':>28}  Normalisation",
            "─" * 70,
        ]
        sense = {
            "CPR":   ("HIGH → ice", "2–98 pct, keep"),
            "DOP":   ("LOW  → ice", "2–98 pct, invert"),
            "Tmean": ("LOW  → ice", "2–98 pct, invert"),
            "ZIT":   ("LOW  → ice", "2–98 pct, invert"),
            "Pump":  ("HIGH → ice", "2–98 pct, keep"),
            "PSR":   ("IN   → ice", "binary 0/1"),
            "DPSR":  ("IN   → ice", "binary 0/1"),
            "Slope": ("LOW  → ice", "2–98 pct, invert"),
        }
        for k, w in weights.items():
            s, n = sense.get(k, ("—", "—"))
            meth.append(f"  {k:<10}  {w:>6.2f}   {s:>28}   {n}")
        meth += [
            f"\n  Total weight = {w_total:.2f}",
            "",
            "References:",
            "  Nozette et al. (1996) Science 274, 1495",
            "  Campbell et al. (2006) Nature 443, 835",
            "  Paige et al. (2010) Science 330, 479",
            "  Hayne et al. (2015) Icarus 255, 58",
            "  Schorghofer (2014) Astrophys. J. 788, 169",
            "  Watson et al. (1961) JGR 66, 3033",
            "  O'Brien & Byrne (2022) Planet. Space Sci. 221, 105566",
        ]
        _text_page(pdf, "Section 6 — Ice Confidence Methodology", meth)

        # ── Section 7: Feature maps ────────────────────────────────────────
        for name in ["Tmean", "ZIT", "Pump", "DEM", "Slope",
                     "PSR", "DPSR", "CPR", "DOP"]:
            img = out_dir / f"{name}_map.png"
            _image_page(pdf, img, f"Section 7 — {name} Map")

        # ── Section 8: Ice Confidence Map ─────────────────────────────────
        _image_page(pdf, out_dir / "Ice_Confidence_Map.png",
                    "Section 8 — Ice Confidence Map")
        _image_page(pdf, out_dir / "IceConfidence_map.png",
                    "Section 8 — Ice Confidence Map (feature-set view)")

        # ── Section 9: Scatter plots ───────────────────────────────────────
        scatter_pairs = [
            ("CPR",   "DOP"),
            ("CPR",   "Tmean"),
            ("CPR",   "ZIT"),
            ("DOP",   "Tmean"),
            ("Tmean", "Pump"),
        ]
        for x, y in scatter_pairs:
            img = out_dir / f"scatter_{x}_vs_{y}.png"
            _image_page(pdf, img, f"Section 9 — Scatter: {x} vs {y}")

        # ── Section 10: Generated output file listing ──────────────────────
        file_lines = ["All files generated by the Diviner pipeline:\n"]
        search_dirs = [out_dir, preview_dir, aligned_dir]
        for d in search_dirs:
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        try:
                            rel = f.relative_to(out_dir.parent.parent)
                        except ValueError:
                            rel = f
                        size_mb = f.stat().st_size / 1024 ** 2
                        file_lines.append(f"  {size_mb:6.1f} MB  {rel}")
        _text_page(pdf, "Section 10 — Generated Output Files", file_lines)

        # PDF document metadata
        info = pdf.infodict()
        info["Title"]   = "Diviner Thermal Integration Report"
        info["Author"]  = "ISRO Hackathon Team"
        info["Subject"] = "Lunar South Pole Ice Detection — Diviner + DFSAR"

    log.info(
        f"  Saved PDF: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)"
    )
