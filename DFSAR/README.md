# `DFSAR/` — DFSAR Product Ingestion & Feature Stack

Ingests the raw **Chandrayaan-2 DFSAR** product tree and catalogues the full
**18-raster** full-polarimetric product set, aligning everything to the LOLA
grid for a broader feature stack (beyond the narrower `cpr/` / `dop/` scripts).

---

## Contents

| Path | Description |
|------|-------------|
| `dfsar_processor.py` | Core DFSAR reader / processor |
| `data_pipeline/` | Multi-product ingestion & LOLA-grid alignment |

Cataloged products include **CPR**, **SRD** (Stokes Radar Decomposition),
**ODD** (odd-bounce), **VOL** (volume), **HLX** (helix), **EVN** (eigenvalue),
**TRT** (total power / trace), plus **GRI / SLI / SRI** levels and their
polarization variants.

DFSAR product levels used across the project:

| Level | Meaning | Used by |
|-------|---------|---------|
| SLI | Single-look complex | `cpr/`, `dop/` |
| GRI | Multilooked, ground-range | `cpr_gri/` |
| SRI | Slant-range | feature stack |

---

## Run

```powershell
python DFSAR/data_pipeline/main.py
```

---

## Output

Aligned DEM/PSR/DPSR layers and the aligned DFSAR products (CPR, SRD, EVN, HLX,
ODD, TRT, …) on the common LOLA grid, for downstream fusion / analysis.

> ⚠️ **Known issue:** `data_pipeline/main.py` currently crashes right after
> dataset discovery with a `UnicodeEncodeError` (box-drawing characters printed
> to a Windows cp1252 console). Figures were recovered from a prior partial run;
> a one-line encoding fix is pending. See `PROJECT_REPORT.md` §7.
