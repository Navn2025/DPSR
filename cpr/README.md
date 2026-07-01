# `cpr/` — Circular Polarization Ratio (DFSAR SLI)

Computes the **Circular Polarization Ratio (CPR)** directly from raw
Chandrayaan-2 **DFSAR full-polarimetric SLI** data, and runs the detailed
**Faustini crater** case study used throughout the report.

```
S_RR (SC) = (S_HH − S_VV + 2j·S_HV) / 2
S_RL (OC) = (S_HH + S_VV) / 2
σ_SC = mean(|S_RR|²)   σ_OC = mean(|S_RL|²)   CPR = σ_SC / σ_OC
```

**Interpretation:** CPR < 1 → surface (Bragg) scattering / bare regolith;
CPR > 1 → volume scattering → ice-grain *or* rough-ejecta candidate. CPR alone
is ambiguous — it is combined with DOP and thermal context downstream
(see the rock-vs-ice logic in the report).

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `reader.py`, `complex_builder.py` | Read DFSAR channels, build scattering matrix |
| `compute_cpr.py`, `cpr.py` | SC/OC powers and CPR |
| `config.py`, `utils.py` | Paths and constants |
| `validate_cpr.py` | Sanity checks vs official product |
| `geo_extract.py` | Crop scenes by geographic footprint |
| `faustini*.py`, `f2_crater.py`, `ice_zoom.py` | Faustini / F2 crater studies |
| `combined_figure.py`, `cpr_dop_scatter.py` | Report figures (CPR map, CPR–DOP scatter) |
| `data/`, `outputs/`, `faustini/` | Inputs / generated rasters & plots |

---

## Two CPR Formulas

| Formula | Definition | Note |
|---------|-----------|------|
| **Default (SC/OC)** | `σ_SC / σ_OC` | Primary; also computed on GRI (`cpr_gri/`) |
| **Research (μc)** | `(√σ_HH+√σ_VV)² / (√σ_HH−√σ_VV)²` | Co-pol only; ill-conditioned where σ_HH ≈ σ_VV, log-rescaled |

---

## Run

```powershell
python cpr/main.py
```

---

## Output

CPR raster + SC/OC power maps, CPR histogram, CPR–DOP scatter, and the Faustini
crater figures under `cpr/outputs/`. The ice-candidate criterion used in the
study is **CPR > 1 AND DOP < 0.13**.

Related: `cpr_gri/` (GRI-level CPR, best pixel correlation to the official
product), `cpr_official/` (official Putrevu-2023 mosaic extraction),
[`dop/`](../dop/), and [`validation/`](../validation/).
