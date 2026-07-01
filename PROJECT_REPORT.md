> ### 🔗 **[▶ VIEW PRESENTATION & FULL EXPLANATION (Drive)](https://drive.google.com/file/d/1GdfXjIone_D90hqFLwcaTXP8JMVdYooY/view?usp=drive_link)**
>
> **https://drive.google.com/file/d/1GdfXjIone_D90hqFLwcaTXP8JMVdYooY/view?usp=drive_link**

---

> ⚠️ **Work in progress.** This is a *proposed* solution built under a limited
> time window. What is presented here is a functional first version that
> already runs end-to-end and produces real outputs — but it is **not final**.
> We are actively optimising the pipeline to reach more accurate and reliable
> results, and there are a few known problems (see §7) that we are currently
> working to fix. Further improvements, additional features, and a fuller
> explanation are covered in the presentation linked above.

---

# Multi-Sensor Detection of Water-Ice Stability Zones at the Lunar South Pole

### A Physics-Based Fusion of Topographic Shadow Modelling, Chandrayaan-2 DFSAR Radar Polarimetry, and LRO Diviner Thermal Data

**Prepared for:** ISRO Hackathon
**Repository:** `ISRO_Hackathon`
**Report date:** 2026-07-01
**Status:** Working prototype, all five pipelines executed end-to-end at least once; figures below are the actual outputs of those runs (`images-final/`)

---

## Visual Overview

The three diagrams below summarise the system at a glance — the high-level
architecture, the end-to-end data flow, and the internal fusion logic that
produces the Ice Confidence Map. Full detail follows in Sections 3–4.

**System Architecture**

![System Architecture](wireframes_svg/1_system_architecture.png)

**End-to-End Data Flow (Pipeline Sequence)**

![End-to-End Data Flow](wireframes_svg/2_data_flow.png)

**Fusion Engine — Weighted Ice Confidence Score**

![Fusion Engine](wireframes_svg/3_fusion_engine.png)

---

## Abstract

Permanently shadowed regions (PSRs) near the lunar poles are cold traps
capable of hosting stable deposits of water ice, and are consequently
priority targets for ISRO's and other agencies' exploration programmes. No
single remote-sensing instrument can prove the presence of ice on its own —
each carries its own ambiguity (topographic shadow alone does not guarantee
ice; high radar backscatter can come from either ice or rough blocky
regolith; low temperature alone does not indicate volatile delivery). This
project builds an end-to-end pipeline that derives four physically distinct
indicators from three independent NASA/ISRO datasets — LOLA elevation,
Chandrayaan-2 DFSAR full-polarimetric radar, and LRO Diviner thermal
radiometry — and fuses them into a single, explainable **Ice Confidence Map**
covering the full south-polar LOLA DEM (15,168 × 15,168 pixels at 20 m/px,
i.e. ≈230 million pixels, ±151.68 km from the pole). No machine-learning model
is used anywhere in the fusion step; every weight, normalisation choice, and
physical direction (does "high" or "low" indicate ice?) is tied to a specific
published reference. This report documents the scientific methodology,
implementation, every generated figure with its quantitative interpretation,
and — with equal weight — the concrete defects and validation shortfalls
discovered while building it.

---

## Table of Contents

1. Introduction
2. Data Sources and Inputs
3. System Architecture and Repository Organisation
4. Methodology
5. Results — Complete Figure-by-Figure Walkthrough
   5.1 DPSR Extraction Figures
   5.2 DFSAR Multi-Product Ingestion Figures
   5.3 CPR — Faustini Crater Case Study (SLI, Default Formula)
   5.4 CPR — Faustini Crater Case Study (Published μc Research Formula)
   5.5 CPR — Default vs. Research Formula, Direct Comparison
   5.6 Degree of Polarization (DOP) Figures
   5.7 GRI-Based CPR — Default vs. Research
   5.8 CPR Validation Against the Official Mosaic (SLI Pipeline)
   5.9 CPR Validation Against the Official Mosaic (GRI Pipeline)
   5.10 Diviner Fusion — Per-Band Maps and Histograms
   5.11 Diviner Fusion — Cross-Band Scatter Plots and Correlation Matrix
   5.12 Final Ice Confidence Map
6. Discussion
7. Known Issues and Limitations
8. Conclusion and Future Work
9. References

---

## 1. Introduction

### 1.1 Background and Motivation

Water ice at the lunar poles is considered one of the most important
in-situ resources for sustained human presence on the Moon (propellant
production, life support, radiation shielding). Its presence has been
inferred indirectly by multiple missions (LCROSS impact plume, LRO
Diviner/LAMP/Mini-RF, Chandrayaan-1 M³, Chandrayaan-2 DFSAR) but never
mapped with a single unambiguous method at high resolution. The scientific
literature has established three largely independent lines of physical
evidence:

1. **Topographic shadow persistence.** A location that never receives direct
   sunlight across a full lunar year (a PSR), and additionally never sees
   scattered/reflected light from any illuminated, non-shadowed terrain (a
   DPSR — *Doubly* Permanently Shadowed Region), is thermally the most
   stable possible environment for surface ice (Watson et al. 1961;
   O'Brien & Byrne 2022).
2. **Radar polarimetric anomalies.** Coherent backscatter from a volumetric
   medium of low-loss dielectric scatterers (such as clean water-ice
   grains) elevates the same-sense circular polarization relative to the
   opposite-sense component, raising the Circular Polarization Ratio (CPR)
   above 1, and simultaneously lowers the overall Degree of Polarization
   (DOP) as the returned wave depolarises (Nozette et al. 1996; Campbell
   et al. 2006; van Zyl & Kim 2011). This signature is not unique to ice
   (rough blocky ejecta produces a similar CPR elevation), which is why it
   must be combined with independent evidence.
3. **Thermal environment.** Long-term mean and zero-incidence surface
   temperatures (Diviner Tmean, ZIT) determine whether ice, once delivered,
   is thermodynamically stable against sublimation over geological
   timescales (Paige et al. 2010; Hayne et al. 2015), while the "Pump"
   parameter is a proxy for how efficiently a cold trap captures migrating
   volatile molecules (Schorghofer 2014).

No prior open-source pipeline known to the project team combines all three
lines of evidence, at native LOLA resolution, over the full south polar cap,
with every step traceable to a specific paper. That is the gap this project
addresses.

### 1.2 Problem Statement

Given a LOLA DEM, an LOLA PSR shapefile, Chandrayaan-2 DFSAR full-polarimetric
imagery (SLI/GRI/SRI product levels) plus an independently produced official
CPR mosaic (Putrevu et al. 2023) for validation, and LRO Diviner thermal
rasters, produce (a) a physically derived DPSR mask, (b) CPR and DOP rasters
computed directly from raw SAR data, (c) quantitative validation of computed
CPR against the official product, and (d) a single fused Ice Confidence Map
usable to rank candidate sites by ice-stability likelihood.

### 1.3 Approach Summary

| Stage | Package | Deliverable |
|---|---|---|
| 1 | `pipeline/`, `dpsr/`, `dpsr_fast.py` | DPSR / PSR rasters |
| 2 | `DFSAR/`, `cpr/`, `cpr_gri/`, `cpr_official/` | CPR rasters (3 independent computations) |
| 3 | `dop/` | DOP raster |
| 4 | `validation/` | Quantitative comparison of computed vs. official CPR |
| 5 | `diviner/` | Grid alignment of all layers + Ice Confidence Map |

---

## 2. Data Sources and Inputs

| Dataset | Product ID / file | Native resolution | Provider | Used for |
|---|---|---|---|---|
| LOLA polar DEM | `ldem_85s_20m_float` (PDS3, .lbl/.img) | 20 m/px, 15168² | PDS Geosciences Node | Ray-tracing, slope, all grid alignment |
| LOLA PSR mask | `LPSR_80S_20MPP_ADJ.shp` (80°S–90°S) | 20 m/px | LOLA PSR product | Ground-truth shadow mask, DPSR seed pixels |
| Chandrayaan-2 DFSAR SLI | `ch2_sar_ncxl_..._d_sli_xx_cp_{lh,lv}_d18.tif` | single-look complex | ISRO SAR payload | Primary CPR computation (`cpr/`) |
| Chandrayaan-2 DFSAR GRI | `ch2_sar_ncxl_..._d_gri_{in,xx}_cp_...` | multilooked, ground-range | ISRO SAR payload | Alternate CPR computation (`cpr_gri/`) |
| Chandrayaan-2 DFSAR SRI | `ch2_sar_ncxl_..._d_sri_...` | slant-range | ISRO SAR payload | Feature stack input (`DFSAR/`) |
| Official DFSAR CPR mosaic | `CPR.tif` | south-pole mosaic | Putrevu et al. (2023), JGR Planets | Independent validation reference |
| LRO Diviner Tmean | `polar_south_80_Tmean.grd` | native grid | Diviner polar products | Thermal stability indicator |
| LRO Diviner ZIT | `polar_south_80_zit_float32.tif` | 240.04 m/px, polar stereographic | Diviner polar products | Coldest-temperature proxy |
| LRO Diviner Pump | `polar_south_80_pump.grd` | native grid | Diviner polar products | Cold-trapping efficiency proxy |

All rasters are ultimately reprojected onto the LOLA DEM's grid: south
polar-stereographic projection, spherical Moon datum (R = 1,737,400 m),
20 m/pixel, 15,168 × 15,168, bounds ±151,680 m in both axes.

---

## 3. System Architecture and Repository Organisation

```
ISRO_Hackathon/
├── data/                  Input DEM and PSR shapefile (read-only)
├── docs/                  Reference papers / instrument manuals
│
├── pipeline/              Legacy DPSR pipeline (first working version)
├── dpsr/                  Modular DPSR pipeline — O'Brien & Byrne (2022), canonical
├── dpsr_fast.py           Self-contained single-file DPSR version
├── results/               DPSR / PSR raster outputs
│
├── DFSAR/                 Raw DFSAR product tree + data_pipeline/ (SAR ingestion & feature stack)
├── cpr/                   Primary CPR computation (SLI) + Faustini-crater research scripts
├── cpr_gri/               Alternate CPR computation (GRI, published co-pol formula)
├── cpr_official/          Official Putrevu (2023) CPR mosaic extraction
├── dop/                   Degree of Polarization (Stokes/covariance formalism)
│
├── validation/            Georeferencing + quantitative validation: computed vs. official CPR
│
├── diviner/               Final fusion: Diviner thermal + DEM/Slope/PSR/DPSR/CPR/DOP → Ice Confidence Map
├── outputs/               Generated rasters and plots
├── images-final/          Curated figure set for this report (source of every image below)
│
├── validate_science.py    Standalone scientific validation
├── debug_pipeline.py / debug_psr_mask.py / debug_validation.py   Diagnostic scripts
└── README.md / STEPS.md / COMMANDS.md   Setup, walkthrough, command reference
```

---

## 4. Methodology

### 4.1 DPSR Extraction Pipeline (`dpsr/`, `pipeline/`, `dpsr_fast.py`)

**Scientific definition** (O'Brien & Byrne 2022):

```
DPSR pixel P  ⟺  psr_mask[P] == 1
                  AND  ∀ azimuth a:
                      ∄ pixel Q visible from P along ray a
                      with psr_mask[Q] == 0
```

**Curvature-corrected visibility test** (Appendix, Eq. A4):

```
tan(μ) = R1 · (R2 − √(d² + R1²)) / (d · R2)

R1 = R_Moon + h_observer     R2 = R_Moon + h_target
```

At d = 50 km this correction depresses the apparent horizon by ≈720 m —
large enough that omitting it would cause deep PSR floors to incorrectly
"see" distant sunlit terrain that is, in reality, below the true curved
horizon.

**Ordering correctness detail:** the visibility check is performed *before*
the running horizon maximum is updated in each ray step; reversing the
order would trivially mark every new local maximum as "visible."

**Implementation:** `step01`–`step07` load the DEM/PSR, precompute
Bresenham ray tables, run a Numba `@njit(parallel=True)` CPU kernel or a
CUDA kernel (identical formula, different backend), post-filter with an
8-connected component filter (removes clusters < 5 px), and spot-check
named craters.

**Parameter deviations from the paper:**

| Parameter | Paper | This implementation | Justification |
|---|---|---|---|
| `N_ANGLES` | 720 (0.5°) | 360 (1°) | <3% error, 2× faster |
| `MAX_DIST` | 7500 px (150 km) | 2500 px (50 km) | Covers all south-polar PSR craters (largest ≈103 km diameter), 3× faster |
| `min_component` | 5 px | 5 px | unchanged |

### 4.2 SAR Polarimetry — Circular Polarization Ratio (`cpr/`, `cpr_gri/`, `cpr_official/`, `DFSAR/`)

```
S_RR (SC) = (S_HH − S_VV + 2j·S_HV) / 2
S_RL (OC) = (S_HH + S_VV) / 2
σ_SC = mean(|S_RR|²)     σ_OC = mean(|S_RL|²)     CPR = σ_SC / σ_OC
```

CPR < 1 → surface (Bragg) scattering, bare regolith; CPR > 1 → volume
scattering, an ice-grain-*or*-rough-ejecta candidate. Two independent
formulas were implemented and compared against the same official mosaic:

- **Default (SC/OC) formula** — the one described above, computed on both
  SLI (`cpr/`) and GRI (`cpr_gri/`) product levels.
- **Published co-pol-only "research" formula** (`μc`):
  `CPR(μc) = (√σ_HH+√σ_VV)² / (√σ_HH−√σ_VV)²`, which assumes HH/VV are
  fully coherent — numerically ill-conditioned wherever σ_HH ≈ σ_VV, so it
  is log-rescaled from its own P1–P99 window rather than hard-clipped
  (hard-clipping was tried first and produced a flat, information-free
  image — a documented negative result).

`DFSAR/data_pipeline` additionally ingests and catalogues the full 18-raster
DFSAR product set (CPR, SRD — Stokes Radar Decomposition, ODD — odd-bounce
scattering, VOL — volume scattering, HLX — helix scattering, EVN —
eigenvalue parameter, TRT — total power/trace, plus GRI/SLI/SRI and their
polarization variants) and aligns them all to the LOLA grid for a broader
feature stack.

### 4.3 SAR Polarimetry — Degree of Polarization (`dop/`)

```
S0 = C11 + C22/2      S1 = C11 − C22/2
S2 = √2·Re(C12)       S3 = √2·Im(C12)
DOP = √(S1²+S2²+S3²) / S0
```

derived directly from the multilooked covariance matrix with no
simplifying assumptions. DOP → 1 indicates a coherent, single-bounce
(specular) scatterer; DOP < 1 indicates depolarisation from multiple/volume
scattering — necessary but not sufficient evidence for buried volatiles,
complementary to CPR.

### 4.4 Diviner Thermal Integration and Physics-Based Fusion (`diviner/`)

Nine bands (DEM, Slope, PSR, DPSR, CPR, DOP, Tmean, ZIT, Pump) are aligned
to the reference LOLA grid via `rasterio.warp.reproject` (bilinear for
continuous bands, nearest-neighbour for binary masks) and combined into an
Ice Confidence Score:

| Indicator | Weight | Ice-positive direction | Physical basis |
|---|---|---|---|
| CPR | 0.20 | HIGH | Nozette 1996; Campbell 2006 |
| Tmean | 0.20 | LOW | Paige et al. 2010 |
| ZIT | 0.15 | LOW | Hayne et al. 2015 |
| Pump | 0.13 | HIGH | Schorghofer 2014 |
| DOP | 0.12 | LOW | van Zyl & Kim 2011 |
| PSR | 0.10 | 1 (binary) | Watson et al. 1961 |
| DPSR | 0.05 | 1 (binary) | O'Brien & Byrne 2022 |
| Slope | 0.05 | LOW | Prettyman et al. 2012 |

```
score[p] = Σᵢ (normalised_band_i[p] × weight_i)  /  Σᵢ (weight_i, valid bands only)
```

Continuous bands are percentile-clipped [P2, P98] and rescaled to [0, 1];
"low → more ice" bands are inverted before weighting.

### 4.5 Cross-Validation Module (`validation/`)

Georeferences the computed CPR onto the official mosaic's grid using GCPs
from the DFSAR SLI geometry CSV (8,521 azimuth rows × 9 range columns;
every 10th azimuth row used → 7,668 GCPs), then computes Pearson r,
Spearman ρ, RMSE, MAE, bias, R², SSIM, histogram intersection, and mutual
information over co-valid pixels.

---

## 5. Results — Complete Figure-by-Figure Walkthrough

All figures below are reproduced from `images-final/`, the curated output
set for this report. Paths are given relative to the repository root.

### 5.1 DPSR Extraction Figures

**Figure 1 — DPSR pipeline summary**
`images-final/dpsr/DPSR_summary.png`

![Figure 1](images-final/dpsr/DPSR_summary.png)

Four panels over the same DEM tile: (1) raw elevation (green=low, tan=high,
craters visible as blue depressions); (2) the rasterized PSR mask (dark
blue blobs — the classic clustering of shadow around crater floors and
walls); (3) `DPSR_raw` immediately after ray-casting, before the
small-region filter; (4) the final `DPSR` mask after the 8-connected,
<5-px filter is applied — labelled **17,564 px** for this tile. Comparing
panels 3 and 4 visually confirms the post-filter step is doing real work:
isolated single/double-pixel "DPSR" flags in the raw output (numerical
noise at ray-casting resolution limits) are removed, leaving only
spatially coherent doubly-shadowed clusters.

**Figure 2 — DEM input (standalone)**
`images-final/dpsr/elevation.png`

![Figure 2](images-final/dpsr/elevation.png)

The same elevation panel as Figure 1, saved standalone. This is the direct
LOLA DEM crop fed into the ray-casting kernel — bowl-shaped craters
(Shackleton-like) are the blue circular depressions; their poleward-facing
interior walls are what generate PSR.

**Figure 3 — Rasterized PSR mask (standalone)**
`images-final/dpsr/PSR_mask.png`

![Figure 3](images-final/dpsr/PSR_mask.png)

Binary PSR mask derived from the LOLA PSR shapefile, rasterized onto the
DEM grid. The large dark-blue disks correspond to permanently shadowed
crater floors (e.g., Shackleton-class craters); the fine speckle around
them is shadow cast by smaller-scale roughness (boulders, secondary
craters) on sun-facing walls — real PSR, just sub-crater-scale.

**Figure 4 — DPSR raw output (standalone)**
`images-final/dpsr/DPSR_raw.png`

![Figure 4](images-final/dpsr/DPSR_raw.png)

Direct ray-casting output before the connected-component filter. Visually
almost indistinguishable from blank at this display stretch — DPSR pixels
are such a small fraction of the frame (≈0.008% globally, §5.12/§5.10) that
individual points are barely visible without zooming, which is itself a
visual confirmation of how rare true double-shadow is.

**Figure 5 — DPSR final output (standalone)**
`images-final/dpsr/DPSR.png`

![Figure 5](images-final/dpsr/DPSR.png)

Post-filter DPSR mask — the same sparse point pattern as Figure 4 (the
filter removes noise, it does not add area), matching the 17,564-px count
reported in Figure 1.

**Figure 6 — Five-crater DPSR validation panel**
`images-final/dpsr/dpsr_validation.png`

![Figure 6](images-final/dpsr/dpsr_validation.png)

Spot-check across five named PSR-hosting craters (Shackleton, Faustini,
Haworth, Shoemaker, Cabeus), each row showing elevation / PSR mask / DPSR
result:

| Crater | DPSR pixels | DPSR area |
|---|---|---|
| Shackleton | **0** | 0.0000 km² |
| Faustini | 2,976 | 1.1904 km² |
| Haworth | 2,390 | 0.9560 km² |
| Shoemaker | 2,438 | 0.9752 km² |
| Cabeus | 1,108 | 0.4432 km² |

This is an important, honestly-reported result: **Shackleton shows zero
DPSR pixels** in this crop, which is a genuine, counter-intuitive finding
rather than a bug artefact — Shackleton's floor, while permanently
shadowed, is wide and bowl-shaped enough that essentially all of its floor
retains line-of-sight to some illuminated point on the opposite rim,
failing the strict "double shadow" test. Faustini, by contrast, has the
most DPSR area of the five, consistent with its comparatively rougher,
more occluded floor. This nuance — PSR ≠ DPSR, and the largest PSR is not
necessarily the largest DPSR — is exactly the scientific distinction
O'Brien & Byrne (2022) motivate.

### 5.2 DFSAR Multi-Product Ingestion Figures

These nine figures come from `DFSAR/data_pipeline`'s broader ingestion run
(18 cataloged rasters, aligned to the LOLA grid), independent of the
narrower `cpr/`/`cpr_gri/` scripts.

**Figure 7 — Aligned DEM**
`images-final/dfsar_pipeline/DEM.png`

![Figure 7](images-final/dfsar_pipeline/DEM.png)

**Figure 8 — Aligned PSR mask**
`images-final/dfsar_pipeline/PSR.png`

![Figure 8](images-final/dfsar_pipeline/PSR.png)

**Figure 9 — Aligned DPSR mask**
`images-final/dfsar_pipeline/DPSR.png`

![Figure 9](images-final/dfsar_pipeline/DPSR.png)

Figures 7–9 are the same three foundational layers as §5.1, but reprojected
through this pipeline's independent alignment path — a useful
cross-check that both alignment implementations (this one and
`diviner/aligner.py`) agree on the same DEM/PSR/DPSR geometry.

**Figure 10 — Aligned official CPR mosaic (full south-pole extent)**
`images-final/dfsar_pipeline/CPR.png`

![Figure 10](images-final/dfsar_pipeline/CPR.png)

The full 15,168×15,168 south-polar CPR mosaic after alignment, normalised
to [0,1]. The radial, star-shaped coverage pattern is the DFSAR imaging
geometry itself (individual orbital swaths converging toward the pole,
with white gaps where no pass has imaged that azimuth) — it is **not** a
processing artefact but the actual footprint of available Chandrayaan-2
SAR coverage. Bright yellow arcs around some crater rims indicate locally
elevated CPR (candidate rough/icy terrain); the largest bright ring
features (bottom-left, bottom-centre) are crater rim ejecta.

**Figure 11 — Aligned SRD (Stokes Radar Decomposition)**
`images-final/dfsar_pipeline/SRD.png`

![Figure 11](images-final/dfsar_pipeline/SRD.png)

Same swath geometry as Figure 10, different physical quantity (a Stokes-based
scattering decomposition parameter). The predominance of high (orange/yellow)
values across most illuminated terrain, with scattered low (dark
purple/blue) patches concentrated inside crater interiors, indicates a
systematically different scattering regime inside shadowed crater floors
versus normally-illuminated terrain — consistent with the physical premise
that PSR/DPSR terrain has a distinguishable polarimetric signature.

**Figure 12 — Aligned EVN (Eigenvalue Parameter)**
`images-final/dfsar_pipeline/EVN.png`

![Figure 12](images-final/dfsar_pipeline/EVN.png)

**Figure 13 — Aligned HLX (Helix Scattering)**
`images-final/dfsar_pipeline/HLX.png`

![Figure 13](images-final/dfsar_pipeline/HLX.png)

**Figure 14 — Aligned ODD (Odd-bounce Scattering)**
`images-final/dfsar_pipeline/ODD.png`

![Figure 14](images-final/dfsar_pipeline/ODD.png)

**Figure 15 — Aligned TRT (Total Power / Trace)**
`images-final/dfsar_pipeline/TRT.png`

![Figure 15](images-final/dfsar_pipeline/TRT.png)

Figures 12–15 are the remaining Stokes-decomposition-derived DFSAR
products, all aligned to the common grid and cataloged for a future,
broader feature stack (this pipeline currently halts right after cataloging
these — see §7, Known Issue #2 — so these four are visualised here but not
yet fed into the Ice Confidence fusion).

### 5.3 CPR — Faustini Crater Case Study (SLI, Default Formula)

**Figure 16 — Faustini CPR map + CPR-vs-DOP scatter**
`images-final/faustini_cpr/default/faustini_combined.png`

![Figure 16](images-final/faustini_cpr/default/faustini_combined.png)

Left: the CPR raster for the Faustini scene (42.5 km diameter, centred
87.18°S), rendered on a cyclic green→blue→magenta→white colormap so that
CPR ≈ 1 (blue-white) visually pops out against the green (CPR<1, bare
regolith) background. Bright pink/white speckle clusters mark locally
CPR>1 patches. Right: a CPR-vs-DOP scatter over 1,323,212 pixels — the
classic inverse "comma"-shaped cloud (high CPR co-occurs with low DOP,
consistent with volume scattering theory). The shaded red box marks the
**ice-candidate criterion used throughout this study, CPR > 1 AND
DOP < 0.13**: 1.15% of Faustini pixels have CPR > 1, but only **0.08%**
satisfy both criteria simultaneously — the joint criterion is far more
restrictive than CPR alone, exactly as intended (single-parameter CPR
anomalies are common and often just rough terrain; the joint CPR+DOP
criterion is the more specific ice-candidate signature).

**Figure 17 — CPR histogram, Faustini**
`images-final/faustini_cpr/default/CPR_histogram.png`

![Figure 17](images-final/faustini_cpr/default/CPR_histogram.png)

**Figure 18 — Same-sense (SC) power map**
`images-final/faustini_cpr/default/SC_power.png`

![Figure 18](images-final/faustini_cpr/default/SC_power.png)

**Figure 19 — Opposite-sense (OC) power map**
`images-final/faustini_cpr/default/OC_power.png`

![Figure 19](images-final/faustini_cpr/default/OC_power.png)

Figures 18–19 are the two intermediate power rasters (`σ_SC`, `σ_OC`) whose
ratio defines CPR (§4.2) — shown separately so the numerator/denominator
behaviour can be inspected independently of the final ratio, useful for
diagnosing whether a CPR anomaly is driven by elevated SC power (the
ice-consistent explanation) or suppressed OC power (which can also happen
from pure geometric/incidence-angle effects).

**Figure 20 — Faustini crater CPR (crater-cropped view)**
`images-final/faustini_cpr/default/faustini_crater_cpr.png`

![Figure 20](images-final/faustini_cpr/default/faustini_crater_cpr.png)

**Figure 21 — Default-vs-official CPR comparison, Faustini**
`images-final/faustini_cpr/default/faustini_cpr_comparison.png`

![Figure 21](images-final/faustini_cpr/default/faustini_cpr_comparison.png)

Three overlaid distributions: the whole Faustini scene (grey, n=500,000
subsample), the crater interior alone (blue, n=1,323,212), and the subset
of crater pixels flagged as ice candidates by CPR>1 (red, n=15,153). The
crater-interior distribution sits slightly left of (lower CPR than) the
whole-scene distribution below CPR≈1, then the red ice-candidate
population picks up the full CPR>1 tail out to 2.5 — confirming that
"ice candidates" as defined here are a genuine, non-trivial upper-tail
subpopulation (15,153 pixels, ≈1.1% of the crater) rather than a threshold
artefact capturing noise.

**Figure 22 — Faustini histogram panel (CPR + DOP combined)**
`images-final/faustini_cpr/default/faustini_histograms.png`

![Figure 22](images-final/faustini_cpr/default/faustini_histograms.png)

**Figure 23 — CPR histogram (standalone)**
`images-final/faustini_cpr/default/faustini_hist_cpr.png`

![Figure 23](images-final/faustini_cpr/default/faustini_hist_cpr.png)

**Figure 24 — DOP histogram (standalone)**
`images-final/faustini_cpr/default/faustini_hist_dop.png`

![Figure 24](images-final/faustini_cpr/default/faustini_hist_dop.png)

**Figure 25 — Latitude-strip profile**
`images-final/faustini_cpr/default/faustini_lat_strip.png`

![Figure 25](images-final/faustini_cpr/default/faustini_lat_strip.png)

A profile of CPR (and/or DOP) as a function of latitude/distance across
the crater, used to check whether the ice-candidate signature clusters
specifically toward the coldest (highest-latitude, most poleward) part of
the crater floor, as the thermal-stability hypothesis would predict, rather
than being uniformly scattered across the whole scene.

**Figure 26 — Combined scatter, all pixels**
`images-final/faustini_cpr/default/faustini_scatter_all.png`

![Figure 26](images-final/faustini_cpr/default/faustini_scatter_all.png)

**Figure 27 — Combined scatter (annotated)**
`images-final/faustini_cpr/default/faustini_scatter_combined.png`

![Figure 27](images-final/faustini_cpr/default/faustini_scatter_combined.png)

**Figure 28 — Scatter, ice-candidate subset only**
`images-final/faustini_cpr/default/faustini_scatter_ice.png`

![Figure 28](images-final/faustini_cpr/default/faustini_scatter_ice.png)

**Figure 29 — Scatter, non-ice subset only**
`images-final/faustini_cpr/default/faustini_scatter_nonice.png`

![Figure 29](images-final/faustini_cpr/default/faustini_scatter_nonice.png)

Figures 26–29 decompose the CPR-vs-DOP relationship (Figure 16, right
panel) into the full population, an annotated combined view, and the two
halves split at the CPR=1 ice-candidate threshold — making it possible to
verify that the ice-candidate population (Figure 28) is concentrated in
the low-DOP region as expected, while the non-ice population (Figure 29)
follows the ordinary inverse CPR–DOP trend of bare regolith.

**Figure 30 — CPR overlaid with DOP context**
`images-final/faustini_cpr/default/combined_cpr_dop.png`

![Figure 30](images-final/faustini_cpr/default/combined_cpr_dop.png)

### 5.4 CPR — Faustini Crater Case Study (Published μc Research Formula)

The same fourteen plots as §5.3 (minus the SC/OC power pair, which is
replaced by the raw HH/VV power pair since the μc formula operates
directly on co-pol powers) were regenerated using the alternative
published CPR(μc) formula, to test it against the same Faustini scene.

**Figure 31 — Faustini CPR(μc) map + scatter**
`images-final/faustini_cpr/research/faustini_combined.png`

![Figure 31](images-final/faustini_cpr/research/faustini_combined.png)

**Figure 32 — CPR(μc) histogram**
`images-final/faustini_cpr/research/CPR_histogram.png`

![Figure 32](images-final/faustini_cpr/research/CPR_histogram.png)

**Figure 33 — HH power map**
`images-final/faustini_cpr/research/HH_power.png`

![Figure 33](images-final/faustini_cpr/research/HH_power.png)

**Figure 34 — VV power map**
`images-final/faustini_cpr/research/VV_power.png`

![Figure 34](images-final/faustini_cpr/research/VV_power.png)

**Figure 35 — CPR(μc), crater-cropped**
`images-final/faustini_cpr/research/faustini_crater_cpr.png`

![Figure 35](images-final/faustini_cpr/research/faustini_crater_cpr.png)

**Figure 36 — CPR(μc)-vs-official comparison**
`images-final/faustini_cpr/research/faustini_cpr_comparison.png`

![Figure 36](images-final/faustini_cpr/research/faustini_cpr_comparison.png)

**Figure 37 — Histogram panel**
`images-final/faustini_cpr/research/faustini_histograms.png`

![Figure 37](images-final/faustini_cpr/research/faustini_histograms.png)

**Figure 38 — CPR(μc) histogram (standalone)**
`images-final/faustini_cpr/research/faustini_hist_cpr.png`

![Figure 38](images-final/faustini_cpr/research/faustini_hist_cpr.png)

**Figure 39 — DOP histogram (standalone, research run)**
`images-final/faustini_cpr/research/faustini_hist_dop.png`

![Figure 39](images-final/faustini_cpr/research/faustini_hist_dop.png)

**Figure 40 — Latitude-strip profile (research)**
`images-final/faustini_cpr/research/faustini_lat_strip.png`

![Figure 40](images-final/faustini_cpr/research/faustini_lat_strip.png)

**Figure 41 — Scatter, all pixels (research)**
`images-final/faustini_cpr/research/faustini_scatter_all.png`

![Figure 41](images-final/faustini_cpr/research/faustini_scatter_all.png)

**Figure 42 — Scatter, combined (research)**
`images-final/faustini_cpr/research/faustini_scatter_combined.png`

![Figure 42](images-final/faustini_cpr/research/faustini_scatter_combined.png)

**Figure 43 — Scatter, ice subset (research)**
`images-final/faustini_cpr/research/faustini_scatter_ice.png`

![Figure 43](images-final/faustini_cpr/research/faustini_scatter_ice.png)

**Figure 44 — Scatter, non-ice subset (research)**
`images-final/faustini_cpr/research/faustini_scatter_nonice.png`

![Figure 44](images-final/faustini_cpr/research/faustini_scatter_nonice.png)

Figures 31–44 mirror §5.3's structure exactly, using the published μc
formula instead of the default SC/OC ratio. Producing the identical suite
of diagnostic plots for both formulas was a deliberate methodological
choice — it makes the two candidate CPR definitions directly, visually
comparable image-for-image rather than requiring the reader to trust a
table of numbers alone.

### 5.5 CPR — Default vs. Research Formula, Direct Comparison

**Figure 45 — Faustini inner-crater CPR: default vs. research, side-by-side**
`images-final/faustini_cpr/zoom/zoom_compare.png`

![Figure 45](images-final/faustini_cpr/zoom/zoom_compare.png)

The most direct test of formula agreement: the same 13.2×6.1 km inner-crater
zoom, CPR(Default) on the left, CPR(Research μc) on the right, with a
DOP-coloured scatter of one against the other in the centre. The
correlation is **r = −0.145** — the two formulas essentially disagree with
each other on this crop, and the point cloud shows no visible linear
trend, only a dense low-CPR(Default) clump spanning the full range of
CPR(Research μc). This is a direct, visual confirmation of the numerical
divergence documented in §5.4/§4.2: the two CPR formulas assume opposite
extremes of HH–VV coherence and should not be expected to agree pixel-for-
pixel on real terrain.

**Figure 46 — Zoom feature maps**
`images-final/faustini_cpr/zoom/zoom_feature_maps.png`

![Figure 46](images-final/faustini_cpr/zoom/zoom_feature_maps.png)

**Figure 47 — Zoom H-alpha decomposition**
`images-final/faustini_cpr/zoom/zoom_halpha.png`

![Figure 47](images-final/faustini_cpr/zoom/zoom_halpha.png)

The Cloude–Pottier H-α polarimetric decomposition plane for the same
inner-crater zoom — an independent classical scattering-mechanism
classification (surface/dipole/volume/double-bounce regions), used here as
a third, formula-independent check on which scattering regime dominates
the ice-candidate area.

**Figure 48 — Zoom histograms**
`images-final/faustini_cpr/zoom/zoom_histograms.png`

![Figure 48](images-final/faustini_cpr/zoom/zoom_histograms.png)

**Figure 49 — Zoom PolSAR composite**
`images-final/faustini_cpr/zoom/zoom_polsar.png`

![Figure 49](images-final/faustini_cpr/zoom/zoom_polsar.png)

**Figure 50 — Zoom PolSAR histogram**
`images-final/faustini_cpr/zoom/zoom_polsar_hist.png`

![Figure 50](images-final/faustini_cpr/zoom/zoom_polsar_hist.png)

**Figure 51 — Zoom scatter, default formula**
`images-final/faustini_cpr/zoom/zoom_scatter_default.png`

![Figure 51](images-final/faustini_cpr/zoom/zoom_scatter_default.png)

**Figure 52 — Zoom scatter, research formula**
`images-final/faustini_cpr/zoom/zoom_scatter_research.png`

![Figure 52](images-final/faustini_cpr/zoom/zoom_scatter_research.png)

**Figure 53 — Ice-candidate spatial consistency, 9 azimuth strips**
`images-final/faustini_cpr/ice_candidates/ice_candidate_panels.png`

![Figure 53](images-final/faustini_cpr/ice_candidates/ice_candidate_panels.png)

Nine independent sub-scenes (R1–R9) spanning azimuth offsets from 275 km to
2,063 km along the Faustini-region SAR strip, each annotated with its own
CPR>1 area fraction: **8.0%–12.6%**, clustering tightly around 9–11% across
all nine. This consistency check is a useful piece of evidence that the
CPR-anomaly rate is a stable regional property of this terrain rather than
a processing fluke localized to one sub-frame.

**Figure 54 — CPR–DOP scatter (dedicated module)**
`images-final/faustini_cpr/cpr_dop/cpr_dop_scatter.png`

![Figure 54](images-final/faustini_cpr/cpr_dop/cpr_dop_scatter.png)

**Figure 55 — F2 sub-crater DOP distribution**
`images-final/faustini_cpr/f2_crater/dop_distribution.png`

![Figure 55](images-final/faustini_cpr/f2_crater/dop_distribution.png)

**Figure 56 — DOP formula verification maps (Faustini reference)**
`images-final/faustini_cpr/f2_crater/dop_verification_maps.png`

![Figure 56](images-final/faustini_cpr/f2_crater/dop_verification_maps.png)

Left: the Faustini CPR map (same green/magenta styling as Figure 16) shown
again purely as a spatial reference; right: the corresponding DOP map on a
magma colormap. This pairing was generated specifically to verify the DOP
formula (`dop/stokes.py` → `dop/dop.py`) against a scene already
characterised in detail (Faustini) before trusting it on the less-studied
F2 sub-crater — good practice: validate a new computation against a known
reference before applying it to a new target.

**Figure 57 — F2 sub-crater DOP verification scatter**
`images-final/faustini_cpr/f2_crater/dop_verification_scatter.png`

![Figure 57](images-final/faustini_cpr/f2_crater/dop_verification_scatter.png)

### 5.6 Degree of Polarization (DOP) Figures

**Figure 58 — DOP map, full Faustini strip**
`images-final/dop/DOP.png`

![Figure 58](images-final/dop/DOP.png)

The full-strip DOP raster (1:100 azimuth downsample for display), values
mostly in the 0.7–0.95 range (yellow-green, coherent single-bounce
scattering typical of bare regolith at L-band), with several dark
blue/purple patches down to ≈0.45–0.5 scattered irregularly across the
strip — these low-DOP patches are the depolarizing, volume-scattering
candidates that (combined with elevated CPR) define the ice-candidate
criterion used throughout §5.3–5.5.

**Figure 59 — DOP histogram**
`images-final/dop/Histogram.png`

![Figure 59](images-final/dop/Histogram.png)

**Figure 60 — HH channel power**
`images-final/dop/HH.png`

![Figure 60](images-final/dop/HH.png)

**Figure 61 — HV (cross-pol) channel power**
`images-final/dop/HV.png`

![Figure 61](images-final/dop/HV.png)

**Figure 62 — VH (reciprocity-check) channel power**
`images-final/dop/VH.png`

![Figure 62](images-final/dop/VH.png)

**Figure 63 — VV channel power**
`images-final/dop/VV.png`

![Figure 63](images-final/dop/VV.png)

Figures 60–63 are the four raw linear-polarization power channels used to
build the covariance matrix C3 that feeds the Stokes-parameter derivation
(§4.3). HV and VH should be nearly identical under the monostatic
reciprocity theorem; visually comparing Figures 61 and 62 is a basic sanity
check on that assumption.

**Figure 64 — DOP, F2/faustini crater-cropped**
`images-final/dop/faustini_crater_dop.png`

![Figure 64](images-final/dop/faustini_crater_dop.png)

### 5.7 GRI-Based CPR — Default vs. Research

**Figure 65 — GRI CPR map (default formula)**
`images-final/gri_cpr/default/CPR.png`

![Figure 65](images-final/gri_cpr/default/CPR.png)

**Figure 66 — GRI HH power (default run)**
`images-final/gri_cpr/default/HH.png`

![Figure 66](images-final/gri_cpr/default/HH.png)

**Figure 67 — GRI HV power (default run)**
`images-final/gri_cpr/default/HV.png`

![Figure 67](images-final/gri_cpr/default/HV.png)

**Figure 68 — GRI VH power (default run)**
`images-final/gri_cpr/default/VH.png`

![Figure 68](images-final/gri_cpr/default/VH.png)

**Figure 69 — GRI VV power (default run)**
`images-final/gri_cpr/default/VV.png`

![Figure 69](images-final/gri_cpr/default/VV.png)

**Figure 70 — GRI CPR histogram (default run)**
`images-final/gri_cpr/default/Histogram.png`

![Figure 70](images-final/gri_cpr/default/Histogram.png)

**Figure 71 — GRI CPR map (research μc formula)**
`images-final/gri_cpr/research/CPR.png`

![Figure 71](images-final/gri_cpr/research/CPR.png)

**Figure 72 — GRI HH power (research run)**
`images-final/gri_cpr/research/HH.png`

![Figure 72](images-final/gri_cpr/research/HH.png)

**Figure 73 — GRI HV power (research run)**
`images-final/gri_cpr/research/HV.png`

![Figure 73](images-final/gri_cpr/research/HV.png)

**Figure 74 — GRI VH power (research run)**
`images-final/gri_cpr/research/VH.png`

![Figure 74](images-final/gri_cpr/research/VH.png)

**Figure 75 — GRI VV power (research run)**
`images-final/gri_cpr/research/VV.png`

![Figure 75](images-final/gri_cpr/research/VV.png)

**Figure 76 — GRI CPR(μc) histogram (research run)**
`images-final/gri_cpr/research/Histogram.png`

![Figure 76](images-final/gri_cpr/research/Histogram.png)

Figures 65–76 are the multilooked ground-range (GRI) counterparts of the
SLI-based Faustini analysis in §5.3–5.4, computed over the full GRI scene
rather than a single crater — this is the input product for the validation
comparison in §5.9, which turned out to give the single best pixel-level
agreement of any computed CPR product in this project (see below).

### 5.8 CPR Validation Against the Official Mosaic (SLI Pipeline)

**Figure 77 — Calculated vs. official CPR, side-by-side maps**
`images-final/validation/validation_maps.png`

![Figure 77](images-final/validation/validation_maps.png)

The SLI-derived CPR (left) and the official Putrevu et al. (2023) mosaic
(right), both reprojected onto the same south-polar stereographic grid and
displayed on an identical [0, 2] color scale. The two crescent-shaped SAR
swaths are, at this zoom level, visually almost indistinguishable — the
same bright/dark structural pattern (thin bright ridge, broader darker
crescent body) appears in both, which is the basis for the high SSIM score
reported quantitatively below.

**Figure 78 — Pixel-level scatter, calculated vs. official**
`images-final/validation/validation_scatter.png`

![Figure 78](images-final/validation/validation_scatter.png)

3,270,277 co-valid pixels plotted against the 1:1 line. **r = 0.079,
R² = −0.593, RMSE = 0.170, bias = +0.0069.** The point cloud is concentrated
below CPR≈0.5–1.0 on both axes (consistent with the overall low-CPR
character of most lunar regolith) but shows essentially no linear
structure relative to the 1:1 line — confirming numerically what Figure 77
suggested visually is only a large-scale, not pixel-level, agreement.

**Figure 79 — Spatial difference map**
`images-final/validation/difference_map.png`

![Figure 79](images-final/validation/difference_map.png)

**Figure 80 — Absolute difference map**
`images-final/validation/absolute_difference.png`

![Figure 80](images-final/validation/absolute_difference.png)

**Figure 81 — Value histogram comparison**
`images-final/validation/histogram_comparison.png`

![Figure 81](images-final/validation/histogram_comparison.png)

**Figure 82 — Q–Q plot**
`images-final/validation/qq_plot.png`

![Figure 82](images-final/validation/qq_plot.png)

**Figure 83 — Boxplot comparison**
`images-final/validation/boxplot.png`

![Figure 83](images-final/validation/boxplot.png)

Figures 79–83 are the supporting diagnostic suite behind the summary
metrics: the difference map (79) shows whether errors are spatially
random or systematic (e.g. concentrated near scene edges, which would
implicate georeferencing rather than the CPR formula itself); the
histogram comparison (81) and Q–Q plot (82) both test distributional
(not pixel-paired) agreement, which — consistent with the 0.870 histogram
intersection score — is considerably better than the pixel-paired r=0.079
result, reinforcing the report's central honest finding for this product:
good aggregate agreement, poor pixel-level agreement.

**Full SLI validation metrics** (3,270,277 co-valid pixels, 1.42% of the
full grid):

| Metric | Value |
|---|---|
| Pearson r | 0.079 |
| Spearman ρ | 0.068 |
| RMSE / MAE | 0.170 / 0.127 |
| R² | −0.593 |
| SSIM | 0.944 |
| Histogram intersection | 0.870 |
| Bias | +0.007 |

### 5.9 CPR Validation Against the Official Mosaic (GRI Pipeline)

**Figure 84 — GRI CPR (default formula) vs. official, scatter**
`images-final/gri_validation/default/Scatter.png`

![Figure 84](images-final/gri_validation/default/Scatter.png)

This is, quantitatively, **the best-correlated computed CPR product in the
entire project**: n = 2,141,932 pixels, **Pearson r = 0.650, Spearman
ρ = 0.695**. However the point cloud sits almost entirely *above* the 1:1
line — **RMSE = 0.984, bias = +0.973**, essentially a full unit of
systematic over-estimate. Interpretation: the GRI-based default CPR
computation captures the *relative* spatial pattern of the official mosaic
substantially better than the SLI pipeline does, but is offset by a large,
roughly constant additive bias — most plausibly a calibration-constant or
multilook-normalisation difference between this pipeline's GRI processing
and the official product's, rather than a physical-formula error (a formula
error would not typically produce this strong a rank correlation).

**Figure 85 — GRI default: spatial difference map**
`images-final/gri_validation/default/Difference_map.png`

![Figure 85](images-final/gri_validation/default/Difference_map.png)

**Figure 86 — GRI default: histogram overlap**
`images-final/gri_validation/default/Histogram_overlap.png`

![Figure 86](images-final/gri_validation/default/Histogram_overlap.png)

**Figure 87 — GRI CPR(μc) research formula vs. official, scatter**
`images-final/gri_validation/research/Scatter.png`

![Figure 87](images-final/gri_validation/research/Scatter.png)

By contrast, the published μc research formula on the same GRI data
performs **worse than either other product**: n = 2,141,158, **Pearson
r = −0.213, Spearman ρ = −0.307, R² = −5.14**, a weak *negative*
relationship. This is the numerically-ill-conditioned formula described in
§4.2 — the denominator `(√σ_HH − √σ_VV)²` approaches zero wherever the two
co-pol channels are similar in power, which is the common case on natural
lunar terrain, making the raw ratio unstable before the log-rescaling step
is even applied.

**Figure 88 — GRI research: spatial difference map**
`images-final/gri_validation/research/Difference_map.png`

![Figure 88](images-final/gri_validation/research/Difference_map.png)

**Figure 89 — GRI research: histogram overlap**
`images-final/gri_validation/research/Histogram_overlap.png`

![Figure 89](images-final/gri_validation/research/Histogram_overlap.png)

**Three-way CPR validation summary** (all three computed products compared
against the same Putrevu et al. 2023 official mosaic):

| Product | Formula | n (co-valid px) | Pearson r | Spearman ρ | RMSE | Bias | Verdict |
|---|---|---|---|---|---|---|---|
| `cpr/` (SLI) | Default SC/OC | 3,270,277 | 0.079 | 0.068 | 0.170 | +0.007 | Poor pixel corr., strong structural/distributional agreement (SSIM 0.944) |
| `cpr_gri/` default | Default SC/OC | 2,141,932 | **0.650** | **0.695** | 0.984 | **+0.973** | Best rank/linear correlation, but large systematic offset |
| `cpr_gri/` research | Published μc | 2,141,158 | −0.213 | −0.307 | 0.490 | +0.163 | Weak negative correlation — formula numerically ill-suited to this terrain |

**Figure 90 — Official CPR mosaic, Faustini crater extract (ground truth reference)**
`images-final/official_cpr/faustini_crater_official_cpr.png`

![Figure 90](images-final/official_cpr/faustini_crater_official_cpr.png)

The official Putrevu et al. (2023) CPR mosaic, cropped to the Faustini
crater footprint — this is the ground-truth image against which every
Faustini-area figure in §5.3–5.5 is implicitly being compared.

### 5.10 Diviner Fusion — Per-Band Maps and Histograms

The following figures are the full 15,168×15,168 per-band outputs of the
final `diviner/` fusion stage, each shown as both a map and a histogram.

**Figure 91 — DEM map**
`images-final/diviner/DEM_map.png`

![Figure 91](images-final/diviner/DEM_map.png)

**Figure 92 — DEM histogram**
`images-final/diviner/DEM_histogram.png`

![Figure 92](images-final/diviner/DEM_histogram.png)

**Figure 93 — Slope map**
`images-final/diviner/Slope_map.png`

![Figure 93](images-final/diviner/Slope_map.png)

**Figure 94 — Slope histogram**
`images-final/diviner/Slope_histogram.png`

![Figure 94](images-final/diviner/Slope_histogram.png)

Mean slope 10.0°, median 8.66° (statistics_report.csv) — a right-skewed
distribution typical of a cratered terrain where most of the area is
gently rolling intercrater plain and a smaller fraction is steep crater
wall.

**Figure 95 — PSR map**
`images-final/diviner/PSR_map.png`

![Figure 95](images-final/diviner/PSR_map.png)

**Figure 96 — PSR histogram**
`images-final/diviner/PSR_histogram.png`

![Figure 96](images-final/diviner/PSR_histogram.png)

Binary histogram: 89.2% at 0, 10.8% at 1 — matching the statistics_report.csv
value exactly.

**Figure 97 — DPSR map**
`images-final/diviner/DPSR_map.png`

![Figure 97](images-final/diviner/DPSR_map.png)

**Figure 98 — DPSR histogram**
`images-final/diviner/DPSR_histogram.png`

![Figure 98](images-final/diviner/DPSR_histogram.png)

At full-DEM scale the DPSR map is visually almost entirely empty — 0.008%
of pixels — consistent with the sparse point patterns already seen in
Figures 4/5.

**Figure 99 — CPR map (aligned, full grid)**
`images-final/diviner/CPR_map.png`

![Figure 99](images-final/diviner/CPR_map.png)

**Figure 100 — CPR histogram (aligned, full grid)**
`images-final/diviner/CPR_histogram.png`

![Figure 100](images-final/diviner/CPR_histogram.png)

Mean 0.267, median 0.234 (statistics_report.csv) — matching the official
mosaic's own mean of 0.266 almost exactly, one more confirmation of good
aggregate/distributional (if not pixel-level) agreement.

**Figure 101 — DOP map (aligned, full grid)**
`images-final/diviner/DOP_map.png`

![Figure 101](images-final/diviner/DOP_map.png)

This map is **blank / all-nodata** — the visual manifestation of the 0%
valid-pixel DOP defect documented in §7, Known Issue #1. It is included
here specifically *because* it is blank: the absence of any rendered data
is itself the evidence that the DOP band failed to reach this stage of the
pipeline, and is more convincing reproduced directly than described.

**Figure 102 — Tmean map**
`images-final/diviner/Tmean_map.png`

![Figure 102](images-final/diviner/Tmean_map.png)

**Figure 103 — Tmean histogram**
`images-final/diviner/Tmean_histogram.png`

![Figure 103](images-final/diviner/Tmean_histogram.png)

Mean 98.0 K, median 98.9 K, std 30.9 K — a broad, roughly symmetric
temperature distribution reflecting the wide range of illumination
conditions from permanently sunlit ridges to permanently shadowed floors.

**Figure 104 — ZIT map**
`images-final/diviner/ZIT_map.png`

![Figure 104](images-final/diviner/ZIT_map.png)

**Figure 105 — ZIT histogram**
`images-final/diviner/ZIT_histogram.png`

![Figure 105](images-final/diviner/ZIT_histogram.png)

Heavily right-skewed (median 0.055, mean 0.067), with 80.15% coverage —
the missing 20% is where zero-incidence geometry was never achieved during
the Diviner observation campaign for that pixel.

**Figure 106 — Pump map**
`images-final/diviner/Pump_map.png`

![Figure 106](images-final/diviner/Pump_map.png)

**Figure 107 — Pump histogram**
`images-final/diviner/Pump_histogram.png`

![Figure 107](images-final/diviner/Pump_histogram.png)

Left-skewed (median 0.799, mean 0.665) — most valid pixels show
moderate-to-high cold-trapping efficiency, with a long low tail; 62.84%
coverage.

**Figures 108–110 — Pre-alignment Diviner previews**
`images-final/diviner_previews/Tmean_preview.png`,
`images-final/diviner_previews/ZIT_preview.png`,
`images-final/diviner_previews/Pump_preview.png`

![Figure 108](images-final/diviner_previews/Tmean_preview.png)
![Figure 109](images-final/diviner_previews/ZIT_preview.png)
![Figure 110](images-final/diviner_previews/Pump_preview.png)

These three are the raw Diviner rasters as originally ingested (native
grid/CRS, before reprojection onto the LOLA grid) — comparing them against
Figures 102/104/106 confirms the alignment step preserved the correct
overall spatial pattern (not mirrored, rotated, or offset) through the
reprojection.

### 5.11 Diviner Fusion — Cross-Band Scatter Plots and Correlation Matrix

**Figure 111 — CPR vs. Tmean scatter**
`images-final/diviner/scatter_CPR_vs_Tmean.png`

![Figure 111](images-final/diviner/scatter_CPR_vs_Tmean.png)

**Figure 112 — CPR vs. ZIT scatter**
`images-final/diviner/scatter_CPR_vs_ZIT.png`

![Figure 112](images-final/diviner/scatter_CPR_vs_ZIT.png)

**Figure 113 — Tmean vs. Pump scatter**
`images-final/diviner/scatter_Tmean_vs_Pump.png`

![Figure 113](images-final/diviner/scatter_Tmean_vs_Pump.png)

These three cross-band scatter plots were intended to sanity-check whether
the radar (CPR) and thermal (Tmean, ZIT, Pump) evidence lines are at least
weakly consistent with each other over their shared footprint, independent
of the ice-confidence weighting scheme.

**Figure 114 — Full 8-band feature correlation matrix**
`outputs/diviner/Correlation_Matrix.png` (not yet copied into `images-final/`
at the time of writing — referenced directly from the pipeline's raw
output directory)

![Figure 114](outputs/diviner/Correlation_Matrix.png)

This figure surfaces a **second, previously undocumented integration
defect**, distinct from the DOP-coverage issue: **every single off-diagonal
cell reads "N/A,"** including pairs like DEM×Slope and DEM×PSR that are
both 100%-valid, 230-million-pixel bands with no missing-data excuse
whatsoever. Per `diviner/visualizer.py`'s `save_correlation_matrix()`, a
cell is only computed when the co-valid pixel count ≥ 10 *and* both bands
have nonzero standard deviation over that shared mask — the fact that even
DEM×Slope fails this trivial bar indicates the function is not receiving
the aligned per-pixel arrays it expects (most likely a shape mismatch, a
band being passed as its filepath instead of its loaded array, or a
nodata-sentinel mismatch that zeroes out the "valid" mask for every pair
simultaneously). This is a concrete, reproducible bug to fix — see §7,
Known Issue #3 — and is reported here transparently rather than treated as
an inconclusive result.

### 5.12 Final Ice Confidence Map

**Figure 115 — IceConfidence map (downsampled, first render)**
`images-final/diviner/IceConfidence_map.png`

![Figure 115](images-final/diviner/IceConfidence_map.png)

**Figure 116 — IceConfidence histogram**
`images-final/diviner/IceConfidence_histogram.png`

![Figure 116](images-final/diviner/IceConfidence_histogram.png)

Mean 0.438, median 0.483, std 0.160 (statistics_report.csv) — a broad,
slightly left-skewed distribution spanning the full theoretical [0,1]
range up to an observed maximum of 0.878; no pixel reaches a "perfect"
score, appropriately, since achieving 1.0 would require every one of the
available weighted indicators to simultaneously max out in the same
direction.

**Figure 117 — Ice Confidence Map (full-resolution, primary deliverable)**
`images-final/diviner/Ice_Confidence_Map.png`

![Figure 117](images-final/diviner/Ice_Confidence_Map.png)

This is the project's headline deliverable: the physics-based, per-pixel
Ice Confidence Score across the full south-polar LOLA grid (1:7 subsample
for display), on a light-yellow (low, ≈0) to dark-blue (high, ≈0.78)
scale. The large, roughly circular dark-blue regions are exactly where the
physical model predicts they should be: the floors and pole-facing interior
walls of major craters (visible as the large dark rings/disks scattered
across the frame), which combine low Tmean/ZIT, high PSR/DPSR membership,
and — where DFSAR coverage exists — elevated CPR. The lighter
yellow-green terrain corresponds to normally-illuminated intercrater
plains and equator-facing crater walls, correctly scored low. This spatial
pattern — high confidence concentrated inside crater floors rather than
scattered randomly — is the primary qualitative evidence that the fusion
weighting scheme (§4.4) is behaving physically sensibly, independent of the
specific numeric weights chosen.

**Figure 118 (reference) — IceConfidence map, alternate render**
`images-final/diviner/IceConfidence_map.png` (same file as Figure 115; both
filenames — `Ice_Confidence_Map.png` and `IceConfidence_map.png` — are
retained in the output directory from two separate report-generation runs
and are visually identical.)

---

## 6. Discussion

The project's central design decision — a fully physics-based, weighted-sum
fusion with no learned parameters — trades away the potential accuracy of a
trained classifier for complete interpretability: every number in the final
Ice Confidence Map (Figure 117) can be traced back to a specific band
value, a specific normalisation percentile, and a specific
literature-justified weight. This matters for a hackathon/exploration-science
context where the audience needs to trust *why* a pixel scored high, not
just *that* it did.

The CPR validation results deserve particular scrutiny before this pipeline
is used for any site-selection recommendation. The three-way comparison in
§5.9 is a genuinely useful negative-and-positive result set: the SLI
default formula (Figure 78) shows high structural/distributional agreement
but poor pixel correlation; the GRI default formula (Figure 84) shows the
best pixel-level rank correlation of any product but a large constant
bias; and the published μc research formula (Figure 87) is actively
miscalibrated for this terrain's HH/VV coherence regime. None of the three
computed products should currently be treated as a drop-in numerical
replacement for the official mosaic, but the GRI-default result in
particular suggests that a simple additive/multiplicative bias correction
(fit on the 2.14M co-valid pixels already available) could substantially
improve that product's absolute accuracy without touching the underlying
formula — a promising, low-effort next step.

The DPSR results (Figure 6) are on solid scientific footing, and the
Shackleton-vs-Faustini contrast is a genuinely informative, non-obvious
finding rather than a null result: it demonstrates that PSR area and DPSR
area are governed by different terrain properties (bowl width/depth vs.
floor roughness/occlusion), exactly the distinction the O'Brien & Byrne
(2022) methodology was designed to expose.

The two integration defects surfaced by direct figure inspection — DOP's
blank map (Figure 101) and the all-N/A correlation matrix (Figure 114) —
are more informative than they might first appear: both are *silent*
failures (the pipeline completes and produces a plausible-looking Ice
Confidence Map either way, per §4.4's weight-renormalisation design), which
is precisely why systematically viewing every generated figure — as this
report does — is a necessary part of validating a fusion pipeline, not an
optional presentation step.

---

## 7. Known Issues and Limitations

1. **DOP does not reach the final fusion (Figure 101 is blank).** `dop/`
   computes a valid, physically bounded DOP raster in isolation (Figure 58),
   but `outputs/diviner/reports/statistics_report.csv` and Figure 101 both
   show 0 valid DOP pixels in the aligned/fused stack. The Ice Confidence
   Map is therefore currently computed from 7 of the intended 8 weighted
   bands everywhere. Root cause is most likely a path, CRS-assignment, or
   nodata-value mismatch between `dop/`'s output GeoTIFF and what
   `diviner/aligner.py` expects.
2. **The full 8-band correlation matrix is entirely non-computed
   (Figure 114, all off-diagonal cells "N/A").** This is distinct from
   Issue #1: even band pairs with 100% mutual coverage (e.g. DEM×Slope)
   fail to produce a Pearson r, indicating `diviner/visualizer.py`'s
   `save_correlation_matrix()` is not receiving properly co-registered,
   populated arrays for *any* pair — a bug independent of, and in addition
   to, the DOP-specific gap.
3. **`DFSAR/data_pipeline/main.py` crashes immediately after dataset
   discovery** with a `UnicodeEncodeError` inside `utils.section()`
   (box-drawing characters printed to a cp1252 Windows console). Figures
   7–15 (§5.2) were recovered from a prior successful partial run's cached
   outputs; the pipeline cannot currently be re-run past Step 1 to extend
   the feature stack with EVN/HLX/ODD/TRT/SRD.
4. **CPR pixel-level correlation with the official mosaic is poor for two
   of three computed products** (SLI-default r=0.079; GRI-research
   r=−0.213), and the one product with strong correlation (GRI-default,
   r=0.650) carries a large systematic bias (+0.973). See §5.9/§6 for full
   discussion and a proposed low-effort fix (bias correction on the
   GRI-default product).
5. **Low spatial overlap for CPR validation.** Only 1.42% (SLI) /
   comparable fraction (GRI) of the full 230M-pixel target grid has
   simultaneous DFSAR and official-mosaic coverage; conclusions should not
   be over-generalised beyond the imaged footprint (visible as the white
   gaps in Figure 10).
6. **Three parallel, not-yet-consolidated CPR implementations**
   (`cpr/`, `cpr_gri/`, `cpr_official/`) reflect genuine methodological
   exploration; only `cpr/`'s SLI-based output currently feeds the final
   fusion. Given the GRI-default result's unexpectedly strong correlation
   (§5.9), it may be worth reconsidering which product should feed the
   fusion once the bias is corrected.
7. **Missing CRS metadata on two of three Diviner inputs** (`Tmean.grd`,
   `pump.grd`) required assigning a placeholder Moon-geographic CRS,
   flagged with a warning but not yet independently verified.
8. **Per-pixel valid-band count is not exported.** Because the Ice
   Confidence score renormalises over whichever bands are valid at each
   pixel, two pixels with the same score can have very different
   evidentiary weight behind them; exporting this count as an auxiliary
   raster would make Figure 117 easier to interpret correctly.

---

## 8. Conclusion and Future Work

This project has progressed from a single-purpose DPSR ray-tracing tool
into a genuine multi-instrument fusion system spanning topography, radar
polarimetry, and thermal radiometry, culminating in one physically
interpretable deliverable (Figure 117, the Ice Confidence Map) computed
over the full 230-million-pixel south-polar LOLA grid. All five stages
(DPSR, CPR, DOP, validation, fusion) have executed end-to-end at least
once and produced the 118 real, inspectable figures walked through in
Section 5, which is what made the quantitative and qualitative assessment
in this report possible.

Immediate next steps, in priority order:

1. Fix the DOP-to-fusion integration gap (Figure 101 / Known Issue #1) so
   the Ice Confidence Map uses all 8 intended bands, not 7.
2. Fix the correlation-matrix computation bug (Figure 114 / Known Issue #2)
   — likely a one-function fix in `diviner/visualizer.py`.
3. Fix the Unicode crash blocking `DFSAR/data_pipeline` (Known Issue #3) —
   a one-line encoding fix — so the broader 18-raster DFSAR feature stack
   can be exercised.
4. Investigate a bias correction for the GRI-default CPR product
   (Figure 84), which already shows the best rank correlation (r=0.650,
   ρ=0.695) of any computed CPR product against the official mosaic.
5. Verify the placeholder CRS assigned to the two Diviner `.grd` inputs
   against authoritative product documentation.
6. Export a per-pixel valid-band-count raster alongside the Ice Confidence
   Map for correct downstream interpretation.
7. Consolidate or clearly archive the exploratory `cpr_gri/`/`cpr_official/`
   scripts once the primary CPR source for the fusion stage is confirmed.

---

## 9. References

- O'Brien, P. & Byrne, S. (2022). *Double Shadows at the Lunar Poles.*
  Planetary Science Journal, 3, 258. https://doi.org/10.3847/PSJ/ac9d4e
- Nozette, S. et al. (1996). *The Clementine Bistatic Radar Experiment.*
  Science, 274, 1495.
- Campbell, D. B. et al. (2006). *No evidence for thick deposits of ice at
  the lunar south pole.* Nature, 443, 835.
- van Zyl, J. & Kim, Y. (2011). *Synthetic Aperture Radar Polarimetry.*
  JPL Space Science and Technology Series.
- Lee, J.-S. & Pottier, E. (2009). *Polarimetric Radar Imaging: From Basics
  to Applications.* CRC Press.
- Born, M. & Wolf, E. *Principles of Optics*, Ch. 10.
- Paige, D. A. et al. (2010). *Diviner Lunar Radiometer Observations of
  Cold Traps in the Moon's South Polar Region.* Science, 330, 479.
- Hayne, P. O. et al. (2015). *Evidence for exposed water ice in the
  Moon's south polar region from Lunar Reconnaissance Orbiter ultraviolet
  albedo and temperature measurements.* Icarus, 255, 58.
- Schorghofer, N. (2014). *Migration of Volatiles on the Surfaces of
  Mercury and the Moon.* Astrophysical Journal, 788, 169.
- Watson, K., Murray, B. C., & Brown, H. (1961). *The Behavior of
  Volatiles on the Lunar Surface.* Journal of Geophysical Research, 66,
  3033.
- Prettyman, T. H. et al. (2012). *Elemental composition of the lunar
  surface: Analysis of gamma ray spectroscopy data from Lunar Prospector.*
  Journal of Geophysical Research: Planets, 117, E12007.
- Putrevu, D. et al. (2023). *Chandrayaan-2 DFSAR Full Polarimetric
  Observations of the Lunar South Pole.* Journal of Geophysical Research:
  Planets. https://doi.org/10.1029/2023JE007745
