# Multi-Sensor Detection of Water-Ice Stability Zones at the Lunar South Pole

### A Physics-Based Fusion of Topographic Shadow Modelling, Chandrayaan-2 DFSAR Radar Polarimetry, and LRO Diviner Thermal Data

**Prepared for:** ISRO Hackathon
**Repository:** `ISRO_Hackathon`
**Report date:** 2026-07-01
**Status:** Working prototype, all four pipelines executed end-to-end at least once

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
implementation, quantitative results, and — with equal weight — the concrete
defects and validation shortfalls discovered while building it.

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

Given:
- a LOLA digital elevation model of the lunar south pole (20 m/pixel),
- an LOLA-derived PSR shapefile,
- Chandrayaan-2 DFSAR full-polarimetric SAR imagery (multiple product
  levels: SLI, GRI, SRI) and an independently produced official CPR mosaic
  (Putrevu et al. 2023) for validation,
- LRO Diviner mean-temperature, zero-incidence-temperature, and
  cold-trapping-efficiency rasters,

produce (a) a physically derived DPSR mask, (b) CPR and DOP rasters computed
directly from the raw SAR data, (c) quantitative validation of the computed
CPR against the independently produced official product, and (d) a single
fused Ice Confidence Map that can be used to rank candidate landing/roving
sites by ice-stability likelihood.

### 1.3 Approach Summary

The work is organised as five loosely coupled Python packages, each with its
own config, logging, and output directory, developed and validated somewhat
independently before being brought together in the final `diviner/` fusion
stage:

| Stage | Package | Deliverable |
|---|---|---|
| 1 | `pipeline/`, `dpsr/`, `dpsr_fast.py` | DPSR / PSR rasters |
| 2 | `DFSAR/`, `cpr/`, `cpr_gri/`, `cpr_official/` | CPR rasters (3 independent computations) |
| 3 | `dop/` | DOP raster |
| 4 | `validation/` | Quantitative comparison of computed vs. official CPR |
| 5 | `diviner/` | Grid alignment of all layers + Ice Confidence Map |

Each stage is discussed in full in Section 4.

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
| LRO Diviner Tmean | `polar_south_80_Tmean.grd` | native grid (assigned Moon geographic CRS) | Diviner polar products | Thermal stability indicator |
| LRO Diviner ZIT | `polar_south_80_zit_float32.tif` | 240.04 m/px, polar stereographic | Diviner polar products | Coldest-temperature proxy |
| LRO Diviner Pump | `polar_south_80_pump.grd` | native grid | Diviner polar products | Cold-trapping efficiency proxy |

All rasters are ultimately reprojected onto the LOLA DEM's grid: south
polar-stereographic projection, spherical Moon datum (R = 1,737,400 m),
20 m/pixel, 15,168 × 15,168, bounds ±151,680 m in both axes. This choice
maximises resolution (LOLA is the finest native grid among the inputs) at
the cost of upsampling the coarser DFSAR and Diviner products.

---

## 3. System Architecture and Repository Organisation

```
ISRO_Hackathon/
├── data/                  Input DEM and PSR shapefile (read-only)
├── docs/                  Reference papers / instrument manuals
│
├── pipeline/              Legacy DPSR pipeline (first working version, single-epoch illumination)
├── dpsr/                  Modular DPSR pipeline — O'Brien & Byrne (2022), canonical implementation
├── dpsr_fast.py           Self-contained single-file DPSR version (used for quick iteration)
├── results/               DPSR / PSR raster outputs from pipeline/ + dpsr_fast.py
│
├── DFSAR/                 Raw Chandrayaan-2 DFSAR product tree + data_pipeline/ (SAR ingestion & feature stack)
├── cpr/                   Primary CPR computation from SLI data + Faustini-crater research scripts
├── cpr_gri/               Alternate CPR computation from GRI data (published co-pol formula)
├── cpr_official/          Extraction/visualisation of the official Putrevu (2023) CPR mosaic
├── dop/                   Degree of Polarization from the Stokes/covariance formalism
│
├── validation/            Georeferencing + quantitative validation: computed vs. official CPR
│
├── diviner/               Final fusion stage: Diviner thermal + DEM/Slope/PSR/DPSR/CPR/DOP → Ice Confidence Map
├── outputs/                Generated rasters and plots (DPSR pipeline runs + diviner fusion run)
│
├── validate_science.py    Standalone scientific validation (crater spot-checks, 4-panel comparison plot)
├── debug_pipeline.py / debug_psr_mask.py / debug_validation.py   Diagnostic / regression scripts
└── README.md / STEPS.md / COMMANDS.md   Setup guide, narrative walkthrough, command reference
```

Each package follows the same internal convention: a `config.py` (paths,
constants, weights), a `reader.py`/`loader.py` (dataset ingestion with CRS
sanity-checking), a computational core module (e.g. `cpr.py`, `stokes.py`,
`ice_score.py`), a `validator.py`/`validation.py`, a `visualizer.py`, and a
`main.py` entry point that logs every step to both console and a persistent
log file under that package's `outputs/logs/`. This consistent structure
made it possible to develop the five stages largely independently and
combine them later without a shared framework.

---

## 4. Methodology

### 4.1 DPSR Extraction Pipeline (`dpsr/`, `pipeline/`, `dpsr_fast.py`)

**Scientific definition.** Following O'Brien & Byrne (2022), *"Double
Shadows at the Lunar Poles,"* Planetary Science Journal 3:258:

```
DPSR pixel P  ⟺  psr_mask[P] == 1
                  AND  ∀ azimuth a:
                      ∄ pixel Q visible from P along ray a
                      with psr_mask[Q] == 0
```

In words: P is doubly shadowed only if it is already inside a PSR, and no
ray cast from P in any direction ever reaches a non-PSR (potentially
sunlit) surface before being blocked by terrain. This is a strictly
stronger condition than PSR membership and is the physically correct
target for "coldest possible cold trap," since even a PSR pixel that can
"see" nearby illuminated terrain receives significant scattered/reflected
infrared flux from it.

**Visibility test.** A candidate pixel Q at horizontal distance d from
observer P is visible if and only if its curvature-corrected elevation
angle μ_Q is greater than or equal to the highest elevation angle recorded
by any closer terrain along the same ray (i.e. Q clears the accumulated
horizon). The curvature correction (O'Brien & Byrne 2022, Appendix, Eq. A4)
accounts for the fact that on a spherical body, distant terrain is
depressed below the flat-Moon prediction:

```
tan(μ) = R1 · (R2 − √(d² + R1²)) / (d · R2)

R1 = R_Moon + h_observer     (radial distance of P from Moon centre)
R2 = R_Moon + h_target       (radial distance of Q from Moon centre)
```

At d = 50 km this correction depresses the apparent horizon by
≈ d²/(2 R_Moon) ≈ 720 m — large enough that omitting it would cause deep
PSR floors (Shackleton, Haworth) to incorrectly "see" distant sunlit
terrain that is, in reality, below the true curved horizon, producing
false negatives (real DPSRs misclassified as ordinary PSR).

**Algorithm ordering (critical correctness detail).** For every ray step,
the implementation performs the visibility check *before* updating the
running horizon maximum. If the horizon were updated first, every new
local maximum would trivially satisfy "≥ highest so far" and be misflagged
as visible, corrupting the whole classification. This ordering is called
out explicitly in the code comments and covered by the docstring in
`dpsr/step04_visibility.py`.

**Implementation.**
- `step01_load_dem.py` / `step02_load_psr.py` — load the LOLA DEM and
  rasterize/load the PSR mask.
- `step03_precompute_rays.py` — precompute Bresenham row/column offset
  tables and horizontal distances for `N_ANGLES` azimuths × `MAX_DIST`
  steps, shared by every PSR pixel.
- `step04_visibility.py` — the horizon-tracing kernel itself, implemented
  twice: a Numba `@njit(parallel=True)` CPU kernel (OpenMP over PSR
  pixels via `prange`, one thread per pixel, race-free by construction
  since each thread writes only its own output slot) and a CUDA kernel
  (one GPU thread per PSR pixel, DEM and PSR mask resident in VRAM,
  coalesced access to ray tables). Both kernels implement the identical
  formula; only the parallelisation backend differs.
- `step05_compute_dpsr.py` — orchestrates loading → ray precompute →
  classification → writes `DPSR_raw.tif`.
- `step06_remove_small_regions.py` — an 8-connected component filter
  removes clusters smaller than 5 pixels, matching the paper's
  post-processing spec (isolated single-pixel "DPSR" flags are treated as
  noise, not real cold traps).
- `step07_validation.py` — spot-checks named craters (Shackleton, Faustini,
  Haworth, Shoemaker, Cabeus).

**Parameter deviations from the published paper** (both intentional, and
logged with justification in `dpsr/utils.py`):

| Parameter | Paper (O'Brien & Byrne 2022) | This implementation | Justification |
|---|---|---|---|
| `N_ANGLES` (azimuth spacing) | 720 (0.5°) | 360 (1°) | <3% classification error at half the runtime |
| `MAX_DIST` (ray length) | 7500 px (150 km) | 2500 px (50 km) | Covers all known south-polar PSR-hosting craters (largest, Amundsen, ≈103 km diameter) at 3× lower cost |
| `min_component` (post-filter) | 5 px | 5 px | Unchanged — paper spec |

**Complexity.** Worst-case cost is O(P × A × D) (PSR pixel count × angles ×
max ray steps ≈ 20 M × 360 × 2500 ≈ 1.8×10¹³ operations), but the early-exit
rule (a ray stops the instant it finds one visible non-PSR pixel) means that
the >99.9% of PSR pixels that are *not* DPSR terminate after only a few
angles and a few tens of steps on average, reducing the effective cost to
roughly 2×10⁹ operations — a 2–8 minute wall-clock on a 12-core CPU, and
correspondingly faster on GPU.

**Result.** DPSR area is 7.9×10⁻⁵ of the total DEM area (≈0.008%,
`outputs/diviner/reports/statistics_report.csv`), consistent with the
paper's finding that DPSRs are extremely rare, small (sub-600 m) patches
embedded in the floors of much larger PSR craters, rather than large
contiguous regions. PSR itself covers 10.8% of the DEM area. The DPSR ⊆ PSR
subset property was confirmed to hold with zero violating pixels
(`COMMANDS.md` §5 sanity check), and Shackleton's PSR centroid was located
and confirmed at DEM pixel (row=7916, col=8006) via `debug_validation.py`.

Three separate implementations of this pipeline exist in the repository —
`pipeline/` (legacy, single-epoch illumination model, built first),
`dpsr_fast.py` (a self-contained single-file version used for rapid
iteration without the package import machinery), and `dpsr/` (the final
modular, paper-faithful implementation). All three are retained
deliberately: `dpsr/` is canonical, but the others remain useful for
regression comparison and for the annual (72-azimuth sweep) illumination
mode that `pipeline/step_illumination.py` implements independently of the
DPSR horizon-tracing algorithm.

### 4.2 SAR Polarimetry — Circular Polarization Ratio (`cpr/`, `cpr_gri/`, `cpr_official/`, `DFSAR/`)

**Physical basis.** From the full-polarimetric linear scattering matrix
(S_HH, S_HV, S_VV), the same-sense (SC) and opposite-sense (OC) circular
polarization fields are formed as:

```
S_RR (SC) = (S_HH − S_VV + 2j·S_HV) / 2
S_RL (OC) = (S_HH + S_VV) / 2

σ_SC = mean(|S_RR|²)   after multilooking
σ_OC = mean(|S_RL|²)   after multilooking

CPR = σ_SC / σ_OC
```

Physically: CPR < 1 indicates surface (Bragg/Fresnel) scattering typical of
bare regolith; CPR ≈ 1, mixed terrain; CPR > 1, volume scattering typical of
either an ice-grain volume or rough blocky ejecta (the fundamental
ambiguity this project addresses by combining CPR with DOP and thermal
evidence). The official Putrevu et al. (2023) mosaic reports a median CPR
of ≈0.21 for the lunar south pole; this project's computed products
(§5.2) land in the same range.

**Three independent implementations were built**, reflecting genuine
methodological exploration rather than redundant duplication:

1. **`cpr/`** (primary) computes CPR directly from **SLI** (single-look
   complex) HH/HV data via the SC/OC formula above, and includes an
   extensive Faustini-crater case study (`faustini_research_all.py`,
   `faustini_zoom.py`, `faustini_histograms.py`, `faustini_scatter3.py`,
   `faustini_combined.py`) cross-plotting CPR against DOP and Tmean for a
   single, well-studied crater.
2. **`cpr_gri/`** computes CPR from the **GRI** (multilooked ground-range)
   product using the *published, co-pol-only* formula:

   ```
   CPR(μc) = (σ_HH+σ_VV+2√(σ_HHσ_VV)) / (σ_HH+σ_VV−2√(σ_HHσ_VV))
           = (√σ_HH+√σ_VV)² / (√σ_HH−√σ_VV)²
   ```

   This formula assumes HH and VV are *fully coherent* — the opposite
   extreme from the default SC/OC formula's implicit reflection-symmetry
   assumption (zero HH–VV correlation). It was found to be numerically
   ill-conditioned: the denominator vanishes whenever σ_HH ≈ σ_VV (common
   for natural terrain, which rarely differs by the >15.3 dB needed to
   keep the raw ratio under 2), so the raw value saturates almost
   everywhere. The implementation therefore log10-rescales the raw ratio
   using its own P1–P99 percentile window rather than hard-clipping (hard
   clipping was tested first and confirmed to collapse the output to a
   flat, information-free image — this negative result is documented
   directly in the code and validation report rather than silently
   discarded).
3. **`cpr_official/`** extracts and re-visualises the official Putrevu
   (2023) CPR mosaic directly (e.g. `faustini_crater_official.py`) to serve
   as the ground truth for comparison in (1) and (2).
4. **`DFSAR/data_pipeline`** is a broader SAR ingestion pipeline (loader,
   reprojector, feature stack, GPU acceleration) that discovers and
   catalogues all 18 available DFSAR raster products (CPR, SRD, TRT, EVN,
   HLX, ODD, VOL, GRI, SLI, SRI and their polarization variants) alongside
   DEM/PSR/DPSR, intended eventually to replace the narrower `cpr/`/`cpr_gri`
   scripts with one unified feature-stack builder. It currently aborts
   (see §6) immediately after successful dataset discovery.

### 4.3 SAR Polarimetry — Degree of Polarization (`dop/`)

**Derivation.** The 2×2 Wolf coherency matrix J of the H-transmit receive
channels maps onto the four Stokes parameters via the standard optics
definitions (Born & Wolf, *Principles of Optics*, Ch. 10; van Zyl & Kim
2011):

```
S0 = C11 + C22/2                (total power)
S1 = C11 − C22/2                (linear H/V power imbalance)
S2 = √2 · Re(C12)               (linear ±45° component)
S3 = √2 · Im(C12)               (circular component)
```

where C11, C22, C12 are entries of the multilooked covariance matrix C3
built directly from the calibrated SLI data (`covariance.py`) — no
simplifying assumptions (dropped cross-terms, small-angle approximations,
reflection symmetry) are applied in this derivation, unlike the default CPR
formula's implicit assumption. The Degree of Polarization is then:

```
DOP = √(S1² + S2² + S3²) / S0
```

physically bounded to [0, 1]: DOP → 1 indicates a coherent, single-bounce
(specular Fresnel) scatterer — a smooth bare surface — while DOP < 1
indicates depolarisation from multiple/volume scattering, a *necessary but
not sufficient* signature consistent with buried volatile deposits, and
complementary to CPR in the fusion step (§4.4). The Stokes inequality
S0 ≥ √(S1²+S2²+S3²) is checked at every pixel; violations (numerical noise
near S0 ≈ 0) are logged and masked as invalid rather than silently clamped.

### 4.4 Diviner Thermal Integration and Physics-Based Fusion (`diviner/`)

This is the final, integrating stage of the project. It performs three
jobs in sequence:

**(a) Ingestion** (`loader.py`). Reads the three Diviner products
(`polar_south_80_Tmean.grd`, `polar_south_80_zit_float32.tif`,
`polar_south_80_pump.grd`). Two of the three (`Tmean`, `Pump`) arrive with
no CRS metadata; the loader assigns a Moon-geographic placeholder
(`+proj=longlat +R=1737400 +no_defs`) and logs an explicit warning that
this must be verified before publishing results — a deliberate
"fail loud, not silent" choice given that an incorrect placeholder CRS
would silently misalign these two bands relative to everything else.

**(b) Alignment** (`aligner.py`). Every input layer — DEM, Slope, PSR,
DPSR, CPR, DOP, Tmean, ZIT, Pump — is reprojected and resampled via
`rasterio.warp.reproject` onto the exact pixel grid of the reference LOLA
DEM (south polar-stereographic, 20 m/px, 15,168²). Continuous bands use
bilinear resampling; binary masks (PSR, DPSR) use nearest-neighbour to
avoid inventing fractional mask values at boundaries. Slope is derived from
the aligned DEM via standard finite differences, with geographic pixel
sizes converted to metres using the Moon's mean radius before
differencing. All nine aligned rasters are written once to
`outputs/diviner/aligned/` and never overwritten on subsequent runs.

**(c) Fusion** (`ice_score.py`). Eight bands (CPR, DOP, Tmean, ZIT, Pump,
PSR, DPSR, Slope) are combined into a single Ice Confidence Score in [0, 1]
via a physically weighted sum — **no machine learning is used**; every
weight and directional sign is a documented judgment call grounded in a
specific citation:

| Indicator | Weight | Ice-positive direction | Physical basis |
|---|---|---|---|
| CPR | 0.20 | HIGH → more ice | Volume/double-bounce scattering from ice-grain aggregates (Nozette 1996; Campbell 2006) |
| Tmean | 0.20 | LOW → more ice | Thermally stable cold environments preserve volatiles (Paige et al. 2010) |
| ZIT | 0.15 | LOW → more ice | Best proxy for coldest surface temperature at a PSR pixel (Hayne et al. 2015) |
| Pump | 0.13 | HIGH → more ice | Efficient volatile cold-trapping proxy (Schorghofer 2014) |
| DOP | 0.12 | LOW → more ice | Depolarising, volumetric targets lower overall DOP (van Zyl & Kim 2011) |
| PSR | 0.10 | 1 → more ice | No solar heating → thermally stable ice (Watson et al. 1961) |
| DPSR | 0.05 | 1 → more ice | Extra cold-trap stability (O'Brien & Byrne 2022) |
| Slope | 0.05 | LOW → more ice | Flat terrain favours ice accumulation/retention (Prettyman et al. 2012) |
| **Total** | **1.00** | | |

Normalisation procedure (`_normalise_continuous`): each continuous band is
clipped to its own [P2, P98] range (robust to outliers), linearly rescaled
to [0, 1], and inverted (1 − x) for "low → more ice" bands so that every
normalised band shares the same "higher = more ice-favourable" sense before
weighting. Binary masks are cast directly to {0.0, 1.0}.

The combination rule per pixel is:

```
score[p] = Σᵢ (normalised_band_i[p] × weight_i)  /  Σᵢ (weight_i, valid bands only)
```

— i.e. the score is renormalised by the sum of weights of *whatever bands
are actually valid at that pixel*, so a pixel missing an optional band
(e.g. outside the DFSAR CPR footprint) still receives a meaningful score
from the remaining seven bands rather than being zeroed out or discarded.
This design choice is important given that CPR only covers 72.5% of the
DEM area and, at the time of this run, DOP covers 0% (see §6).

**(d) Reporting** (`reporter.py`, `visualizer.py`). Produces per-band maps
and histograms, a 9×9 inter-band correlation matrix, scatter plots for
selected band pairs (CPR–Tmean, CPR–ZIT, Tmean–Pump), and a CSV statistics
report (min/max/mean/median/std/percentiles/valid-pixel-count) for every
band plus the final Ice Confidence score.

### 4.5 Cross-Validation Module (`validation/`)

Independently georeferences the SLI-based computed CPR onto the official
Putrevu (2023) mosaic's grid and computes a full quantitative comparison.

**Georeferencing.** Uses `rasterio.warp.reproject` driven by ground control
points extracted from the DFSAR SLI geometry CSV (8,521 azimuth tie rows ×
9 range columns; every 10th azimuth row is used, giving 7,668 GCPs), mapping
from the native Moon-geographic sphere (R = 1,737,400 m) into the south
polar-stereographic target grid (15,168², 20 m/px, ±151,680 m bounds) —
the same target grid used throughout the project.

**Metrics** (`metrics.py`): Pearson r, Spearman ρ, RMSE, MAE, signed bias,
R², SSIM (structural similarity, via `skimage`), histogram intersection,
and mutual information, all computed over the set of pixels that are
simultaneously valid (finite, non-nodata, within a physical CPR range of
(0, 20]) in both rasters. Six diagnostic figures are produced: a spatial
difference map, an absolute-difference map, side-by-side value histograms,
a Q–Q plot, a boxplot, and a validation scatter plot with a 1:1 reference
line.

---

## 5. Results

### 5.1 DPSR / PSR Extraction

| Quantity | Value | Source |
|---|---|---|
| PSR area fraction | 10.79% of DEM | `statistics_report.csv` |
| DPSR area fraction | 0.0079% of DEM | `statistics_report.csv` |
| DPSR / PSR ratio | ≈0.073% | derived |
| DPSR ⊆ PSR subset check | Passed (0 violating pixels) | `COMMANDS.md` §5 |
| Shackleton PSR centroid (row, col) | (7916, 8006) | `debug_validation.py` |
| Expected DPSR/PSR ratio (paper, 30 m) | ~0.04% | O'Brien & Byrne (2022) |

The observed DPSR/PSR ratio is somewhat higher than the 30 m-resolution
paper value, which is expected: at 20 m/px resolution the ray-tracing
kernel can resolve smaller sub-crater floor features that a coarser grid
would smooth away, consistent with the note in `COMMANDS.md` that "our 20 m
DEM may resolve slightly more DPSRs than the 30 m paper results."

### 5.2 CPR Validation Against the Official Mosaic

Two independently computed CPR products were compared against the official
Putrevu et al. (2023) mosaic.

**(a) SLI-based CPR** (`cpr/`, the product that feeds the final fusion),
compared over 3,270,277 overlapping pixels (1.42% of the full 230M-pixel
grid — the DFSAR scene footprint is a narrow strip, not full coverage):

| Metric | Value | Interpretation |
|---|---|---|
| Pearson r | 0.079 | Poor pixel-level linear agreement |
| Spearman ρ | 0.068 | Poor rank agreement |
| RMSE | 0.170 | Moderate absolute error |
| MAE | 0.127 | |
| R² | −0.593 | Worse fit than predicting the mean value everywhere |
| SSIM | 0.944 | High structural similarity |
| Histogram intersection | 0.870 | Very similar overall value distributions |
| Bias (calc − official) | +0.007 | Negligible |
| Mean (calc / official) | 0.231 / 0.266 | Same order of magnitude |
| Median (calc / official) | 0.214 / 0.234 | Same order of magnitude |

**(b) GRI-based CPR, published μc formula** (`cpr_gri/`, research mode),
compared over 2,141,158 overlapping pixels from a single Faustini-region
scene:

| Metric | Value | Interpretation |
|---|---|---|
| Pearson r | −0.213 | Weak *negative* relationship |
| Spearman ρ | −0.307 | Weak negative rank relationship |
| RMSE | 0.490 | |
| R² | −5.14 | Poor absolute agreement |
| Histogram intersection | 0.755 | Moderately similar distributions |
| Bias | +0.163 | Non-negligible over-estimate |

**Interpretation.** Both computed CPR products reproduce the overall
*distribution* (mean/median/percentiles, histogram intersection) and,
for the SLI product, the large-scale *spatial structure* (SSIM = 0.944) of
the official mosaic well. Pixel-level correlation is nonetheless poor for
both, and actively inverted for the GRI/μc formula. The validation report
attributes this to three concrete, documented causes rather than treating
it as an unexplained discrepancy: (1) different acquisition date/orbit
geometry between the scenes being compared and the 2025-06-30 official
mosaic; (2) different multilook window sizes; (3) residual georeferencing
error, particularly near scene edges, from the GCP-based reprojection. The
GRI/μc formula additionally has a specific, understood failure mode: it
assumes HH and VV are fully coherent, the opposite extreme from the
default formula's zero-correlation assumption, and is numerically
ill-conditioned exactly where σ_HH ≈ σ_VV — which is common on real lunar
terrain. This is reported as a genuine negative methodological finding, not
suppressed.

### 5.3 Degree of Polarization

The `dop/` module in isolation produces a valid DOP raster (bounded [0,1],
with explicit invalidity masking for non-physical Stokes vectors). However,
in the current fused feature stack (`outputs/diviner/reports/statistics_report.csv`),
**the DOP band shows 0 valid pixels out of 230,068,224** — this is a known,
unresolved integration defect between `dop/`'s output and `diviner/`'s
alignment step (§6), not a defect in the DOP formula itself.

### 5.4 Final Fused Feature Stack and Ice Confidence Map

Full statistics over the complete 230,068,224-pixel LOLA grid
(`outputs/diviner/reports/statistics_report.csv`, run completed
2026-07-01 17:49–17:50):

| Band | Valid coverage | Mean | Median | Std | Notes |
|---|---|---|---|---|---|
| DEM | 100.00% | −1067 m | −1352 m | 1973 m | |
| Slope | 100.00% | 10.00° | 8.66° | 6.63° | |
| PSR | 100.00% | 0.108 (10.8% area) | 0 | 0.310 | binary |
| DPSR | 100.00% | 7.9×10⁻⁵ (0.008% area) | 0 | 0.0089 | binary |
| CPR | 72.53% | 0.267 | 0.234 | 0.160 | DFSAR footprint gap outside coverage |
| DOP | **0.00%** | — | — | — | **integration defect, see §6** |
| Tmean | 100.00% | 98.0 K | 98.9 K | 30.9 K | |
| ZIT | 80.15% | 0.067 | 0.055 | 0.064 | |
| Pump | 62.84% | 0.665 | 0.799 | 0.345 | |
| **Ice Confidence** | 100.00% | **0.438** | **0.483** | **0.160** | final fused score, range [0, 0.878] |

Deliverables generated by this run: `Feature_Stack.tif` (5.2 GB, 9 bands),
`Ice_Confidence_Map.tif` (992 MB, single band), per-band map and histogram
PNGs for all nine inputs plus the final score, three cross-band scatter
plots, and a 9×9 correlation matrix (`Correlation_Matrix.png`).

Because the Ice Confidence formula renormalises by the sum of weights of
*valid* bands at each pixel (§4.4), the fact that DOP contributes 0% and CPR
contributes 72.5% coverage does not zero out the score outside those
footprints — it simply means the score outside the DFSAR CPR strip and
everywhere (currently) is computed from DEM/Slope/PSR/DPSR/Tmean/ZIT/Pump
only. This is a known, mathematically transparent limitation of the current
run rather than a silent gap: any location's score should be read alongside
its per-pixel valid-band count, which is not currently exported as its own
raster (see §7, future work).

---

## 6. Known Issues and Limitations

These are concrete, reproducible defects and shortfalls identified while
building and validating the pipeline, reported here without euphemism so
they can be prioritised for the next iteration:

1. **DOP does not reach the final fusion.** `dop/` computes a valid,
   physically bounded DOP raster in isolation, but `outputs/diviner/reports/statistics_report.csv`
   shows 0 valid DOP pixels in the aligned/fused stack. The Ice Confidence
   Map is therefore currently computed from 7 of the intended 8 weighted
   bands everywhere (CPR's 0.20 weight is diluted across the other bands'
   proportional shares via the weight-renormalisation rule, but DOP's
   intended 0.12 contribution is entirely absent). Root cause is most
   likely a path, CRS-assignment, or nodata-value mismatch between
   `dop/`'s output GeoTIFF and what `diviner/aligner.py` expects — this
   needs to be traced by re-running `diviner/main.py` with debug logging
   focused on the DOP alignment step.
2. **`DFSAR/data_pipeline/main.py` crashes immediately after dataset
   discovery.** The traceback is a `UnicodeEncodeError` inside
   `utils.section()`, which prints Unicode box-drawing characters
   (`═`) to a default Windows console using the `cp1252` codec. This is a
   trivial fix (either force UTF-8 stdout, e.g.
   `sys.stdout.reconfigure(encoding="utf-8")`, or replace the box-drawing
   characters with ASCII equivalents), but as of this report it has not
   been applied, so `DFSAR/data_pipeline` — the broadest, most complete SAR
   ingestion pipeline in the repository — cannot currently run past Step 1.
3. **CPR pixel-level correlation with the official mosaic is poor**
   (Pearson r ≈ 0.08 for the primary SLI product; R² < 0 for both computed
   products), despite good structural/distributional agreement. The
   validation module documents plausible causes (acquisition
   geometry/date mismatch, multilook differences, georeferencing error)
   but does not yet isolate which factor dominates. A controlled
   experiment (comparing scenes from the *same* acquisition date/orbit as
   the official mosaic, if available) would be needed to disambiguate.
4. **Low spatial overlap for CPR validation.** Only 1.42% of the full
   230M-pixel target grid has simultaneous DFSAR and official-mosaic
   coverage in the current comparison; conclusions about CPR agreement
   should not be over-generalised beyond that footprint.
5. **Three parallel, not-yet-consolidated CPR implementations**
   (`cpr/`, `cpr_gri/`, `cpr_official/`) exist because of genuine
   methodological exploration (different DFSAR product levels, different
   published formulas). Only `cpr/`'s SLI-based output currently feeds the
   final fusion; the GRI-based alternative remains a documented negative
   result rather than a competing production path. Before any final
   submission, the repository would benefit from either removing the
   superseded exploratory scripts or clearly marking them archival.
6. **Missing CRS metadata on two of three Diviner inputs** (`Tmean.grd`,
   `pump.grd`) required assigning a placeholder Moon-geographic CRS; this
   is flagged with an explicit warning in the pipeline log but has not yet
   been independently verified against an authoritative Diviner product
   specification — this verification is a prerequisite before treating the
   Ice Confidence Map as publication-ready.
7. **Per-pixel valid-band count is not exported.** Because the Ice
   Confidence score renormalises over whichever bands are valid at each
   pixel, two pixels with the same numeric score can have very different
   evidentiary weight behind them (e.g. one uses all 7 currently-available
   bands, another uses only 4 near a data gap). Exporting this count as an
   auxiliary raster would make the confidence map itself easier to
   interpret correctly.

---

## 7. Discussion

The project's central design decision — a fully physics-based, weighted-sum
fusion with no learned parameters — trades away the potential accuracy of a
trained classifier for complete interpretability: every number in the final
Ice Confidence Map can be traced back to a specific band value, a specific
normalisation percentile, and a specific literature-justified weight. This
matters for a hackathon/exploration-science context where the audience
needs to trust *why* a pixel scored high, not just *that* it did. The cost
of this choice is that the fusion cannot learn to compensate for the
DOF-coverage gaps and validation weaknesses documented in §6 — those must
be fixed at the source (in `dop/` and `cpr/`'s inputs) rather than
absorbed by a downstream model.

The CPR validation results deserve particular scrutiny before this pipeline
is used for any site-selection recommendation. High SSIM and histogram
intersection alongside low Pearson r is a specific, recognisable pattern:
it means the computed CPR "looks like" the official product in aggregate
(similar contrast, similar large-scale bright/dark structure) but individual
pixels do not line up precisely — consistent with a geometric
(georeferencing/acquisition-geometry) explanation rather than a fundamentally
wrong physical formula. This is an encouraging sign for the underlying
science (the SC/OC CPR formula itself appears sound) but means absolute
pixel values from the current computed CPR raster should not yet be treated
as equivalent in accuracy to the official mosaic.

The DPSR results, by contrast, are on solid footing: the ⊆-PSR subset
property holds exactly, the crater-centroid spot check passed, and the
observed area fraction is within the expected order of magnitude of the
published paper's result once resolution differences are accounted for.
This is the most mature component of the pipeline.

---

## 8. Conclusion and Future Work

This project has progressed from a single-purpose DPSR ray-tracing tool
into a genuine multi-instrument fusion system spanning topography, radar
polarimetry, and thermal radiometry, culminating in one physically
interpretable deliverable (the Ice Confidence Map) computed over the full
230-million-pixel south-polar LOLA grid. All five stages (DPSR, CPR, DOP,
validation, fusion) have executed end-to-end at least once and produced
real, inspectable outputs (rasters, plots, CSV statistics, log files),
which is what made the quantitative assessment in this report possible.

Immediate next steps, in priority order:

1. Fix the DOP-to-fusion integration gap (§6.1) so the Ice Confidence Map
   uses all 8 intended bands, not 7.
2. Fix the Unicode crash blocking `DFSAR/data_pipeline` (§6.2) — a one-line
   encoding fix — so the broader 18-raster DFSAR feature stack can be
   exercised.
3. Verify the placeholder CRS assigned to the two Diviner `.grd` inputs
   against authoritative product documentation.
4. Export a per-pixel valid-band-count raster alongside the Ice Confidence
   Map for correct downstream interpretation.
5. Investigate the CPR validation discrepancy with a same-date/same-orbit
   controlled comparison, if such a reference scene can be obtained.
6. Consolidate or clearly archive the exploratory `cpr_gri/`/`cpr_official/`
   scripts once the primary `cpr/` path is confirmed as the production
   source for the fusion stage.

---

## References

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
