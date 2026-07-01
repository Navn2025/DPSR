# `diviner/` — Thermal Integration & Physics-Based Fusion

The final stage: aligns **LRO Diviner** thermal rasters plus every upstream
layer (DEM, Slope, PSR, DPSR, CPR, DOP) onto the common LOLA grid and fuses
them into the project's headline deliverable — the **Ice Confidence Map**.

**No machine learning.** Every weight and direction (does high or low indicate
ice?) is tied to a published reference, so any pixel's score decomposes back
into its exact per-band contributions.

---

## Fusion Weights

| Indicator | Weight | Ice-positive | Basis |
|-----------|--------|--------------|-------|
| CPR | 0.20 | HIGH | Nozette 1996; Campbell 2006 |
| Tmean | 0.20 | LOW | Paige et al. 2010 |
| ZIT | 0.15 | LOW | Hayne et al. 2015 |
| Pump | 0.13 | HIGH | Schorghofer 2014 |
| DOP | 0.12 | LOW | van Zyl & Kim 2011 |
| PSR | 0.10 | 1 (binary) | Watson et al. 1961 |
| DPSR | 0.05 | 1 (binary) | O'Brien & Byrne 2022 |
| Slope | 0.05 | LOW | Prettyman et al. 2012 |

```
score[p] = Σᵢ (norm_band_i[p] × wᵢ) / Σᵢ (wᵢ over valid bands only)
```

Continuous bands are percentile-clipped [P2, P98] → [0, 1]; "low → ice" bands
are inverted before weighting. Renormalising by valid-band weights handles
partial coverage gracefully.

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `loader.py` | Load Diviner grids + all upstream layers |
| `aligner.py` | Reproject all bands to the LOLA grid |
| `ice_score.py` | Weighted Ice Confidence fusion |
| `visualizer.py` | Per-band maps, histograms, scatter, correlation matrix |
| `reporter.py` | `statistics_report.csv` |
| `config.py`, `utils.py` | Weights, paths, constants |

Alignment: `rasterio.warp.reproject` — bilinear for continuous bands,
nearest-neighbour for binary masks.

---

## Run

```powershell
python diviner/main.py
```

---

## Output

`Ice_Confidence_Map` (full 15,168² grid) + per-band maps/histograms and
`statistics_report.csv`, under `outputs/diviner/`. High confidence concentrates
in crater floors and pole-facing walls — the expected physical pattern.

> ⚠️ **Known issues:** (1) DOP arrives with 0 valid pixels (fusion currently
> uses 7 of 8 bands); (2) the correlation matrix reads all N/A (arrays not
> co-registered into `save_correlation_matrix()`). Both are silent failures —
> see `PROJECT_REPORT.md` §7.
