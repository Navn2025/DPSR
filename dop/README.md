# `dop/` — Degree of Polarization

Computes the **Degree of Polarization (DOP)** from Chandrayaan-2 DFSAR data via
the Stokes / covariance-matrix formalism, with no simplifying assumptions.

```
S0 = C11 + C22/2     S1 = C11 − C22/2
S2 = √2·Re(C12)      S3 = √2·Im(C12)
DOP = √(S1² + S2² + S3²) / S0
```

**Interpretation:** DOP → 1 = coherent, single-bounce (specular) scattering
(typical of bare regolith at L-band); DOP < 1 = depolarisation from
multiple/volume scattering — necessary but not sufficient evidence for buried
volatiles, and complementary to CPR. The joint criterion **CPR > 1 AND
DOP < 0.13** is the ice-candidate signature.

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `reader.py`, `complex_builder.py` | Read DFSAR channels, build complex products |
| `covariance.py` | Multilooked covariance matrix C |
| `stokes.py` | Stokes parameters S0–S3 |
| `dop.py` | DOP raster |
| `validator.py` | Formula verification against a known scene |
| `visualizer.py` | DOP maps, histograms, per-channel power |
| `faustini_crater.py` | Faustini-cropped DOP |
| `config.py`, `utils.py` | Paths and constants |
| `outputs/` | Generated rasters and plots |

---

## Run

```powershell
python dop/main.py
```

---

## Output

`DOP.tif` plus DOP map, histogram, the four linear-pol power channels
(HH/HV/VH/VV — HV≈VH is a reciprocity sanity check), and Faustini-cropped DOP,
under `dop/outputs/`.

> ⚠️ **Known issue:** DOP is valid in isolation but currently shows 0 valid
> pixels once aligned into the fusion stack (`diviner/`) — a path/CRS/nodata
> mismatch under investigation. See `PROJECT_REPORT.md` §7.
