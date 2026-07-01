# `validation/` — Computed vs. Official CPR Validation

Georeferences the **computed CPR** onto the grid of the **official
Putrevu et al. (2023) DFSAR CPR mosaic** and quantifies agreement, so the
project's CPR products can be trusted (or their limits understood).

---

## How It Works

1. **Georeference** the computed CPR using GCPs from the DFSAR SLI geometry CSV
   (8,521 azimuth rows × 9 range cols; every 10th azimuth row → 7,668 GCPs).
2. **Warp / co-register** onto the official mosaic's grid.
3. **Compute metrics** over co-valid pixels: Pearson r, Spearman ρ, RMSE, MAE,
   bias, R², SSIM, histogram intersection, mutual information.
4. **Plot**: side-by-side maps, scatter vs 1:1, difference maps, Q–Q, boxplot.

---

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `reader.py` | Load computed + official CPR |
| `georeference.py`, `warp.py` | GCP georeferencing and reprojection |
| `metrics.py` | Statistical agreement metrics |
| `plots.py` | Validation figures |
| `report.py` | Metrics summary / report |
| `config.py`, `utils.py` | Paths and constants |
| `outputs/` | Generated metrics and plots |

---

## Run

```powershell
python validation/main.py
```

---

## Results Summary

| Product | Formula | Pearson r | Spearman ρ | Bias | Verdict |
|---------|---------|-----------|-----------|------|---------|
| `cpr/` (SLI) | Default SC/OC | 0.079 | 0.068 | +0.007 | Strong structural (SSIM 0.944), weak pixel corr. |
| `cpr_gri/` | Default SC/OC | **0.650** | **0.695** | **+0.973** | Best correlation, large additive bias |
| `cpr_gri/` | Research μc | −0.213 | −0.307 | +0.163 | Ill-conditioned for this terrain |

**Next step:** a simple additive/multiplicative bias correction on the
GRI-default product (fit on the 2.14 M co-valid pixels) should substantially
improve absolute accuracy. See `PROJECT_REPORT.md` §5.8–5.9.
