# `pipeline/` — Legacy DPSR + Illumination Pipeline

The **first working** DPSR pipeline for the lunar south pole, plus the solar
**illumination / shadow-map** step. Superseded by the cleaner [`dpsr/`](../dpsr/)
module for the canonical DPSR product, but kept because it also produces the
illumination raster and an annual (multi-azimuth) shadow model.

---

## Files

| File | Description |
|------|-------------|
| `config.py`, `utils.py` | Shared constants and paths |
| `step01_load.py` | Load DEM + PSR |
| `step02_precompute_rays.py` | Bresenham ray tables |
| `step03_numba_raytrace.py` | Numba DPSR kernel |
| `step04_parallel_processing.py` | Multiprocessing fallback |
| `step05_generate_dpsr.py` | CPU pipeline entry point |
| `step06_gpu_dpsr.py` | CUDA GPU entry point |
| `step_dpsr_obrien.py` | O'Brien & Byrne variant |
| `step_illumination.py` | Solar shadow / illumination map |
| `step1_legacy.py` | Original prototype step |
| `datasets/`, `outputs/` | Local inputs / generated rasters |

---

## Run

From the repository root:

```powershell
python main.py               # auto-detect GPU, else CPU
python main.py --annual      # all 72 sun azimuths (best accuracy)
python main.py --cpu         # force Numba parallel CPU
python main.py --gpu         # force Numba CUDA
python main.py --redo        # recompute everything
```

Steps are skipped automatically if their output already exists.

---

## Output

| File | Description |
|------|-------------|
| `results/PSR_mask.tif` | Rasterized PSR shapefile |
| `results/slope.tif`, `results/aspect.tif` | Terrain derivatives |
| `results/hillshade.tif` | Visualisation only |
| `results/illumination.tif` | Solar shadow map (1 = lit, 0 = shadowed) |
| `results/DPSR.tif` | DPSR map (1 = persistently shadowed) |

---

## Runtimes

| Mode | Total |
|------|-------|
| CPU (single epoch) | ~5–10 min |
| CPU (annual, 72 azimuths) | ~2–4 hrs |
| GPU (annual) | ~30–90 min |

> First run adds ~20 s for Numba JIT compilation. See `../STEPS.md` and
> `../COMMANDS.md` for the full walkthrough.
