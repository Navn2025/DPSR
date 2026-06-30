# DPSR Extraction Pipeline — Lunar South Pole

Detects **Directly Persistently Shadowed Regions (DPSR)** at the lunar south pole using physically based solar ray-casting on the LOLA DEM.

A pixel is DPSR if it lies inside a PSR **and** no ray toward the sun clears the surrounding terrain from any azimuth across the lunar year.

---

## What It Does

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load LOLA DEM (15168×15168, 20 m/px) | in memory |
| 2 | Rasterize LOLA PSR shapefile | `results/PSR_mask.tif` |
| 3 | Compute slope and aspect | `results/slope.tif`, `results/aspect.tif` |
| 4 | Compute hillshade (visualisation) | `results/hillshade.tif` |
| 5 | Solar shadow map via ray-casting | `results/illumination.tif` |
| 6 | DPSR classification (Numba / CUDA) | `results/DPSR.tif` |
| 7 | Spot-check validation | printed to console |

---

## Project Structure

```
ISRO_Hackathon/
├── data/                   Input data (do not modify)
│   ├── ldem_85s_20m_float.lbl       LOLA DEM — PDS3 label
│   ├── ldem_85s_20m_float.img       LOLA DEM — binary data
│   └── LOLA_PSR_75S_120M_82S_060M_5KM2_FINAL.shp  PSR shapefile
├── pipeline/               Modular pipeline steps
│   ├── utils.py                Shared constants & paths
│   ├── step01_load.py          DEM + PSR loading
│   ├── step02_precompute_rays.py   Bresenham ray tables
│   ├── step03_numba_raytrace.py    Numba DPSR kernel
│   ├── step04_parallel_processing.py  Multiprocessing fallback
│   ├── step05_generate_dpsr.py    CPU pipeline entry point
│   ├── step06_gpu_dpsr.py         CUDA GPU entry point
│   └── step_illumination.py       Solar shadow map
├── notebooks/
│   ├── 01_basic_pipeline.ipynb    Step-by-step exploration
│   └── 02_full_pipeline.ipynb     Optimised Numba pipeline
├── docs/                   Reference documents (DFSAR, OHRC manuals)
├── results/                Output rasters (.tif)
├── images/                 Output plots (.png)
├── main.py                 Single entry point — runs all 7 steps
├── dpsr_fast.py            Self-contained single-file version
├── COMMANDS.md             Full commands reference
└── STEPS.md                Step-by-step guide from scratch
```

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install rasterio numpy geopandas matplotlib scipy numba
```

Verify:
```powershell
python -c "import rasterio, numpy, geopandas, matplotlib, numba; print('All OK')"
```

---

## Usage

```powershell
# Auto-detect GPU; fall back to CPU
python main.py

# Annual illumination — all 72 sun azimuths (best accuracy, slowest)
python main.py --annual

# Force CPU (Numba parallel)
python main.py --cpu

# Force GPU (Numba CUDA)
python main.py --gpu

# Recompute everything from scratch
python main.py --redo

# Annual + GPU — best accuracy, fastest runtime
python main.py --annual --gpu
```

Steps are skipped automatically if their output file already exists.

---

## Key Parameters

Edit in `pipeline/utils.py` (or at the top of `dpsr_fast.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SUN_ELEVATION` | `1.54°` | Peak solar elevation at 89.5°S |
| `SUN_AZIMUTH` | `0.0°` | 0 = North, 90 = East (single-epoch only) |
| `MAX_DISTANCE` | `2500 px` | Ray length — 2500 × 20 m = 50 km |
| `N_ANGLES` | `72` | DPSR rays per pixel (one every 5°) |
| `CELLSIZE` | `20.0 m` | DEM pixel size |

---

## Expected Runtimes

| Mode | Shadow Map | DPSR Step | Total |
|------|-----------|-----------|-------|
| CPU (single epoch) | 1–3 min | 1–5 min | ~5–10 min |
| CPU (annual, 72 azimuths) | 1–3 hrs | 1–5 min | ~2–4 hrs |
| GPU (annual) | 20–60 min | 30–120 sec | ~30–90 min |

> First run takes ~20 s extra for Numba JIT compilation. Subsequent runs skip it.

---

## Output

| File | Description |
|------|-------------|
| `results/PSR_mask.tif` | Rasterized PSR shapefile (uint8) |
| `results/slope.tif` | Slope in degrees (float32) |
| `results/aspect.tif` | Aspect in degrees (float32) |
| `results/hillshade.tif` | Hillshade — visualisation only |
| `results/illumination.tif` | Solar shadow map — 1 = lit, 0 = shadowed |
| `results/DPSR.tif` | **Final DPSR map** — 1 = persistently shadowed |
| `images/DPSR_summary.png` | PSR / illumination / DPSR comparison plot |

Quick check after a run:
```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/DPSR.tif') as ds:
    d = ds.read(1)
    print('DPSR pixels:', np.sum(d == 1))
    print('DPSR %     :', round(np.mean(d==1)*100, 3))
"
```

---

## Data Sources

- **DEM**: LOLA 20 m/pixel polar DEM — `ldem_85s_20m_float` (PDS3 format)
- **PSR shapefile**: `LOLA_PSR_75S_120M_82S_060M_5KM2_FINAL.shp` — permanently shadowed regions poleward of 75°S, ≥5 km²

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `.\venv\Scripts\Activate.ps1` first |
| `PermissionError` on .tif | Close QGIS or Jupyter, then retry |
| GPU not detected | Use `python main.py --cpu` |
| Numba compilation error | Delete `pipeline/__pycache__`, rerun |
| DPSR = 0 pixels | Check `results/illumination.tif` — may be all zeros |

See `COMMANDS.md` for the full commands reference and `STEPS.md` for a detailed walkthrough.
