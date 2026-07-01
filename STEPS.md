# DPSR Extraction — Step-by-Step Guide

What you need to do, in order, from a fresh machine to a final DPSR map.

---

## Phase 1 — One-time Setup

### Step 1: Verify your data files

Make sure these files exist in the `data/` folder:

```
data/
  ldem_85s_20m_float.lbl     ← PDS3 label file
  ldem_85s_20m_float.img     ← actual DEM data (opened via .lbl)
  LPSR_80S_20MPP_ADJ.shp
  LPSR_80S_20MPP_ADJ.dbf
  LPSR_80S_20MPP_ADJ.prj
  LPSR_80S_20MPP_ADJ.shx
```

If any shapefile is missing, re-download it from the PDS Geosciences Node.

---

### Step 2: Create and activate the virtual environment

Open PowerShell, then:

```powershell
cd C:\Users\navne\Desktop\ISRO_Hackathon
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.
**Do this every time you open a new terminal.**

---

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install rasterio numpy geopandas matplotlib scipy numba
```

Verify:

```powershell
python -c "import rasterio, numpy, geopandas, matplotlib, numba; print('All OK')"
```

---

### Step 4: Check GPU (optional but recommended)

```powershell
python -c "from numba import cuda; print('GPU:', cuda.is_available())"
```

- If it prints `GPU: True` — you can use `--gpu` for the fastest run.
- If `False` — the CPU pipeline still works fine (1–5 minutes).

---

## Phase 2 — Run the Pipeline

### Step 5: Run everything with one command

```powershell
python main.py
```

This runs all 7 steps automatically and skips any step whose output already exists:

| Step | What it does | Output |
|------|-------------|--------|
| 1 | Load DEM (15168×15168, 20 m/px) | in memory |
| 2 | Rasterize PSR shapefile | `results/PSR_mask.tif` |
| 3 | Compute slope and aspect | `results/slope.tif`, `aspect.tif` |
| 4 | Compute hillshade | `results/hillshade.tif` (visualisation only) |
| 5 | Solar shadow map | `results/illumination.tif` |
| 6 | DPSR ray-casting (Numba) | `results/DPSR.tif` |
| 7 | Spot-check validation | printed to console |

For best scientific accuracy, use annual illumination (longer but more correct):

```powershell
python main.py --annual
```

---

### Step 6: Check the output

After the run finishes:

```powershell
Get-ChildItem results\
```

You should see: `PSR_mask.tif`, `slope.tif`, `aspect.tif`, `hillshade.tif`,
`illumination.tif`, `DPSR.tif`.

Check the DPSR pixel count:

```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/DPSR.tif') as ds:
    d = ds.read(1)
    print('DPSR pixels:', np.sum(d == 1))
    print('DPSR %     :', round(np.mean(d==1)*100, 3))
"
```

Check the summary image:

```powershell
start images\DPSR_summary.png
```

---

## Phase 3 — Validation

### Step 7: Update spot-check coordinates

The validation step checks two pixels and prints whether they match expectations.
The default coordinates are placeholders — update them with real values.

**How to find correct coordinates:**

1. Open `results/DPSR.tif` in QGIS.
2. Also open `data/ldem_85s_20m_float.lbl` (DEM) and `data/*.shp` (PSR).
3. Click on a pixel near the rim of Shackleton crater — note its row and column.
4. Click on a pixel deep inside Shackleton — note its row and column.
5. Open `main.py`, find the `validate_spot_checks()` function, update the coordinates.
6. Also update the same coordinates in `dpsr_fast.py` and the notebooks.

Expected behaviour:
- **Rim pixel**: `illumination=1`, `psr=0`, `dpsr=0` — it can see sunlit terrain.
- **Interior pixel**: `illumination=0`, `psr=1`, `dpsr=1` — completely shadowed.

---

### Step 8: Compare with published results

DPSR maps for the lunar south pole have been published by:
- Hayne et al. (2015), JGR Planets — compare your DPSR/PSR ratio
- Mazarico et al. (2011), Icarus — Shackleton crater illumination fraction

A quick sanity check: the DPSR area should be a subset of the PSR area.
The DPSR/PSR ratio is typically 5–30% depending on the illumination model used.

---

## Phase 4 — Improve Results (optional)

### Step 9: Use annual illumination

The default single-epoch illumination uses one fixed sun position.
Annual illumination sweeps all 72 azimuths at the peak solar elevation (1.54°)
and marks any pixel as "ever illuminated" if it receives sunlight from at least
one direction. This is the physically correct definition for DPSR.

```powershell
python main.py --redo --annual --gpu
```

This deletes the existing illumination.tif and recomputes it.
Runtime: 30–90 minutes with GPU; several hours on CPU.

---

### Step 10: Tune sun elevation angle

The default `SUN_ELEVATION = 1.54°` is the peak solar elevation at 89.5°S.
For a specific date/epoch, compute the actual solar elevation using:

```python
# Approximate formula for lunar south pole
# latitude = -89.5 degrees
# sun_elevation_max ≈ 90 - abs(latitude) = 0.5 deg
# (varies with lunar orbital inclination ~1.54 deg max)
```

To change it, edit `SUN_ELEVATION` in `pipeline/utils.py` and rerun:

```powershell
python main.py --redo
```

---

### Step 11: Adjust search radius

The search radius is set to 50 km (2500 pixels × 20 m).
This covers Amundsen crater (~103 km diameter).
If you only care about smaller craters like Shackleton (~21 km), you can reduce
it to 1000 pixels (20 km) for a faster run:

Edit `MAX_DISTANCE = 1000` in `pipeline/utils.py`, then:

```powershell
python main.py --redo
```

---

## Phase 5 — Explore in Notebooks

### Step 12: Open the notebooks

```powershell
.\venv\Scripts\Activate.ps1
jupyter notebook notebooks\
```

Run them in this order:

1. **`01_basic_pipeline.ipynb`** — Step-by-step: load DEM, rasterize PSR,
   compute hillshade, compute solar shadow map, extract DPSR.
   Good for understanding each step before optimisation.

2. **`02_full_pipeline.ipynb`** — Full optimised pipeline with Numba CPU + CUDA GPU.
   Includes Bresenham ray precomputation, parallel classification, timing.

---

## Project Folder Reference

```
ISRO_Hackathon/
├── data/               Input DEM and PSR shapefile (do not modify)
├── docs/               PDF manuals and reference documents
├── notebooks/          Jupyter notebooks for exploration
│   ├── 01_basic_pipeline.ipynb
│   └── 02_full_pipeline.ipynb
├── pipeline/           Modular Python pipeline
│   ├── utils.py            Shared constants and paths
│   ├── step01_load.py      DEM + PSR loading
│   ├── step02_precompute_rays.py   Bresenham ray tables
│   ├── step03_numba_raytrace.py    Numba DPSR kernel
│   ├── step04_parallel_processing.py  Multiprocessing fallback
│   ├── step05_generate_dpsr.py    CPU pipeline entry point
│   ├── step06_gpu_dpsr.py         GPU pipeline entry point
│   └── step_illumination.py       Solar shadow map (new)
├── results/            Output rasters (.tif, .npy)
├── images/             Output plots (.png)
├── main.py             Single command to run everything
├── dpsr_fast.py        Self-contained single-file version
├── COMMANDS.md         All commands reference
└── STEPS.md            This file
```

---

## Common Issues

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | venv not active | Run `.\venv\Scripts\Activate.ps1` |
| `No such file: ldem_85s_20m_float.lbl` | Data missing | Check `data/` folder |
| `PermissionError` on .tif file | File open in QGIS | Close QGIS, try again |
| GPU not detected | No CUDA / wrong drivers | Use `python main.py --cpu` |
| Numba compilation error | Cache corrupt | Delete `pipeline/__pycache__` |
| Very slow on first run | Numba JIT compiling | Wait ~20 sec; next run is fast |
| DPSR = 0 pixels | Illumination map all zeros | Check shadow map output |
| DPSR = all PSR pixels | Illumination map all zeros or inverted | Rerun with `--redo` |
