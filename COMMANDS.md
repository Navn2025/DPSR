# DPSR Pipeline — Commands Reference

> Always activate the virtual environment first before running any command.

---

## 1. Setup (first time only)

### Create virtual environment
```powershell
cd C:\Users\navne\Desktop\ISRO_Hackathon
python -m venv venv
```

### Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
> Run this every time you open a new terminal.

### Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install rasterio numpy geopandas matplotlib scipy numba pyproj
```

### Verify installation
```powershell
python -c "import rasterio, numpy, geopandas, matplotlib, numba, pyproj; print('All OK')"
```

### Check GPU availability
```powershell
python -c "from numba import cuda; print('GPU available:', cuda.is_available()); [print(' GPU', i, ':', cuda.gpus[i].name) for i in range(len(cuda.gpus))]"
```

---

## 2. Run the Full Pipeline

### Recommended — GPU + annual illumination
```powershell
python main.py --annual --gpu
```

### Force CPU (if GPU causes issues)
```powershell
python main.py --annual --cpu
```

### Recompute everything from scratch
```powershell
python main.py --redo --annual --gpu
```

### Rerun only DPSR (illumination already cached)
```powershell
Remove-Item results\DPSR.tif -Force
python main.py --annual --gpu
```

### Rerun only illumination + DPSR
```powershell
Remove-Item results\illumination.tif, results\DPSR.tif -Force
python main.py --annual --gpu
```

---

## 3. Pipeline Steps and Outputs

Each step is **skipped automatically** if the output already exists.
PNG previews are saved to `images/` after every step.

| Step | Output file | Estimated time (GPU) |
|------|-------------|----------------------|
| 1 — Load DEM | *(in memory)* | ~5 s |
| 2 — PSR mask | `results/PSR_mask.tif` | ~30 s |
| 3 — Slope + aspect | `results/slope.tif`, `aspect.tif` | ~2 min |
| 4 — Hillshade | `results/hillshade.tif` | ~1 min |
| 5 — Annual illumination | `results/illumination.tif` | ~5–15 min |
| 6 — DPSR (O'Brien & Byrne) | `results/DPSR.tif` | ~10–30 min |
| 7 — Spot-check validation | *(console output)* | <1 s |

> First run compiles Numba/CUDA kernels (~20–30 s). Subsequent runs skip compilation.

---

## 4. DPSR Algorithm — O'Brien & Byrne (2022)

The DPSR step implements *"Double Shadows at the Lunar Poles"*,
PSJ 3:258. Key parameters (set in `pipeline/step_dpsr_obrien.py`):

| Parameter | Default | Paper value | Meaning |
|-----------|---------|-------------|---------|
| `n_angles` | 360 | 720 | Azimuth directions (1° vs 0.5° spacing) |
| `max_dist` | 2500 px | 7500 px | Max ray length (50 km vs 150 km) |
| `min_component` | 5 px | 5 px | Min DPSR cluster size (8-connected) |
| `moon_r` | 1 737 400 m | 1 737 400 m | Lunar reference radius |

**Algorithm (Section 2.3):** For every PSR pixel, cast rays in all directions.
A pixel is visible if its curvature-corrected elevation angle ≥ the maximum
angle from all closer terrain. If ANY visible pixel has `psr_mask == 0`
(non-PSR surface) → NOT doubly shadowed.

**Curvature correction (Appendix, Eq. A4):**
```
tan(mu) = R1 * (R2 - sqrt(d^2 + R1^2)) / (d * R2)
```
At 50 km, this corrects apparent elevation by ~720 m — critical for accuracy.

---

## 5. Inspect Outputs

### List results
```powershell
Get-ChildItem results\ | Select-Object Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}, LastWriteTime
```

### List preview images
```powershell
Get-ChildItem images\ | Select-Object Name, LastWriteTime
```

### Open summary image
```powershell
start images\DPSR_summary.png
```

### DPSR pixel statistics
```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/PSR_mask.tif') as ds:  psr = ds.read(1)
with rasterio.open('results/DPSR.tif') as ds:      dpsr = ds.read(1)
print('PSR  pixels:', f'{int(psr.sum()):,}', f'({100*psr.mean():.2f}%)')
print('DPSR pixels:', f'{int(dpsr.sum()):,}', f'({100*dpsr.mean():.4f}%)')
print('DPSR/PSR   :', f'{100*dpsr.sum()/psr.sum():.4f}%')
print('DPSR subset:', 'OK' if int(((dpsr==1)&(psr==0)).sum())==0 else 'FAIL')
"
```

### Illumination statistics
```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/illumination.tif') as ds: ill = ds.read(1)
print('Lit   :', f'{int(ill.sum()):,}', f'({100*ill.mean():.2f}%)')
print('Dark  :', f'{int((ill==0).sum()):,}', f'({100*(ill==0).mean():.2f}%)')
print('Unique:', np.unique(ill).tolist())
"
```

---

## 6. Debugging and Validation Scripts

### Full scientific validation (subset check, crater tests, 4-panel plot)
```powershell
python validate_science.py
```
Outputs `images/science_validation.png` and checks Faustini, Haworth, Shackleton.

### Debug PSR mask and coordinate conversion
```powershell
python debug_validation.py
```
Confirms Shackleton PSR centroid at row=7916, col=8006.

### Debug illumination and pipeline
```powershell
python debug_pipeline.py
```

### Smoke test DPSR on Shackleton crop (fast, ~30 s)
```powershell
python -c "
import sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, '.')
from pipeline.step_dpsr_obrien import compute_dpsr
with rasterio.open('data/ldem_85s_20m_float.lbl') as ds:
    elev = ds.read(1).astype('float32') * 1000.0
with rasterio.open('results/PSR_mask.tif') as ds:
    psr = ds.read(1)
# Crop around Shackleton PSR (row=7916, col=8006)
e = np.ascontiguousarray(elev[7416:8416, 7506:8506])
p = np.ascontiguousarray(psr[7416:8416, 7506:8506])
d = compute_dpsr(e, p, n_angles=72, max_dist=500, use_gpu=True, min_component=5)
print('DPSR pixels:', int(d.sum()), '/', int(p.sum()), 'PSR')
"
```

---

## 7. Run Individual Pipeline Modules

### Illumination only — single epoch
```powershell
python -m pipeline.step_illumination --az 0 --el 1.54
```

### Illumination only — annual (72 azimuths), GPU
```powershell
python -m pipeline.step_illumination --annual --gpu
```

### Illumination only — annual, CPU
```powershell
python -m pipeline.step_illumination --annual --cpu
```

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | venv not active | `.\venv\Scripts\Activate.ps1` |
| `No such file: ldem_85s_20m_float.lbl` | Data missing | Check `data/` folder |
| `PermissionError` on `.tif` | File open in QGIS | Close QGIS, try again |
| GPU not detected | No CUDA / wrong driver | `python main.py --cpu` |
| Numba compilation error | Cache corrupt | `Remove-Item -Recurse pipeline\__pycache__` |
| `illumination.tif` all zeros | Old corrupt file cached | `Remove-Item results\illumination.tif -Force` |
| DPSR = 0 after filter | Components all < 5 px | Normal for small crops; run on full DEM |
| Unicode encode error | Terminal not UTF-8 | Check for `->` vs `→` in print statements |
| Out of GPU VRAM | DEM too large for GPU | Use `--cpu` or tile the DEM |

### Clear Numba cache (force recompile)
```powershell
Remove-Item -Recurse -Force pipeline\__pycache__
Remove-Item -Recurse -Force __pycache__
```

### Delete all generated outputs and restart
```powershell
Remove-Item results\*.tif -Force
Remove-Item results\*.aux.xml -Force
Remove-Item images\*.png -Force
python main.py --annual --gpu
```

---

## 9. Key Parameters

### `pipeline/utils.py` — illumination and shared constants

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SUN_ELEVATION` | `1.54` | Peak solar elevation at south pole (degrees) |
| `SUN_AZIMUTH` | `0.0` | Single-epoch azimuth (0=North, 90=East) |
| `N_ANGLES` | `72` | Illumination sweep azimuths (annual mode) |
| `MAX_DISTANCE` | `2500` | Illumination ray length in pixels (50 km) |
| `CELLSIZE` | `20.0` | DEM pixel size in metres |

### `pipeline/step_dpsr_obrien.py` — DPSR kernel

| Parameter | Default | Paper | Meaning |
|-----------|---------|-------|---------|
| `n_angles` | `360` | `720` | DPSR ray directions |
| `max_dist` | `2500` | `7500` | DPSR ray length in pixels |
| `moon_r` | `1737400.0` | `1737400.0` | Lunar radius in metres |
| `min_component` | `5` | `5` | Min DPSR cluster size (pixels) |

---

## 10. Project File Structure

```
ISRO_Hackathon/
├── data/
│   ├── ldem_85s_20m_float.lbl          LOLA DEM (PDS3 label)
│   ├── ldem_85s_20m_float.img          LOLA DEM (binary data)
│   └── LPSR_80S_20MPP_ADJ.shp/.dbf/.prj/.shx  PSR shapefile (80°S–90°S)
├── pipeline/
│   ├── utils.py                         Shared constants and paths
│   ├── step01_load.py                   DEM + PSR loading helpers
│   ├── step02_precompute_rays.py        Bresenham ray tables
│   ├── step03_numba_raytrace.py         Legacy CPU DPSR kernel
│   ├── step_illumination.py             Annual solar shadow map (GPU/CPU)
│   └── step_dpsr_obrien.py              O'Brien & Byrne (2022) DPSR — active
├── results/                             Generated TIF outputs
├── images/                              Generated PNG previews
├── main.py                              Single command to run all steps
├── debug_validation.py                  Shackleton coordinate diagnostics
├── debug_pipeline.py                    10-check pipeline sanity test
├── validate_science.py                  Scientific validation + crater checks
├── COMMANDS.md                          This file
└── STEPS.md                             Narrative step-by-step guide
```

---

## 11. Expected Scientific Results

Per O'Brien & Byrne (2022) for the lunar south pole at 30 m resolution:

| Quantity | Expected | Notes |
|----------|----------|-------|
| PSR / total DEM area | ~6–8% | Depends on latitude coverage |
| DPSR / PSR area | ~0.04% | Extremely rare — small embedded craters |
| DPSR / total area | ~0.005% | |
| Largest DPSR | ~0.27 km² | Fibiger / Nansen F crater |
| DPSR location | Floors of large PSR craters | Shackleton, Haworth, Faustini |

Our 20 m DEM may resolve slightly more DPSRs than the 30 m paper results.
The open PSR floor of Shackleton is NOT itself DPSR — the DPSRs within
Shackleton are small sub-craters (< 600 m across) on the floor.
