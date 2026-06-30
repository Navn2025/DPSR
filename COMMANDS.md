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
python -m pip install rasterio numpy geopandas matplotlib scipy numba
```

### Verify installation
```powershell
python -c "import rasterio, numpy, geopandas, matplotlib, numba; print('All OK')"
```

### Check GPU availability
```powershell
python -c "from numba import cuda; print('GPU available:', cuda.is_available())"
```

---

## 2. Run the Full Pipeline

### Recommended — single-epoch illumination, auto GPU/CPU
```powershell
python main.py
```

### Best accuracy — annual illumination (sweeps all 72 sun azimuths)
```powershell
python main.py --annual
```

### Annual illumination + GPU (fastest, best accuracy)
```powershell
python main.py --annual --gpu
```

### Force CPU only
```powershell
python main.py --cpu
```

### Force GPU only
```powershell
python main.py --gpu
```

### Recompute everything from scratch
```powershell
python main.py --redo
```

### Recompute with annual illumination
```powershell
python main.py --redo --annual
```

### Recompute with GPU
```powershell
python main.py --redo --gpu
```

---

## 3. Quick Single-File Version

Runs everything in one self-contained script (no pipeline/ imports needed).
Edit `ANNUAL_MODE = True` inside the file for annual illumination.

```powershell
python dpsr_fast.py
```

---

## 4. Run Individual Pipeline Steps

### Step 1 — Load and inspect DEM
```powershell
python -c "from pipeline.step01_load import load_dem; e, m = load_dem(); print(e.shape, e.min(), e.max())"
```

### Step 2 — Precompute Bresenham ray offsets
```powershell
python -c "from pipeline.step02_precompute_rays import precompute_rays; dr,dc,dd,rl = precompute_rays(); print('rays:', dr.shape)"
```

### Solar shadow map only (single epoch)
```powershell
python -m pipeline.step_illumination --az 0 --el 1.54
```

### Solar shadow map — annual (all 72 azimuths)
```powershell
python -m pipeline.step_illumination --annual
```

### CPU DPSR extraction only
```powershell
python -m pipeline.step05_generate_dpsr
```

### CPU DPSR — multiprocessing fallback
```powershell
python -m pipeline.step05_generate_dpsr --mp
```

### GPU DPSR extraction only
```powershell
python -m pipeline.step06_gpu_dpsr
```

---

## 5. Open Jupyter Notebooks

```powershell
.\venv\Scripts\Activate.ps1
jupyter notebook notebooks\
```

| Notebook | Purpose |
|----------|---------|
| `01_basic_pipeline.ipynb` | Step-by-step exploration, original approach |
| `02_full_pipeline.ipynb` | Complete optimised pipeline with Numba |

---

## 6. Output Files

| File | Folder | Description |
|------|--------|-------------|
| `PSR_mask.tif` | `results/` | Rasterized PSR shapefile |
| `slope.tif` | `results/` | Slope map (degrees) |
| `aspect.tif` | `results/` | Aspect map (degrees) |
| `hillshade.tif` | `results/` | Hillshade — visualisation only |
| `illumination.tif` | `results/` | Solar shadow map (single epoch) |
| `illumination_annual.tif` | `results/` | Solar shadow map (annual union) |
| `DPSR.tif` | `results/` | **Final DPSR output** |
| `DPSR_summary.png` | `images/` | PSR / illumination / DPSR comparison |
| `DPSR_comparison.png` | `images/` | PSR vs DPSR side-by-side |

---

## 7. Inspect Outputs

### List results folder
```powershell
Get-ChildItem results\ | Select-Object Name, Length, LastWriteTime
```

### List images folder
```powershell
Get-ChildItem images\ | Select-Object Name, LastWriteTime
```

### Inspect DPSR raster
```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/DPSR.tif') as ds:
    d = ds.read(1)
    print('Shape :', ds.width, 'x', ds.height)
    print('CRS   :', ds.crs)
    print('DPSR pixels :', np.sum(d == 1))
    print('DPSR %       :', round(np.mean(d==1)*100, 2))
"
```

### Check illumination coverage
```powershell
python -c "
import rasterio, numpy as np
with rasterio.open('results/illumination.tif') as ds:
    ill = ds.read(1)
    print('Illuminated :', np.sum(ill==1), '/', ill.size)
    print('Lit %       :', round(np.mean(ill)*100, 2))
"
```

---

## 8. Troubleshooting

### pip installs to wrong Python
```powershell
python -m pip install <package>       # always use python -m pip, never bare pip
```

### Clear Numba cache (force recompile)
```powershell
Remove-Item -Recurse -Force pipeline\__pycache__
Remove-Item -Recurse -Force __pycache__
python main.py
```

### Out of GPU memory
```powershell
python main.py --cpu
```

### Locked output files (cannot delete)
Close QGIS, Jupyter, or any other tool that has the file open, then:
```powershell
Remove-Item results\DPSR.tif
python main.py --redo
```

### Check which Python is active
```powershell
Get-Command python
python --version
```

### Module not found errors
Make sure you are in the project root and the venv is active:
```powershell
cd C:\Users\navne\Desktop\ISRO_Hackathon
.\venv\Scripts\Activate.ps1
python main.py
```

---

## 9. Expected Runtimes

| Step | Mode | Estimated Time |
|------|------|---------------|
| Shadow map (single epoch) | CPU Numba | 1 – 3 min |
| Shadow map (annual, 72 azimuths) | CPU Numba | 1 – 3 hours |
| DPSR ray-casting | CPU Numba parallel | 1 – 5 min |
| DPSR ray-casting | GPU CUDA | 30 – 120 sec |
| Full pipeline (`python main.py`) | CPU | ~5 – 10 min |
| Full pipeline (`--annual --gpu`) | GPU | ~30 – 90 min |

> First run compiles Numba kernels (~20 sec). Subsequent runs skip compilation.
> `MAX_DISTANCE = 2500` (50 km radius) — increase computation vs original 10 km.

---

## 10. Key Parameters (edit in `pipeline/utils.py` or `dpsr_fast.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SUN_ELEVATION` | `1.54` | Sun elevation angle in degrees (peak at 89.5°S) |
| `SUN_AZIMUTH` | `0.0` | Sun azimuth in degrees (0=North, 90=East) |
| `MAX_DISTANCE` | `2500` | Ray length in pixels (2500 × 20 m = 50 km) |
| `N_ANGLES` | `72` | DPSR rays per pixel (one every 5°) |
| `CELLSIZE` | `20.0` | DEM pixel size in metres |
