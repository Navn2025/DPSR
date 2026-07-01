# `dpsr/` — Double Permanently Shadowed Region Extraction

Canonical, modular implementation of the **DPSR** algorithm from
**O'Brien & Byrne (2022)**, *Double Shadows at the Lunar Poles* (PSJ 3, 258).

A pixel is **DPSR** if it is inside a PSR **and** from every azimuth it has no
line-of-sight to any illuminated (non-PSR) terrain — i.e. it is shadowed *and*
sees only shadowed terrain. This is the coldest, most ice-stable class of
lunar surface.

```
DPSR pixel P  ⟺  psr_mask[P] == 1
                  AND  ∀ azimuth a:  ∄ visible Q along a with psr_mask[Q] == 0
```

---

## How It Works

| Step | File | Description |
|------|------|-------------|
| 1 | `step01_load_dem.py` | Load LOLA DEM (15168², 20 m/px) |
| 2 | `step02_load_psr.py` | Rasterize LOLA PSR shapefile to the DEM grid |
| 3 | `step03_precompute_rays.py` | Precompute Bresenham ray tables (`N_ANGLES`, `MAX_DIST`) |
| 4 | `step04_visibility.py` | Curvature-corrected horizon / visibility test |
| 5 | `step05_compute_dpsr.py` | Numba `@njit(parallel=True)` / CUDA DPSR kernel |
| 6 | `step06_remove_small_regions.py` | 8-connected filter, drops clusters < 5 px |
| 7 | `step07_validation.py` | Spot-check named craters (Shackleton, Faustini, …) |

**Curvature correction** (Eq. A4): `tan(μ) = R1·(R2 − √(d²+R1²)) / (d·R2)`
depresses the apparent horizon by ≈720 m at 50 km — omitting it would let deep
PSR floors falsely "see" distant sunlit terrain below the true curved horizon.

**Ordering detail:** the visibility check runs *before* the running horizon
maximum is updated each ray step; reversing it would mark every new local
maximum as visible.

---

## Run

```powershell
python -m dpsr.run_pipeline
```

Configuration (angles, max distance, paths) lives in `utils.py`.

---

## Parameters (deviations from the paper)

| Parameter | Paper | Here | Reason |
|-----------|-------|------|--------|
| `N_ANGLES` | 720 (0.5°) | 360 (1°) | < 3 % error, 2× faster |
| `MAX_DIST` | 7500 px (150 km) | 2500 px (50 km) | Covers all south-polar PSR craters, 3× faster |
| `min_component` | 5 px | 5 px | unchanged |

---

## Output

`DPSR.tif` — binary mask (1 = doubly-shadowed). Globally ≈ 0.008 % of pixels;
Shackleton's floor yields **0** DPSR px (wide bowl retains rim line-of-sight),
while Faustini/Haworth/Shoemaker have the most — a real, non-obvious result.

See also `pipeline/` (the earlier/legacy DPSR implementation) and `dpsr_fast.py`
(single-file version).
