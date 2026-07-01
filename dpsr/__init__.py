"""
dpsr  —  DPSR extraction pipeline following O'Brien & Byrne (2022).

Reference
---------
O'Brien, P. & Byrne, S. (2022). Double Shadows at the Lunar Poles.
Planetary Science Journal, 3, 258.  https://doi.org/10.3847/PSJ/ac9d4e

Module structure
----------------
step01_load_dem.py          Load LOLA DEM → float32 metres
step02_load_psr.py          Load binary PSR mask → uint8
step03_precompute_rays.py   Bresenham ray tables → int32/float32 arrays
step04_visibility.py        Curvature-corrected visibility kernel (CPU + GPU)
step05_compute_dpsr.py      Orchestrate steps 01–04; save raw DPSR
step06_remove_small_regions Remove < 5-pixel connected components (paper spec)
step07_validation.py        Validate against Shackleton, Faustini, etc.
utils.py                    Constants, paths, logging, parameter justifications

Quick start
-----------
    python -m dpsr.step05_compute_dpsr           # CPU, default params
    python -m dpsr.step05_compute_dpsr --gpu     # CUDA GPU
    python -m dpsr.step06_remove_small_regions
    python -m dpsr.step07_validation

Or run the full pipeline in one command:
    python -m dpsr.run_pipeline
"""

__version__ = "1.0.0"
__author__  = "ISRO Hackathon Team"
__ref__     = "O'Brien & Byrne (2022), PSJ 3:258"
