"""
main.py
=======
Entry point for the Chandrayaan-2 DFSAR Full-Pol SLI -> Degree of
Polarization (DOP) pipeline.

Run from the project root:
    python dop/main.py

or from inside the dop/ folder:
    python main.py

Pipeline steps
--------------
1.  Read metadata from all four SLI GeoTIFFs (HH, HV, VH, VV).
2.  Validate geometric consistency (dimensions, CRS, transform).
3.  Construct the complex scattering matrix S = [[HH, HV], [VH, VV]].
4.  Compute amplitude, power, phase per polarisation.
5.  Build the multilooked covariance matrix C3 (lexicographic basis).
6.  Compute Stokes parameters S0, S1, S2, S3 from C3.
7.  Compute DOP = sqrt(S1^2+S2^2+S3^2) / S0, masking invalid pixels.
8.  (Speckle reduction is intrinsic to step 5/6 -- see note below.)
9.  Statistics: min/max/mean/median/std/percentiles/histogram.
10. Publication-quality preview PNGs.
11. Write Calculated_DOP.tif (float32, LZW).
12. Validation / diagnostic report.

Formula
-------
    S_XP  = (S_HV + S_VH) / 2                         (reciprocity average)
    C3    = < [HH, sqrt2*XP, VV]^T . [HH, sqrt2*XP, VV]^H >   (multilooked)
    S0    = C11 + C22/2
    S1    = C11 - C22/2
    S2    = sqrt2 * Re(C12)
    S3    = sqrt2 * Im(C12)
    DOP   = sqrt(S1^2 + S2^2 + S3^2) / S0

References
----------
    Lee, J.S. & Pottier, E. (2009), "Polarimetric Radar Imaging",
    CRC Press -- covariance/coherency matrix formulation.
    van Zyl, J.J. & Kim, Y. (2011), "Synthetic Aperture Radar
    Polarimetry", JPL Space Science & Technology Series -- Stokes
    scattering operator for a fixed transmit illumination.
    Born, M. & Wolf, E., "Principles of Optics", Ch. 10 -- classical
    Stokes parameters and degree of polarization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
from rasterio.windows import Window

import config as cfg
from utils import (
    setup_logger, Timer, array_stats, log_stats, percentiles,
    section, memory_mb, MemoryTracker,
)
from reader import read_metadata, print_metadata, load_complex
from validator import validate_rasters
from complex_builder import amplitude_power_phase, multilook_power
from covariance import build_covariance_matrix
from stokes import compute_stokes_parameters
from dop import compute_dop
from visualizer import save_power_previews, save_dop_preview, save_dop_histogram


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
for d in (cfg.PREV_DIR, cfg.DOP_DIR, cfg.LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

log = setup_logger(cfg.LOG_DIR)
pipeline_timer = Timer()
mem = MemoryTracker()


# ===========================================================================
# STEP 1  --  Read metadata
# ===========================================================================
section("STEP 1  |  Read SLI Metadata", log)

metas: dict = {}
for pol, path in cfg.SLI_PATHS.items():
    meta = read_metadata(path)
    metas[pol] = meta
    log.info("-" * 50)
    print_metadata(pol, meta, log)

ref = metas["HH"]
H, W = ref["height"], ref["width"]
log.info(f"\n  Scene dimensions : {W} samples (range) x {H} lines (azimuth)")
log.info(f"  Pixel spacing    : {ref['res'][0]} m (range) x {ref['res'][1]} m (azimuth)")

# Any channel with an explicit NoData sentinel? (SLI is typically
# ungeoreferenced with nodata=None, but check honestly rather than assume.)
has_nodata_sentinel = any(m["nodata"] is not None for m in metas.values())
log.info(f"  Explicit NoData sentinel present in any channel: {has_nodata_sentinel}")


# ===========================================================================
# STEP 2  --  Validate consistency
# ===========================================================================
section("STEP 2  |  Validate Raster Consistency", log)
validate_rasters(metas)


# ===========================================================================
# STEP 3  --  Construct the complex scattering matrix S
# ===========================================================================
section("STEP 3  |  Construct Complex Scattering Matrix S", log)

S = {}
input_invalid = np.zeros((H, W), dtype=bool)
for pol in ("HH", "HV", "VH", "VV"):
    t0 = Timer()
    arr = load_complex(cfg.SLI_PATHS[pol], pol)
    S[pol] = arr
    mem.add(arr)
    log.info(f"  {pol} loaded in {t0}  ({memory_mb(arr):.0f} MB)")

    nodata_val = metas[pol]["nodata"]
    invalid = ~np.isfinite(arr.real) | ~np.isfinite(arr.imag)
    if nodata_val is not None:
        invalid |= (arr.real == nodata_val) | (arr.imag == nodata_val)
    input_invalid |= invalid

log.info(
    f"  S = [[S_HH, S_HV], [S_VH, S_VV]] assembled.  "
    f"Input-invalid pixels: {int(input_invalid.sum()):,} / {H * W:,}"
)


# ===========================================================================
# STEP 4  --  Amplitude / Power / Phase per polarisation
# ===========================================================================
section("STEP 4  |  Amplitude, Power, Phase per Polarisation", log)

ml_powers = {}
for pol in ("HH", "HV", "VH", "VV"):
    apr = amplitude_power_phase(S[pol], pol)
    st_amp = array_stats(apr["amplitude"], nodata=cfg.NODATA, label=f"{pol} amplitude")
    log_stats(st_amp, log)

    ml_powers[pol] = multilook_power(apr["power"], cfg.MULTILOOK_WINDOW, pol)
    mem.add(ml_powers[pol])
    del apr   # single-look amplitude/power/phase not needed beyond reporting


# ===========================================================================
# STEP 5  --  Multilooked covariance matrix C3
# ===========================================================================
section("STEP 5  |  Build Multilooked Covariance Matrix (C3)", log)

t0 = Timer()
C3 = build_covariance_matrix(S["HH"], S["HV"], S["VH"], S["VV"], cfg.MULTILOOK_WINDOW)
log.info(f"  C3 built in {t0}")

for k in ("C11", "C22", "C33", "C12", "C13", "C23", "span"):
    mem.add(C3[k])

for pol in ("HH", "HV", "VH", "VV"):
    mem.remove(S[pol])
S.clear()
log.info("  Single-look scattering matrix S freed from memory.")


# ===========================================================================
# STEP 6  --  Stokes parameters
# ===========================================================================
section("STEP 6  |  Compute Stokes Parameters (S0, S1, S2, S3)", log)

t0 = Timer()
stokes = compute_stokes_parameters(C3)
log.info(f"  Stokes parameters computed in {t0}")
for k in ("S0", "S1", "S2", "S3"):
    mem.add(stokes[k])


# ===========================================================================
# STEP 7  --  Degree of Polarization
# ===========================================================================
section("STEP 7  |  Compute DOP = sqrt(S1^2+S2^2+S3^2) / S0", log)

log.info(f"  epsilon = {cfg.EPSILON}   nodata = {cfg.NODATA}")
t0 = Timer()
dop = compute_dop(
    stokes["S0"], stokes["S1"], stokes["S2"], stokes["S3"],
    epsilon=cfg.EPSILON, nodata=cfg.NODATA,
    input_invalid_mask=input_invalid if has_nodata_sentinel else None,
)
mem.add(dop)
log.info(f"  DOP computed in {t0}")


# ===========================================================================
# STEP 8  --  Speckle reduction (multilook)
# ===========================================================================
section("STEP 8  |  Speckle Reduction via Multilook", log)

az, rg = cfg.MULTILOOK_WINDOW
log.info(
    f"  DOP was NOT computed from single-look pixels: the covariance matrix "
    f"C3 (STEP 5) was built by multilooking the per-pixel Hermitian products "
    f"with a ({az}az x {rg}rg) = {az * rg}-look box-car window -- the same "
    f"window used by the CPR pipeline (cpr/config.py MULTILOOK_WINDOW), so "
    f"CPR and DOP are directly comparable at the same effective resolution."
)


# ===========================================================================
# STEP 9  --  DOP statistics
# ===========================================================================
section("STEP 9  |  DOP Statistics & Histogram Data", log)

dop_stats = array_stats(dop, nodata=cfg.NODATA, label="DOP")
log_stats(dop_stats, log)

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
dop_pcts = percentiles(dop, cfg.NODATA, pcts)
log.info("  Percentiles:")
for p in pcts:
    log.info(f"    P{p:02d} = {dop_pcts[p]:.4f}")


# ===========================================================================
# STEP 10  --  Preview images
# ===========================================================================
section("STEP 10  |  Save Preview Images", log)

log.info(f"Downsample factor: 1:{cfg.PREVIEW_DOWNSAMPLE}")
save_power_previews(ml_powers, cfg.PREV_DIR, downsample=cfg.PREVIEW_DOWNSAMPLE, nodata=0.0)
save_dop_preview(dop, cfg.PREV_DIR, downsample=cfg.PREVIEW_DOWNSAMPLE, nodata=cfg.NODATA)
save_dop_histogram(dop, cfg.PREV_DIR, nodata=cfg.NODATA)
log.info(f"  All previews saved to: {cfg.PREV_DIR}")


# ===========================================================================
# STEP 11  --  Write DOP GeoTIFF
# ===========================================================================
section("STEP 11  |  Save Calculated_DOP.tif", log)

out_tif = cfg.DOP_DIR / cfg.DOP_OUTPUT_NAME
out_profile = ref["profile"].copy()
out_profile.update(
    dtype    = "float32",
    count    = 1,
    nodata   = cfg.NODATA,
    compress = "lzw",
)

log.info(f"Writing {out_tif.name} ({W}W x {H}H, float32, LZW) ...")
t0 = Timer()

with rasterio.open(out_tif, "w", **out_profile) as dst:
    for row_start in range(0, H, cfg.WRITE_BLOCK):
        row_end = min(row_start + cfg.WRITE_BLOCK, H)
        nrows   = row_end - row_start
        win     = Window(0, row_start, W, nrows)
        dst.write(dop[row_start:row_end, :], 1, window=win)

    dst.update_tags(
        PIPELINE     = "Chandrayaan-2 DFSAR Full-Pol SLI -> DOP",
        FORMULA      = "sqrt(S1^2+S2^2+S3^2) / S0  (H-transmit Stokes parameters)",
        REFERENCE    = "van Zyl & Kim (2011) SAR Polarimetry; Lee & Pottier (2009)",
        MULTILOOK    = str(cfg.MULTILOOK_WINDOW),
        EPSILON      = str(cfg.EPSILON),
        NODATA       = str(cfg.NODATA),
        INPUT_HH     = cfg.SLI_PATHS["HH"].name,
        INPUT_HV     = cfg.SLI_PATHS["HV"].name,
        INPUT_VH     = cfg.SLI_PATHS["VH"].name,
        INPUT_VV     = cfg.SLI_PATHS["VV"].name,
        DOP_MEDIAN   = f"{dop_stats['median']:.6f}",
        DOP_MEAN     = f"{dop_stats['mean']:.6f}",
    )

log.info(f"  Done in {t0}")
log.info(f"  Saved : {out_tif}")
log.info(f"  Size  : {out_tif.stat().st_size / 1024**2:.1f} MB")


# ===========================================================================
# STEP 12  --  Validation report
# ===========================================================================
section("STEP 12  |  Validation Report", log)

n_total  = H * W
n_valid  = int(dop_stats["valid"])
n_nodata = int((dop == cfg.NODATA).sum())
n_nan    = int(dop_stats["nan"])
n_inf    = int(dop_stats["inf"])
n_invalid = n_total - n_valid

log.info(f"""
  -------------------------------------------------------
  Chandrayaan-2 DFSAR Full-Pol SLI -> DOP Pipeline
  -------------------------------------------------------
  Scene         : {cfg.SCENE}  ({cfg._DATE})
  Formula       : DOP = sqrt(S1^2+S2^2+S3^2) / S0
                  S0 = C11 + C22/2   S1 = C11 - C22/2
                  S2 = sqrt2*Re(C12) S3 = sqrt2*Im(C12)
  References    : Lee & Pottier (2009); van Zyl & Kim (2011)

  INPUT RASTERS
  -------------------------------------------------------
  Dimensions    : {W} x {H}  (range x azimuth)
  Bands         : 2 per file (Band1=Real, Band2=Imag)
  Dtype         : float32 (complex SLC, I/Q)

  PROCESSING
  -------------------------------------------------------
  Multilook     : {cfg.MULTILOOK_WINDOW}  az x rg  = {az * rg} looks
  Epsilon       : {cfg.EPSILON}

  DOP STATISTICS
  -------------------------------------------------------
  Min           : {dop_stats['min']:.6f}
  Max           : {dop_stats['max']:.6f}
  Mean          : {dop_stats['mean']:.6f}
  Median        : {dop_stats['median']:.6f}
  Std           : {dop_stats['std']:.6f}
  Valid pixels  : {n_valid:,} / {n_total:,}  ({100*n_valid/n_total:.2f}%)
  Invalid pixels: {n_invalid:,}  ({100*n_invalid/n_total:.2f}%)
  NoData        : {n_nodata:,}
  NaN / Inf     : {n_nan:,} / {n_inf:,}

  RESOURCE USAGE
  -------------------------------------------------------
  Peak tracked array memory : {mem.peak_mb:.0f} MB
  Total runtime              : {pipeline_timer}

  OUTPUT FILES
  -------------------------------------------------------
  DOP GeoTIFF   : {out_tif}
  HH/HV/VH/VV   : {cfg.PREV_DIR / 'HH.png'} (+ HV/VH/VV)
  DOP preview   : {cfg.PREV_DIR / 'DOP.png'}
  DOP histogram : {cfg.PREV_DIR / 'Histogram.png'}
  Log file      : {cfg.LOG_DIR / 'dop_pipeline.log'}
  -------------------------------------------------------
""")

log.info("Pipeline completed successfully.")
