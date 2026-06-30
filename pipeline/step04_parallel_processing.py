"""
step04_parallel_processing.py — Multiprocessing fallback using SharedMemory.

When to use this module vs step03
----------------------------------
  step03 (Numba prange)          ← RECOMMENDED for most users
    • Single process, all cores via OpenMP threads
    • Shared memory automatically — no overhead
    • Simpler to run

  step04 (this module)           ← ALTERNATIVE / FALLBACK
    • N separate OS processes, each running a Numba serial kernel
    • Useful when:
        – Numba parallel=True does not scale on your CPU/OS
        – You want to mix Numba-compiled and non-Numba workers
        – You need fine-grained progress reporting per chunk
    • Uses Python's SharedMemory so the 920 MB DEM is NOT copied
      per process — all workers read the same physical RAM pages.

Architecture
------------
  Main process
    │
    ├── Creates SharedMemory blocks for elevation + illumination
    ├── Splits PSR pixel list into CHUNK_SIZE chunks
    │
    └── ProcessPoolExecutor  (N_WORKERS = cpu_count())
           ├── Worker 0  → classify_chunk(chunk_0)
           ├── Worker 1  → classify_chunk(chunk_1)
           ├── ...
           └── Worker N  → classify_chunk(chunk_N)
    │
    └── Collects results → concatenate → dpsr_flags uint8 (P,)

Memory model
------------
  Large arrays (elevation, illumination) live in SharedMemory blocks.
  Each worker attaches to the same blocks by name — zero-copy sharing.
  Small arrays (PSR chunk rows/cols, ray tables) are passed normally
  via pickle (they are small: a 200 k-pixel chunk of int32 = ~0.8 MB).

Complexity
----------
  Same O(P × A × D) algorithm; parallelism is at the process level.
  Effective wall time ≈ O(P × A × D) / N_WORKERS  (ignoring overhead).

Windows compatibility
---------------------
  Windows uses 'spawn' (not 'fork') for multiprocessing.
  All worker initialisation is done inside worker functions (not at module
  import time) to avoid pickling errors under 'spawn'.
  SharedMemory is fully supported on Windows (Python 3.8+).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).parent.parent          # ISRO_Hackathon/
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from multiprocessing import get_context

import numpy as np

from pipeline.utils import CHUNK_SIZE, get_logger

log = get_logger(__name__)


# ── Worker function (runs in each subprocess) ──────────────────────────────────

def _worker(
    chunk_rows:     np.ndarray,   # int32 (chunk_size,)
    chunk_cols:     np.ndarray,   # int32 (chunk_size,)
    shm_elev_name:  str,
    shm_ill_name:   str,
    elev_shape:     tuple,
    ill_shape:      tuple,
    ray_dr:         np.ndarray,
    ray_dc:         np.ndarray,
    ray_dist:       np.ndarray,
    ray_len:        np.ndarray,
) -> np.ndarray:                  # uint8 (chunk_size,)
    """
    Process one chunk of PSR pixels inside a worker process.

    Input
    -----
    chunk_rows / chunk_cols : PSR pixel indices for this chunk
    shm_elev_name           : SharedMemory name for elevation array
    shm_ill_name            : SharedMemory name for illumination array
    elev_shape / ill_shape  : array shapes (needed to reconstruct ndarray)
    ray_dr/dc/dist/len      : precomputed Bresenham ray tables

    Output
    ------
    uint8 array of length chunk_size  (1=DPSR, 0=non-DPSR)

    Notes
    -----
    • Imports are inside the function so 'spawn' workers don't need
      the parent's module state.
    • SharedMemory blocks are attached (not created) here; the main
      process owns them and is responsible for cleanup.
    """
    # Attach to shared memory (zero-copy)
    shm_e = shared_memory.SharedMemory(name=shm_elev_name)
    shm_i = shared_memory.SharedMemory(name=shm_ill_name)

    elevation    = np.ndarray(elev_shape, dtype=np.float32, buffer=shm_e.buf)
    illumination = np.ndarray(ill_shape,  dtype=np.uint8,   buffer=shm_i.buf)

    # Import Numba serial kernel (compiled on first call per process;
    # cache=True means compiled binary is reused from disk thereafter)
    from step03_numba_raytrace import classify_chunk

    result = classify_chunk(
        elevation, illumination,
        chunk_rows, chunk_cols,
        ray_dr, ray_dc, ray_dist, ray_len,
    )

    # Detach from shared memory (do NOT unlink — main process owns it)
    shm_e.close()
    shm_i.close()

    return result


# ── Public interface ───────────────────────────────────────────────────────────

def run_parallel(
    elevation:    np.ndarray,   # float32 (H, W)
    illumination: np.ndarray,   # uint8   (H, W)
    psr_rows:     np.ndarray,   # int32   (P,)
    psr_cols:     np.ndarray,   # int32   (P,)
    ray_dr:       np.ndarray,
    ray_dc:       np.ndarray,
    ray_dist:     np.ndarray,
    ray_len:      np.ndarray,
    n_workers:    int | None = None,
    chunk_size:   int        = CHUNK_SIZE,
) -> np.ndarray:                # uint8   (P,)
    """
    Classify all PSR pixels using N worker processes.

    Input
    -----
    elevation / illumination : full raster arrays (kept in SharedMemory)
    psr_rows / psr_cols      : PSR pixel indices  (split into chunks)
    ray_*                    : Bresenham ray tables from step02
    n_workers                : number of worker processes (default: cpu_count())
    chunk_size               : PSR pixels per work unit

    Output
    ------
    dpsr_flags : uint8 (P,)  1=DPSR  0=non-DPSR

    Complexity : O(P × A × D / N_WORKERS) wall time
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)   # keep one core for OS

    n_psr   = len(psr_rows)
    result  = np.zeros(n_psr, dtype=np.uint8)

    # ── Create shared memory for large arrays ──────────────────────────────
    log.info("Creating shared memory blocks …")
    shm_e = shared_memory.SharedMemory(create=True, size=elevation.nbytes)
    shm_i = shared_memory.SharedMemory(create=True, size=illumination.nbytes)

    try:
        # Copy arrays into shared memory (done once — NOT per chunk)
        np.ndarray(elevation.shape,    dtype=elevation.dtype,
                   buffer=shm_e.buf)[:] = elevation
        np.ndarray(illumination.shape, dtype=illumination.dtype,
                   buffer=shm_i.buf)[:] = illumination
        log.info(
            "Shared memory ready: elev=%.0f MB  illum=%.0f MB  workers=%d",
            elevation.nbytes / 1e6, illumination.nbytes / 1e6, n_workers,
        )

        # ── Split PSR pixels into chunks ───────────────────────────────────
        chunk_starts = range(0, n_psr, chunk_size)
        n_chunks     = len(list(chunk_starts))
        log.info("Processing %s PSR pixels in %d chunks of %s …",
                 f"{n_psr:,}", n_chunks, f"{chunk_size:,}")

        t0          = time.perf_counter()
        done        = 0

        ctx = get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            # Submit all chunks
            futures = {}
            for start in range(0, n_psr, chunk_size):
                end   = min(start + chunk_size, n_psr)
                fut   = pool.submit(
                    _worker,
                    psr_rows[start:end].copy(),   # small array — fast pickle
                    psr_cols[start:end].copy(),
                    shm_e.name,
                    shm_i.name,
                    elevation.shape,
                    illumination.shape,
                    ray_dr, ray_dc, ray_dist, ray_len,
                )
                futures[fut] = (start, end)

            # Collect results as they complete
            for fut in as_completed(futures):
                start, end = futures[fut]
                result[start:end] = fut.result()
                done += (end - start)

                elapsed = time.perf_counter() - t0
                rate    = done / max(elapsed, 1e-6)
                eta     = (n_psr - done) / max(rate, 1)
                log.info(
                    "  %6.2f%%  %s / %s px  %.0f px/s  ETA %.0f s",
                    100 * done / n_psr,
                    f"{done:,}", f"{n_psr:,}",
                    rate, eta,
                )

        elapsed = time.perf_counter() - t0
        log.info(
            "Parallel processing done in %.1f s (%.1f min)  DPSR=%s",
            elapsed, elapsed / 60, f"{result.sum():,}",
        )

    finally:
        # Always release shared memory, even on exception
        shm_e.close(); shm_e.unlink()
        shm_i.close(); shm_i.unlink()

    return result
