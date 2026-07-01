"""
utils.py  --  Logging, timing, and helper utilities for the validation pipeline.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, name: str = "validation") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.info(f"Log file: {(log_dir / name).with_suffix('.log')}")
    return log


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def __str__(self) -> str:
        e = self.elapsed()
        if e < 60:
            return f"{e:.2f} s"
        return f"{e/60:.1f} min"

    def reset(self):
        self._start = time.perf_counter()


# ---------------------------------------------------------------------------
# Array statistics
# ---------------------------------------------------------------------------

def array_stats(arr: np.ndarray, nodata=None, label: str = "") -> dict:
    """Return a statistics dict for valid pixels in arr."""
    a = arr.astype(np.float64).ravel()
    mask = np.isfinite(a)
    if nodata is not None and not np.isnan(nodata):
        mask &= (a != nodata)
    mask &= (a >= 0)   # CPR must be non-negative

    valid = a[mask]
    n_tot = a.size
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())

    stats = {
        "label":  label,
        "total":  n_tot,
        "valid":  valid.size,
        "nan":    n_nan,
        "inf":    n_inf,
        "nodata": n_tot - valid.size - n_nan - n_inf,
    }
    if valid.size > 0:
        pcts = np.percentile(valid, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        stats.update({
            "min":    float(valid.min()),
            "max":    float(valid.max()),
            "mean":   float(valid.mean()),
            "median": float(np.median(valid)),
            "std":    float(valid.std()),
            "p01": pcts[0], "p05": pcts[1], "p10": pcts[2],
            "p25": pcts[3], "p50": pcts[4], "p75": pcts[5],
            "p90": pcts[6], "p95": pcts[7], "p99": pcts[8],
        })
    return stats


def log_stats(stats: dict, log: logging.Logger):
    lbl = stats.get("label", "")
    log.info(f"  [{lbl}]")
    for k in ("valid", "total", "nan", "inf", "min", "max", "mean", "median", "std"):
        if k in stats:
            v = stats[k]
            if isinstance(v, float):
                log.info(f"    {k:8s}: {v:.6g}")
            else:
                log.info(f"    {k:8s}: {v:,}")
    pct_keys = [f"p{p:02d}" for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
    pcts = {k: stats[k] for k in pct_keys if k in stats}
    if pcts:
        log.info("    Percentiles:")
        for k, v in pcts.items():
            log.info(f"      {k} = {v:.4f}")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024**2


def section(title: str, log: logging.Logger):
    bar = "=" * 62
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


def mask_valid(arr: np.ndarray, nodata) -> np.ndarray:
    """Return boolean mask: True where arr is a valid CPR pixel."""
    m = np.isfinite(arr) & (arr >= 0)
    if nodata is not None and not np.isnan(nodata):
        m &= (arr != nodata)
    return m
