"""
utils.py
========
Shared utilities: logging, timing, array statistics, memory accounting.

Deliberately uses only numpy / logging / pathlib / time so the whole
pipeline stays within the approved library list (no psutil, no GDAL).
"""

import logging
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, name: str = "dop_pipeline") -> logging.Logger:
    """Create a logger that writes to both a log file and stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dop_pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicate handlers on re-runs

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Lightweight wall-clock timer."""

    def __init__(self) -> None:
        self._start: float = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def __str__(self) -> str:
        e = self.elapsed()
        if e < 60:
            return f"{e:.2f} s"
        return f"{e / 60:.1f} min"


# ---------------------------------------------------------------------------
# Array statistics
# ---------------------------------------------------------------------------

def array_stats(arr: np.ndarray, nodata: float = -9999.0, label: str = "") -> dict:
    """
    Compute descriptive statistics on a floating-point array,
    ignoring nodata, NaN, and Inf values.
    """
    mask = np.isfinite(arr) & (arr != nodata)
    valid = arr[mask]

    if valid.size == 0:
        return {
            "label": label, "min": np.nan, "max": np.nan,
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "nan": int(np.isnan(arr).sum()), "inf": int(np.isinf(arr).sum()),
            "valid": 0, "total": int(arr.size),
        }

    return {
        "label":  label,
        "min":    float(valid.min()),
        "max":    float(valid.max()),
        "mean":   float(valid.mean()),
        "median": float(np.median(valid)),
        "std":    float(valid.std()),
        "nan":    int(np.isnan(arr).sum()),
        "inf":    int(np.isinf(arr).sum()),
        "valid":  int(valid.size),
        "total":  int(arr.size),
    }


def percentiles(arr: np.ndarray, nodata: float, pcts) -> dict:
    """Return {p: value} for the requested percentiles over valid pixels."""
    valid = arr[(arr != nodata) & np.isfinite(arr)]
    if valid.size == 0:
        return {p: float("nan") for p in pcts}
    vals = np.percentile(valid, pcts)
    return {p: float(v) for p, v in zip(pcts, vals)}


def log_stats(stats: dict, logger: logging.Logger) -> None:
    """Print a statistics dict through the logger."""
    lbl = stats.get("label", "")
    prefix = f"  [{lbl}]" if lbl else " "
    logger.info(f"{prefix}")
    logger.info(f"    Min    : {stats['min']:.6e}")
    logger.info(f"    Max    : {stats['max']:.6e}")
    logger.info(f"    Mean   : {stats['mean']:.6e}")
    logger.info(f"    Median : {stats['median']:.6e}")
    logger.info(f"    Std    : {stats['std']:.6e}")
    logger.info(f"    NaN    : {stats['nan']}")
    logger.info(f"    Inf    : {stats['inf']}")
    logger.info(f"    Valid  : {stats['valid']} / {stats['total']}")


def memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024 ** 2


class MemoryTracker:
    """
    Tracks cumulative peak array memory footprint without psutil.

    Not a substitute for real process RSS, but gives a reproducible,
    dependency-free "data memory usage" figure for the validation report
    (Step 12), computed purely from the nbytes of arrays we register.
    """

    def __init__(self) -> None:
        self._current_mb = 0.0
        self._peak_mb = 0.0

    def add(self, *arrays: np.ndarray) -> None:
        self._current_mb += sum(memory_mb(a) for a in arrays)
        self._peak_mb = max(self._peak_mb, self._current_mb)

    def remove(self, *arrays: np.ndarray) -> None:
        self._current_mb -= sum(memory_mb(a) for a in arrays)
        self._current_mb = max(self._current_mb, 0.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def current_mb(self) -> float:
        return self._current_mb


def section(title: str, logger: logging.Logger, width: int = 62) -> None:
    bar = "=" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)
