"""
utils.py
========
Shared utilities: logging, timing, array statistics, memory accounting.

Mirrors dop/utils.py conventions so outputs are visually consistent
across every sub-module in this project.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, name: str = "diviner_pipeline") -> logging.Logger:
    """Create a dual-sink logger: DEBUG to file, INFO to stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "diviner_pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
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
    """Lightweight wall-clock timer (same interface as dop/utils.py)."""

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

_STATS_MAX_PX = 5_000_000   # sample cap to avoid OOM on 230M-pixel arrays


def array_stats(
    arr: np.ndarray,
    nodata: float = -9999.0,
    label: str = "",
) -> dict:
    """
    Compute descriptive statistics on a floating-point array, ignoring
    nodata, NaN, and Inf values.

    For arrays with more than _STATS_MAX_PX valid pixels a random subsample
    is used so that numpy's float64 promotions stay within memory limits.
    Min and max are always computed on the full valid set.
    """
    mask  = np.isfinite(arr) & (arr != nodata)
    valid = arr[mask]

    if valid.size == 0:
        return {
            "label":  label,
            "min":    float("nan"),
            "max":    float("nan"),
            "mean":   float("nan"),
            "median": float("nan"),
            "std":    float("nan"),
            "nan":    int(np.isnan(arr).sum()),
            "inf":    int(np.isinf(arr).sum()),
            "valid":  0,
            "total":  int(arr.size),
        }

    v_min = float(valid.min())
    v_max = float(valid.max())
    n_valid = int(valid.size)

    # Subsample for large arrays to keep float64 promotions within RAM
    sample = valid
    if valid.size > _STATS_MAX_PX:
        idx    = np.random.default_rng(seed=0).choice(valid.size, _STATS_MAX_PX, replace=False)
        sample = valid[idx]

    return {
        "label":  label,
        "min":    v_min,
        "max":    v_max,
        "mean":   float(sample.mean()),
        "median": float(np.median(sample)),
        "std":    float(sample.std(dtype=np.float64)),
        "nan":    int(np.isnan(arr).sum()),
        "inf":    int(np.isinf(arr).sum()),
        "valid":  n_valid,
        "total":  int(arr.size),
    }


def percentile_stats(
    arr: np.ndarray,
    nodata: float,
    pcts: Sequence[float],
) -> Dict[float, float]:
    """Return {percentile: value} for valid pixels (sampled if very large)."""
    valid = arr[(arr != nodata) & np.isfinite(arr)]
    if valid.size == 0:
        return {p: float("nan") for p in pcts}
    if valid.size > _STATS_MAX_PX:
        valid = np.random.default_rng(seed=0).choice(valid, _STATS_MAX_PX, replace=False)
    vals = np.percentile(valid, list(pcts))
    return {p: float(v) for p, v in zip(pcts, vals)}


def log_stats(stats: dict, logger: logging.Logger) -> None:
    """Print a statistics dict through the logger."""
    lbl    = stats.get("label", "")
    prefix = f"  [{lbl}]" if lbl else " "
    logger.info(prefix)
    logger.info(f"    Min    : {stats['min']:.6e}")
    logger.info(f"    Max    : {stats['max']:.6e}")
    logger.info(f"    Mean   : {stats['mean']:.6e}")
    logger.info(f"    Median : {stats['median']:.6e}")
    logger.info(f"    Std    : {stats['std']:.6e}")
    logger.info(f"    NaN    : {stats['nan']}")
    logger.info(f"    Inf    : {stats['inf']}")
    logger.info(f"    Valid  : {stats['valid']:,} / {stats['total']:,}")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024 ** 2


class MemoryTracker:
    """
    Tracks cumulative peak array footprint without psutil.
    Values are approximate — nbytes of registered numpy arrays only.
    """

    def __init__(self) -> None:
        self._current_mb = 0.0
        self._peak_mb    = 0.0

    def add(self, *arrays: np.ndarray) -> None:
        self._current_mb += sum(memory_mb(a) for a in arrays)
        self._peak_mb     = max(self._peak_mb, self._current_mb)

    def remove(self, *arrays: np.ndarray) -> None:
        self._current_mb -= sum(memory_mb(a) for a in arrays)
        self._current_mb  = max(self._current_mb, 0.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def current_mb(self) -> float:
        return self._current_mb


# ---------------------------------------------------------------------------
# Section banner (matches dop/utils.py)
# ---------------------------------------------------------------------------

def section(title: str, logger: logging.Logger, width: int = 62) -> None:
    bar = "=" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)
