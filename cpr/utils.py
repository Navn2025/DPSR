"""
utils.py
========
Shared utilities: logging, timing, array statistics.
"""

import logging
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, name: str = "cpr_pipeline") -> logging.Logger:
    """Create a logger that writes to both a log file and stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cpr_pipeline.log"

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


def section(title: str, logger: logging.Logger, width: int = 62) -> None:
    bar = "=" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)
