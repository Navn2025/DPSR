"""
utils.py
========
Shared utilities: logging, timing, memory reporting, console display.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Attach two handlers to the root logger:
      FileHandler   → DEBUG level  (every detail saved to disk)
      StreamHandler → INFO  level  (clean console output)

    Clears existing handlers so repeated calls in interactive sessions
    do not produce duplicate output.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt     = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    root.addHandler(fh)
    root.addHandler(ch)

    logger = logging.getLogger("pipeline")
    logger.info(f"Log → {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped under 'pipeline.<name>'."""
    return logging.getLogger(f"pipeline.{name}")


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """
    Context-manager stopwatch.

    with Timer("My step", log) as t:
        heavy_work()
    print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self, label: str, logger: Optional[logging.Logger] = None):
        self.label   = label
        self.logger  = logger
        self.elapsed = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._t0
        msg = f"[{self.label}] done in {self.elapsed:.2f}s"
        (self.logger.info if self.logger else print)(msg)


# ── Memory ────────────────────────────────────────────────────────────────────

def memory_mb() -> float:
    """Current process RSS in MB; -1 if psutil unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return -1.0


def memory_str() -> str:
    mb = memory_mb()
    return f"{mb:.1f} MB" if mb >= 0 else "N/A (pip install psutil)"


def warn_large(label: str, nbytes: int, threshold_mb: float) -> None:
    mb = nbytes / 1_048_576
    if mb > threshold_mb:
        get_logger("utils").warning(
            f"[{label}] large array: {mb:.0f} MB — consider tiled processing."
        )


# ── Console display ───────────────────────────────────────────────────────────

def section(title: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title: str, width: int = 70) -> None:
    print(f"\n{'-' * width}\n  {title}\n{'-' * width}")


def progress(idx: int, total: int, label: str) -> None:
    w = len(str(total))
    print(f"  [{idx:>{w}}/{total}]  {label}")
