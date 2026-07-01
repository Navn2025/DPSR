"""
normalizer.py
=============
Robust per-layer normalisation.

  normalize_continuous() — percentile-clipped stretch to [0, 1]
  normalize_mask()       — binary float (0.0 / 1.0), NaN preserved
  normalize_layer()      — dispatcher used by the pipeline
  print_norm_report()    — per-band pre/post statistics
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from config import NORM_P_LOW, NORM_P_HIGH
from utils import get_logger

log = get_logger("normalizer")


# -- Core normalisers ----------------------------------------------------------

def normalize_continuous(
    array:  np.ndarray,
    p_low:  int = NORM_P_LOW,
    p_high: int = NORM_P_HIGH,
) -> np.ndarray:
    """
    Stretch *array* to [0, 1] via robust percentile clipping.

    NaN values are preserved in the output.
    Finite values outside [p_low, p_high] percentile are clamped, not discarded.

    Parameters
    ----------
    array  : float32/64 2-D array, may contain NaN
    p_low  : lower percentile (default 2)
    p_high : upper percentile (default 98)
    """
    valid = array[np.isfinite(array)]

    if valid.size == 0:
        log.warning("Array contains no finite values — returning NaN array.")
        return np.full_like(array, np.nan, dtype="float32")

    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))

    if hi <= lo:
        log.warning(
            f"Percentile range collapsed (lo={lo:.4g}, hi={hi:.4g}) "
            f"— returning zero array."
        )
        out = np.zeros_like(array, dtype="float32")
        out[~np.isfinite(array)] = np.nan
        return out

    clipped = np.clip(array, lo, hi)
    normed  = (clipped - lo) / (hi - lo)
    normed  = normed.astype("float32")
    normed[~np.isfinite(array)] = np.nan
    return normed


def normalize_mask(array: np.ndarray) -> np.ndarray:
    """
    Convert a mask-like array to binary float32.

    Any finite value > 0  ->  1.0
    0 or non-positive     ->  0.0
    NaN / Inf             ->  NaN
    """
    finite = np.isfinite(array)
    out    = np.where(finite & (array > 0), 1.0, 0.0).astype("float32")
    out[~finite] = np.nan
    return out


# -- Dispatcher ----------------------------------------------------------------

def normalize_layer(
    array:   np.ndarray,
    label:   str,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Apply the appropriate normalisation strategy for *label*.

    is_mask=True  -> binary normalisation (PSR, DPSR)
    is_mask=False -> percentile stretch   (DEM, Slope, SAR products, …)
    """
    finite = array[np.isfinite(array)]
    pre_min = float(np.min(finite)) if finite.size else float("nan")
    pre_max = float(np.max(finite)) if finite.size else float("nan")

    if is_mask:
        normed = normalize_mask(array)
        log.info(
            f"[{label}] binary mask  raw=[{pre_min:.4g}, {pre_max:.4g}]  "
            f"-> [0, 1]"
        )
    else:
        normed = normalize_continuous(array)
        post_valid = normed[np.isfinite(normed)]
        post_min = float(np.min(post_valid)) if post_valid.size else float("nan")
        post_max = float(np.max(post_valid)) if post_valid.size else float("nan")
        log.info(
            f"[{label}] percentile stretch  raw=[{pre_min:.4g}, {pre_max:.4g}]  "
            f"-> [{post_min:.4g}, {post_max:.4g}]"
        )

    return normed


# -- Reporting -----------------------------------------------------------------

def print_norm_report(
    raw:    dict[str, Optional[np.ndarray]],
    normed: dict[str, Optional[np.ndarray]],
) -> None:
    """Print a compact before/after normalisation table."""
    from utils import section
    section("STEP 7 — NORMALISATION REPORT")
    header = f"  {'Layer':<14}  {'Raw Min':>12}  {'Raw Max':>12}  {'Norm Min':>10}  {'Norm Max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label in raw:
        r = raw.get(label)
        n = normed.get(label)

        def _stat(arr: Optional[np.ndarray], func) -> str:
            if arr is None:
                return "N/A"
            v = arr[np.isfinite(arr)]
            return f"{func(v):.4g}" if v.size else "empty"

        print(
            f"  {label:<14}  "
            f"{_stat(r, np.min):>12}  {_stat(r, np.max):>12}  "
            f"{_stat(n, np.min):>10}  {_stat(n, np.max):>10}"
        )
