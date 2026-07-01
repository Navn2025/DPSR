"""
validator.py
============
Dataset integrity verification.

Each raster/vector is tested for:
  - File openability
  - CRS presence
  - Non-zero dimensions
  - Valid affine resolution
  - Readable pixel data (small sample read)
  - Excessive NoData coverage

Validation never raises — errors are collected and reported so the
pipeline can decide whether to skip or attempt best-effort processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import rasterio

from loader import RasterInfo, VectorInfo
from utils import get_logger, section

log = get_logger("validator")


# -- Result container ----------------------------------------------------------

@dataclass
class ValidationResult:
    label:    str
    passed:   bool
    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)


# -- Per-type validators -------------------------------------------------------

def validate_raster(info: RasterInfo) -> ValidationResult:
    """Run all raster integrity checks."""
    W: list[str] = []
    E: list[str] = []

    if not info.valid:
        E.append(f"File could not be opened: {info.error}")
        return ValidationResult(info.label, False, W, E)

    # CRS
    if info.crs in ("Not set", "Unknown", "None"):
        W.append("No CRS — reprojection will assume source matches target CRS.")

    # Dimensions
    if info.width == 0 or info.height == 0:
        E.append(f"Invalid dimensions: {info.width} x {info.height}.")

    # Band count
    if info.band_count == 0:
        E.append("Zero bands reported.")

    # Resolution
    if np.isnan(info.res_x) or np.isnan(info.res_y):
        W.append("NaN resolution — affine transform may be identity.")

    if len(E) > 0:
        return ValidationResult(info.label, False, W, E)

    # Sample read — catch truncated / locked files early
    try:
        with rasterio.open(info.path) as src:
            win = rasterio.windows.Window(
                0, 0,
                min(128, src.width),
                min(128, src.height),
            )
            sample = src.read(1, window=win).astype("float64")
            nd = src.nodata

            if nd is not None:
                valid_px = np.sum(sample != nd)
            else:
                valid_px = np.sum(np.isfinite(sample))

            total_px = sample.size
            nodata_pct = 100.0 * (total_px - valid_px) / max(total_px, 1)

            if valid_px == 0:
                W.append("Sample block is 100% NoData / non-finite.")
            elif nodata_pct > 80:
                W.append(f"Sample block is {nodata_pct:.0f}% NoData.")
    except Exception as exc:
        E.append(f"Sample read failed: {exc}")

    passed = len(E) == 0
    return ValidationResult(info.label, passed, W, E)


def validate_vector(info: VectorInfo) -> ValidationResult:
    """Run all vector integrity checks."""
    W: list[str] = []
    E: list[str] = []

    if not info.valid:
        E.append(f"File could not be opened: {info.error}")
        return ValidationResult(info.label, False, W, E)

    if info.crs in ("Not set", "Unknown", "None"):
        W.append("No CRS — will reproject assuming source matches target.")

    if info.feature_count == 0:
        W.append("Layer contains zero features — will produce an empty mask.")

    passed = len(E) == 0
    return ValidationResult(info.label, passed, W, E)


# -- Batch validator -----------------------------------------------------------

def validate_catalog(
    rasters: list[RasterInfo],
    vectors: list[VectorInfo],
) -> dict[str, ValidationResult]:
    """
    Validate all datasets; return a results dict keyed by label.
    Execution continues regardless of individual failures.
    """
    results: dict[str, ValidationResult] = {}

    for ri in rasters:
        r = validate_raster(ri)
        results[ri.label] = r
        _log_result(r)

    for vi in vectors:
        r = validate_vector(vi)
        results[vi.label] = r
        _log_result(r)

    return results


def _log_result(r: ValidationResult) -> None:
    tag = "PASS" if r.passed else "FAIL"
    log.info(f"[{tag}] {r.label}")
    for w in r.warnings:
        log.warning(f"      {r.label} -> {w}")
    for e in r.errors:
        log.error(f"      {r.label} -> {e}")


# -- Summary display -----------------------------------------------------------

def print_validation_summary(results: dict[str, ValidationResult]) -> None:
    section("STEP 3 — VALIDATION SUMMARY")
    passed = sum(1 for r in results.values() if r.passed)
    failed = len(results) - passed
    print(f"  Passed : {passed}   Failed : {failed}\n")

    for label, r in results.items():
        mark = "OK" if r.passed else "X"
        print(f"  {mark}  {label}")
        for w in r.warnings:
            print(f"       WARNING : {w}")
        for e in r.errors:
            print(f"       ERROR   : {e}")
