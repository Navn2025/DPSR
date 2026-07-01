"""
loader.py
=========
Dataset discovery, classification, and metadata extraction.

DatasetCatalog scans all configured directories, classifies every file
by dataset type and product code, reads rasterio / geopandas metadata,
and exposes the results through typed attributes.

No filenames are hardcoded — discovery is entirely pattern-driven.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
import geopandas as gpd

from config import (
    DEM_DIR, PSR_DIR, DPSR_DIR, DFSAR_DIR,
    DEM_FILE, PSR_FILE, DPSR_FILE,
    RASTER_EXTENSIONS, VECTOR_EXTENSIONS,
    DFSAR_PRODUCT_KEYWORDS, DFSAR_EXCLUDE_DIRS,
)
from utils import get_logger, section, subsection

log = get_logger("loader")


# -- Typed metadata containers -------------------------------------------------

@dataclass
class RasterInfo:
    """All metadata extracted from one raster file."""
    path:       Path
    label:      str
    width:      int   = 0
    height:     int   = 0
    band_count: int   = 0
    dtype:      str   = ""
    crs:        str   = "Unknown"
    res_x:      float = float("nan")
    res_y:      float = float("nan")
    bounds:     tuple = field(default_factory=tuple)
    nodata:     Optional[float] = None
    valid:      bool  = True
    error:      str   = ""

    def print(self) -> None:
        status = "" if self.valid else "  *** INVALID ***"
        print(f"    Path       : {self.path}")
        print(f"    Label      : {self.label}{status}")
        print(f"    CRS        : {self.crs}")
        print(f"    Size       : {self.width} x {self.height} px")
        print(f"    Bands      : {self.band_count}")
        print(f"    Dtype      : {self.dtype}")
        print(f"    Resolution : {self.res_x:.8g} x {self.res_y:.8g}")
        print(f"    Bounds     : {self.bounds}")
        print(f"    NoData     : {self.nodata}")
        if not self.valid:
            print(f"    Error      : {self.error}")
        print()


@dataclass
class VectorInfo:
    """Metadata for a vector (shapefile / GeoPackage) layer."""
    path:          Path
    label:         str
    crs:           str   = "Unknown"
    feature_count: int   = 0
    bounds:        tuple = field(default_factory=tuple)
    valid:         bool  = True
    error:         str   = ""

    def print(self) -> None:
        status = "" if self.valid else "  *** INVALID ***"
        print(f"    Path     : {self.path}")
        print(f"    Label    : {self.label}{status}")
        print(f"    CRS      : {self.crs}")
        print(f"    Features : {self.feature_count}")
        print(f"    Bounds   : {self.bounds}")
        if not self.valid:
            print(f"    Error    : {self.error}")
        print()


# -- File discovery helpers ----------------------------------------------------

def _scan(directory: Path, extensions: frozenset[str]) -> list[Path]:
    """Return sorted list of files matching *extensions* in *directory* tree."""
    if not directory.exists():
        log.warning(f"Directory not found: {directory}")
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def _scan_dfsar(directory: Path, extensions: frozenset[str]) -> list[Path]:
    """
    Like _scan but excludes sub-directories listed in DFSAR_EXCLUDE_DIRS so
    the pipeline's own outputs and the data_pipeline project folder are not
    re-ingested as DFSAR products.
    """
    if not directory.exists():
        log.warning(f"DFSAR root not found: {directory}")
        return []
    results: list[Path] = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        # Exclude paths that pass through a blocked directory
        blocked = any(part in DFSAR_EXCLUDE_DIRS for part in p.parts)
        if not blocked:
            results.append(p)
    return sorted(results)


def _classify_dfsar(path: Path) -> str:
    """
    Derive a canonical product label from a DFSAR filename.

    CH-2 DFSAR filename convention:
      ch2_sar_<mode>_<ts>_d_<product>_<aux>_<pol>_<ver>.tif
                                       ↑ token[5]

    Tries token[5] first, then falls back to a full-stem keyword search.
    """
    stem   = path.stem
    tokens = stem.split("_")

    # Primary: check the product-code position in the standard convention
    if len(tokens) > 5:
        candidate = tokens[5].lower()
        if candidate in DFSAR_PRODUCT_KEYWORDS:
            return DFSAR_PRODUCT_KEYWORDS[candidate]

    # Secondary: keyword search across the whole stem
    stem_lower = stem.lower()
    for keyword, label in DFSAR_PRODUCT_KEYWORDS.items():
        if keyword in stem_lower:
            return label

    # Fallback: keep a readable auto-label
    return f"DFSAR_{stem[:24]}"


# -- Metadata readers ----------------------------------------------------------

def _read_raster(path: Path, label: str) -> RasterInfo:
    info = RasterInfo(path=path, label=label)
    try:
        with rasterio.open(path) as src:
            info.width      = src.width
            info.height     = src.height
            info.band_count = src.count
            info.dtype      = str(src.dtypes[0])
            info.crs        = str(src.crs) if src.crs else "Not set"
            info.nodata     = src.nodata
            if src.res:
                info.res_x, info.res_y = float(src.res[0]), float(src.res[1])
            info.bounds = tuple(src.bounds)
    except Exception as exc:
        info.valid = False
        info.error = str(exc)
        log.error(f"Cannot open raster {path.name}: {exc}")
    return info


def _read_vector(path: Path, label: str) -> VectorInfo:
    info = VectorInfo(path=path, label=label)
    try:
        gdf = gpd.read_file(path)
        info.crs           = str(gdf.crs) if gdf.crs else "Not set"
        info.feature_count = len(gdf)
        if not gdf.empty:
            info.bounds = tuple(float(v) for v in gdf.total_bounds)
    except Exception as exc:
        info.valid = False
        info.error = str(exc)
        log.error(f"Cannot open vector {path.name}: {exc}")
    return info


# -- Catalog -------------------------------------------------------------------

class DatasetCatalog:
    """
    Discovers and catalogues every dataset across all configured directories.

    Attributes
    ----------
    dem         : RasterInfo for the primary DEM
    psr         : RasterInfo or VectorInfo for the PSR layer
    dpsr        : RasterInfo or VectorInfo for the DPSR layer
    dfsar       : dict  label -> RasterInfo  for every DFSAR product
    all_rasters : flat list of all RasterInfo objects (DEM + PSR + DPSR + DFSAR)
    all_vectors : flat list of all VectorInfo objects
    """

    def __init__(self) -> None:
        self.dem:         Optional[RasterInfo]                      = None
        self.psr:         Optional[Union[RasterInfo, VectorInfo]]   = None
        self.dpsr:        Optional[Union[RasterInfo, VectorInfo]]   = None
        self.dfsar:       dict[str, RasterInfo]                     = {}
        self.all_rasters: list[RasterInfo]                          = []
        self.all_vectors: list[VectorInfo]                          = []

    # -- Public ----------------------------------------------------------------

    def discover(self) -> None:
        """Scan all directories and populate the catalog."""
        log.info("=== Dataset Discovery START ===")
        self._load_dem()
        self._load_psr()
        self._load_dpsr()
        self._load_dfsar()
        log.info(
            f"Discovery complete — "
            f"{len(self.all_rasters)} raster(s), {len(self.all_vectors)} vector(s)"
        )

    def print_catalog(self) -> None:
        """Pretty-print every discovered dataset to stdout."""
        section("STEP 1 — DISCOVERED DATASETS")

        subsection("DEM")
        if self.dem:
            self.dem.print()
        else:
            print("  [!] No DEM found in datasets/DEM/")

        subsection("PSR")
        if self.psr:
            self.psr.print()
        else:
            print("  [!] No PSR found in datasets/PSR/")

        subsection("DPSR")
        if self.dpsr:
            self.dpsr.print()
        else:
            print("  [!] No DPSR found in datasets/DPSR/")

        subsection(f"DFSAR Products  ({len(self.dfsar)} found)")
        if self.dfsar:
            for label, info in self.dfsar.items():
                info.print()
        else:
            print("  [!] No DFSAR products found under DFSAR root")

    # -- Private ---------------------------------------------------------------

    def _load_dem(self) -> None:
        # Explicit file override takes priority
        if DEM_FILE is not None and Path(DEM_FILE).exists():
            info = _read_raster(Path(DEM_FILE), "DEM")
            self.dem = info
            self.all_rasters.append(info)
            log.info(f"DEM (override): {Path(DEM_FILE).name}  "
                     f"({info.width}x{info.height}, CRS={info.crs[:40]}…)")
            return

        # Fall back to directory scan
        files = _scan(DEM_DIR, RASTER_EXTENSIONS)
        if not files:
            log.warning(
                "No DEM found.  Run setup_dem.py first to convert the "
                "LOLA PDS3 DEM to GeoTIFF."
            )
            return
        info = _read_raster(files[0], "DEM")
        self.dem = info
        self.all_rasters.append(info)
        if len(files) > 1:
            log.warning(
                f"Multiple DEMs found; using '{files[0].name}'. "
                f"Ignored: {[f.name for f in files[1:]]}"
            )
        log.info(f"DEM: {files[0].name}  ({info.width}x{info.height})")

    def _load_psr(self) -> None:
        # Explicit file override
        if PSR_FILE is not None:
            p = Path(PSR_FILE)
            if p.exists():
                if p.suffix.lower() in VECTOR_EXTENSIONS:
                    info = _read_vector(p, "PSR")
                    self.psr = info
                    self.all_vectors.append(info)
                else:
                    info = _read_raster(p, "PSR")
                    self.psr = info
                    self.all_rasters.append(info)
                log.info(f"PSR (override): {p.name}")
                return
            else:
                log.warning(f"PSR_FILE set but not found: {p}")

        # Directory scan
        rasters = _scan(PSR_DIR, RASTER_EXTENSIONS)
        vectors = _scan(PSR_DIR, VECTOR_EXTENSIONS)
        if vectors:                          # prefer vector for PSR
            info = _read_vector(vectors[0], "PSR")
            self.psr = info
            self.all_vectors.append(info)
            log.info(f"PSR vector: {vectors[0].name}")
        elif rasters:
            info = _read_raster(rasters[0], "PSR")
            self.psr = info
            self.all_rasters.append(info)
            log.info(f"PSR raster: {rasters[0].name}")
        else:
            log.warning("No PSR data found.")

    def _load_dpsr(self) -> None:
        # Explicit file override
        if DPSR_FILE is not None:
            p = Path(DPSR_FILE)
            if p.exists():
                if p.suffix.lower() in VECTOR_EXTENSIONS:
                    info = _read_vector(p, "DPSR")
                    self.dpsr = info
                    self.all_vectors.append(info)
                else:
                    info = _read_raster(p, "DPSR")
                    self.dpsr = info
                    self.all_rasters.append(info)
                log.info(f"DPSR (override): {p.name}")
                return
            else:
                log.warning(f"DPSR_FILE set but not found: {p}")

        # Directory scan
        rasters = _scan(DPSR_DIR, RASTER_EXTENSIONS)
        vectors = _scan(DPSR_DIR, VECTOR_EXTENSIONS)
        if rasters:
            info = _read_raster(rasters[0], "DPSR")
            self.dpsr = info
            self.all_rasters.append(info)
            log.info(f"DPSR raster: {rasters[0].name}")
        elif vectors:
            info = _read_vector(vectors[0], "DPSR")
            self.dpsr = info
            self.all_vectors.append(info)
            log.info(f"DPSR vector: {vectors[0].name}")
        else:
            log.warning("No DPSR data found.")

    def _load_dfsar(self) -> None:
        files = _scan_dfsar(DFSAR_DIR, RASTER_EXTENSIONS)
        log.info(f"DFSAR root scan found {len(files)} raster(s).")

        for path in files:
            label = _classify_dfsar(path)

            # Deduplicate: if the same label was already assigned, suffix with
            # the last 8 chars of the stem so both files are kept.
            if label in self.dfsar:
                label = f"{label}_{path.stem[-8:]}"

            info = _read_raster(path, label)
            self.dfsar[label] = info
            self.all_rasters.append(info)
            log.info(f"DFSAR [{label}]: {path.name}")
