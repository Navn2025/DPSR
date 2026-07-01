"""
config.py  --  Single source of truth for all validation pipeline paths and parameters.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent   # ISRO_Hackathon/
VAL_DIR    = Path(__file__).resolve().parent          # validation/
OUT_DIR    = VAL_DIR / "outputs"
FIG_DIR    = OUT_DIR / "figures"
LOG_DIR    = OUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Scene selection — mirrors cpr/config.py
# ---------------------------------------------------------------------------
SCENE = "faustini"   # "faustini" | "south_pole_oct25"

# ---------------------------------------------------------------------------
# Input CPR products
# ---------------------------------------------------------------------------
if SCENE == "faustini":
    _DATE    = "20210506"
    _STEM    = "ch2_sar_ncxl_20210506t022608652"
    _CPR_DIR = ROOT / "cpr" / "faustini" / "outputs" / "cpr"
    _DATA_DIR = ROOT / "cpr" / "faustini" / "data" / "calibrated" / _DATE
    GEOM_DIR  = ROOT / "cpr" / "faustini" / "geometry" / "calibrated" / _DATE
    SLI_HEIGHT = 252825
    SLI_WIDTH  = 244
else:
    _DATE    = "20251025"
    _STEM    = "ch2_sar_ncxl_20251025t211236510"
    _CPR_DIR = ROOT / "cpr" / "outputs" / "cpr"
    _DATA_DIR = ROOT / "cpr" / "data" / "data" / "calibrated" / _DATE
    GEOM_DIR  = ROOT / "cpr" / "data" / "geometry" / "calibrated" / _DATE
    SLI_HEIGHT = 272631
    SLI_WIDTH  = 244

CALC_CPR_PATH = _CPR_DIR / "Calculated_CPR.tif"

# Official DFSAR aligned CPR (15168 x 15168, 20m, Moon South Pole Stereo)
OFFICIAL_CPR_PATH = (
    ROOT / "DFSAR" / "data_pipeline" / "outputs" / "aligned" / "CPR.tif"
)

# Geometry files
SLI_GEOM_CSV = GEOM_DIR / f"{_STEM}_g_sli_xx_fp_xx_d18.csv"
GRI_GEOM_CSV = GEOM_DIR / f"{_STEM}_g_gri_xx_fp_xx_d18.csv"

# SRI product (already projected at 25m -- used to validate georeferencing)
SRI_REF_PATH = _DATA_DIR / f"{_STEM}_d_sri_xx_fp_hh_d18.tif"

# Tie-point grid dimensions  (auto-detected from CSV; shown here for reference)
N_AZ_TIES  = 7902 if SCENE == "faustini" else 8521   # azimuth tie rows
N_RNG_TIES = 9                                         # range tie columns

# ---------------------------------------------------------------------------
# GCP subsampling  (every GCP_AZ_STRIDE-th azimuth tie row)
# ---------------------------------------------------------------------------
GCP_AZ_STRIDE = 10    # faustini: 7902/10≈790 × 9 = 7110 GCPs

# ---------------------------------------------------------------------------
# Moon coordinate reference systems
# ---------------------------------------------------------------------------
MOON_GEO_WKT = (
    'GEOGCS["GCS_Moon",'
    'DATUM["D_Moon",SPHEROID["Moon",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433]]'
)

# CRS of the official aligned CPR — copied exactly from rasterio metadata
MOON_STEREO_WKT = (
    'PROJCS["POLAR_STEREOGRAPHIC MOON",'
    'GEOGCS["GCS_MOON",'
    'DATUM["D_MOON",SPHEROID["MOON",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1],'
    'AXIS["Easting",NORTH],'
    'AXIS["Northing",NORTH]]'
)

# ---------------------------------------------------------------------------
# Target grid  (matches the official aligned CPR exactly)
# ---------------------------------------------------------------------------
TARGET_WIDTH     = 15168
TARGET_HEIGHT    = 15168
TARGET_RES       = 20.0          # metres per pixel
TARGET_BOUNDS    = (-151680.0, -151680.0, 151680.0, 151680.0)  # (W, S, E, N)

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------
NODATA_CALC    = -9999.0         # nodata in Calculated_CPR.tif
NODATA_GEOREF  = float("nan")    # nodata in output georeferenced product
CPR_VALID_RANGE = (0.0, 20.0)    # physical bounds; outside = outlier

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
GEOREF_OUTPUT_NAME   = "Calculated_CPR_Georeferenced.tif"
REPORT_NAME          = "validation_report.txt"

# Histogram parameters
HIST_BINS  = 300
HIST_RANGE = (0.0, 3.0)
