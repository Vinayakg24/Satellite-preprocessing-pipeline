"""
config.py
All paths, parameters, and constants are defined here.
"""

import os
from pathlib import Path
from datetime import date

# Project Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SENTINEL1_DIR = DATA_DIR / "sentinel1"
SENTINEL2_DIR = DATA_DIR / "sentinel2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAR_OUTPUT_DIR = OUTPUT_DIR / "sar"
OPTICAL_OUTPUT_DIR = OUTPUT_DIR / "optical"
PLOTS_DIR = OUTPUT_DIR / "plots"


AOI_BBOX = [80.20, 26.30, 80.50, 26.60]

# AOI as WKT polygon (for ASF search)
AOI_WKT = (
    f"POLYGON(({AOI_BBOX[0]} {AOI_BBOX[1]}, {AOI_BBOX[2]} {AOI_BBOX[1]}, "
    f"{AOI_BBOX[2]} {AOI_BBOX[3]}, {AOI_BBOX[0]} {AOI_BBOX[3]}, "
    f"{AOI_BBOX[0]} {AOI_BBOX[1]}))"
)

# Time Window
START_DATE = date(2023, 5, 1)
END_DATE = date(2023, 5, 15)

# ASF (Alaska Satellite Facility) Credentials for Sentinel-1 Download
ASF_USERNAME = "YOUR_ASF_USERNAME"
ASF_PASSWORD = "YOUR_ASF_PASSWORD"

# CDSE API Credentials
CDSE_USERNAME = "YOUR_CDSE_EMAIL"
CDSE_PASSWORD = "YOUR_CDSE_PASSWORD"

# CDSE API Endpoints
CDSE_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
