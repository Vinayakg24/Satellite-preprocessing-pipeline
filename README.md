# Satellite Preprocessing Pipeline

A modular Python pipeline that automatically acquires, cleans, and preprocesses multi-sensor satellite imagery (Sentinel-1 SAR and Sentinel-2 Optical). The pipeline aligns both data sources to a single grid to produce analysis-ready datasets for mapping and geospatial analytics.

## Features

- **Automated Download**: Search and download of Sentinel-1 GRD (via ASF) and Sentinel-2 L2A (via Copernicus CDSE).
- **SAR Processing**: Speckle noise filtering (Lee filter) and high-quality GLCM texture feature generation.
- **Optical Processing**: NDVI computation and cloud masking using the Scene Classification Layer (SCL).
- **Image Coregistration**: Automatic spatial alignment (warping) of SAR data to the Sentinel-2 optical grid for pixel-perfect overlays.
- **Visual Validation**: Generates false-color overlay plots and before/after comparisons for easy QA.

## Project Structure

```
satellite-pipeline/
├── main.py                    # Pipeline entry point
├── config.py                  # Central configuration (AOI, periods, credentials)
├── data_download.py           # Sentinel-1 & Sentinel-2 API download logic
├── preprocessing_sar.py       # SAR: speckle filter + GLCM textures
├── preprocessing_optical.py   # Optical: NDVI + cloud masking
├── coregistration.py          # Spatial alignment of SAR to Optical
├── utils.py                   # Shared I/O and plotting utilities
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Raw downloaded .SAFE products
└── outputs/                   # Processed results
    ├── sar/                   # Filtered SAR, textures, and Coregistered layers
    ├── optical/               # NDVI and cloud-mask GeoTIFFs
    └── plots/                 # QA plots and visualizations
```

## Setup & Installation

### 1. Prerequisites

- Python 3.9 or higher
- [Copernicus Data Space](https://dataspace.copernicus.eu/) account
- [ASF Earthdata](https://urs.earthdata.nasa.gov/) account

### 2. Configuration

Open `config.py` and update your API credentials:

```python
ASF_USERNAME = "your_username"
ASF_PASSWORD = "your_password"

CDSE_USERNAME = "your_email"
CDSE_PASSWORD = "your_password"
```

You can also adjust the `AOI_BBOX` and date ranges in this file.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline with a single command:

```bash
python main.py
```

### Optional Arguments

```bash
python main.py --skip-sar       # Only process Sentinel-2 Optical
python main.py --skip-optical    # Only process Sentinel-1 SAR
```

## Outputs

### GeoTIFF Files

| File | Description |
|------|-------------|
| `COREG_SAR_VV_lee_filtered_dB.tif` | SAR VV aligned to the Optical grid (Analysis Ready) |
| `SAR_VV_GLCM_*.tif` | Textural features (Contrast, Correlation, etc.) |
| `NDVI_cloud_masked.tif` | Sentinel-2 NDVI with clouds removed |
| `cloud_mask.tif` | Binary mask used for clearing optical data |

### Visualizations (`outputs/plots/`)

- `coregistration_overlay.png`: RGB composite verifying SAR and Optical alignment.
- `SAR_VV_speckle_comparison.png`: Comparison of raw vs filtered SAR data.
- `S2_NDVI_cloud_mask_comparison.png`: Comparison of original vs cloud-cleared NDVI.

## Methodology

### Image Coregistration
Uses `rasterio.warp` to reproject SAR pixels onto the Sentinel-2 grid. This ensures that a pixel at location (X, Y) in the SAR data represents the exact same physical spot as the pixel at (X, Y) in the Optical data.

### Speckle Filtering (Lee Filter)
Reduces multiplicative noise while preserving edges using adaptive local statistics.

### GLCM Textures
Computes second-order spatial statistics (Contrast, Correlation, Energy) to identify surface patterns in SAR backscatter.



## License

Developed as part of the GalaxEye technical assessment.
