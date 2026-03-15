# preprocessing_optical.py

import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

import config
import utils


S2_BANDS = {
    "B02": {"resolution": "R10m", "description": "Blue (490 nm)"},
    "B03": {"resolution": "R10m", "description": "Green (560 nm)"},
    "B04": {"resolution": "R10m", "description": "Red (665 nm)"},
    "B08": {"resolution": "R10m", "description": "NIR (842 nm)"},
    "SCL": {"resolution": "R20m", "description": "Scene Classification Layer"},
}

# Cloud Masking - SCL classes to mask out
# 0=No Data, 1=Saturated, 2=Dark/Shadow, 3=Cloud Shadow,
# 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Thin Cirrus, 11=Snow/Ice
SCL_MASK_CLASSES = [0, 1, 2, 3, 8, 9, 10, 11]


def _find_band_file(safe_path, band_name, resolution):
    safe_path = Path(safe_path)

    patterns = [
        str(
            safe_path
            / "GRANULE"
            / "*"
            / "IMG_DATA"
            / resolution
            / f"*_{band_name}_*.jp2"
        ),
        str(
            safe_path
            / "GRANULE"
            / "*"
            / "IMG_DATA"
            / resolution
            / f"*_{band_name}_*.tif"
        ),
        # Sometimes nested .SAFE
        str(
            safe_path
            / "*.SAFE"
            / "GRANULE"
            / "*"
            / "IMG_DATA"
            / resolution
            / f"*_{band_name}_*.jp2"
        ),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])

    print(f"Band {band_name} at {resolution} not found in {safe_path.name}")
    return None


def load_sentinel2_bands(safe_path, target_resolution=10, aoi_bbox=None):

    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    safe_path = Path(safe_path)
    aoi_bbox = aoi_bbox or config.AOI_BBOX
    print(f"Loading Sentinel-2 bands from: {safe_path.name}")
    print(f"\n>>> Loading S2 bands, cropping to AOI: {aoi_bbox}")

    bands = {}
    ref_profile = None

    for band_name, band_info in S2_BANDS.items():
        resolution = band_info["resolution"]
        description = band_info["description"]

        band_file = _find_band_file(safe_path, band_name, resolution)
        if band_file is None:
            print(f"  Skipping {band_name} ({description}) - file not found")
            continue

        with rasterio.open(band_file) as src:
            # Reproject AOI bbox from WGS84 to the raster's CRS
            src_bbox = transform_bounds("EPSG:4326", src.crs, *aoi_bbox)

            # Get the window (pixel coordinates) for the AOI
            window = from_bounds(*src_bbox, transform=src.transform)

            data = src.read(1, window=window).astype(np.float32)

            # Build a new profile for this cropped extent
            cropped_transform = rasterio.windows.transform(window, src.transform)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=cropped_transform,
            )

            print(
                f"    {band_name} ({description}): {data.shape} pixels (cropped from {src.height}x{src.width})"
            )
            print(
                f"  Loaded {band_name} ({description}) | "
                f"{resolution} | Cropped shape: {data.shape}"
            )

            if resolution != f"R{target_resolution}m" and ref_profile is not None:
                target_h = ref_profile["height"]
                target_w = ref_profile["width"]
                data_resampled = src.read(
                    1,
                    window=window,
                    out_shape=(target_h, target_w),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                print(
                    f"    {band_name}: Resampled {resolution} -> R{target_resolution}m | {data_resampled.shape}"
                )
                data = data_resampled

            if resolution == f"R{target_resolution}m" and ref_profile is None:
                ref_profile = profile

        bands[band_name] = data

    bands["profile"] = ref_profile
    print(f">>> Loaded {len(bands) - 1} band(s) successfully.\n")
    print(f"Loaded {len(bands) - 1} band(s) successfully.")

    return bands


def compute_ndvi(nir, red):

    print("Computing NDVI = (NIR - Red) / (NIR + Red)...")

    nir = nir.astype(np.float64)
    red = red.astype(np.float64)

    denominator = nir + red
    ndvi = np.where(
        denominator != 0,
        (nir - red) / denominator,
        np.nan,
    )

    ndvi = np.clip(ndvi, -1.0, 1.0)

    valid = ndvi[~np.isnan(ndvi)]
    print(
        f"NDVI computed | "
        f"Range: [{np.min(valid):.4f}, {np.max(valid):.4f}] | "
        f"Mean: {np.mean(valid):.4f} | "
        f"Valid pixels: {len(valid)}/{ndvi.size} ({100*len(valid)/ndvi.size:.1f}%)"
    )

    return ndvi.astype(np.float32)


def create_cloud_mask(scl_band):
    print("Creating cloud mask from SCL band...")
    print(f"  Masking SCL classes: {SCL_MASK_CLASSES}")

    mask = np.ones_like(scl_band, dtype=np.uint8)

    for cls in SCL_MASK_CLASSES:
        count = np.sum(scl_band == cls)
        if count > 0:
            mask[scl_band == cls] = 0
            cls_names = {
                0: "No Data",
                1: "Saturated",
                2: "Dark/Shadow",
                3: "Cloud Shadow",
                8: "Cloud Med",
                9: "Cloud High",
                10: "Cirrus",
                11: "Snow/Ice",
            }
            print(
                f"    Class {cls} ({cls_names.get(cls, 'Unknown')}): "
                f"{count} pixels masked"
            )

    clear_pct = 100 * np.sum(mask == 1) / mask.size
    print(f"  Clear pixels: {clear_pct:.1f}%")

    return mask


def apply_cloud_mask(data, mask, nodata=np.nan):
    masked_data = data.copy().astype(np.float32)
    masked_data[mask == 0] = nodata
    masked_count = np.sum(mask == 0)
    print(f"Applied cloud mask: {masked_count} pixels masked to {nodata}")
    return masked_data

def process_sentinel2(safe_path):

    print("=" * 60)
    print("SENTINEL-2 OPTICAL PREPROCESSING")
    print("=" * 60)

    outputs = {}

    # Step 1: Load bands
    bands = load_sentinel2_bands(safe_path)
    profile = bands.get("profile")

    if profile is None:
        raise ValueError("Could not load reference profile from Sentinel-2 bands.")

    # Step 2: Create RGB composite (for visualization)
    if all(b in bands for b in ["B04", "B03", "B02"]):
        rgb_path = config.PLOTS_DIR / "S2_RGB_composite.png"
        utils.plot_rgb_composite(
            bands["B04"],
            bands["B03"],
            bands["B02"],
            "Sentinel-2 RGB Composite (B4-B3-B2)",
            rgb_path,
        )
        outputs["rgb_composite"] = rgb_path

    # Step 3: Cloud mask
    if "SCL" in bands:
        cloud_mask = create_cloud_mask(bands["SCL"])

        # Save cloud mask
        mask_path = config.OPTICAL_OUTPUT_DIR / "cloud_mask.tif"
        utils.save_geotiff(
            cloud_mask.astype(np.float32),
            profile,
            mask_path,
            "Cloud Mask (1=clear, 0=cloudy)",
        )
        outputs["cloud_mask"] = mask_path

        # Plot cloud mask
        mask_plot = config.PLOTS_DIR / "S2_cloud_mask.png"
        utils.plot_raster(
            cloud_mask,
            "Cloud Mask (White=Clear, Black=Cloudy)",
            mask_plot,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        outputs["cloud_mask_plot"] = mask_plot
    else:
        print("SCL band not found — skipping cloud masking.")
        cloud_mask = None

    # Step 4: Compute NDVI
    if "B08" in bands and "B04" in bands:
        ndvi = compute_ndvi(bands["B08"], bands["B04"])

        # Save raw NDVI
        ndvi_path = config.OPTICAL_OUTPUT_DIR / "NDVI.tif"
        utils.save_geotiff(ndvi, profile, ndvi_path, "NDVI")
        outputs["ndvi"] = ndvi_path

        # Plot raw NDVI
        ndvi_plot = config.PLOTS_DIR / "S2_NDVI.png"
        utils.plot_raster(
            ndvi,
            "NDVI (Normalized Difference Vegetation Index)",
            ndvi_plot,
            cmap="RdYlGn",
            vmin=-0.2,
            vmax=0.8,
        )
        outputs["ndvi_plot"] = ndvi_plot

        # Step 5: Apply cloud mask to NDVI
        if cloud_mask is not None:
            ndvi_masked = apply_cloud_mask(ndvi, cloud_mask)

            # Save masked NDVI
            ndvi_masked_path = config.OPTICAL_OUTPUT_DIR / "NDVI_cloud_masked.tif"
            utils.save_geotiff(
                ndvi_masked, profile, ndvi_masked_path, "NDVI (Cloud Masked)"
            )
            outputs["ndvi_masked"] = ndvi_masked_path

            # Plot masked NDVI
            ndvi_masked_plot = config.PLOTS_DIR / "S2_NDVI_cloud_masked.png"
            utils.plot_raster(
                ndvi_masked,
                "NDVI (Cloud Masked)",
                ndvi_masked_plot,
                cmap="RdYlGn",
                vmin=-0.2,
                vmax=0.8,
            )
            outputs["ndvi_masked_plot"] = ndvi_masked_plot

            # Comparison plot: NDVI before and after cloud masking
            comparison_plot = config.PLOTS_DIR / "S2_NDVI_cloud_mask_comparison.png"
            utils.plot_comparison(
                ndvi,
                ndvi_masked,
                ("NDVI (Original)", "NDVI (Cloud Masked)"),
                comparison_plot,
                cmap="RdYlGn",
            )
            outputs["ndvi_comparison_plot"] = comparison_plot
    else:
        print("B08 (NIR) or B04 (Red) not found — cannot compute NDVI!")

    print("\nSentinel-2 optical preprocessing complete!")
    print(f"Outputs saved to: {config.OPTICAL_OUTPUT_DIR}")
    return outputs
