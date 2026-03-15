# preprocessing_sar.py 

import glob
import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter

import config
import utils

def read_sar_data(safe_path, aoi_bbox=None):

    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, Window
    from rasterio.transform import rowcol

    safe_path = Path(safe_path)
    aoi_bbox = aoi_bbox or config.AOI_BBOX
    measurement_dir = safe_path / "measurement"

    if not measurement_dir.exists():
        raise FileNotFoundError(
            f"Measurement directory not found in {safe_path}. "
            f"Ensure this is a valid Sentinel-1 .SAFE product."
        )

    # Find VV and VH TIFF files
    tiff_files = list(measurement_dir.glob("*.tiff")) + list(
        measurement_dir.glob("*.tif")
    )
    print(f"Found {len(tiff_files)} measurement file(s) in {safe_path.name}")
    print(f"\n>>> Loading S1 SAR data, cropping to AOI: {aoi_bbox}")

    sar_data = {"vv": None, "vh": None, "profile": None}

    for tiff in tiff_files:
        fname_lower = tiff.name.lower()
        with rasterio.open(tiff) as src:

            if src.crs is not None and src.transform != rasterio.Affine.identity():
                src_bbox = transform_bounds("EPSG:4326", src.crs, *aoi_bbox)
                window = from_bounds(*src_bbox, transform=src.transform)
                cropped_transform = rasterio.windows.transform(window, src.transform)
            elif src.gcps[0]:
                gcps, gcp_crs = src.gcps

                from rasterio.transform import from_gcps

                approx_transform = from_gcps(gcps)

                min_lon, min_lat, max_lon, max_lat = aoi_bbox

                rows, cols = rowcol(
                    approx_transform, [min_lon, max_lon], [max_lat, min_lat]
                )

                row_min, row_max = min(rows), max(rows)
                col_min, col_max = min(cols), max(cols)

                # Ensure within bounds
                row_min = max(0, row_min)
                col_min = max(0, col_min)
                row_max = min(src.height, row_max)
                col_max = min(src.width, col_max)

                window = Window(
                    col_off=col_min,
                    row_off=row_min,
                    width=col_max - col_min,
                    height=row_max - row_min,
                )
                cropped_transform = rasterio.windows.transform(window, approx_transform)
            else:
                print(f"No CRS or GCPs found in {tiff.name}. Reading full image.")
                window = Window(0, 0, src.width, src.height)
                cropped_transform = src.transform
            if window.width <= 0 or window.height <= 0:
                print(
                    f"Calculated window for AOI is empty or invalid. Reading full image. Window: {window}"
                )
                window = Window(0, 0, src.width, src.height)
                cropped_transform = src.transform

            data = src.read(1, window=window).astype(np.float32)

            profile = src.profile.copy()

            nodata_val = profile.get("nodata")

            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=cropped_transform,
                crs=src.gcps[1] if src.gcps[0] else src.crs or "EPSG:4326",
            )

            if nodata_val is not None:
                if not np.isnan(nodata_val):
                    data[data == nodata_val] = np.nan
            data[data == 0] = np.nan

            print(
                f"    {tiff.name}: {data.shape} pixels (cropped from {src.height}x{src.width})"
            )

            if "vv" in fname_lower:
                sar_data["vv"] = data
                sar_data["profile"] = profile
                print(f"  Loaded VV: {tiff.name} | Cropped shape: {data.shape}")
            elif "vh" in fname_lower:
                sar_data["vh"] = data
                if sar_data["profile"] is None:
                    sar_data["profile"] = profile
                print(f"  Loaded VH: {tiff.name} | Cropped shape: {data.shape}")

    if sar_data["vv"] is None and sar_data["vh"] is None:
        raise ValueError(f"No VV or VH bands found in {measurement_dir}")

    return sar_data


def convert_to_db(linear_data):

    # Avoid log of zero/negative values
    data = np.where(linear_data > 0, linear_data, np.nan)
    db_data = 10.0 * np.log10(data)
    print(
        f"Converted to dB | Range: [{np.nanmin(db_data):.2f}, {np.nanmax(db_data):.2f}]"
    )
    return db_data


def apply_lee_filter(image, window_size=7):
    """
    Formula:
        filtered = mean + K * (pixel - mean)
        K = max(0, (local_var - noise_var) / local_var)

    Parameters
    ----------
    image : np.ndarray
        2D SAR image (can be in linear or dB scale).
    window_size : int, optional
        Size of the sliding window (must be odd). Defaults to 7.
    """
    print(f"Applying Lee speckle filter (window={window_size}x{window_size})...")

    mask = np.isnan(image)
    img = np.where(mask, 0.0, image).astype(np.float64)

    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  

    overall_var = np.nanvar(img[img > 0])
    overall_mean = np.nanmean(img[img > 0])
    noise_var = overall_var / (overall_mean**2 + 1e-10) * local_mean**2

    # Compute the adaptive weight K
    denominator = local_var + 1e-10
    k = np.clip((local_var - noise_var) / denominator, 0, 1)

    # Apply the Lee filter formula
    filtered = local_mean + k * (img - local_mean)

    filtered[mask] = np.nan

    print(
        f"Lee filter applied | "
        f"Input range: [{np.nanmin(image):.4f}, {np.nanmax(image):.4f}] | "
        f"Output range: [{np.nanmin(filtered):.4f}, {np.nanmax(filtered):.4f}]"
    )

    return filtered.astype(np.float32)


def compute_glcm_textures(
    image,
    distances=(1, 2),
    angles=(0, 45, 90, 135),
    properties=("contrast", "correlation", "energy"),
    levels=64,
):
    """
    Parameters
    ----------
    image : np.ndarray
        2D SAR image (should be filtered first).
    distances : tuple of int, optional
        Pixel distances for GLCM. Defaults to (1, 2).
    angles : tuple of float, optional
        Angles in degrees. Defaults to (0, 45, 90, 135).
    properties : tuple of str, optional
        GLCM properties to compute. Defaults to standard 6 properties.
    levels : int, optional
        Number of quantization levels. Defaults to 64.

     """
    from skimage.feature import graycomatrix, graycoprops

    angles_deg = angles

    angles_rad = [np.deg2rad(a) for a in angles_deg]

    print(
        f"Computing GLCM textures...\n"
        f"  Distances: {distances}\n"
        f"  Angles: {angles_deg}°\n"
        f"  Properties: {properties}\n"
        f"  Quantization levels: {levels}"
    )

    img = image.copy()
    mask = np.isnan(img)
    img[mask] = 0
    img_min = np.nanmin(image[~mask]) if np.any(~mask) else 0
    img_max = np.nanmax(image[~mask]) if np.any(~mask) else 1
    img_scaled = np.clip(
        (img - img_min) / (img_max - img_min + 1e-10) * (levels - 1),
        0,
        levels - 1,
    ).astype(np.uint8)

    win = 21  # Window size for local GLCM computation
    half_win = win // 2
    rows, cols = img_scaled.shape

    textures = {prop: np.full_like(image, np.nan) for prop in properties}

    step = 5  # Compute every 5th pixel for high quality texture (final run)

    print(f"  Computing GLCM on grid with step={step}, window={win}x{win}...")

    for r in range(half_win, rows - half_win, step):
        for c in range(half_win, cols - half_win, step):
            window = img_scaled[
                r - half_win : r + half_win + 1,
                c - half_win : c + half_win + 1,
            ]

            if np.sum(window == 0) > (win * win * 0.5):
                continue

            glcm = graycomatrix(
                window,
                distances=distances,
                angles=angles_rad,
                levels=levels,
                symmetric=True,
                normed=True,
            )

            for prop in properties:
                value = graycoprops(glcm, prop).mean()
                r_end = min(r + step, rows)
                c_end = min(c + step, cols)
                textures[prop][r:r_end, c:c_end] = value
    for prop in properties:
        textures[prop][mask] = np.nan

    print(f"GLCM texture computation complete. Properties: {list(textures.keys())}")

    return textures


def process_sentinel1(safe_path):
    print("=" * 60)
    print("SENTINEL-1 SAR PREPROCESSING")
    print("=" * 60)
    sar_data = read_sar_data(safe_path)
    profile = sar_data["profile"]
    outputs = {}
    for pol_name, pol_data in [("VV", sar_data["vv"]), ("VH", sar_data["vh"])]:
        if pol_data is None:
            print(f"  {pol_name} band not available, skipping.")
            continue

        print(f"\n--- Processing {pol_name} polarization ---")

        data_db = convert_to_db(pol_data)

        raw_path = config.SAR_OUTPUT_DIR / f"SAR_{pol_name}_raw_dB.tif"
        utils.save_geotiff(
            data_db, profile, raw_path, f"SAR {pol_name} Backscatter (dB)"
        )
        outputs[f"{pol_name}_raw"] = raw_path

        filtered_db = apply_lee_filter(data_db, window_size=7)

        filtered_path = config.SAR_OUTPUT_DIR / f"SAR_{pol_name}_lee_filtered_dB.tif"
        utils.save_geotiff(
            filtered_db, profile, filtered_path, f"SAR {pol_name} Lee Filtered (dB)"
        )
        outputs[f"{pol_name}_filtered"] = filtered_path

        plot_path = config.PLOTS_DIR / f"SAR_{pol_name}_speckle_comparison.png"
        utils.plot_comparison(
            data_db,
            filtered_db,
            (f"{pol_name} Raw (dB)", f"{pol_name} Lee Filtered (dB)"),
            plot_path,
            cmap="gray",
        )
        outputs[f"{pol_name}_comparison_plot"] = plot_path

        if pol_name == "VV":
            print("Computing GLCM textures on filtered VV band...")
            textures = compute_glcm_textures(filtered_db)

            for tex_name, tex_data in textures.items():
                tex_path = config.SAR_OUTPUT_DIR / f"SAR_VV_GLCM_{tex_name}.tif"
                utils.save_geotiff(tex_data, profile, tex_path, f"GLCM {tex_name}")
                outputs[f"GLCM_{tex_name}"] = tex_path

                tex_plot = config.PLOTS_DIR / f"SAR_VV_GLCM_{tex_name}.png"
                utils.plot_raster(
                    tex_data,
                    f"GLCM Texture: {tex_name.capitalize()}",
                    tex_plot,
                    cmap="inferno",
                )

    print("\nSentinel-1 SAR preprocessing complete!")
    print(f"Outputs saved to: {config.SAR_OUTPUT_DIR}")
    return outputs
