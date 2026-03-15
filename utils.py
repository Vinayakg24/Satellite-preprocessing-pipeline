# utils.py 

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config


def create_output_dirs():
    dirs = [
        config.DATA_DIR,
        config.SENTINEL1_DIR,
        config.SENTINEL2_DIR,
        config.OUTPUT_DIR,
        config.SAR_OUTPUT_DIR,
        config.OPTICAL_OUTPUT_DIR,
        config.PLOTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {d}")
    print("All output directories created.")


def save_geotiff(data, profile, output_path, description=""):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # Add band dimension

    num_bands = data.shape[0]

    profile_copy = profile.copy()
    profile_copy.update(
        dtype=data.dtype,
        count=num_bands,
        driver="GTiff",
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile_copy) as dst:
        for i in range(num_bands):
            dst.write(data[i], i + 1)
            if description:
                dst.set_band_description(i + 1, description)

    print(f"Saved GeoTIFF: {output_path} ({num_bands} band(s))")


def plot_raster(data, title, output_path, cmap="viridis", vmin=None, vmax=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    masked_data = np.ma.masked_invalid(data)

    im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_comparison(before, after, titles, output_path, cmap="gray"):

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, title in zip(axes, [before, after], titles):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Before vs After Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


def plot_rgb_composite(red, green, blue, title, output_path, percentile_clip=2):
    def stretch(band):
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            return np.zeros_like(band)
        low = np.percentile(valid, percentile_clip)
        high = np.percentile(valid, 100 - percentile_clip)
        stretched = np.clip((band - low) / (high - low + 1e-10), 0, 1)
        return stretched

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RGB composite: {output_path}")
