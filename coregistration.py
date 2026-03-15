# coregistration.py 

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.exposure import equalize_hist

import config
import utils


def coregister_sar_to_optical(
    master_path, slave_path, output_path, resampling_method=Resampling.bilinear
):
    master_path = Path(master_path)
    slave_path = Path(slave_path)
    output_path = Path(output_path)

    print(f"Coregistering {slave_path.name} to {master_path.name}...")

    with rasterio.open(master_path) as master:
        master_meta = master.meta.copy()
        master_crs = master.crs
        master_transform = master.transform
        master_width = master.width
        master_height = master.height

    with rasterio.open(slave_path) as slave:

        # Determine the source CRS
        src_crs = slave.crs if slave.crs else "EPSG:4326"

        slave_data = slave.read(1)

        destination = np.zeros(
            (master_height, master_width), dtype=master_meta["dtype"]
        )

        reproject(
            source=slave_data,
            destination=destination,
            src_transform=slave.transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=master_transform,
            dst_crs=master_crs,
            dst_nodata=np.nan,
            resampling=resampling_method,
        )

        master_meta.update({"nodata": np.nan})

    # Save the coregistered image
    utils.save_geotiff(
        destination, master_meta, output_path, f"Coregistered {slave_path.stem}"
    )
    print(f"Saved coregistered image to {output_path}")

    return output_path


def run_coregistration_pipeline(optical_outputs, sar_outputs):

    if not optical_outputs or not sar_outputs:
        print("Both Optical and SAR outputs are required for coregistration. Skipping.")
        return {}

    print("=" * 60)
    print("IMAGE COREGISTRATION (SAR to OPTICAL)")
    print("=" * 60)

    # NDVI as master reference
    master_reference = optical_outputs.get("ndvi_masked") or optical_outputs.get("ndvi")

    if not master_reference:
        print("No valid master reference found in optical outputs.")
        return {}

    coregistered_outputs = {}

    layers_to_coregister = [
        "VV_filtered",
        "VH_filtered",
        "GLCM_contrast",
        "GLCM_correlation",
        "GLCM_energy",
    ]

    for layer in layers_to_coregister:
        if layer in sar_outputs:
            slave_img = sar_outputs[layer]

            coreg_filename = f"COREG_{Path(slave_img).name}"
            output_path = config.SAR_OUTPUT_DIR / coreg_filename

            try:
                out_path = coregister_sar_to_optical(
                    master_reference, slave_img, output_path
                )
                coregistered_outputs[f"coreg_{layer}"] = out_path
            except Exception as e:
                print(f"Failed to coregister {layer}: {e}")

    if "coreg_VV_filtered" in coregistered_outputs:
        print("Generating QA overlay plot to verify coregistration...")
        try:
            with rasterio.open(master_reference) as src_s2:
                s2_img = np.nan_to_num(src_s2.read(1), 0)
            with rasterio.open(coregistered_outputs["coreg_VV_filtered"]) as src_s1:
                s1_img = np.nan_to_num(src_s1.read(1), 0)

            s2_eq = equalize_hist(s2_img)
            s1_eq = equalize_hist(s1_img)

            rgb = np.zeros((s2_img.shape[0], s2_img.shape[1], 3))
            rgb[:, :, 0] = s1_eq
            rgb[:, :, 1] = s2_eq
            rgb[:, :, 2] = s2_eq

            plot_path = config.PLOTS_DIR / "coregistration_overlay.png"
            plt.figure(figsize=(12, 12))
            plt.imshow(rgb)
            plt.title("Coregistration Overlay\nRed: SAR VV | Cyan (GB): Optical NDVI")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved coregistration overlay plot to {plot_path}")
        except Exception as e:
            print(f"Failed to generate coregistration overlay plot: {e}")

    print("Coregistration complete!")
    return coregistered_outputs
