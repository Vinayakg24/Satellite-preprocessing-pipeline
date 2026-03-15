"""
main.py

This script orchestrates the complete workflow:
1. Data Download (Sentinel-1 GRD and Sentinel-2 L2A)
2. SAR Processing (Speckle filtering + GLCM texture generation)
3. Optical Processing (NDVI + Cloud masking)
4. Image Coregistration
"""
import sys
import time
import argparse
from pathlib import Path

import config
import utils
from data_download import ingest_data
from preprocessing_sar import process_sentinel1
from preprocessing_optical import process_sentinel2
from coregistration import run_coregistration_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Satellite Imagery Pipeline",
    )
    parser.add_argument(
        "--skip-sar",
        action="store_true",
        help="Skip Sentinel-1 SAR processing.",
    )
    parser.add_argument(
        "--skip-optical",
        action="store_true",
        help="Skip Sentinel-2 optical processing.",
    )
    return parser.parse_args()


def run_pipeline(args):

    start_time = time.time()

    print("Initializing Preprocessing Pipeline...")
    print(f"AOI BBox: {config.AOI_BBOX}")
    print(f"Output Dir: {config.OUTPUT_DIR}")

    utils.create_output_dirs()

    print("Starting Data Download...")
    s1_path, s2_path = ingest_data()

    optical_outputs = {}
    if not args.skip_optical:
        if s2_path:
            print("Processing Sentinel-2 Optical Data...")
            try:
                optical_outputs = process_sentinel2(s2_path)
            except Exception as e:
                print(f"Sentinel-2 processing failed: {e}")
        else:
            print("No Sentinel-2 data available — skipping optical processing.")
    else:
        print("Sentinel-2 optical processing skipped.")

    sar_outputs = {}
    if not args.skip_sar:
        if s1_path:
            print("Processing Sentinel-1 SAR Data...")
            try:
                sar_outputs = process_sentinel1(s1_path)
            except Exception as e:
                print(f"Sentinel-1 processing failed: {e}")
        else:
            print("No Sentinel-1 data available — skipping SAR processing.")
    else:
        print("Sentinel-1 SAR processing skipped.")

    coregistered_outputs = {}
    if optical_outputs and sar_outputs:
        print("Running Image Coregistration...")
        try:
            coregistered_outputs = run_coregistration_pipeline(
                optical_outputs, sar_outputs
            )
        except Exception as e:
            print(f"Coregistration failed: {e}")
    else:
        print("Skipping coregistration (both Optical and SAR outputs required).")

    elapsed = time.time() - start_time
    print("Pipeline Complete!")
    print(f"Total time elapsed: {elapsed/60:.1f} minutes")
    print(f"Outputs saved to: {config.OUTPUT_DIR}")

    # List all generated files
    if sar_outputs or optical_outputs:
        print("\nGenerated files:")
        for name, path in {**sar_outputs, **optical_outputs}.items():
            print(f"  [{name}] {path}")

    return {"sar": sar_outputs, "optical": optical_outputs}


def main():
    args = parse_args()

    try:
        results = run_pipeline(args)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
