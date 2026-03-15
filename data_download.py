"""
data_download.py
"""

import zipfile
import shutil
from pathlib import Path

import requests

import config


def search_sentinel1(aoi_wkt=None, start_date=None, end_date=None):

    import asf_search as asf

    aoi_wkt = aoi_wkt or config.AOI_WKT
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE

    print(
        f"Searching ASF for Sentinel-1 GRD products...\n"
        f"  AOI: {aoi_wkt[:60]}...\n"
        f"  Date range: {start_date} to {end_date}"
    )

    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        intersectsWith=aoi_wkt,
        start=str(start_date),
        end=str(end_date),
    )

    print(f"Found {len(results)} Sentinel-1 GRD product(s).")
    for r in results:
        print(f"  - {r.properties['fileName']} | {r.properties['startTime']}")

    return results


def download_sentinel1(results, output_dir=None):

    import asf_search as asf

    output_dir = Path(output_dir or config.SENTINEL1_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.ASF_USERNAME or not config.ASF_PASSWORD:
        raise ValueError(
            "ASF credentials not set! Please set ASF_USERNAME and ASF_PASSWORD "
            "environment variables. Register at: https://urs.earthdata.nasa.gov/users/new"
        )

    session = asf.ASFSession().auth_with_creds(config.ASF_USERNAME, config.ASF_PASSWORD)

    print(f"Downloading {len(results)} Sentinel-1 product(s) to {output_dir}...")

    product = results[0] if results else None
    if product is None:
        print("No products to download.")
        return []

    product.download(path=str(output_dir), session=session)
    zip_file = output_dir / product.properties["fileName"]
    print(f"Downloaded: {zip_file}")

    safe_dirs = _extract_safe(zip_file, output_dir)
    return safe_dirs


def search_sentinel2(bbox=None, start_date=None, end_date=None):
    bbox = bbox or config.AOI_BBOX
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE

    print(
        f"Searching CDSE for Sentinel-2 L2A products...\n"
        f"  BBox: {bbox}\n"
        f"  Date range: {start_date} to {end_date}"
    )

    aoi_filter = (
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
        f"{bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},"
        f"{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},"
        f"{bbox[0]} {bbox[1]}))')"
    )

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
            f"'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
            f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
            f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
            f"{aoi_filter} and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 20.0)"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": 5,
    }

    response = requests.get(config.CDSE_CATALOG_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    products = data.get("value", [])
    print(f"Found {len(products)} Sentinel-2 L2A product(s).")
    for p in products:
        print(f"  - {p['Name']} | {p['ContentDate']['Start']}")

    return products


def _get_cdse_access_token():
    if not config.CDSE_USERNAME or not config.CDSE_PASSWORD:
        raise ValueError(
            "CDSE credentials not set! Please set CDSE_USERNAME and CDSE_PASSWORD "
            "environment variables. Register at: https://dataspace.copernicus.eu/"
        )

    data = {
        "client_id": "cdse-public",
        "username": config.CDSE_USERNAME,
        "password": config.CDSE_PASSWORD,
        "grant_type": "password",
    }

    response = requests.post(config.CDSE_TOKEN_URL, data=data, timeout=30)
    response.raise_for_status()
    token = response.json()["access_token"]
    print("CDSE access token obtained successfully.")
    return token


def download_sentinel2(products, output_dir=None):
    output_dir = Path(output_dir or config.SENTINEL2_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = _get_cdse_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    product = products[0] if products else None
    if product is None:
        print("No products to download.")
        return []

    product_id = product["Id"]
    product_name = product["Name"]
    download_url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/"
        f"Products({product_id})/$value"
    )

    zip_path = output_dir / f"{product_name}.zip"
    print(f"Downloading {product_name}...")

    with requests.get(download_url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16):
                f.write(chunk)

    print(f"Downloaded: {zip_path}")

    safe_dirs = _extract_safe(zip_path, output_dir)
    return safe_dirs


def _extract_safe(zip_path, output_dir):
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    safe_dirs = []

    if zip_path.suffix == ".zip" and zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)

        for item in output_dir.iterdir():
            if item.is_dir() and item.suffix == ".SAFE":
                safe_dirs.append(item)
                print(f"  Extracted: {item.name}")

    return safe_dirs


def ingest_data():

    s1_path = None
    s2_path = None

    print("Downloading data from remote APIs...")

    try:
        s1_results = search_sentinel1()
        if s1_results:
            s1_dirs = download_sentinel1(s1_results)
            s1_path = s1_dirs[0] if s1_dirs else None
        else:
            print("No Sentinel-1 GRD products found within parameters.")
    except Exception as e:
        print(f"Sentinel-1 search/download failed: {e}")

    try:
        s2_results = search_sentinel2()
        if s2_results:
            s2_dirs = download_sentinel2(s2_results)
            s2_path = s2_dirs[0] if s2_dirs else None
        else:
            print("No Sentinel-2 L2A products found within parameters.")
    except Exception as e:
        print(f"Sentinel-2 search/download failed: {e}")

    print(f"Final Sentinel-1 data path: {s1_path or 'NOT AVAILABLE'}")
    print(f"Final Sentinel-2 data path: {s2_path or 'NOT AVAILABLE'}")

    return s1_path, s2_path
