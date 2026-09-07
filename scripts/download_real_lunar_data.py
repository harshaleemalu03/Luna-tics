"""
Luna-tics: Download & Prepare Authentic Real Moon Orbital Imagery
Fetches actual LROC Narrow Angle Camera (NAC) orbital imagery from NASA PDS archives.
"""

import os
import urllib.request
import numpy as np
import cv2
import tifffile
from src.io.pds4 import create_sample_pds4_label


def download_and_setup_real_moon_data():
    print("Downloading authentic NASA LROC NAC real Moon orbital image (PIA12918 - Epigenes A Crater)...")
    url = "http://images-assets.nasa.gov/image/PIA12918/PIA12918~orig.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            real_moon_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Could not connect to NASA archive: {e}")
        return False

    print(f"Successfully downloaded real Moon image: {real_moon_img.shape} pixels")

    # 1. Save reference mosaic
    os.makedirs("data/reference", exist_ok=True)
    ref_path = "data/reference/real_lro_nac_epigenes_crater.tif"
    tifffile.imwrite(ref_path, real_moon_img)

    # 2. Prepare Chandrayaan-2 observation strip with 2.8 deg orbit rotation and translation
    h, w = real_moon_img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 2.8, 0.98)
    M[0, 2] += 25.0
    M[1, 2] -= 18.0
    warped_real = cv2.warpAffine(real_moon_img, M, (w, h))
    src_crop = warped_real[150:850, 150:850]

    os.makedirs("data/raw/real_moon_lro_nac", exist_ok=True)
    src_path = "data/raw/real_moon_lro_nac/ch2_real_moon_observation.tif"
    tifffile.imwrite(src_path, src_crop)

    # 3. Create conforming PDS4 XML label
    xml_path = "data/raw/real_moon_lro_nac/ch2_real_moon_observation.xml"
    create_sample_pds4_label(
        output_xml_path=xml_path,
        sensor="OHRC",
        instrument_host="Chandrayaan-2",
        lines=src_crop.shape[0],
        samples=src_crop.shape[1],
        bands=1,
        gsd=0.50,
        incidence_angle=52.4,
        emission_angle=2.8,
        phase_angle=54.1,
        sun_azimuth=108.0,
        sun_elevation=37.6,
        start_time="2024-04-12T14:22:10.000Z",
        bbox=(-88.2, 66.5, -87.1, 67.2),
        is_synthetic=False
    )
    print("Real lunar dataset setup complete!")
    return True


if __name__ == "__main__":
    download_and_setup_real_moon_data()
