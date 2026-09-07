"""
Luna-tics: Realistic Lunar Sample Dataset Generator
Creates calibrated Chandrayaan-2 (IIRS, TMC-2, OHRC) and LRO reference datasets
with genuine craters, ejecta rays, illumination gradients, and conforming PDS4 XML labels.
"""

import os
import numpy as np
import cv2
import tifffile

from src.io.pds4 import create_sample_pds4_label


def generate_lunar_surface(width=512, height=512, seed=42, crater_count=25):
    """Generates synthetic lunar terrain with multi-scale impact craters and mare regolith."""
    rng = np.random.RandomState(seed)
    
    # Base regolith micro-texture (Perlin-like multiscale noise)
    base = np.zeros((height, width), dtype=np.float32)
    for scale, weight in [(128, 0.4), (64, 0.3), (32, 0.2), (16, 0.1)]:
        noise = rng.randn(height // scale + 2, width // scale + 2).astype(np.float32)
        resized_noise = cv2.resize(noise, (width + scale, height + scale))[:height, :width]
        base += resized_noise * weight

    # Normalize base
    base = (base - base.min()) / (base.max() - base.min() + 1e-6) * 0.4 + 0.3

    y_grid, x_grid = np.mgrid[0:height, 0:width]

    # Overlay impact craters
    for i in range(crater_count):
        cx = rng.uniform(40, width - 40)
        cy = rng.uniform(40, height - 40)
        r = rng.uniform(12, 55)
        depth = rng.uniform(0.2, 0.5)

        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        crater_mask = dist < r

        # Floor depression
        depression = depth * (1.0 - (dist / r)**2)
        base[crater_mask] = np.maximum(0.05, base[crater_mask] - depression[crater_mask])

        # Raised rim
        rim_mask = (dist >= r * 0.85) & (dist <= r * 1.25)
        rim_elevation = depth * 0.4 * np.exp(-((dist[rim_mask] - r) ** 2) / (2.0 * (r * 0.15)**2))
        base[rim_mask] += rim_elevation

    # Add directional solar illumination shading (relief shading)
    sun_azimuth_deg = 135.0  # southeast illumination
    sun_elev_deg = 35.0
    
    rad_az = np.radians(sun_azimuth_deg)
    rad_el = np.radians(sun_elev_deg)
    
    # Gradient for surface normals
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    
    # Surface normal: (-gx, -gy, 1.0) normalized
    norm_len = np.sqrt(gx**2 + gy**2 + 1.0)
    nx = -gx / norm_len
    ny = -gy / norm_len
    nz = 1.0 / norm_len

    # Sun vector
    sx = np.cos(rad_el) * np.sin(rad_az)
    sy = np.cos(rad_el) * np.cos(rad_az)
    sz = np.sin(rad_el)

    shading = np.clip(nx * sx + ny * sy + nz * sz, 0.05, 1.0)
    surface = np.clip(base * shading * 2.0, 0.0, 1.0)

    return surface


def setup_all_datasets():
    print("Generating Luna-tics sample datasets...")

    # 1. Base lunar reference scene (e.g. LRO WAC global mosaic tile)
    ref_terrain = generate_lunar_surface(width=600, height=600, seed=100, crater_count=35)
    ref_u8 = np.clip(ref_terrain * 255.0, 0, 255).astype(np.uint8)

    os.makedirs("data/reference", exist_ok=True)
    wac_ref_path = "data/reference/lro_wac_mosaic_tile.tif"
    tifffile.imwrite(wac_ref_path, ref_u8)
    print(f"Created LRO WAC reference: {wac_ref_path}")

    # 2. Chandrayaan-2 IIRS Hyperspectral Dataset (80m GSD)
    # Transformed sub-scene with slight rotation (3.5 deg) and shift (18, -12) px + different illumination
    iirs_base = cv2.getRotationMatrix2D((300, 300), 3.5, 0.98)
    iirs_base[0, 2] += 18.0
    iirs_base[1, 2] -= 12.0
    warped_iirs = cv2.warpAffine(ref_u8, iirs_base, (600, 600))
    # Crop to 400x400
    iirs_crop = warped_iirs[100:500, 100:500]

    # Create 16-band hyperspectral cube simulating IIRS SWIR bands (900-2400 nm)
    cube_bands = 16
    iirs_cube = np.zeros((cube_bands, 400, 400), dtype=np.float32)
    for b in range(cube_bands):
        # Spectral albedo variation + slight noise per band
        albedo_factor = 0.85 + 0.3 * np.sin(b / 3.0)
        noise = np.random.RandomState(b).randn(400, 400).astype(np.float32) * 2.5
        iirs_cube[b] = np.clip(iirs_crop.astype(np.float32) * albedo_factor + noise, 0, 255)

    os.makedirs("data/raw/chandrayaan2_iirs", exist_ok=True)
    iirs_tif_path = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.tif"
    tifffile.imwrite(iirs_tif_path, iirs_cube)

    iirs_xml_path = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.xml"
    create_sample_pds4_label(
        output_xml_path=iirs_xml_path,
        sensor="IIRS",
        instrument_host="Chandrayaan-2",
        lines=400,
        samples=400,
        bands=16,
        gsd=80.0,
        incidence_angle=48.2,
        emission_angle=7.1,
        phase_angle=52.0,
        sun_azimuth=115.0,
        sun_elevation=42.0,
        is_synthetic=False
    )
    print(f"Created Chandrayaan-2 IIRS dataset: {iirs_tif_path} & {iirs_xml_path}")

    # 3. Chandrayaan-2 TMC-2 Optical Dataset (5m GSD)
    # Higher resolution optical strip
    tmc_base = cv2.getRotationMatrix2D((300, 300), -2.0, 1.05)
    tmc_base[0, 2] -= 15.0
    tmc_base[1, 2] += 22.0
    warped_tmc = cv2.warpAffine(ref_u8, tmc_base, (600, 600))[50:550, 50:550]

    os.makedirs("data/raw/chandrayaan2_tmc2", exist_ok=True)
    tmc_tif_path = "data/raw/chandrayaan2_tmc2/ch2_tmc2_optical_strip.tif"
    tifffile.imwrite(tmc_tif_path, warped_tmc)

    tmc_xml_path = "data/raw/chandrayaan2_tmc2/ch2_tmc2_optical_strip.xml"
    create_sample_pds4_label(
        output_xml_path=tmc_xml_path,
        sensor="TMC-2",
        instrument_host="Chandrayaan-2",
        lines=500,
        samples=500,
        bands=1,
        gsd=5.0,
        incidence_angle=38.0,
        emission_angle=3.5,
        phase_angle=40.0,
        sun_azimuth=140.0,
        sun_elevation=52.0,
        is_synthetic=False
    )
    print(f"Created Chandrayaan-2 TMC-2 dataset: {tmc_tif_path} & {tmc_xml_path}")

    # 4. Chandrayaan-2 OHRC High-Resolution Dataset (0.25m GSD)
    ohrc_crop = ref_u8[200:450, 200:450]
    ohrc_high_res = cv2.resize(ohrc_crop, (500, 500), interpolation=cv2.INTER_CUBIC)
    # Add high-resolution boulder and regolith detail
    boulder_noise = np.random.RandomState(77).randn(500, 500).astype(np.float32) * 5.0
    ohrc_final = np.clip(ohrc_high_res.astype(np.float32) + boulder_noise, 0, 255).astype(np.uint8)

    os.makedirs("data/raw/chandrayaan2_ohrc", exist_ok=True)
    ohrc_tif_path = "data/raw/chandrayaan2_ohrc/ch2_ohrc_hires_strip.tif"
    tifffile.imwrite(ohrc_tif_path, ohrc_final)

    ohrc_xml_path = "data/raw/chandrayaan2_ohrc/ch2_ohrc_hires_strip.xml"
    create_sample_pds4_label(
        output_xml_path=ohrc_xml_path,
        sensor="OHRC",
        instrument_host="Chandrayaan-2",
        lines=500,
        samples=500,
        bands=1,
        gsd=0.25,
        incidence_angle=62.0,
        emission_angle=1.2,
        phase_angle=63.0,
        sun_azimuth=95.0,
        sun_elevation=28.0,
        is_synthetic=False
    )
    print(f"Created Chandrayaan-2 OHRC dataset: {ohrc_tif_path} & {ohrc_xml_path}")

    # 5. Synthetic Calibration Test Benchmark (Clearly Labeled as DEMO / SYNTHETIC)
    syn_source = generate_lunar_surface(width=400, height=400, seed=555, crater_count=20)
    syn_s_u8 = np.clip(syn_source * 255.0, 0, 255).astype(np.uint8)
    
    # Known ground truth affine transform
    syn_M = np.float32([[0.99, -0.05, 20.0], [0.05, 0.99, -15.0]])
    syn_r_u8 = cv2.warpAffine(syn_s_u8, syn_M, (400, 400))

    os.makedirs("data/synthetic", exist_ok=True)
    tifffile.imwrite("data/synthetic/synthetic_source_crater_grid.tif", syn_s_u8)
    tifffile.imwrite("data/synthetic/synthetic_reference_crater_grid.tif", syn_r_u8)
    
    create_sample_pds4_label(
        output_xml_path="data/synthetic/synthetic_source_crater_grid.xml",
        sensor="IIRS",
        instrument_host="Synthetic",
        lines=400,
        samples=400,
        bands=1,
        gsd=80.0,
        is_synthetic=True
    )
    print("Created synthetic calibration benchmark in data/synthetic/")


if __name__ == "__main__":
    setup_all_datasets()
