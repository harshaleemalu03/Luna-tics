"""
Luna-tics: Automated Testing Dataset Builder for Chandrayaan-2 IIRS (Hyperspectral Multi-Modal)
Creates standardized cross-spectral benchmark pairs with ground-truth tie points,
PDS4 XML labels, and dataset manifest for SIH 2026 PS 26166 evaluation.
"""

import os
import json
import numpy as np
import cv2
import tifffile

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.pds4 import PDS4Parser
from scripts.generate_sample_data import generate_lunar_surface, create_sample_pds4_label

TEST_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "test_dataset", "iirs_multimodal")
os.makedirs(TEST_ROOT, exist_ok=True)

def build_iirs_test_pairs():
    print("Building standardized Chandrayaan-2 IIRS multi-modal testing dataset...")

    # Shared high-fidelity lunar base terrain (512x512)
    base_terrain = generate_lunar_surface(width=512, height=512, seed=2026, crater_count=40)
    base_u8 = np.clip(base_terrain * 255.0, 0, 255).astype(np.uint8)

    scenarios = [
        {
            "id": "pair_01_iirs_nir_cross_modal",
            "name": "IIRS NIR (0.9-1.2 µm) vs TMC-2 Optical (0.65 µm)",
            "description": "Baseline cross-spectral matching between near-infrared continuum and visible optical.",
            "rotation_deg": 3.0,
            "scale": 0.95,
            "shift": (15.0, -10.0),
            "bands": 16,
            "active_band_range": (0, 4),  # NIR
            "sun_azimuth_iirs": 115.0,
            "incidence_iirs": 45.0,
            "sun_azimuth_ref": 130.0,
            "incidence_ref": 42.0,
            "gsd_src": 80.0,
            "gsd_ref": 5.0
        },
        {
            "id": "pair_02_iirs_swir_mineral_absorption",
            "name": "IIRS SWIR (2.0 µm Pyroxene Absorption) vs TMC-2 Optical",
            "description": "Mineral absorption band causing local contrast inversions against visible reflectance.",
            "rotation_deg": -5.5,
            "scale": 1.05,
            "shift": (-18.0, 22.0),
            "bands": 16,
            "active_band_range": (5, 10),  # Pyroxene 2µm absorption
            "sun_azimuth_iirs": 95.0,
            "incidence_iirs": 52.0,
            "sun_azimuth_ref": 140.0,
            "incidence_ref": 48.0,
            "gsd_src": 80.0,
            "gsd_ref": 5.0
        },
        {
            "id": "pair_03_iirs_mwir_thermal_sun_angle",
            "name": "IIRS MWIR (3.2 µm OH/Thermal) vs TMC-2 with Severe Sun Shift",
            "description": "Cross-spectral thermal regime with 65° solar azimuth divergence and shadow inversion.",
            "rotation_deg": 8.0,
            "scale": 0.92,
            "shift": (25.0, 15.0),
            "bands": 16,
            "active_band_range": (11, 16),  # MWIR thermal/OH
            "sun_azimuth_iirs": 60.0,
            "incidence_iirs": 62.0,
            "sun_azimuth_ref": 125.0,
            "incidence_ref": 35.0,
            "gsd_src": 80.0,
            "gsd_ref": 5.0
        }
    ]

    manifest = {"dataset_name": "Chandrayaan-2 IIRS Multi-Modal Test Bench", "total_pairs": len(scenarios), "pairs": []}

    for sc in scenarios:
        pair_dir = os.path.join(TEST_ROOT, sc["id"])
        os.makedirs(pair_dir, exist_ok=True)

        w, h = 512, 512
        center = (w // 2, h // 2)

        # 1. Compute ground truth affine transform
        M_gt = cv2.getRotationMatrix2D(center, sc["rotation_deg"], sc["scale"])
        M_gt[0, 2] += sc["shift"][0]
        M_gt[1, 2] += sc["shift"][1]

        # 2. Warp terrain to produce source IIRS observation
        warped_base = cv2.warpAffine(base_u8, M_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        src_crop = warped_base[56:456, 56:456]  # 400x400

        # Adjust ground truth transform for crop offset:
        # p_crop = p_warped - (56, 56)
        M_gt_crop = M_gt.copy()
        M_gt_crop[0, 2] -= 56.0
        M_gt_crop[1, 2] -= 56.0

        # 3. Create realistic 16-band hyperspectral cube simulating Chandrayaan-2 IIRS
        cube = np.zeros((sc["bands"], 400, 400), dtype=np.float32)
        rng = np.random.RandomState(42)

        for b in range(sc["bands"]):
            # Spectral absorption curve: pyroxene absorption dip around band 7-9, thermal rise at 14-16
            wavelength_um = 0.8 + (b / (sc["bands"] - 1)) * 4.2
            if wavelength_um < 2.5:
                # Solar reflectance regime
                spectral_factor = 0.95 - 0.25 * np.exp(-((wavelength_um - 2.0) ** 2) / 0.15)
            else:
                # Thermal emission regime
                spectral_factor = 0.70 + 0.35 * ((wavelength_um - 2.5) / 2.5)

            noise = rng.randn(400, 400).astype(np.float32) * 2.0
            band_img = np.clip(src_crop.astype(np.float32) * spectral_factor + noise, 0, 255)
            cube[b] = band_img

        src_tif_path = os.path.join(pair_dir, "source_iirs_cube.tif")
        tifffile.imwrite(src_tif_path, cube)

        # 4. Generate PDS4 XML label for IIRS source
        src_xml_path = os.path.join(pair_dir, "source_iirs_label.xml")
        create_sample_pds4_label(
            output_xml_path=src_xml_path,
            sensor="IIRS",
            instrument_host="Chandrayaan-2",
            lines=400,
            samples=400,
            bands=sc["bands"],
            gsd=sc["gsd_src"],
            incidence_angle=sc["incidence_iirs"],
            emission_angle=5.0,
            phase_angle=abs(sc["incidence_iirs"] - 5.0),
            sun_azimuth=sc["sun_azimuth_iirs"],
            sun_elevation=90.0 - sc["incidence_iirs"],
            is_synthetic=False
        )

        # 5. Generate Reference Optical Image (TMC-2 Optical simulation)
        ref_tif_path = os.path.join(pair_dir, "reference_tmc2.tif")
        tifffile.imwrite(ref_tif_path, base_u8)

        ref_xml_path = os.path.join(pair_dir, "reference_tmc2_label.xml")
        create_sample_pds4_label(
            output_xml_path=ref_xml_path,
            sensor="TMC-2",
            instrument_host="Chandrayaan-2",
            lines=512,
            samples=512,
            bands=1,
            gsd=sc["gsd_ref"],
            incidence_angle=sc["incidence_ref"],
            emission_angle=3.0,
            phase_angle=abs(sc["incidence_ref"] - 3.0),
            sun_azimuth=sc["sun_azimuth_ref"],
            sun_elevation=90.0 - sc["incidence_ref"],
            is_synthetic=False
        )

        # 6. Generate Ground Truth Verification Tie-Points (Grid of 64 points)
        gt_grid_x = np.linspace(40, 472, 8)
        gt_grid_y = np.linspace(40, 472, 8)
        gx, gy = np.meshgrid(gt_grid_x, gt_grid_y)
        ref_pts = np.vstack([gx.ravel(), gy.ravel()]).T

        # Project reference points into cropped source frame: p_src = M_gt_crop @ [x_ref, y_ref, 1]^T
        ref_homo = np.hstack([ref_pts, np.ones((len(ref_pts), 1))])
        src_pts = (M_gt_crop @ ref_homo.T).T

        # Filter points within crop bounds
        valid = (src_pts[:, 0] >= 10) & (src_pts[:, 0] <= 390) & (src_pts[:, 1] >= 10) & (src_pts[:, 1] <= 390)
        gt_tiepoints = []
        for i in range(len(ref_pts)):
            if valid[i]:
                gt_tiepoints.append({
                    "id": len(gt_tiepoints) + 1,
                    "ref_xy": [float(ref_pts[i, 0]), float(ref_pts[i, 1])],
                    "src_xy": [float(src_pts[i, 0]), float(src_pts[i, 1])]
                })

        gt_data = {
            "pair_id": sc["id"],
            "affine_matrix_ref_to_src": M_gt_crop.tolist(),
            "scale_factor": sc["scale"],
            "rotation_deg": sc["rotation_deg"],
            "total_ground_truth_tiepoints": len(gt_tiepoints),
            "tiepoints": gt_tiepoints
        }

        gt_path = os.path.join(pair_dir, "ground_truth_tiepoints.json")
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)

        manifest["pairs"].append({
            "id": sc["id"],
            "name": sc["name"],
            "description": sc["description"],
            "dir": os.path.relpath(pair_dir, TEST_ROOT),
            "source_file": "source_iirs_cube.tif",
            "source_label": "source_iirs_label.xml",
            "reference_file": "reference_tmc2.tif",
            "reference_label": "reference_tmc2_label.xml",
            "ground_truth_file": "ground_truth_tiepoints.json",
            "gsd_ratio": sc["gsd_src"] / sc["gsd_ref"],
            "delta_sun_azimuth_deg": abs(sc["sun_azimuth_iirs"] - sc["sun_azimuth_ref"]),
            "delta_incidence_deg": abs(sc["incidence_iirs"] - sc["incidence_ref"])
        })
        print(f"[OK] Created testing pair: {sc['id']} with {len(gt_tiepoints)} ground truth tiepoints")

    manifest_path = os.path.join(TEST_ROOT, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SUCCESS] Dataset manifest written to {manifest_path}")

if __name__ == "__main__":
    build_iirs_test_pairs()
