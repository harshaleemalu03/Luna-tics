"""
Chandrayaan-2 IIRS Multi-Modal Test Suite Runner
Simple script to evaluate any matcher on the IIRS test dataset.
Usage: python run_tests.py
"""

import os
import json
import numpy as np
import tifffile
import cv2

def run_evaluation():
    manifest_file = "manifest.json"
    if not os.path.exists(manifest_file):
        print("Error: manifest.json not found.")
        return

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    print("=" * 90)
    print(f" {manifest['dataset_name']}")
    print("=" * 90)

    for sc in manifest["scenes"]:
        print(f"\nEvaluating Scene: {sc['title']}")
        print(f"  Region: {sc['region']}")
        print(f"  Spectral: {sc['spectral_regime']}")
        
        cube_path = os.path.join(sc["directory"], sc["iirs_cube"])
        ref_path = os.path.join(sc["directory"], sc["tmc_reference"])
        gt_path = os.path.join(sc["directory"], sc["ground_truth_json"])

        cube = tifffile.imread(cube_path)
        ref = tifffile.imread(ref_path)
        with open(gt_path, "r") as f:
            gt = json.load(f)

        print(f"  Loaded IIRS cube: {cube.shape} (Bands: {cube.shape[0]}), Reference: {ref.shape}")
        print(f"  Ground truth tiepoints: {len(gt['tiepoints'])} points")

if __name__ == "__main__":
    run_evaluation()
