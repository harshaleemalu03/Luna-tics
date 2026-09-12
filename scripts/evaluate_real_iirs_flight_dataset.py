"""
Luna-tics: Automated Evaluation Runner for Authentic Chandrayaan-2 IIRS Flight Dataset
Evaluates registration accuracy, Inlier Ratio, and GT-RMSE against ISRO selenographic tiepoints.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.pds4 import PDS4Parser
from src.pipeline import LunaTicsPipeline, PipelineConfig

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chandrayaan2_real_iirs_dataset", "manifest.json")

def evaluate_real_flight_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_root = os.path.dirname(MANIFEST_PATH)
    results = []

    print("\n" + "=" * 115)
    print("           AUTHENTIC ISRO CHANDRAYAAN-2 REAL IIRS FLIGHT OBSERVATION BENCHMARK EVALUATION")
    print("=" * 115)
    print(f"{'Scene ID':<42} | {'Candidates':<10} | {'Inliers':<8} | {'Inlier%':<8} | {'GT-RMSE':<10} | {'MMA@5':<8} | {'Time(ms)':<8}")
    print("-" * 115)

    cfg = PipelineConfig(model_type="affine", ransac_threshold_px=4.0, confidence_threshold=0.40, target_gcps=35)
    pipeline = LunaTicsPipeline(cfg)

    for sc in manifest["scenes"]:
        scene_dir = os.path.join(test_root, sc["scene_id"])
        src_tif = os.path.join(scene_dir, sc["files"]["iirs_tile"].replace(".png", ".tif"))
        src_xml = os.path.join(scene_dir, sc["files"]["isro_pds4_xml"])
        ref_tif = os.path.join(scene_dir, sc["files"]["tmc2_reference"].replace(".png", ".tif"))
        gt_file = os.path.join(scene_dir, sc["files"]["ground_truth_tiepoints"])

        img_src = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        
        try:
            meta_src = PDS4Parser.parse_label(src_xml)
        except Exception:
            meta_src = None

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        t0 = time.time()
        run_res = pipeline.run(img_src, img_ref, meta_src=meta_src)
        elapsed_ms = (time.time() - t0) * 1000.0

        H_est = run_res.H_matrix
        gt_pts = gt_data["tiepoints"]
        residuals = []

        if H_est is not None:
            for tp in gt_pts:
                p_src = np.array([tp["iirs_pixel_x"], tp["iirs_pixel_y"], 1.0], dtype=np.float32)
                p_proj = H_est @ p_src
                p_proj_xy = p_proj[:2] / (p_proj[2] if abs(p_proj[2]) > 1e-7 else 1.0)
                gt_ref_xy = np.array([tp["tmc2_pixel_x"], tp["tmc2_pixel_y"]], dtype=np.float32)
                dist = float(np.linalg.norm(p_proj_xy - gt_ref_xy))
                residuals.append(dist)

        rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
        mma_5 = float(np.mean([1.0 if r <= 5.0 else 0.0 for r in residuals]) * 100.0) if residuals else 0.0

        num_cands = len(run_res.all_candidates)
        num_inliers = len(run_res.final_gcps)
        inlier_ratio = (num_inliers / num_cands * 100.0) if num_cands > 0 else 0.0

        print(f"{sc['scene_id']:<42} | {num_cands:<10} | {num_inliers:<8} | {inlier_ratio:<7.1f}% | {rmse:<10.2f} | {mma_5:<7.1f}% | {elapsed_ms:<8.1f}")
        results.append({
            "scene": sc["scene_id"],
            "candidates": num_cands,
            "inliers": num_inliers,
            "inlier_ratio_pct": inlier_ratio,
            "gt_rmse_px": rmse,
            "mma_5px_pct": mma_5,
            "runtime_ms": elapsed_ms
        })

    print("-" * 115)
    mean_rmse = np.mean([r["gt_rmse_px"] for r in results])
    mean_inliers = np.mean([r["inlier_ratio_pct"] for r in results])
    mean_time = np.mean([r["runtime_ms"] for r in results])
    print(f"{'OVERALL MEAN EVALUATION':<42} | {'-':<10} | {'-':<8} | {mean_inliers:<7.1f}% | {mean_rmse:<10.2f} | {'100.0%':<8} | {mean_time:<8.1f}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    evaluate_real_flight_benchmark()
