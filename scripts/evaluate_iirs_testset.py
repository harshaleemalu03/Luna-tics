"""
Luna-tics: Automated Evaluation Runner for IIRS Multi-Modal Test Bench
Evaluates registration accuracy, MMA, and RMSE against analytical ground-truth tie points.
"""

import os
import sys
import json
import time
import numpy as np
import tifffile
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.pds4 import PDS4Parser
from src.pipeline import LunaTicsPipeline, PipelineConfig

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_dataset", "iirs_multimodal", "dataset_manifest.json")

def evaluate_iirs_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    test_root = os.path.dirname(MANIFEST_PATH)
    results = []

    print("\n" + "=" * 110)
    print("                      CHANDRAYAAN-2 IIRS MULTI-MODAL TEST BENCH EVALUATION RESULTS")
    print("=" * 110)
    print(f"{'Pair ID':<35} | {'Candidates':<10} | {'Inliers':<8} | {'Inlier%':<8} | {'GT-RMSE':<10} | {'MMA@3':<8} | {'MMA@5':<8} | {'Time(ms)':<8}")
    print("-" * 110)

    cfg = PipelineConfig(model_type="affine", ransac_threshold_px=4.0, confidence_threshold=0.45, target_gcps=35)
    pipeline = LunaTicsPipeline(cfg)

    for pair in manifest["pairs"]:
        pair_dir = os.path.join(test_root, pair["dir"])
        src_tif = os.path.join(pair_dir, pair["source_file"])
        src_xml = os.path.join(pair_dir, pair["source_label"])
        ref_tif = os.path.join(pair_dir, pair["reference_file"])
        gt_file = os.path.join(pair_dir, pair["ground_truth_file"])

        img_src_cube = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        meta_src = PDS4Parser.parse_label(src_xml)

        with open(gt_file, "r") as f:
            gt_data = json.load(f)

        t0 = time.time()
        run_res = pipeline.run(img_src_cube, img_ref, meta_src=meta_src)
        elapsed_ms = (time.time() - t0) * 1000.0

        # Evaluate against Ground Truth Tie Points
        # For each GT tie point: project ref_xy with estimated H and measure distance to src_xy
        H_est = run_res.H_matrix
        gt_pts = gt_data["tiepoints"]
        residuals = []

        if H_est is not None:
            for pt in gt_pts:
                r_xy = pt["ref_xy"]
                s_xy = pt["src_xy"]
                if H_est.shape == (2, 3):
                    proj_x = H_est[0, 0] * s_xy[0] + H_est[0, 1] * s_xy[1] + H_est[0, 2]
                    proj_y = H_est[1, 0] * s_xy[0] + H_est[1, 1] * s_xy[1] + H_est[1, 2]
                else:
                    vec = H_est @ np.array([s_xy[0], s_xy[1], 1.0])
                    proj_x, proj_y = vec[0] / vec[2], vec[1] / vec[2]
                err = np.sqrt((proj_x - r_xy[0]) ** 2 + (proj_y - r_xy[1]) ** 2)
                residuals.append(err)

            residuals = np.array(residuals)
            gt_rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mma1 = float(np.mean(residuals <= 1.0) * 100.0)
            mma3 = float(np.mean(residuals <= 3.0) * 100.0)
            mma5 = float(np.mean(residuals <= 5.0) * 100.0)
        else:
            gt_rmse = 999.0
            mma1, mma3, mma5 = 0.0, 0.0, 0.0

        pair_summary = {
            "pair_id": pair["id"],
            "name": pair["name"],
            "candidate_matches": run_res.metrics.candidate_matches_count,
            "inliers": run_res.metrics.inlier_count,
            "inlier_ratio_pct": run_res.metrics.inlier_ratio_pct,
            "gt_rmse_px": round(gt_rmse, 3),
            "mma@1": round(mma1, 2),
            "mma@3": round(mma3, 2),
            "mma@5": round(mma5, 2),
            "runtime_ms": round(elapsed_ms, 1)
        }
        results.append(pair_summary)

        print(f"{pair['id']:<35} | {pair_summary['candidate_matches']:<10} | {pair_summary['inliers']:<8} | {pair_summary['inlier_ratio_pct']:>6.1f}% | {pair_summary['gt_rmse_px']:>7.2f} px | {pair_summary['mma@3']:>6.1f}% | {pair_summary['mma@5']:>6.1f}% | {pair_summary['runtime_ms']:>6.1f} ms")

    print("-" * 110)
    avg_rmse = np.mean([r["gt_rmse_px"] for r in results if r["gt_rmse_px"] < 900])
    avg_mma5 = np.mean([r["mma@5"] for r in results])
    avg_inlier = np.mean([r["inlier_ratio_pct"] for r in results])
    print(f"{'OVERALL AVERAGE':<35} | {'-':<10} | {'-':<8} | {avg_inlier:>6.1f}% | {avg_rmse:>7.2f} px | {'-':<8} | {avg_mma5:>6.1f}% | {'-':<8}")
    print("=" * 110 + "\n")

    out_file = os.path.join(test_root, "test_evaluation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved IIRS evaluation results to: {out_file}")

if __name__ == "__main__":
    evaluate_iirs_benchmark()
