"""
Unit tests for Luna-tics Geometry module
(Coarse Alignment, USAC-MAGSAC Robust Estimation, Sub-pixel Refinement)
"""

import numpy as np
import pytest
import cv2

from src.io.metadata import LunarMetadata
from src.matching.primary import CandidateMatch
from src.geometry.coarse_alignment import CoarseAligner
from src.geometry.robust_estimation import RobustGeometricEstimator
from src.geometry.refinement import SubpixelRefiner


def test_coarse_alignment_geographic():
    meta_src = LunarMetadata("IIRS", "Chandrayaan-2", (500, 500), footprint_bbox=(10.0, -5.0, 11.0, -4.0))
    meta_ref = LunarMetadata("LRO_WAC", "LRO", (1000, 1000), footprint_bbox=(9.0, -6.0, 12.0, -3.0))

    img_src = np.zeros((500, 500), dtype=np.uint8)
    img_ref = np.zeros((1000, 1000), dtype=np.uint8)

    M, info = CoarseAligner.estimate_coarse_alignment(img_src, img_ref, meta_src, meta_ref)
    assert info["alignment_mode"] == "GEOGRAPHIC_FOOTPRINT_OVERLAP"
    assert M.shape == (2, 3)


def test_robust_estimation_usac_magsac():
    # Ground truth affine transformation: translation (15, -8), slight scale 1.02
    A = np.float32([[1.02, 0.0, 15.0], [0.0, 1.02, -8.0]])

    matches = []
    rng = np.random.RandomState(42)
    # Generate 30 clean inliers + 10 random outliers
    for i in range(30):
        xs = float(rng.uniform(50, 450))
        ys = float(rng.uniform(50, 450))
        pt_s = np.array([xs, ys, 1.0], dtype=np.float32)
        pt_r = A @ pt_s
        matches.append(CandidateMatch(i, (xs, ys), (float(pt_r[0]), float(pt_r[1])), 5.0, 5.0, 0.0, 0.0, 10.0, 0.4))

    for i in range(30, 40):
        xs = float(rng.uniform(50, 450))
        ys = float(rng.uniform(50, 450))
        xr = float(rng.uniform(50, 450))
        yr = float(rng.uniform(50, 450))
        matches.append(CandidateMatch(i, (xs, ys), (xr, yr), 5.0, 5.0, 0.0, 0.0, 50.0, 0.9))

    H, inliers, residuals, info = RobustGeometricEstimator.estimate_geometry(
        matches, model_type="affine", ransac_thresh=3.0
    )

    assert H is not None
    assert info["status"] == "SUCCESS"
    assert info["inlier_count"] >= 28  # detected almost all inliers
    assert info["rmse_px"] < 1.0       # very low sub-pixel residual error on clean points


def test_subpixel_refinement():
    img_src = np.ones((100, 100), dtype=np.uint8) * 120
    # Paint a distinct feature (crater) at (50, 50)
    cv2.circle(img_src, (50, 50), 10, 20, -1)
    cv2.circle(img_src, (50, 50), 4, 240, -1)

    # Shift reference by exactly 5 pixels
    img_ref = np.zeros_like(img_src)
    M = np.float32([[1, 0, 5], [0, 1, 0]])
    img_ref = cv2.warpAffine(img_src, M, (100, 100))

    # Candidate match with slight initial offset (e.g. 54.8 instead of 55.0)
    m = CandidateMatch(0, (50.0, 50.0), (54.7, 50.1), 5.0, 5.0, 0.0, 0.0, 5.0, 0.3)
    refined, stats = SubpixelRefiner.refine_match_locations(img_src, img_ref, [m])

    assert len(refined) == 1
    # Refined point should move closer to ground truth (55.0, 50.0)
    ref_x = refined[0].pt_ref[0]
    ref_y = refined[0].pt_ref[1]
    assert abs(ref_x - 55.0) < 0.25
    assert abs(ref_y - 50.0) < 0.25
