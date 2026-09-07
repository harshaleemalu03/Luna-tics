"""
Luna-tics: Robust Geometric Verification (USAC-MAGSAC / RANSAC)
Evaluates spatial models (Affine vs Homography) and computes per-match geometric residuals.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2

from src.matching.primary import CandidateMatch


class RobustGeometricEstimator:
    """Estimates rigid, affine, or projective transformation using USAC-MAGSAC."""

    @staticmethod
    def estimate_geometry(
        matches: List[CandidateMatch],
        model_type: str = "auto",  # 'auto', 'affine', or 'homography'
        ransac_thresh: float = 4.0,
        max_iters: int = 5000,
        confidence: float = 0.995
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Estimate geometric transformation and compute residuals.

        Returns:
            Tuple of:
            - H_matrix (3x3 or None if failed)
            - inlier_mask (boolean array of shape len(matches))
            - residuals (float32 array of Euclidean reprojection errors in pixels)
            - info_dict (parameters, model used, inlier count, RMSE)
        """
        n_pts = len(matches)
        if n_pts < 4:
            return None, np.zeros(n_pts, dtype=bool), np.full(n_pts, 999.0, dtype=np.float32), {
                "status": "FAILED_TOO_FEW_MATCHES",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "model_used": "none"
            }

        pts_src = np.array([m.pt_src for m in matches], dtype=np.float32)
        pts_ref = np.array([m.pt_ref for m in matches], dtype=np.float32)

        # Detect USAC_MAGSAC availability in OpenCV
        method_flag = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        method_name = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC"

        best_H = None
        best_inliers = np.zeros(n_pts, dtype=bool)
        best_residuals = np.full(n_pts, 999.0, dtype=np.float32)
        chosen_model = model_type

        # Decide between Affine (6 DOF) and Homography (8 DOF)
        # Affine is more stable for nadir/near-nadir lunar orbital cameras
        if model_type in ("affine", "auto"):
            # Estimate Affine (2x3 -> convert to 3x3)
            M_aff, aff_inliers = cv2.estimateAffine2D(
                pts_src, pts_ref,
                method=method_flag,
                ransacReprojThreshold=ransac_thresh,
                maxIters=max_iters,
                confidence=confidence
            )

            if M_aff is not None and aff_inliers is not None:
                H_aff = np.eye(3, dtype=np.float32)
                H_aff[:2, :] = M_aff
                
                # Compute reprojection residuals
                pts_src_homo = np.hstack([pts_src, np.ones((n_pts, 1), dtype=np.float32)])
                proj_pts = (H_aff @ pts_src_homo.T).T
                res_aff = np.linalg.norm(proj_pts[:, :2] - pts_ref, axis=1)

                best_H = H_aff
                best_inliers = (aff_inliers.ravel() == 1)
                best_residuals = res_aff.astype(np.float32)
                chosen_model = "affine"

        if model_type in ("homography", "auto") and n_pts >= 6:
            H_homo, homo_inliers = cv2.findHomography(
                pts_src, pts_ref,
                method=method_flag,
                ransacReprojThreshold=ransac_thresh,
                maxIters=max_iters,
                confidence=confidence
            )

            if H_homo is not None and homo_inliers is not None:
                pts_src_homo = np.hstack([pts_src, np.ones((n_pts, 1), dtype=np.float32)])
                proj = (H_homo @ pts_src_homo.T).T
                # Avoid division by zero
                denom = np.where(np.abs(proj[:, 2:3]) < 1e-6, 1e-6, proj[:, 2:3])
                proj_norm = proj[:, :2] / denom
                res_homo = np.linalg.norm(proj_norm - pts_ref, axis=1)
                
                inliers_count_homo = int(np.sum(homo_inliers.ravel() == 1))
                inliers_count_aff = int(np.sum(best_inliers)) if best_H is not None else 0

                # Select Homography if explicitly requested or if it significantly improves inliers without warping degenerate points
                if model_type == "homography" or (model_type == "auto" and inliers_count_homo > inliers_count_aff + 3):
                    best_H = H_homo
                    best_inliers = (homo_inliers.ravel() == 1)
                    best_residuals = res_homo.astype(np.float32)
                    chosen_model = "homography"

        if best_H is None:
            return None, np.zeros(n_pts, dtype=bool), np.full(n_pts, 999.0, dtype=np.float32), {
                "status": "GEOMETRIC_FIT_FAILED",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "model_used": chosen_model
            }

        inlier_count = int(np.sum(best_inliers))
        inlier_ratio = float(inlier_count / max(1, n_pts))
        inlier_residuals = best_residuals[best_inliers] if inlier_count > 0 else np.array([999.0])
        rmse = float(np.sqrt(np.mean(inlier_residuals ** 2)))

        info = {
            "status": "SUCCESS",
            "solver": method_name,
            "model_used": chosen_model,
            "threshold_px": ransac_thresh,
            "total_candidates": n_pts,
            "inlier_count": inlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "rmse_px": round(rmse, 3),
            "median_residual_px": round(float(np.median(inlier_residuals)), 3),
            "p95_residual_px": round(float(np.percentile(inlier_residuals, 95)), 3)
        }

        return best_H, best_inliers, best_residuals, info
