"""
Luna-tics: Local Sub-pixel Correspondence Refinement
Refines GCP locations using Normalized Cross-Correlation (NCC) with 2D quadratic peak interpolation.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2

from src.matching.primary import CandidateMatch


class SubpixelRefiner:
    """Performs localized template matching and quadratic surface peak interpolation."""

    @staticmethod
    def refine_match_locations(
        img_src_u8: np.ndarray,
        img_ref_u8: np.ndarray,
        matches: List[CandidateMatch],
        template_radius: int = 12,
        search_radius: int = 4
    ) -> Tuple[List[CandidateMatch], Dict[str, Any]]:
        """
        Refine each candidate match's reference point using sub-pixel NCC surface fitting.
        """
        refined_matches: List[CandidateMatch] = []
        h_s, w_s = img_src_u8.shape[:2]
        h_r, w_r = img_ref_u8.shape[:2]

        offsets = []
        correlations = []

        t_rad = template_radius
        s_rad = search_radius
        t_size = 2 * t_rad + 1

        for m in matches:
            xs, ys = int(round(m.pt_src[0])), int(round(m.pt_src[1]))
            xr, yr = int(round(m.pt_ref[0])), int(round(m.pt_ref[1]))

            # Check template bounds in source
            if (xs - t_rad < 0 or xs + t_rad >= w_s or
                ys - t_rad < 0 or ys + t_rad >= h_s):
                refined_matches.append(m)
                continue

            # Check search area bounds in reference
            r_box_x0 = xr - t_rad - s_rad
            r_box_x1 = xr + t_rad + s_rad + 1
            r_box_y0 = yr - t_rad - s_rad
            r_box_y1 = yr + t_rad + s_rad + 1

            if (r_box_x0 < 0 or r_box_x1 > w_r or
                r_box_y0 < 0 or r_box_y1 > h_r):
                refined_matches.append(m)
                continue

            src_patch = img_src_u8[ys - t_rad:ys + t_rad + 1, xs - t_rad:xs + t_rad + 1]
            ref_search = img_ref_u8[r_box_y0:r_box_y1, r_box_x0:r_box_x1]

            # Normalized Cross Correlation
            res_ncc = cv2.matchTemplate(ref_search, src_patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_ncc)

            best_u, best_v = max_loc

            # Sub-pixel peak estimation via 2D quadratic interpolation if inside margin
            dx_sub, dy_sub = 0.0, 0.0
            if (0 < best_u < res_ncc.shape[1] - 1 and
                0 < best_v < res_ncc.shape[0] - 1):
                # 1D parabolic fit along x
                denom_x = 2 * (res_ncc[best_v, best_u - 1] - 2 * res_ncc[best_v, best_u] + res_ncc[best_v, best_u + 1])
                if abs(denom_x) > 1e-5:
                    dx_sub = (res_ncc[best_v, best_u - 1] - res_ncc[best_v, best_u + 1]) / denom_x

                # 1D parabolic fit along y
                denom_y = 2 * (res_ncc[best_v - 1, best_u] - 2 * res_ncc[best_v, best_u] + res_ncc[best_v + 1, best_u])
                if abs(denom_y) > 1e-5:
                    dy_sub = (res_ncc[best_v - 1, best_u] - res_ncc[best_v + 1, best_u]) / denom_y

            # Absolute refined reference coordinate
            refined_xr = r_box_x0 + t_rad + best_u + dx_sub
            refined_yr = r_box_y0 + t_rad + best_v + dy_sub

            disp = np.hypot(refined_xr - m.pt_ref[0], refined_yr - m.pt_ref[1])
            offsets.append(disp)
            correlations.append(max_val)

            refined_matches.append(
                CandidateMatch(
                    id=m.id,
                    pt_src=m.pt_src,
                    pt_ref=(float(refined_xr), float(refined_yr)),
                    scale_src=m.scale_src,
                    scale_ref=m.scale_ref,
                    angle_src=m.angle_src,
                    angle_ref=m.angle_ref,
                    descriptor_distance=m.descriptor_distance,
                    ratio_score=m.ratio_score,
                    is_rescued=m.is_rescued
                )
            )

        mean_shift = float(np.mean(offsets)) if offsets else 0.0
        mean_ncc = float(np.mean(correlations)) if correlations else 0.0

        stats = {
            "refinement_points_count": len(refined_matches),
            "mean_subpixel_shift_px": round(mean_shift, 3),
            "mean_peak_ncc_score": round(mean_ncc, 3),
            "refinement_technique": "NCC_Quadratic_Interpolation"
        }

        return refined_matches, stats
