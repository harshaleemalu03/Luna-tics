"""
Luna-tics: Spatially Distributed Ground Control Point (GCP) Network Optimization
Partitions lunar scene into spatial cells and optimizes for high confidence, low residual error,
and maximum spatial distribution and coverage.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

from src.matching.primary import CandidateMatch
from src.matching.confidence import MatchEvidence


class SpatialGCPOptimizer:
    """
    Selects an optimal, spatially dispersed GCP network across the lunar footprint.
    Prevents degenerate clusters where dozens of points crowd around a single crater rim.
    """

    @staticmethod
    def optimize_gcp_network(
        matches: List[CandidateMatch],
        evidences: List[MatchEvidence],
        inlier_mask: np.ndarray,
        residuals: np.ndarray,
        image_shape: Tuple[int, int],
        grid_rows: int = 6,
        grid_cols: int = 6,
        max_points_per_cell: int = 2,
        target_total_gcps: int = 40
    ) -> Tuple[List[CandidateMatch], List[MatchEvidence], Dict[str, Any]]:
        """
        Optimize GCP selection across spatial cells.

        Returns:
            Tuple of:
            - selected_gcp_matches: List[CandidateMatch]
            - selected_gcp_evidences: List[MatchEvidence]
            - distribution_metrics: Dict of coverage, uniformity, and occupancy stats
        """
        h, w = image_shape
        cell_w = max(1.0, w / grid_cols)
        cell_h = max(1.0, h / grid_rows)

        # Filter to inliers only
        inlier_indices = [i for i in range(len(matches)) if inlier_mask[i] and evidences[i].is_accepted]

        if not inlier_indices:
            # Fallback to top accepted matches if no strict inliers
            inlier_indices = [i for i in range(len(matches)) if evidences[i].is_accepted]

        if not inlier_indices:
            return [], [], {
                "spatial_coverage_pct": 0.0,
                "grid_occupancy_pct": 0.0,
                "spatial_uniformity_index": 0.0,
                "total_gcps": 0
            }

        # Group inliers into spatial grid bins
        bins: Dict[Tuple[int, int], List[int]] = {}
        for idx in inlier_indices:
            m = matches[idx]
            c = int(np.clip(m.pt_src[0] // cell_w, 0, grid_cols - 1))
            r = int(np.clip(m.pt_src[1] // cell_h, 0, grid_rows - 1))
            bins.setdefault((r, c), []).append(idx)

        # Sort matches within each cell by joint criterion:
        # Score = Confidence - 0.2 * (Normalized Residual) + Centrality Bonus
        selected_indices: List[int] = []

        for (r, c), idxs in bins.items():
            cell_center_x = (c + 0.5) * cell_w
            cell_center_y = (r + 0.5) * cell_h

            def cell_score(idx: int) -> float:
                conf = evidences[idx].final_confidence
                res = residuals[idx]
                pt = matches[idx].pt_src
                dist_to_center = np.hypot(pt[0] - cell_center_x, pt[1] - cell_center_y)
                center_bonus = 0.1 * (1.0 - min(1.0, dist_to_center / (cell_w + 1e-4)))
                return conf - 0.15 * min(1.0, res / 5.0) + center_bonus

            # Sort descending
            sorted_idxs = sorted(idxs, key=cell_score, reverse=True)
            chosen = sorted_idxs[:max_points_per_cell]
            selected_indices.extend(chosen)

        # If total selected exceeds target_total_gcps, take best across unique cells
        if len(selected_indices) > target_total_gcps:
            selected_indices = sorted(
                selected_indices,
                key=lambda idx: evidences[idx].final_confidence - 0.1 * residuals[idx],
                reverse=True
            )[:target_total_gcps]

        final_matches = [matches[i] for i in selected_indices]
        final_evidences = [evidences[i] for i in selected_indices]

        # Calculate Spatial Distribution Statistics
        occupied_cells = len(bins)
        total_cells = grid_rows * grid_cols
        grid_occupancy_pct = float((occupied_cells / total_cells) * 100.0)

        # Convex hull coverage area
        if len(final_matches) >= 3:
            src_coords = np.array([m.pt_src for m in final_matches], dtype=np.float32)
            try:
                hull = ConvexHull(src_coords)
                hull_area = hull.volume  # 2D volume is area
                total_area = float(w * h)
                spatial_coverage_pct = float(np.clip((hull_area / total_area) * 100.0, 0.0, 100.0))
            except Exception:
                spatial_coverage_pct = grid_occupancy_pct
        else:
            spatial_coverage_pct = grid_occupancy_pct

        # Spatial Uniformity Index: 1 - Coefficient of Variation of nearest neighbor distances
        if len(final_matches) >= 4:
            coords = np.array([m.pt_src for m in final_matches], dtype=np.float32)
            dists = distance_matrix(coords, coords)
            np.fill_diagonal(dists, np.inf)
            nn_dists = np.min(dists, axis=1)
            mean_nn = float(np.mean(nn_dists))
            std_nn = float(np.std(nn_dists))
            cv = (std_nn / (mean_nn + 1e-6))
            uniformity_index = float(np.clip(1.0 - cv * 0.5, 0.1, 1.0))
        else:
            uniformity_index = 0.5

        metrics = {
            "total_gcps": len(final_matches),
            "occupied_grid_cells": occupied_cells,
            "total_grid_cells": total_cells,
            "grid_occupancy_pct": round(grid_occupancy_pct, 2),
            "spatial_coverage_pct": round(spatial_coverage_pct, 2),
            "spatial_uniformity_index": round(uniformity_index, 3)
        }

        return final_matches, final_evidences, metrics
