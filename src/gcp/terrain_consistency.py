"""
Luna-tics: Lunar Terrain & Crater Neighborhood Consistency Filter
Rejects deceptive matches in repetitive lunar terrain using local graph geometric consensus.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.matching.primary import CandidateMatch


class TerrainConsistencyFilter:
    """
    Evaluates topological and geometric consistency among neighboring lunar crater features.
    Deceptive crater matches violate relative distance and angular relationships with adjacent landmarks.
    """

    @staticmethod
    def evaluate_neighborhood_consistency(
        matches: List[CandidateMatch],
        k_neighbors: int = 6,
        distance_ratio_tol: float = 0.35,
        angular_tol_deg: float = 25.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Compute terrain consistency score T_i in [0, 1] and diagnostic neighborhood data.

        Returns:
            Tuple of:
            - scores: np.ndarray of shape (len(matches),) with T_i in [0, 1]
            - inspectable_details: list of dicts with neighbor consensus data
        """
        n = len(matches)
        if n <= 3:
            return np.ones(n, dtype=np.float32), [{"consensus_ratio": 1.0, "reason": "Too few points for graph"} for _ in range(n)]

        pts_src = np.array([m.pt_src for m in matches], dtype=np.float32)
        pts_ref = np.array([m.pt_ref for m in matches], dtype=np.float32)

        # Build k-NN graph in source image space
        k = min(k_neighbors + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(pts_src)
        _, indices = nbrs.kneighbors(pts_src)

        scores = np.zeros(n, dtype=np.float32)
        details = []

        for i in range(n):
            neighbor_idxs = [idx for idx in indices[i] if idx != i]
            if not neighbor_idxs:
                scores[i] = 0.5
                details.append({"consensus_ratio": 0.5, "reason": "Isolated point"})
                continue

            # Compute pairwise vectors to neighbors
            v_src = pts_src[neighbor_idxs] - pts_src[i]
            v_ref = pts_ref[neighbor_idxs] - pts_ref[i]

            dist_src = np.linalg.norm(v_src, axis=1) + 1e-6
            dist_ref = np.linalg.norm(v_ref, axis=1) + 1e-6

            ratios = dist_ref / dist_src
            median_ratio = np.median(ratios)

            # Pairwise orientation angles
            angles_src = np.degrees(np.arctan2(v_src[:, 1], v_src[:, 0]))
            angles_ref = np.degrees(np.arctan2(v_ref[:, 1], v_ref[:, 0]))
            angle_diffs = np.abs((angles_ref - angles_src + 180) % 360 - 180)
            median_angle_diff = np.median(angle_diffs)

            # Consensus votes
            ratio_consistent = np.abs(ratios - median_ratio) < (distance_ratio_tol * max(0.5, median_ratio))
            angle_consistent = np.abs(angle_diffs - median_angle_diff) < angular_tol_deg

            valid_neighbor_votes = np.logical_and(ratio_consistent, angle_consistent)
            consensus_ratio = float(np.sum(valid_neighbor_votes) / len(neighbor_idxs))

            # Scale to continuous [0, 1] score
            score = float(np.clip(0.2 + 0.8 * consensus_ratio, 0.0, 1.0))
            scores[i] = score

            details.append({
                "match_id": matches[i].id,
                "neighbor_count": len(neighbor_idxs),
                "neighbor_indices": [int(idx) for idx in neighbor_idxs],
                "consensus_ratio": round(consensus_ratio, 3),
                "median_distance_ratio": round(float(median_ratio), 3),
                "median_relative_rotation_deg": round(float(median_angle_diff), 2),
                "consistent_neighbors_count": int(np.sum(valid_neighbor_votes)),
                "is_terrain_consistent": consensus_ratio >= 0.5
            })

        return scores, details
