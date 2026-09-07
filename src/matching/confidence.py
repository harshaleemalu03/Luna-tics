"""
Luna-tics: Evidence-Guided Correspondence Fusion Engine
Combines Photometric (P), Scale (S), Geometric (G), Terrain (T), Descriptor (D),
and Spatial (U) evidence into a unified confidence metric C_i in [0, 1].
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.matching.primary import CandidateMatch


@dataclass
class MatchEvidence:
    """Breakdown of all 6 evidence factors for an individual correspondence."""
    match_id: int
    pt_src: Tuple[float, float]
    pt_ref: Tuple[float, float]
    descriptor_evidence: float      # D_i in [0, 1]
    scale_evidence: float           # S_i in [0, 1]
    geometry_evidence: float        # G_i in [0, 1]
    terrain_evidence: float         # T_i in [0, 1]
    photometric_evidence: float     # P_i in [0, 1]
    spatial_contribution: float     # U_i in [0, 1]
    final_confidence: float         # C_i in [0, 1]
    is_accepted: bool = True
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceFusionEngine:
    """
    Computes multi-factor evidence fusion:
    C_i = (wP * P_i + wS * S_i + wG * G_i + wT * T_i + wD * D_i + wU * U_i) / sum(w)
    """

    def __init__(
        self,
        weight_photometric: float = 0.15,
        weight_scale: float = 0.15,
        weight_geometry: float = 0.25,
        weight_terrain: float = 0.20,
        weight_descriptor: float = 0.15,
        weight_spatial: float = 0.10,
        confidence_threshold: float = 0.55
    ):
        self.wP = weight_photometric
        self.wS = weight_scale
        self.wG = weight_geometry
        self.wT = weight_terrain
        self.wD = weight_descriptor
        self.wU = weight_spatial
        self.total_weight = self.wP + self.wS + self.wG + self.wT + self.wD + self.wU
        self.threshold = confidence_threshold

    def calculate_descriptor_evidence(self, match: CandidateMatch) -> float:
        """
        Descriptor confidence D_i based on Lowe ratio score.
        D_i approaches 1.0 for highly distinctive matches, drops towards 0 for ambiguous ones.
        """
        ratio = match.ratio_score
        # Inverse quadratic scaling
        score = np.clip(1.0 - (ratio ** 2), 0.0, 1.0)
        return float(score)

    def calculate_spatial_contribution(
        self,
        matches: List[CandidateMatch],
        image_shape: Tuple[int, int],
        grid_bins: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Spatial contribution U_i: points in sparse cells receive high utility (up to 1.0),
        while points in redundant, dense clusters receive lower utility (e.g. 0.4 - 0.6).
        """
        h, w = image_shape
        rows, cols = grid_bins
        cell_w = max(1.0, w / cols)
        cell_h = max(1.0, h / rows)

        # Count occupancy per cell
        grid_counts = np.zeros((rows, cols), dtype=int)
        cell_indices = []
        for m in matches:
            c = int(np.clip(m.pt_src[0] // cell_w, 0, cols - 1))
            r = int(np.clip(m.pt_src[1] // cell_h, 0, rows - 1))
            grid_counts[r, c] += 1
            cell_indices.append((r, c))

        # Utility inversely proportional to cell saturation
        u_scores = []
        for r, c in cell_indices:
            count = grid_counts[r, c]
            # 1 to 3 points in cell -> highest marginal utility
            # > 10 points -> diminishing utility
            utility = 1.0 / (1.0 + 0.15 * max(0, count - 1))
            u_scores.append(float(np.clip(utility, 0.2, 1.0)))

        return np.array(u_scores, dtype=np.float32)

    def fuse_evidence(
        self,
        matches: List[CandidateMatch],
        photometric_scores: np.ndarray,
        scale_scores: np.ndarray,
        geometry_scores: np.ndarray,
        terrain_scores: np.ndarray,
        spatial_scores: np.ndarray,
        active_weights: Optional[Dict[str, float]] = None
    ) -> List[MatchEvidence]:
        """
        Execute unified evidence fusion across all candidate correspondences.
        """
        wP = active_weights.get("wP", self.wP) if active_weights else self.wP
        wS = active_weights.get("wS", self.wS) if active_weights else self.wS
        wG = active_weights.get("wG", self.wG) if active_weights else self.wG
        wT = active_weights.get("wT", self.wT) if active_weights else self.wT
        wD = active_weights.get("wD", self.wD) if active_weights else self.wD
        wU = active_weights.get("wU", self.wU) if active_weights else self.wU
        sum_w = max(1e-6, wP + wS + wG + wT + wD + wU)

        evidence_list: List[MatchEvidence] = []

        for i, m in enumerate(matches):
            d_val = self.calculate_descriptor_evidence(m)
            p_val = float(photometric_scores[i]) if i < len(photometric_scores) else 0.5
            s_val = float(scale_scores[i]) if i < len(scale_scores) else 0.5
            g_val = float(geometry_scores[i]) if i < len(geometry_scores) else 0.5
            t_val = float(terrain_scores[i]) if i < len(terrain_scores) else 0.5
            u_val = float(spatial_scores[i]) if i < len(spatial_scores) else 0.5

            conf = (wP * p_val + wS * s_val + wG * g_val + wT * t_val + wD * d_val + wU * u_val) / sum_w
            conf = float(np.clip(conf, 0.0, 1.0))

            # Determine rejection reason if below threshold
            accepted = conf >= self.threshold
            rejection_reason = None
            if not accepted:
                # Identify dominant weak factor
                scores_dict = {
                    "Descriptor ambiguity": d_val,
                    "Inconsistent scale ratio": s_val,
                    "High geometric residual": g_val,
                    "Neighborhood terrain violation": t_val,
                    "Poor photometric illumination / shadow": p_val
                }
                weakest = min(scores_dict, key=scores_dict.get)
                rejection_reason = f"Rejected: {weakest} (score: {scores_dict[weakest]:.2f}, final: {conf:.2f})"

            evidence_list.append(
                MatchEvidence(
                    match_id=m.id,
                    pt_src=m.pt_src,
                    pt_ref=m.pt_ref,
                    descriptor_evidence=round(d_val, 3),
                    scale_evidence=round(s_val, 3),
                    geometry_evidence=round(g_val, 3),
                    terrain_evidence=round(t_val, 3),
                    photometric_evidence=round(p_val, 3),
                    spatial_contribution=round(u_val, 3),
                    final_confidence=round(conf, 3),
                    is_accepted=accepted,
                    rejection_reason=rejection_reason
                )
            )

        return evidence_list
