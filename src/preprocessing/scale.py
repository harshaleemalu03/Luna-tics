"""
Luna-tics: Sensor-Aware Scale Estimation and Adaptive Scale Pyramid
Uses GSD priors to determine optimal search levels and verify scale consistency.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2

from src.io.metadata import LunarMetadata


class ScaleHandler:
    """Calculates GSD-based scale prior and manages adaptive scale pyramid search."""

    @staticmethod
    def estimate_scale_prior(
        source_meta: Optional[LunarMetadata],
        reference_meta: Optional[LunarMetadata],
        fallback_source_gsd: float = 80.0,
        fallback_ref_gsd: float = 100.0
    ) -> Dict[str, Any]:
        """
        Estimate theoretical scale ratio from GSDs:
        scale_ratio = reference_GSD / source_GSD
        
        Example:
        source = TMC-2 (5 m/px), reference = LRO_WAC (100 m/px)
        scale_ratio = 100 / 5 = 20.0 (source is 20x higher resolution than reference).
        """
        src_gsd = source_meta.get_effective_gsd(fallback=fallback_source_gsd) if source_meta else fallback_source_gsd
        ref_gsd = reference_meta.get_effective_gsd(fallback=fallback_ref_gsd) if reference_meta else fallback_ref_gsd

        if src_gsd <= 0 or ref_gsd <= 0:
            scale_ratio = 1.0
            is_prior_valid = False
        else:
            scale_ratio = float(ref_gsd / src_gsd)
            is_prior_valid = True

        return {
            "source_gsd": src_gsd,
            "reference_gsd": ref_gsd,
            "estimated_scale_ratio": scale_ratio,
            "is_prior_valid": is_prior_valid
        }

    @staticmethod
    def plan_adaptive_pyramid_levels(
        scale_ratio: float,
        max_levels: int = 4
    ) -> List[float]:
        """
        Given the expected scale ratio, plans a focused set of pyramid scale factors
        centered around the expected prior rather than an exhaustive brute-force search.
        """
        if scale_ratio <= 0.05:
            scale_ratio = 1.0

        # Create localized search band around scale_ratio (e.g. 0.7x, 1.0x, 1.3x of prior)
        if 0.8 <= scale_ratio <= 1.25:
            # Similar GSDs (e.g. IIRS 80m vs WAC 100m)
            levels = [1.0, 0.8, 1.25]
        else:
            # Disparate GSDs (e.g. TMC-2 vs WAC, or OHRC vs NAC)
            levels = [
                scale_ratio * 0.75,
                scale_ratio * 1.0,
                scale_ratio * 1.33
            ]

        # Filter out extreme scaling factors that degenerate image
        valid_levels = [round(float(s), 3) for s in levels if 0.05 <= s <= 50.0]
        if not valid_levels:
            valid_levels = [1.0]

        return valid_levels[:max_levels]

    @staticmethod
    def compute_match_scale_consistency(
        kp_src_size: float,
        kp_ref_size: float,
        prior_scale_ratio: float
    ) -> float:
        """
        Calculate scale consistency score S_i in [0, 1] for a candidate match.
        
        Evaluates how closely the keypoint octave size ratio matches the expected physical GSD ratio:
        S_i = exp( - (ln(s_observed) - ln(s_prior))^2 / (2 * sigma^2) )
        """
        if kp_src_size <= 1e-4 or kp_ref_size <= 1e-4 or prior_scale_ratio <= 1e-4:
            return 0.5  # neutral score

        observed_ratio = kp_ref_size / kp_src_size
        log_diff = np.abs(np.log(observed_ratio) - np.log(prior_scale_ratio))

        # sigma = 0.8 allows moderate octave deviation while penalizing unrealistic scale jumps
        sigma = 0.8
        score = float(np.exp(- (log_diff ** 2) / (2.0 * (sigma ** 2))))
        return float(np.clip(score, 0.0, 1.0))
