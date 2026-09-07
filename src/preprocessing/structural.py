"""
Luna-tics: Common Structural Representation Engine
Converts multimodal optical imagery into illumination-invariant structural representations.
"""

from typing import Tuple
import numpy as np
import cv2

from src.preprocessing.normalization import ImageNormalizer


class StructuralRepresentationBuilder:
    """
    Transforms raw or normalized lunar radiance into a modality-tolerant structural map.
    Invariant to monotonic radiometric variations, contrasting Sun directions, and sensor albedo differences.
    """

    @classmethod
    def generate_structural_representation(
        cls,
        image_01: np.ndarray,
        use_phase_congruency_approx: bool = True,
        edge_weight: float = 0.55,
        contrast_clip: float = 2.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Build common structural representation.

        Combines:
        1. Multi-scale directional gradient energy (Scharr filters)
        2. Structural edge preservation (Laplacian of Gaussian / Phase Congruency moment approximation)
        3. CLAHE-enhanced base intensity
        """
        arr = np.clip(image_01, 0.0, 1.0).astype(np.float32)

        # 1. CLAHE enhanced baseline
        clahe_base = ImageNormalizer.apply_clahe(arr, clip_limit=contrast_clip)
        base_01 = clahe_base.astype(np.float32) / 255.0

        # 2. Multi-scale Gradient Magnitude (Scharr preserves high-frequency crater rims)
        gx = cv2.Scharr(arr, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(arr, cv2.CV_32F, 0, 1)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Robust gradient scaling
        p98 = np.percentile(grad_mag, 98)
        if p98 > 1e-5:
            grad_norm = np.clip(grad_mag / p98, 0.0, 1.0)
        else:
            grad_norm = np.zeros_like(grad_mag)

        # 3. Structural Ridge / Edge Energy (Phase Congruency approximation via multi-scale LoG)
        if use_phase_congruency_approx:
            g1 = cv2.GaussianBlur(arr, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(arr, (9, 9), 2.0)
            dog = np.abs(g1 - g2)
            dog_p98 = np.percentile(dog, 98)
            if dog_p98 > 1e-5:
                struct_energy = np.clip(dog / dog_p98, 0.0, 1.0)
            else:
                struct_energy = np.zeros_like(dog)
        else:
            struct_energy = grad_norm

        # 4. Fuse structural components: Edge features dominate, modulated by normalized contrast
        fused = (1.0 - edge_weight) * base_01 + (edge_weight * 0.5) * grad_norm + (edge_weight * 0.5) * struct_energy
        fused_01 = np.clip(fused, 0.0, 1.0)

        # Convert to 8-bit uint8 representation for standard feature extractors (SIFT)
        structural_u8 = np.clip(fused_01 * 255.0, 0, 255).astype(np.uint8)

        stats = {
            "mean_structure_val": float(np.mean(structural_u8)),
            "std_structure_val": float(np.std(structural_u8)),
            "phase_congruency_enabled": use_phase_congruency_approx,
            "edge_weight": edge_weight
        }

        return structural_u8, stats
