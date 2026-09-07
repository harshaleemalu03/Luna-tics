"""
Luna-tics: Illumination-Aware Photometric Reliability Processing
Models lunar photometric physics and generates continuous reliability maps in [0, 1].
"""

from typing import Optional, Tuple
import numpy as np
import cv2

from src.io.metadata import LunarMetadata


class PhotometricReliabilityEstimator:
    """
    Computes dense illumination reliability maps based on physical solar geometry
    and image-derived radiometric stability metrics.
    """

    @staticmethod
    def compute_physical_prior(metadata: Optional[LunarMetadata]) -> float:
        """
        Calculates global physical illumination reliability factor [0, 1] from
        incidence, emission, and phase angles using Lommel-Seeliger scattering formulation.
        """
        if metadata is None or not metadata.has_illumination_geometry():
            return 0.70  # default neutral prior when geometry is missing

        inc = metadata.incidence_angle_deg or 45.0
        em = metadata.emission_angle_deg or 5.0
        phase = metadata.phase_angle_deg or 45.0

        # Convert to radians
        rad_inc = np.radians(np.clip(inc, 0.0, 89.9))
        rad_em = np.radians(np.clip(em, 0.0, 89.9))

        cos_i = np.cos(rad_inc)
        cos_e = np.cos(rad_em)

        # Lommel-Seeliger lunar scattering factor
        ls_factor = cos_i / (cos_i + cos_e + 1e-6)

        # Grazing angle penalty: incidence > 75 deg casts deep, unstable shadows
        if inc > 75.0:
            shadow_risk = 1.0 - ((inc - 75.0) / 15.0) * 0.5
        else:
            shadow_risk = 1.0

        # High phase angle penalty: phase > 80 deg creates stark contrast disparities
        if phase > 80.0:
            phase_risk = 1.0 - ((phase - 80.0) / 40.0) * 0.4
        else:
            phase_risk = 1.0

        score = float(np.clip(ls_factor * shadow_risk * phase_risk, 0.2, 1.0))
        return score

    @classmethod
    def compute_reliability_map(
        cls,
        image_01: np.ndarray,
        metadata: Optional[LunarMetadata] = None,
        shadow_threshold: float = 0.08,
        saturation_threshold: float = 0.92,
        ksize: int = 15
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate continuous pixel-wise reliability map in [0, 1].

        High reliability = well-illuminated, textured terrain with stable gradients.
        Low reliability = pitch-black cast shadows, clipped highlights, or flat featureless plains.
        """
        arr = np.clip(image_01, 0.0, 1.0).astype(np.float32)
        h, w = arr.shape[:2]

        # 1. Shadow Penalty: smooth transition near shadow_threshold
        shadow_weight = 1.0 / (1.0 + np.exp(-15.0 * (arr - shadow_threshold)))

        # 2. Saturation Penalty: smooth transition near saturation_threshold
        saturation_weight = 1.0 / (1.0 + np.exp(20.0 * (arr - saturation_threshold)))

        # 3. Local Gradient Energy / Texture richness
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Smooth gradient over local neighborhood (ksize)
        local_grad = cv2.GaussianBlur(grad_mag, (ksize, ksize), 0)
        grad_max = np.percentile(local_grad, 98)
        if grad_max > 1e-4:
            texture_weight = np.clip(local_grad / grad_max, 0.0, 1.0)
        else:
            texture_weight = np.zeros_like(arr)

        # Blend texture and intensity bounds
        image_reliability = shadow_weight * saturation_weight * (0.35 + 0.65 * texture_weight)

        # Modulate with physical prior
        phys_prior = cls.compute_physical_prior(metadata)
        final_map = np.clip(image_reliability * (0.6 + 0.4 * phys_prior), 0.0, 1.0)

        # Calculate statistics
        shadow_pct = float(np.mean(arr < shadow_threshold) * 100.0)
        sat_pct = float(np.mean(arr > saturation_threshold) * 100.0)
        mean_rel = float(np.mean(final_map))

        info = {
            "physical_prior": phys_prior,
            "mean_reliability": mean_rel,
            "shadow_area_percent": shadow_pct,
            "saturated_area_percent": sat_pct,
            "physics_available": metadata.has_illumination_geometry() if metadata else False
        }

        return final_map.astype(np.float32), info

    @staticmethod
    def sample_keypoint_reliability(
        keypoints,
        reliability_map: np.ndarray
    ) -> np.ndarray:
        """Sample reliability values at keypoint (x, y) coordinates."""
        h, w = reliability_map.shape[:2]
        scores = []
        for kp in keypoints:
            pt = kp.pt if hasattr(kp, "pt") else kp
            x, y = int(round(pt[0])), int(round(pt[1]))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            scores.append(float(reliability_map[y, x]))
        return np.array(scores, dtype=np.float32)
