"""
Luna-tics: Robust Image Normalization for Lunar Remote Sensing
Applies percentile clipping (cosmic ray rejection) and adaptive contrast enhancement.
"""

from typing import Tuple
import numpy as np
import cv2


class ImageNormalizer:
    """Robust radiometric pre-conditioning for planetary surface imagery."""

    @staticmethod
    def percentile_clip_normalize(
        img: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> np.ndarray:
        """
        Clip extreme outliers (cosmic rays, sensor dead pixels) and scale to [0, 1].
        """
        arr = img.astype(np.float32)
        valid_mask = np.isfinite(arr)
        if not np.any(valid_mask):
            return np.zeros_like(arr, dtype=np.float32)

        p_low = np.percentile(arr[valid_mask], lower_percentile)
        p_high = np.percentile(arr[valid_mask], upper_percentile)

        if p_high - p_low < 1e-6:
            clipped = np.clip(arr, p_low, p_high) - p_low
        else:
            clipped = (np.clip(arr, p_low, p_high) - p_low) / (p_high - p_low)

        return np.nan_to_num(clipped, nan=0.0).astype(np.float32)

    @staticmethod
    def apply_clahe(
        normalized_img_01: np.ndarray,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Contrast-Limited Adaptive Histogram Equalization (CLAHE) to reveal subtle regolith details.
        Returns uint8 array in [0, 255].
        """
        u8_img = np.clip(normalized_img_01 * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(u8_img)
        return enhanced
