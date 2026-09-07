"""
Luna-tics: Coarse Spatial Alignment
Determines initial search bounds using geographic footprints or image-derived coarse correlation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from src.io.metadata import LunarMetadata


class CoarseAligner:
    """Estimates coarse spatial transformation and overlap region between source and reference."""

    @staticmethod
    def estimate_coarse_alignment(
        img_src: np.ndarray,
        img_ref: np.ndarray,
        meta_src: Optional[LunarMetadata],
        meta_ref: Optional[LunarMetadata]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Estimate initial affine transformation matrix (2x3) and alignment mode.
        """
        # Mode 1: Geographic footprint overlap if both have bounding boxes
        if (meta_src and meta_src.has_spatial_coordinates() and
            meta_ref and meta_ref.has_spatial_coordinates()):
            bbox_src = meta_src.footprint_bbox  # (min_lon, min_lat, max_lon, max_lat)
            bbox_ref = meta_ref.footprint_bbox

            # Calculate theoretical scale and translation from coordinate frames
            w_src_deg = max(1e-4, bbox_src[2] - bbox_src[0])
            h_src_deg = max(1e-4, bbox_src[3] - bbox_src[1])
            w_ref_deg = max(1e-4, bbox_ref[2] - bbox_ref[0])
            h_ref_deg = max(1e-4, bbox_ref[3] - bbox_ref[1])

            sx = (img_ref.shape[1] / w_ref_deg) * (w_src_deg / img_src.shape[1])
            sy = (img_ref.shape[0] / h_ref_deg) * (h_src_deg / img_src.shape[0])

            tx = ((bbox_src[0] - bbox_ref[0]) / w_ref_deg) * img_ref.shape[1]
            ty = ((bbox_ref[3] - bbox_src[3]) / h_ref_deg) * img_ref.shape[0]

            M_coarse = np.float32([[sx, 0, tx], [0, sy, ty]])
            info = {
                "alignment_mode": "GEOGRAPHIC_FOOTPRINT_OVERLAP",
                "estimated_translation": (float(tx), float(ty)),
                "estimated_scale": (float(sx), float(sy)),
                "confidence": 0.85
            }
            return M_coarse, info

        # Mode 2: Image-based coarse alignment (multiscale phase correlation on downsampled thumbnails)
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        thumb_size = 256
        s_scale = thumb_size / max(h_s, w_s)
        r_scale = thumb_size / max(h_r, w_r)

        s_thumb = cv2.resize(img_src, (int(w_s * s_scale), int(h_s * s_scale)))
        r_thumb = cv2.resize(img_ref, (int(w_r * r_scale), int(h_r * r_scale)))

        # Ensure both thumbnails have equal shape for phase correlation
        max_h = max(s_thumb.shape[0], r_thumb.shape[0])
        max_w = max(s_thumb.shape[1], r_thumb.shape[1])

        s_pad = np.zeros((max_h, max_w), dtype=np.float32)
        r_pad = np.zeros((max_h, max_w), dtype=np.float32)

        s_pad[:s_thumb.shape[0], :s_thumb.shape[1]] = s_thumb.astype(np.float32)
        r_pad[:r_thumb.shape[0], :r_thumb.shape[1]] = r_thumb.astype(np.float32)

        # Windowing function (Hanning) to reduce boundary spectral leakage
        hann = cv2.createHanningWindow((max_w, max_h), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(s_pad * hann, r_pad * hann)

        # Map back to full-resolution coordinates
        dx = shift[0] / r_scale
        dy = shift[1] / r_scale

        # Identity with coarse translation fallback
        M_coarse = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        info = {
            "alignment_mode": "IMAGE_BASED_COARSE_ALIGNMENT",
            "estimated_translation": (float(dx), float(dy)),
            "phase_correlation_peak": float(response),
            "confidence": float(np.clip(response * 2.0, 0.3, 0.9))
        }

        return M_coarse, info
