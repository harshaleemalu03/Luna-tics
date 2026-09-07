"""
Luna-tics: Primary Local Feature Extraction and Matching Pipeline
Extracts SIFT features on structural representations with ratio testing and mutual consistency.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class CandidateMatch:
    """Detailed record of a matched keypoint pair with physical and structural attributes."""
    id: int
    pt_src: Tuple[float, float]          # (x, y) in source coordinate system
    pt_ref: Tuple[float, float]          # (x, y) in reference coordinate system
    scale_src: float                     # Keypoint octave size / scale in source
    scale_ref: float                     # Keypoint octave size / scale in reference
    angle_src: float                     # Orientation in degrees [0, 360)
    angle_ref: float                     # Orientation in degrees [0, 360)
    descriptor_distance: float           # Euclidean distance between descriptors
    ratio_score: float                   # Distance ratio (d_1 / d_2) from Lowe ratio test
    is_rescued: bool = False             # True if added via Selective Learned Rescue


class PrimaryMatcher:
    """Executes interpretable, robust primary correspondence search."""

    def __init__(
        self,
        max_features: int = 2500,
        ratio_threshold: float = 0.80,
        mutual_check: bool = True
    ):
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.mutual_check = mutual_check
        self.sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=10)

    def extract_features(self, structural_image_u8: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect SIFT keypoints and compute 128-D descriptors on structural image."""
        kps, descs = self.sift.detectAndCompute(structural_image_u8, None)
        if descs is None:
            descs = np.empty((0, 128), dtype=np.float32)
        return kps, descs

    def match(
        self,
        kps_src: List[cv2.KeyPoint],
        descs_src: np.ndarray,
        kps_ref: List[cv2.KeyPoint],
        descs_ref: np.ndarray
    ) -> List[CandidateMatch]:
        """
        Perform KNN matching (k=2), Lowe's ratio test, and optional mutual consistency.
        """
        if len(kps_src) == 0 or len(kps_ref) == 0 or descs_src.shape[0] == 0 or descs_ref.shape[0] == 0:
            return []

        # FLANN matcher for fast descriptor distance computation
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Forward match (src -> ref)
        raw_matches = matcher.knnMatch(descs_src, descs_ref, k=2)

        forward_good = []
        for m_tuple in raw_matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                ratio = m.distance / (n.distance + 1e-7)
                if ratio < self.ratio_threshold:
                    forward_good.append((m, ratio))
            elif len(m_tuple) == 1:
                forward_good.append((m_tuple[0], 0.70))

        # Backward match for mutual consistency check if enabled
        if self.mutual_check and forward_good:
            backward_matches = matcher.knnMatch(descs_ref, descs_src, k=1)
            backward_dict = {m[0].queryIdx: m[0].trainIdx for m in backward_matches if len(m) > 0}
            
            mutual_good = []
            for m, ratio in forward_good:
                src_idx = m.queryIdx
                ref_idx = m.trainIdx
                if backward_dict.get(ref_idx) == src_idx:
                    mutual_good.append((m, ratio))
            final_pairs = mutual_good
        else:
            final_pairs = forward_good

        # Package into CandidateMatch objects
        candidates = []
        for idx, (m, ratio) in enumerate(final_pairs):
            kp_s = kps_src[m.queryIdx]
            kp_r = kps_ref[m.trainIdx]

            candidates.append(
                CandidateMatch(
                    id=idx,
                    pt_src=(float(kp_s.pt[0]), float(kp_s.pt[1])),
                    pt_ref=(float(kp_r.pt[0]), float(kp_r.pt[1])),
                    scale_src=float(kp_s.size),
                    scale_ref=float(kp_r.size),
                    angle_src=float(kp_s.angle),
                    angle_ref=float(kp_r.angle),
                    descriptor_distance=float(m.distance),
                    ratio_score=float(ratio)
                )
            )

        return candidates
