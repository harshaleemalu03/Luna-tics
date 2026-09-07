"""
Unit tests for Luna-tics Matching module
(Primary SIFT matcher, Learned Rescue detection, Evidence Fusion scoring)
"""

import numpy as np
import pytest
import cv2

from src.matching.primary import PrimaryMatcher, CandidateMatch
from src.matching.learned_rescue import LearnedRescueEngine
from src.matching.confidence import EvidenceFusionEngine, MatchEvidence


def test_primary_matcher():
    # Create two synthetic lunar image patches with known craters
    img1 = np.ones((200, 200), dtype=np.uint8) * 100
    cv2.circle(img1, (60, 60), 20, 220, -1)
    cv2.circle(img1, (140, 130), 25, 40, -1)
    cv2.circle(img1, (80, 150), 15, 180, -1)

    # Shift image by (10, 5) pixels
    M = np.float32([[1, 0, 10], [0, 1, 5]])
    img2 = cv2.warpAffine(img1, M, (200, 200))

    matcher = PrimaryMatcher(max_features=500, ratio_threshold=0.85, mutual_check=True)
    kps1, descs1 = matcher.extract_features(img1)
    kps2, descs2 = matcher.extract_features(img2)

    assert len(kps1) > 0
    assert len(kps2) > 0

    matches = matcher.match(kps1, descs1, kps2, descs2)
    assert len(matches) > 0
    # Inspect first match
    m0 = matches[0]
    assert isinstance(m0, CandidateMatch)
    assert m0.ratio_score <= 0.85


def test_learned_rescue_safe_fallback():
    rescue_engine = LearnedRescueEngine(enabled=True)
    # Backend status should be clear and truthful
    assert isinstance(rescue_engine.backend_name, str)

    # Test difficult ROI detection
    matches = [
        CandidateMatch(0, (10, 10), (15, 15), 5.0, 5.0, 0.0, 0.0, 10.0, 0.5),
        CandidateMatch(1, (20, 20), (25, 25), 5.0, 5.0, 0.0, 0.0, 12.0, 0.6)
    ]
    difficult_rois = rescue_engine.identify_difficult_rois(
        image_shape=(200, 200),
        existing_matches=matches,
        grid_rows=2,
        grid_cols=2,
        min_matches_per_cell=1
    )
    # The top-left cell has 2 matches; the other 3 cells should be marked as difficult
    assert len(difficult_rois) == 3

    # Test safe execution
    rescued, info = rescue_engine.execute_rescue(
        np.zeros((200, 200), dtype=np.uint8),
        np.zeros((200, 200), dtype=np.uint8),
        difficult_rois
    )
    assert isinstance(rescued, list)
    assert "status" in info


def test_evidence_fusion_engine():
    fusion = EvidenceFusionEngine(confidence_threshold=0.60)
    matches = [
        CandidateMatch(0, (50, 50), (60, 55), 10.0, 10.0, 0.0, 0.0, 15.0, 0.4),  # good match
        CandidateMatch(1, (100, 100), (190, 20), 10.0, 10.0, 0.0, 0.0, 45.0, 0.85) # bad match
    ]

    p_scores = np.array([0.9, 0.3], dtype=np.float32)
    s_scores = np.array([0.95, 0.2], dtype=np.float32)
    g_scores = np.array([0.92, 0.1], dtype=np.float32)
    t_scores = np.array([0.88, 0.15], dtype=np.float32)
    u_scores = np.array([0.80, 0.5], dtype=np.float32)

    evidences = fusion.fuse_evidence(
        matches=matches,
        photometric_scores=p_scores,
        scale_scores=s_scores,
        geometry_scores=g_scores,
        terrain_scores=t_scores,
        spatial_scores=u_scores
    )

    assert len(evidences) == 2
    ev0, ev1 = evidences[0], evidences[1]

    # Good match should have high confidence and be accepted
    assert ev0.final_confidence > 0.75
    assert ev0.is_accepted is True
    assert ev0.rejection_reason is None

    # Bad match should have low confidence and be rejected with a reason
    assert ev1.final_confidence < 0.50
    assert ev1.is_accepted is False
    assert ev1.rejection_reason is not None
