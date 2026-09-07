"""
Unit tests for Luna-tics GCP module
(Terrain Consistency Filter, Spatially Distributed GCP Optimizer)
"""

import numpy as np
import pytest

from src.matching.primary import CandidateMatch
from src.matching.confidence import MatchEvidence
from src.gcp.terrain_consistency import TerrainConsistencyFilter
from src.gcp.spatial_selection import SpatialGCPOptimizer


def test_terrain_consistency_filter():
    matches = []
    # 10 points on a grid with consistent translation (dx=10, dy=5)
    for i, (x, y) in enumerate([(20, 20), (40, 20), (60, 20), (20, 50), (40, 50), (60, 50), (20, 80), (40, 80), (60, 80), (80, 80)]):
        matches.append(CandidateMatch(i, (float(x), float(y)), (float(x + 10), float(y + 5)), 5.0, 5.0, 0.0, 0.0, 10.0, 0.4))

    # Add 1 deceptive false match to a similar crater far away
    matches.append(CandidateMatch(10, (50.0, 50.0), (190.0, 190.0), 5.0, 5.0, 0.0, 0.0, 10.0, 0.4))

    scores, details = TerrainConsistencyFilter.evaluate_neighborhood_consistency(matches, k_neighbors=4)

    # First 10 should have high consensus
    assert np.mean(scores[:10]) > 0.80
    # False match should have low consensus
    assert scores[10] < 0.40
    assert details[10]["is_terrain_consistent"] is False


def test_spatial_gcp_optimizer():
    matches = []
    evidences = []
    # Create 30 matches clustered in top-left cell, plus 10 spread across other cells
    idx = 0
    # Clustered in (10..40, 10..40)
    for _ in range(30):
        m = CandidateMatch(idx, (25.0 + np.random.randn(), 25.0 + np.random.randn()), (35.0, 35.0), 5.0, 5.0, 0.0, 0.0, 10.0, 0.4)
        ev = MatchEvidence(idx, m.pt_src, m.pt_ref, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.85, is_accepted=True)
        matches.append(m)
        evidences.append(ev)
        idx += 1

    # Distributed points across other parts of (200, 200) image
    for (x, y) in [(150, 30), (180, 50), (40, 160), (120, 140), (170, 170)]:
        m = CandidateMatch(idx, (float(x), float(y)), (float(x + 10), float(y + 10)), 5.0, 5.0, 0.0, 0.0, 10.0, 0.4)
        ev = MatchEvidence(idx, m.pt_src, m.pt_ref, 0.85, 0.85, 0.85, 0.85, 0.85, 0.9, 0.83, is_accepted=True)
        matches.append(m)
        evidences.append(ev)
        idx += 1

    inliers = np.ones(len(matches), dtype=bool)
    res = np.ones(len(matches), dtype=np.float32) * 0.5

    sel_matches, sel_ev, metrics = SpatialGCPOptimizer.optimize_gcp_network(
        matches=matches,
        evidences=evidences,
        inlier_mask=inliers,
        residuals=res,
        image_shape=(200, 200),
        grid_rows=4,
        grid_cols=4,
        max_points_per_cell=2,
        target_total_gcps=20
    )

    # The top-left cell should be capped at max_points_per_cell (2), preventing clustering!
    tl_points = [m for m in sel_matches if m.pt_src[0] < 50 and m.pt_src[1] < 50]
    assert len(tl_points) <= 2
    assert metrics["grid_occupancy_pct"] > 0
    assert metrics["spatial_coverage_pct"] > 0
