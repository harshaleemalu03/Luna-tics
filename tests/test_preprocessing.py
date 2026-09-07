"""
Unit tests for Luna-tics Preprocessing module
(Normalization, Photometric Reliability, Structural Representation, Scale)
"""

import numpy as np
import pytest

from src.io.metadata import LunarMetadata
from src.preprocessing.normalization import ImageNormalizer
from src.preprocessing.photometric import PhotometricReliabilityEstimator
from src.preprocessing.structural import StructuralRepresentationBuilder
from src.preprocessing.scale import ScaleHandler


def test_percentile_clip_and_clahe():
    # Synthetic lunar patch with cosmic ray spikes
    img = np.ones((100, 100), dtype=np.float32) * 50.0
    img[20:30, 20:30] = 80.0
    img[0, 0] = 9999.0  # cosmic ray spike
    img[99, 99] = -500.0 # dead pixel

    norm = ImageNormalizer.percentile_clip_normalize(img, 1.0, 99.0)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

    clahe = ImageNormalizer.apply_clahe(norm)
    assert clahe.dtype == np.uint8
    assert clahe.shape == (100, 100)


def test_photometric_reliability():
    meta = LunarMetadata(
        sensor="IIRS",
        instrument_host="Chandrayaan-2",
        image_dimensions=(128, 128),
        incidence_angle_deg=50.0,
        emission_angle_deg=10.0,
        phase_angle_deg=55.0
    )

    # Patch with shadow on left, textured center, saturated right
    patch = np.zeros((128, 128), dtype=np.float32)
    patch[:, :30] = 0.02  # shadow
    patch[:, 30:90] = 0.5 + 0.1 * np.random.RandomState(42).randn(128, 60) # textured
    patch[:, 90:] = 0.98  # saturated

    rel_map, stats = PhotometricReliabilityEstimator.compute_reliability_map(patch, metadata=meta)
    assert rel_map.shape == (128, 128)
    assert 0.0 <= rel_map.min() <= rel_map.max() <= 1.0
    assert stats["physics_available"] is True

    # Textured region should have higher reliability than deep shadow
    mean_shadow_rel = np.mean(rel_map[:, :30])
    mean_textured_rel = np.mean(rel_map[:, 30:90])
    assert mean_textured_rel > mean_shadow_rel


def test_structural_representation():
    patch = np.zeros((100, 100), dtype=np.float32)
    # Circular crater rim
    y, x = np.ogrid[:100, :100]
    dist_from_center = np.sqrt((x - 50)**2 + (y - 50)**2)
    patch[(dist_from_center >= 25) & (dist_from_center <= 30)] = 0.9

    struct_u8, stats = StructuralRepresentationBuilder.generate_structural_representation(patch)
    assert struct_u8.dtype == np.uint8
    assert struct_u8.shape == (100, 100)
    assert stats["phase_congruency_enabled"] is True
    # Crater rim should have high intensity in structural representation
    rim_val = np.mean(struct_u8[(dist_from_center >= 25) & (dist_from_center <= 30)])
    center_val = np.mean(struct_u8[dist_from_center < 10])
    assert rim_val > center_val


def test_scale_handler():
    meta_src = LunarMetadata(
        sensor="TMC-2",
        instrument_host="Chandrayaan-2",
        image_dimensions=(500, 500),
        gsd=5.0
    )
    meta_ref = LunarMetadata(
        sensor="LRO_WAC",
        instrument_host="LRO",
        image_dimensions=(500, 500),
        gsd=100.0
    )

    scale_info = ScaleHandler.estimate_scale_prior(meta_src, meta_ref)
    assert scale_info["estimated_scale_ratio"] == pytest.approx(20.0, rel=1e-2)

    levels = ScaleHandler.plan_adaptive_pyramid_levels(scale_info["estimated_scale_ratio"])
    assert len(levels) > 0

    # Test scale consistency metric
    # Observed ratio matches prior (e.g. 20.0) -> high score ~ 1.0
    s_consistent = ScaleHandler.compute_match_scale_consistency(kp_src_size=1.0, kp_ref_size=20.0, prior_scale_ratio=20.0)
    # Observed ratio is wrong (e.g. 1.0 vs 20.0) -> low score
    s_inconsistent = ScaleHandler.compute_match_scale_consistency(kp_src_size=1.0, kp_ref_size=1.0, prior_scale_ratio=20.0)
    assert s_consistent > 0.95
    assert s_inconsistent < 0.05
