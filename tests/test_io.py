"""
Unit tests for Luna-tics IO module (PDS4, QUB, GeoTIFF, Metadata)
"""

import os
import tempfile
import numpy as np
import pytest

from src.io.metadata import LunarMetadata
from src.io.pds4 import PDS4Parser, create_sample_pds4_label
from src.io.qub import HyperspectralProcessor
from src.io.geotiff import GeoTIFFHandler


def test_metadata_creation_and_fallback():
    meta = LunarMetadata(
        sensor="IIRS",
        instrument_host="Chandrayaan-2",
        image_dimensions=(512, 512),
        bands=256,
        gsd=None,  # missing
        incidence_angle_deg=42.5,
        emission_angle_deg=10.0,
        phase_angle_deg=45.0
    )
    assert meta.has_illumination_geometry() is True
    assert meta.has_spatial_coordinates() is False
    effective_gsd = meta.get_effective_gsd(fallback=80.0)
    assert effective_gsd == 80.0
    assert "gsd" in meta.fallback_flags


def test_pds4_label_generation_and_parsing():
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = os.path.join(tmpdir, "ch2_iirs_test.xml")
        create_sample_pds4_label(
            output_xml_path=xml_path,
            sensor="IIRS",
            instrument_host="Chandrayaan-2",
            lines=256,
            samples=256,
            bands=256,
            gsd=80.0,
            incidence_angle=55.2,
            emission_angle=8.4,
            phase_angle=60.1,
            sun_azimuth=135.0,
            sun_elevation=34.8,
            is_synthetic=False
        )

        parsed = PDS4Parser.parse_label(xml_path)
        assert parsed.sensor == "IIRS"
        assert parsed.instrument_host == "Chandrayaan-2"
        assert parsed.image_dimensions == (256, 256)
        assert parsed.bands == 256
        assert parsed.gsd == 80.0
        assert parsed.incidence_angle_deg == pytest.approx(55.2, rel=1e-2)
        assert parsed.emission_angle_deg == pytest.approx(8.4, rel=1e-2)
        assert parsed.data_source_type == "REAL"


def test_hyperspectral_qub_reduction():
    # Simulate a small 16-band hyperspectral cube (16, 64, 64)
    rng = np.random.RandomState(42)
    cube = rng.rand(16, 64, 64).astype(np.float32)

    # Test informative selection
    reduced_sel, meta_sel = HyperspectralProcessor.reduce_spectral_bands(cube, method="informative_selection")
    assert reduced_sel.shape == (64, 64)
    assert meta_sel["reduction_method"] == "informative_selection"

    # Test band averaging
    reduced_avg, meta_avg = HyperspectralProcessor.reduce_spectral_bands(cube, method="band_averaging")
    assert reduced_avg.shape == (64, 64)
    assert meta_avg["reduction_method"] == "band_averaging"

    # Test PCA
    reduced_pca, meta_pca = HyperspectralProcessor.reduce_spectral_bands(cube, method="pca")
    assert reduced_pca.shape == (64, 64)
    assert "pca" in meta_pca["reduction_method"]


def test_geotiff_read_write_window():
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = os.path.join(tmpdir, "test_lunar.tif")
        data = np.ones((200, 200), dtype=np.float32) * 128.0
        data[50:100, 50:100] = 220.0  # simulated crater

        GeoTIFFHandler.save_geotiff(tif_path, data)
        assert os.path.exists(tif_path)

        # Windowed read
        window = GeoTIFFHandler.read_window(tif_path, col_off=40, row_off=40, width=50, height=50)
        assert window.shape == (50, 50)
        assert window[15, 15] == pytest.approx(220.0, rel=1e-2)
