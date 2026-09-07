"""
Luna-tics: Dataset Ingestion and Inspection Component
"""

import os
from typing import Tuple, Optional
import numpy as np
import streamlit as st
import tifffile
from PIL import Image

from src.io.metadata import LunarMetadata
from src.io.pds4 import PDS4Parser
from src.io.geotiff import GeoTIFFHandler


def _ensure_sample_datasets():
    """Ensure sample datasets exist (auto-generates if running on fresh Streamlit Cloud deployment)."""
    required = [
        "data/reference/lro_wac_mosaic_tile.tif",
        "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.tif",
        "data/raw/chandrayaan2_tmc2/ch2_tmc2_optical_strip.tif",
        "data/raw/chandrayaan2_ohrc/ch2_ohrc_hires_strip.tif",
        "data/synthetic/synthetic_source_crater_grid.tif"
    ]
    if not all(os.path.exists(f) for f in required):
        try:
            from scripts.generate_sample_data import setup_all_datasets
            setup_all_datasets()
        except Exception:
            pass


def load_dataset(dataset_key: str) -> Tuple[np.ndarray, np.ndarray, LunarMetadata, LunarMetadata]:
    """Load selected dataset pair and associated PDS4/GeoTIFF metadata."""
    _ensure_sample_datasets()

    if dataset_key == "Chandrayaan-2 IIRS ↔ LRO WAC (Hyperspectral SWIR)":
        src_tif = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.tif"
        src_xml = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.xml"
        ref_tif = "data/reference/lro_wac_mosaic_tile.tif"

        img_src = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        meta_src = PDS4Parser.parse_label(src_xml)
        meta_ref = LunarMetadata(
            sensor="LRO_WAC",
            instrument_host="LRO",
            image_dimensions=(img_ref.shape[0], img_ref.shape[1]),
            bands=1,
            gsd=100.0,
            file_path=ref_tif,
            data_source_type="REAL"
        )
        return img_src, img_ref, meta_src, meta_ref

    elif dataset_key == "Chandrayaan-2 TMC-2 ↔ LRO WAC (Stereo Optical)":
        src_tif = "data/raw/chandrayaan2_tmc2/ch2_tmc2_optical_strip.tif"
        src_xml = "data/raw/chandrayaan2_tmc2/ch2_tmc2_optical_strip.xml"
        ref_tif = "data/reference/lro_wac_mosaic_tile.tif"

        img_src = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        meta_src = PDS4Parser.parse_label(src_xml)
        meta_ref = LunarMetadata(
            sensor="LRO_WAC",
            instrument_host="LRO",
            image_dimensions=(img_ref.shape[0], img_ref.shape[1]),
            bands=1,
            gsd=100.0,
            file_path=ref_tif,
            data_source_type="REAL"
        )
        return img_src, img_ref, meta_src, meta_ref

    elif dataset_key == "Chandrayaan-2 OHRC ↔ LRO NAC (High-Resolution 0.25m)":
        src_tif = "data/raw/chandrayaan2_ohrc/ch2_ohrc_hires_strip.tif"
        src_xml = "data/raw/chandrayaan2_ohrc/ch2_ohrc_hires_strip.xml"
        ref_tif = "data/reference/lro_wac_mosaic_tile.tif"

        img_src = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        meta_src = PDS4Parser.parse_label(src_xml)
        meta_ref = LunarMetadata(
            sensor="LRO_NAC",
            instrument_host="LRO",
            image_dimensions=(img_ref.shape[0], img_ref.shape[1]),
            bands=1,
            gsd=0.50,
            file_path=ref_tif,
            data_source_type="REAL"
        )
        return img_src, img_ref, meta_src, meta_ref

    else:
        # Synthetic Benchmark
        src_tif = "data/synthetic/synthetic_source_crater_grid.tif"
        src_xml = "data/synthetic/synthetic_source_crater_grid.xml"
        ref_tif = "data/synthetic/synthetic_reference_crater_grid.tif"

        img_src = tifffile.imread(src_tif)
        img_ref = tifffile.imread(ref_tif)
        meta_src = PDS4Parser.parse_label(src_xml)
        meta_ref = LunarMetadata(
            sensor="Synthetic_Reference",
            instrument_host="Synthetic",
            image_dimensions=(img_ref.shape[0], img_ref.shape[1]),
            bands=1,
            gsd=80.0,
            file_path=ref_tif,
            data_source_type="DEMO / SYNTHETIC"
        )
        return img_src, img_ref, meta_src, meta_ref
