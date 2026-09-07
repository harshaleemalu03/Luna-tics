"""
End-to-end integration and ablation tests for Luna-tics Pipeline
"""

import os
import tempfile
import numpy as np
import pytest
import tifffile

from src.io.pds4 import PDS4Parser
from src.pipeline import LunaTicsPipeline, PipelineConfig
from src.evaluation.benchmark import BenchmarkRunner


def test_pipeline_end_to_end_iirs():
    xml_path = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.xml"
    tif_path = "data/raw/chandrayaan2_iirs/ch2_iirs_calibrated_cube.tif"
    ref_path = "data/reference/lro_wac_mosaic_tile.tif"

    assert os.path.exists(xml_path)
    assert os.path.exists(tif_path)
    assert os.path.exists(ref_path)

    meta_src = PDS4Parser.parse_label(xml_path)
    img_src = tifffile.imread(tif_path)
    img_ref = tifffile.imread(ref_path)

    with tempfile.TemporaryDirectory() as tmp_runs:
        pipeline = LunaTicsPipeline()
        result = pipeline.run(
            img_src=img_src,
            img_ref=img_ref,
            meta_src=meta_src,
            meta_ref=None,
            output_dir=tmp_runs
        )

        assert result.metrics.status == "SUCCESS"
        assert result.metrics.inlier_count >= 10
        assert result.metrics.rmse_px < 4.0
        assert result.metrics.final_gcp_count >= 5
        assert result.H_matrix is not None
        assert result.H_matrix.shape in ((2, 3), (3, 3))

        # Check exported run directory contents
        assert os.path.exists(os.path.join(result.run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(result.run_dir, "gcps.csv"))
        assert os.path.exists(os.path.join(result.run_dir, "registered.tif"))
        assert os.path.exists(os.path.join(result.run_dir, "report.html"))


def test_ablation_modes():
    tif_path = "data/synthetic/synthetic_source_crater_grid.tif"
    ref_path = "data/synthetic/synthetic_reference_crater_grid.tif"

    img_src = tifffile.imread(tif_path)
    img_ref = tifffile.imread(ref_path)

    with tempfile.TemporaryDirectory() as tmp_runs:
        # 1. Full Luna-tics
        pipe_full = LunaTicsPipeline(PipelineConfig())
        res_full = pipe_full.run(img_src, img_ref, output_dir=tmp_runs)
        assert res_full.metrics.status == "SUCCESS"

        # 2. No Terrain Consistency
        pipe_no_terrain = LunaTicsPipeline(PipelineConfig(disable_terrain=True))
        res_no_terrain = pipe_no_terrain.run(img_src, img_ref, output_dir=tmp_runs)
        assert res_no_terrain.metrics.status == "SUCCESS"

        # 3. No Spatial GCP Optimization
        pipe_no_gcp = LunaTicsPipeline(PipelineConfig(disable_gcp_opt=True))
        res_no_gcp = pipe_no_gcp.run(img_src, img_ref, output_dir=tmp_runs)
        assert res_no_gcp.metrics.status == "SUCCESS"


def test_benchmark_runner():
    tif_path = "data/synthetic/synthetic_source_crater_grid.tif"
    ref_path = "data/synthetic/synthetic_reference_crater_grid.tif"

    img_src = tifffile.imread(tif_path)
    img_ref = tifffile.imread(ref_path)

    sift_res = BenchmarkRunner.run_sift_ransac_baseline(img_src, img_ref)
    assert sift_res["method"] == "SIFT + RANSAC"
    assert "rmse_px" in sift_res

    sp_res = BenchmarkRunner.run_superpoint_lightglue_baseline(img_src, img_ref)
    assert "status" in sp_res
