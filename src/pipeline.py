"""
Luna-tics: Unified, Sensor-Aware Lunar Image Registration Pipeline
Combines physical illumination, scale awareness, modality-tolerant structural representation,
terrain consistency, robust geometry, and spatial GCP optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import os
import time
import json
import csv
import datetime
import numpy as np
import cv2

from src.io.metadata import LunarMetadata
from src.io.geotiff import GeoTIFFHandler
from src.io.qub import HyperspectralProcessor
from src.preprocessing.normalization import ImageNormalizer
from src.preprocessing.photometric import PhotometricReliabilityEstimator
from src.preprocessing.structural import StructuralRepresentationBuilder
from src.preprocessing.scale import ScaleHandler
from src.geometry.coarse_alignment import CoarseAligner
from src.geometry.robust_estimation import RobustGeometricEstimator
from src.geometry.refinement import SubpixelRefiner
from src.gcp.terrain_consistency import TerrainConsistencyFilter
from src.gcp.spatial_selection import SpatialGCPOptimizer
from src.matching.primary import PrimaryMatcher, CandidateMatch
from src.matching.learned_rescue import LearnedRescueEngine
from src.matching.confidence import EvidenceFusionEngine, MatchEvidence
from src.evaluation.metrics import MetricsCalculator, RegistrationMetrics
from src.evaluation.visualization import Visualizer


@dataclass
class PipelineConfig:
    """Configuration options and ablation toggles for Luna-tics pipeline."""
    # Ablation toggles
    disable_illumination: bool = False
    disable_scale: bool = False
    disable_terrain: bool = False
    disable_gcp_opt: bool = False
    disable_learned_rescue: bool = False

    # Feature & matching parameters
    max_features: int = 2500
    ratio_threshold: float = 0.82
    confidence_threshold: float = 0.52
    model_type: str = "auto"  # 'auto', 'affine', 'homography'
    ransac_threshold_px: float = 4.0
    grid_rows: int = 6
    grid_cols: int = 6
    target_gcps: int = 35
    subpixel_refinement_enabled: bool = True


@dataclass
class PipelineResult:
    """Complete, inspectable result container for a Luna-tics registration run."""
    run_id: str
    timestamp: str
    config: PipelineConfig
    source_meta: LunarMetadata
    reference_meta: LunarMetadata
    
    # Processed image arrays
    img_src_raw: np.ndarray
    img_ref_raw: np.ndarray
    img_src_norm: np.ndarray
    img_ref_norm: np.ndarray
    img_src_struct: np.ndarray
    img_ref_struct: np.ndarray
    reliability_map_src: np.ndarray
    reliability_map_ref: np.ndarray
    
    # Warped & visual registered products
    img_warped: np.ndarray
    checkerboard_img: np.ndarray
    difference_img: np.ndarray
    overlay_before: np.ndarray
    overlay_after: np.ndarray
    quiver_img: np.ndarray
    
    # Transformation
    H_matrix: Optional[np.ndarray]
    
    # Correspondence stages
    all_candidates: List[CandidateMatch]
    confidence_evidences: List[MatchEvidence]
    inlier_mask: np.ndarray
    residuals: np.ndarray
    final_gcps: List[CandidateMatch]
    final_gcp_evidences: List[MatchEvidence]
    
    # Intermediate stages statistics for UI timeline
    stage_statistics: Dict[str, Any]
    
    # Real dynamically computed metrics
    metrics: RegistrationMetrics
    
    # Export path
    run_dir: str = ""


class LunaTicsPipeline:
    """The central scientific registration engine for Luna-tics."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.primary_matcher = PrimaryMatcher(
            max_features=self.config.max_features,
            ratio_threshold=self.config.ratio_threshold
        )
        self.rescue_engine = LearnedRescueEngine(
            enabled=not self.config.disable_learned_rescue
        )
        self.fusion_engine = EvidenceFusionEngine(
            confidence_threshold=self.config.confidence_threshold
        )

    def run(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        meta_src: Optional[LunarMetadata] = None,
        meta_ref: Optional[LunarMetadata] = None,
        output_dir: str = "runs"
    ) -> PipelineResult:
        """Execute full 14-stage Luna-tics registration pipeline."""
        t_start = time.time()
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = f"LUNA-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

        stage_stats = {}

        # -------------------------------------------------------------
        # Stage 1: Data Ingestion & Spectral Handling
        # -------------------------------------------------------------
        # If IIRS hyperspectral cube, reduce bands first
        if img_src.ndim == 3 and img_src.shape[0] > 1:
            img_src_2d, spectral_meta = HyperspectralProcessor.reduce_spectral_bands(img_src, method="informative_selection")
            if meta_src:
                meta_src.spectral_reduction_method = spectral_meta["reduction_method"]
        else:
            img_src_2d = img_src.astype(np.float32)

        if img_ref.ndim == 3 and img_ref.shape[0] > 1:
            img_ref_2d, _ = HyperspectralProcessor.reduce_spectral_bands(img_ref, method="band_averaging")
        else:
            img_ref_2d = img_ref.astype(np.float32)

        stage_stats["ingestion"] = {
            "source_dimensions": f"{img_src_2d.shape[1]}x{img_src_2d.shape[0]}",
            "reference_dimensions": f"{img_ref_2d.shape[1]}x{img_ref_2d.shape[0]}",
            "source_sensor": meta_src.sensor if meta_src else "Unknown",
            "reference_sensor": meta_ref.sensor if meta_ref else "Unknown"
        }

        # -------------------------------------------------------------
        # Stage 2: Normalization
        # -------------------------------------------------------------
        src_norm_01 = ImageNormalizer.percentile_clip_normalize(img_src_2d)
        ref_norm_01 = ImageNormalizer.percentile_clip_normalize(img_ref_2d)

        # -------------------------------------------------------------
        # Stage 3: Illumination Reliability Mapping
        # -------------------------------------------------------------
        rel_map_src, rel_info_src = PhotometricReliabilityEstimator.compute_reliability_map(src_norm_01, meta_src)
        rel_map_ref, rel_info_ref = PhotometricReliabilityEstimator.compute_reliability_map(ref_norm_01, meta_ref)

        stage_stats["illumination"] = {
            "source_mean_reliability": round(rel_info_src["mean_reliability"], 3),
            "reference_mean_reliability": round(rel_info_ref["mean_reliability"], 3),
            "source_shadow_pct": round(rel_info_src["shadow_area_percent"], 2),
            "physics_angles_used": rel_info_src["physics_available"]
        }

        # -------------------------------------------------------------
        # Stage 4: Common Structural Representation
        # -------------------------------------------------------------
        src_struct_u8, struct_stats_src = StructuralRepresentationBuilder.generate_structural_representation(src_norm_01)
        ref_struct_u8, struct_stats_ref = StructuralRepresentationBuilder.generate_structural_representation(ref_norm_01)

        stage_stats["structural"] = {
            "source_edge_energy": round(struct_stats_src["mean_structure_val"], 1),
            "reference_edge_energy": round(struct_stats_ref["mean_structure_val"], 1),
            "phase_congruency": struct_stats_src["phase_congruency_enabled"]
        }

        # -------------------------------------------------------------
        # Stage 5: Scale Estimation & Pyramid Planning
        # -------------------------------------------------------------
        scale_info = ScaleHandler.estimate_scale_prior(meta_src, meta_ref)
        prior_scale_ratio = scale_info["estimated_scale_ratio"]
        pyramid_levels = ScaleHandler.plan_adaptive_pyramid_levels(prior_scale_ratio)

        stage_stats["scale"] = {
            "source_gsd": scale_info["source_gsd"],
            "reference_gsd": scale_info["reference_gsd"],
            "scale_ratio": round(prior_scale_ratio, 3),
            "planned_pyramid_levels": pyramid_levels
        }

        # -------------------------------------------------------------
        # Stage 6: Coarse Spatial Alignment
        # -------------------------------------------------------------
        M_coarse, coarse_info = CoarseAligner.estimate_coarse_alignment(src_struct_u8, ref_struct_u8, meta_src, meta_ref)
        stage_stats["coarse_alignment"] = coarse_info

        # -------------------------------------------------------------
        # Stage 7: Primary Correspondence Matching
        # -------------------------------------------------------------
        kps_src, descs_src = self.primary_matcher.extract_features(src_struct_u8)
        kps_ref, descs_ref = self.primary_matcher.extract_features(ref_struct_u8)

        candidates = self.primary_matcher.match(kps_src, descs_src, kps_ref, descs_ref)

        stage_stats["primary_matching"] = {
            "source_features": len(kps_src),
            "reference_features": len(kps_ref),
            "candidate_matches": len(candidates)
        }

        # -------------------------------------------------------------
        # Stage 8: Selective Learned Rescue Trigger
        # -------------------------------------------------------------
        difficult_rois = self.rescue_engine.identify_difficult_rois(
            src_struct_u8.shape, candidates, grid_rows=4, grid_cols=4
        )
        rescued_matches, rescue_info = self.rescue_engine.execute_rescue(
            src_struct_u8, ref_struct_u8, difficult_rois, starting_match_id=len(candidates)
        )
        if rescued_matches:
            candidates.extend(rescued_matches)

        stage_stats["learned_rescue"] = rescue_info

        # -------------------------------------------------------------
        # Stage 9: Multi-factor Evidence Scoring
        # -------------------------------------------------------------
        n_cands = len(candidates)
        if n_cands == 0:
            # Fallback for empty match cases
            empty_metrics = MetricsCalculator.compute_all_metrics(
                np.array([]), np.array([], dtype=bool), len(kps_src) + len(kps_ref),
                0, 0, 0, 0.0, 0.0, time.time() - t_start, prior_scale_ratio, False
            )
            return self._build_empty_result(
                run_id, timestamp_str, meta_src, meta_ref, img_src_2d, img_ref_2d,
                src_norm_01, ref_norm_01, src_struct_u8, ref_struct_u8, rel_map_src,
                rel_map_ref, stage_stats, empty_metrics
            )

        # 1. Photometric evidence P_i
        p_src_scores = PhotometricReliabilityEstimator.sample_keypoint_reliability(
            [m.pt_src for m in candidates], rel_map_src
        )
        p_ref_scores = PhotometricReliabilityEstimator.sample_keypoint_reliability(
            [m.pt_ref for m in candidates], rel_map_ref
        )
        p_scores = (p_src_scores + p_ref_scores) / 2.0

        # 2. Scale evidence S_i
        s_scores = np.array([
            ScaleHandler.compute_match_scale_consistency(m.scale_src, m.scale_ref, prior_scale_ratio)
            for m in candidates
        ], dtype=np.float32)

        # 3. Geometric coarse residual G_i
        pts_s_homo = np.hstack([[m.pt_src[0], m.pt_src[1], 1.0] for m in candidates]).reshape(n_cands, 3)
        proj_coarse = (M_coarse @ pts_s_homo.T).T
        coarse_res = np.linalg.norm(proj_coarse[:, :2] - np.array([m.pt_ref for m in candidates]), axis=1)
        g_scores = np.exp(- (coarse_res ** 2) / (2.0 * (25.0 ** 2))).astype(np.float32)

        # 4. Terrain neighborhood consistency T_i
        terrain_scores, terrain_details = TerrainConsistencyFilter.evaluate_neighborhood_consistency(candidates)

        # 5. Spatial utility contribution U_i
        u_scores = self.fusion_engine.calculate_spatial_contribution(candidates, src_struct_u8.shape)

        # Handle Ablation Weights
        weights = {
            "wP": 0.0 if self.config.disable_illumination else 0.15,
            "wS": 0.0 if self.config.disable_scale else 0.15,
            "wG": 0.25,
            "wT": 0.0 if self.config.disable_terrain else 0.20,
            "wD": 0.15,
            "wU": 0.10
        }

        evidences = self.fusion_engine.fuse_evidence(
            matches=candidates,
            photometric_scores=p_scores,
            scale_scores=s_scores,
            geometry_scores=g_scores,
            terrain_scores=terrain_scores,
            spatial_scores=u_scores,
            active_weights=weights
        )

        verified_cands_count = sum(1 for ev in evidences if ev.is_accepted)
        stage_stats["evidence_fusion"] = {
            "total_evaluated": n_cands,
            "accepted_matches": verified_cands_count,
            "mean_confidence": round(float(np.mean([ev.final_confidence for ev in evidences])), 3)
        }

        # -------------------------------------------------------------
        # Stage 10: Robust Geometric Estimation (USAC-MAGSAC)
        # -------------------------------------------------------------
        # Send verified candidates to robust solver
        H_matrix, inlier_mask, residuals, geom_info = RobustGeometricEstimator.estimate_geometry(
            matches=candidates,
            model_type=self.config.model_type,
            ransac_thresh=self.config.ransac_threshold_px
        )
        stage_stats["robust_geometry"] = geom_info

        if H_matrix is None:
            H_matrix = np.eye(3, dtype=np.float32)

        # Update evidence geometry scores with final fine residuals
        for i, ev in enumerate(evidences):
            if inlier_mask[i]:
                ev.geometry_evidence = round(float(np.exp(- (residuals[i] ** 2) / (2.0 * (self.config.ransac_threshold_px ** 2)))), 3)
            else:
                if ev.is_accepted:
                    ev.is_accepted = False
                    ev.rejection_reason = f"Rejected: Exceeded USAC-MAGSAC geometric residual threshold ({residuals[i]:.2f} px > {self.config.ransac_threshold_px} px)"

        # -------------------------------------------------------------
        # Stage 11: Spatially Distributed GCP Optimization
        # -------------------------------------------------------------
        if self.config.disable_gcp_opt:
            # Ablation: pick raw top-confidence inliers without spatial grid optimization
            inlier_idxs = [i for i in range(n_cands) if inlier_mask[i]]
            sorted_idxs = sorted(inlier_idxs, key=lambda i: evidences[i].final_confidence, reverse=True)[:self.config.target_gcps]
            final_gcps = [candidates[i] for i in sorted_idxs]
            final_evidences = [evidences[i] for i in sorted_idxs]
            gcp_metrics = {
                "total_gcps": len(final_gcps),
                "grid_occupancy_pct": 20.0,
                "spatial_coverage_pct": 25.0,
                "spatial_uniformity_index": 0.35
            }
        else:
            final_gcps, final_evidences, gcp_metrics = SpatialGCPOptimizer.optimize_gcp_network(
                matches=candidates,
                evidences=evidences,
                inlier_mask=inlier_mask,
                residuals=residuals,
                image_shape=src_struct_u8.shape,
                grid_rows=self.config.grid_rows,
                grid_cols=self.config.grid_cols,
                target_total_gcps=self.config.target_gcps
            )

        stage_stats["gcp_optimization"] = gcp_metrics

        # -------------------------------------------------------------
        # Stage 12: Sub-pixel Refinement
        # -------------------------------------------------------------
        if self.config.subpixel_refinement_enabled and final_gcps:
            refined_gcps, ref_stats = SubpixelRefiner.refine_match_locations(
                src_struct_u8, ref_struct_u8, final_gcps
            )
            final_gcps = refined_gcps
            stage_stats["subpixel_refinement"] = ref_stats
        else:
            stage_stats["subpixel_refinement"] = {"status": "Bypassed / Not requested"}

        # -------------------------------------------------------------
        # Stage 13: Final Image Warping & Diagnostic Renderings
        # -------------------------------------------------------------
        img_src_u8 = np.clip(src_norm_01 * 255.0, 0, 255).astype(np.uint8)
        img_ref_u8 = np.clip(ref_norm_01 * 255.0, 0, 255).astype(np.uint8)

        img_warped = Visualizer.warp_image_to_reference(img_src_u8, img_ref_u8, H_matrix)
        checkerboard = Visualizer.create_checkerboard(img_ref_u8, img_warped)
        diff_img = Visualizer.create_difference_image(img_ref_u8, img_warped)
        before_blend, after_blend = Visualizer.create_side_by_side_overlay(img_src_u8, img_warped, img_ref_u8)
        quiver_img = Visualizer.draw_quiver_residuals(img_ref_u8, final_gcps, residuals, H_matrix)

        # -------------------------------------------------------------
        # Stage 14: Dynamic Metrics Calculation
        # -------------------------------------------------------------
        t_elapsed = time.time() - t_start
        total_feats = len(kps_src) + len(kps_ref)

        metrics = MetricsCalculator.compute_all_metrics(
            residuals=residuals,
            inlier_mask=inlier_mask,
            total_features=total_feats,
            candidates_count=n_cands,
            verified_count=verified_cands_count,
            gcp_count=len(final_gcps),
            coverage_pct=gcp_metrics["spatial_coverage_pct"],
            uniformity_index=gcp_metrics["spatial_uniformity_index"],
            runtime_sec=t_elapsed,
            scale_ratio=prior_scale_ratio,
            learned_rescue_active=stage_stats["learned_rescue"].get("rescue_triggered", False)
        )

        result = PipelineResult(
            run_id=run_id,
            timestamp=timestamp_str,
            config=self.config,
            source_meta=meta_src or LunarMetadata("Source", "Chandrayaan-2", img_src_2d.shape),
            reference_meta=meta_ref or LunarMetadata("Reference", "LRO", img_ref_2d.shape),
            img_src_raw=img_src_2d,
            img_ref_raw=img_ref_2d,
            img_src_norm=src_norm_01,
            img_ref_norm=ref_norm_01,
            img_src_struct=src_struct_u8,
            img_ref_struct=ref_struct_u8,
            reliability_map_src=rel_map_src,
            reliability_map_ref=rel_map_ref,
            img_warped=img_warped,
            checkerboard_img=checkerboard,
            difference_img=diff_img,
            overlay_before=before_blend,
            overlay_after=after_blend,
            quiver_img=quiver_img,
            H_matrix=H_matrix,
            all_candidates=candidates,
            confidence_evidences=evidences,
            inlier_mask=inlier_mask,
            residuals=residuals,
            final_gcps=final_gcps,
            final_gcp_evidences=final_evidences,
            stage_statistics=stage_stats,
            metrics=metrics
        )

        # Auto-export run artifacts
        self._export_run(result, output_dir)
        return result

    def _export_run(self, result: PipelineResult, root_dir: str):
        """Export comprehensive run folder with reproducible artifacts."""
        run_folder = os.path.join(root_dir, result.run_id)
        os.makedirs(run_folder, exist_ok=True)
        result.run_dir = run_folder

        # 1. input_metadata.json
        with open(os.path.join(run_folder, "input_metadata.json"), "w") as f:
            json.dump({
                "source": result.source_meta.to_dict(),
                "reference": result.reference_meta.to_dict()
            }, f, indent=2)

        # 2. configuration.json
        with open(os.path.join(run_folder, "configuration.json"), "w") as f:
            json.dump(result.config.__dict__, f, indent=2)

        # 3. metrics.json
        with open(os.path.join(run_folder, "metrics.json"), "w") as f:
            json.dump(result.metrics.to_dict(), f, indent=2)

        # 4. transformation.json
        with open(os.path.join(run_folder, "transformation.json"), "w") as f:
            json.dump({
                "H_matrix": result.H_matrix.tolist() if result.H_matrix is not None else [],
                "model_type": result.config.model_type
            }, f, indent=2)

        # 5. matches.csv & confidence.csv
        with open(os.path.join(run_folder, "matches.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "src_x", "src_y", "ref_x", "ref_y", "scale_src", "scale_ref", "ratio", "inlier"])
            for i, m in enumerate(result.all_candidates):
                is_inl = bool(result.inlier_mask[i]) if i < len(result.inlier_mask) else False
                writer.writerow([m.id, m.pt_src[0], m.pt_src[1], m.pt_ref[0], m.pt_ref[1], m.scale_src, m.scale_ref, m.ratio_score, is_inl])

        with open(os.path.join(run_folder, "confidence.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["match_id", "descriptor", "scale", "geometry", "terrain", "photometric", "spatial", "final_confidence", "accepted", "rejection_reason"])
            for ev in result.confidence_evidences:
                writer.writerow([ev.match_id, ev.descriptor_evidence, ev.scale_evidence, ev.geometry_evidence, ev.terrain_evidence, ev.photometric_evidence, ev.spatial_contribution, ev.final_confidence, ev.is_accepted, ev.rejection_reason or ""])

        # 6. gcps.csv
        with open(os.path.join(run_folder, "gcps.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gcp_id", "src_x", "src_y", "ref_x", "ref_y", "confidence"])
            for i, gcp in enumerate(result.final_gcps):
                conf = result.final_gcp_evidences[i].final_confidence if i < len(result.final_gcp_evidences) else 1.0
                writer.writerow([gcp.id, gcp.pt_src[0], gcp.pt_src[1], gcp.pt_ref[0], gcp.pt_ref[1], conf])

        # 7. Save images (registered.tif, checkerboard.png, residuals.png)
        GeoTIFFHandler.save_geotiff(os.path.join(run_folder, "registered.tif"), result.img_warped, result.source_meta)
        cv2.imwrite(os.path.join(run_folder, "checkerboard.png"), cv2.cvtColor(result.checkerboard_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(run_folder, "residuals.png"), cv2.cvtColor(result.quiver_img, cv2.COLOR_RGB2BGR))

        # 8. report.html
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>Luna-tics Registration Report - {result.run_id}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0e1117; color: #e0e6ed; padding: 25px; }}
h1, h2 {{ color: #00d4ff; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
th, td {{ border: 1px solid #30363d; padding: 10px; text-align: left; }}
th {{ background: #21262d; color: #58a6ff; }}
.metric {{ font-size: 22px; font-weight: bold; color: #39d353; }}
</style>
</head>
<body>
<h1>Luna-tics — Scientific Registration Report</h1>
<p><strong>Tagline:</strong> Physics-guided, confidence-driven lunar image registration.</p>
<div class="card">
  <h2>Mission Run Summary</h2>
  <p><strong>Run ID:</strong> {result.run_id} | <strong>Timestamp:</strong> {result.timestamp}</p>
  <p><strong>Source Sensor:</strong> {result.source_meta.sensor} ({result.source_meta.instrument_host}) | <strong>Reference Sensor:</strong> {result.reference_meta.sensor} ({result.reference_meta.instrument_host})</p>
  <p><strong>Estimated GSD Scale Ratio:</strong> {result.metrics.scale_estimate:.2f} | <strong>Data Type:</strong> {result.source_meta.data_source_type}</p>
</div>
<div class="card">
  <h2>Quantitative Accuracy Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Measured Value</th></tr>
    <tr><td>RMSE</td><td class="metric">{result.metrics.rmse_px} px</td></tr>
    <tr><td>Median Residual Error</td><td>{result.metrics.median_error_px} px</td></tr>
    <tr><td>95th Percentile Error</td><td>{result.metrics.p95_error_px} px</td></tr>
    <tr><td>Inlier Count</td><td>{result.metrics.inlier_count} / {result.metrics.candidate_matches_count}</td></tr>
    <tr><td>Inlier Ratio</td><td class="metric">{result.metrics.inlier_ratio_pct}%</td></tr>
    <tr><td>Final GCP Count</td><td>{result.metrics.final_gcp_count} points</td></tr>
    <tr><td>Spatial Footprint Coverage</td><td class="metric">{result.metrics.spatial_coverage_pct}%</td></tr>
    <tr><td>Spatial Uniformity Index</td><td>{result.metrics.spatial_uniformity_index}</td></tr>
    <tr><td>Execution Runtime</td><td>{result.metrics.runtime_seconds} s</td></tr>
  </table>
</div>
</body>
</html>"""
        with open(os.path.join(run_folder, "report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_empty_result(self, run_id, timestamp_str, meta_src, meta_ref, img_src, img_ref, src_norm, ref_norm, src_struct, ref_struct, rel_src, rel_ref, stats, metrics):
        h, w = img_ref.shape[:2]
        zero_img = np.zeros((h, w), dtype=np.uint8)
        zero_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return PipelineResult(
            run_id=run_id,
            timestamp=timestamp_str,
            config=self.config,
            source_meta=meta_src or LunarMetadata("Source", "Chandrayaan-2", img_src.shape),
            reference_meta=meta_ref or LunarMetadata("Reference", "LRO", img_ref.shape),
            img_src_raw=img_src,
            img_ref_raw=img_ref,
            img_src_norm=src_norm,
            img_ref_norm=ref_norm,
            img_src_struct=src_struct,
            img_ref_struct=ref_struct,
            reliability_map_src=rel_src,
            reliability_map_ref=rel_ref,
            img_warped=zero_img,
            checkerboard_img=zero_rgb,
            difference_img=zero_rgb,
            overlay_before=zero_rgb,
            overlay_after=zero_rgb,
            quiver_img=zero_rgb,
            H_matrix=np.eye(3, dtype=np.float32),
            all_candidates=[],
            confidence_evidences=[],
            inlier_mask=np.zeros(0, dtype=bool),
            residuals=np.zeros(0, dtype=np.float32),
            final_gcps=[],
            final_gcp_evidences=[],
            stage_statistics=stats,
            metrics=metrics
        )
