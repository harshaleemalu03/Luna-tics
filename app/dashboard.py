"""
Luna-tics: Scientific Planetary Remote Sensing Registration Workstation
SIH 2026 — PS 26166
“Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)”
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import matplotlib.pyplot as plt

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.header import render_header
from app.components.data_selector import load_dataset
from src.pipeline import LunaTicsPipeline, PipelineConfig, PipelineResult
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.visualization import Visualizer


# Streamlit Page Setup
st.set_page_config(
    page_title="Luna-tics | Lunar Image Registration",
    page_icon="🌔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Global Styled Header
render_header()

# Initialize Session State
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = "Chandrayaan-2 IIRS ↔ LRO WAC (Hyperspectral SWIR)"
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "ablation_results" not in st.session_state:
    st.session_state.ablation_results = None

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🛰️ Mission Parameters")
    
    dataset_choice = st.selectbox(
        "Select Lunar Data Pair",
        [
            "Chandrayaan-2 IIRS ↔ LRO WAC (Hyperspectral SWIR)",
            "Chandrayaan-2 TMC-2 ↔ LRO WAC (Stereo Optical)",
            "Chandrayaan-2 OHRC ↔ LRO NAC (High-Resolution 0.25m)",
            "Synthetic Calibration Crater Benchmark"
        ],
        index=0
    )
    st.session_state.selected_dataset = dataset_choice

    # Load metadata for display
    img_src_raw, img_ref_raw, meta_src, meta_ref = load_dataset(dataset_choice)

    # Data Source Badge
    if meta_src.data_source_type == "REAL":
        st.markdown('<span class="mission-badge badge-real">DATA SOURCE: REAL ISRO ARCHIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="mission-badge badge-synthetic">DATA SOURCE: DEMO / SYNTHETIC DATA</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Configuration")
    model_choice = st.selectbox("Transformation Model", ["auto", "affine", "homography"], index=0)
    ransac_thresh = st.slider("USAC-MAGSAC Threshold (px)", 1.0, 10.0, 4.0, 0.5)
    conf_thresh = st.slider("Evidence Fusion Threshold", 0.30, 0.85, 0.52, 0.02)
    max_gcps = st.slider("Target Spatially Distributed GCPs", 10, 60, 35, 5)

    st.markdown("---")
    # Primary Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_clicked = st.button("🚀 RUN LUNA-TICS", use_container_width=True, type="primary")
    with col_btn2:
        demo_clicked = st.button("⭐ JUDGE DEMO", use_container_width=True)

    st.markdown("---")
    nav_selection = st.radio(
        "Navigation",
        [
            "1. Mission Control",
            "2. Data & Sensors",
            "3. Live Processing Timeline",
            "4. Correspondence Analysis",
            "5. Why This Match? (Explainability)",
            "6. Registration Results",
            "7. Method Benchmark",
            "8. Ablation Study",
            "9. Export & Run History"
        ]
    )

# Execution Logic
if run_clicked or demo_clicked:
    with st.spinner("Executing Luna-tics sensor-aware registration pipeline..."):
        cfg = PipelineConfig(
            model_type=model_choice,
            ransac_threshold_px=ransac_thresh,
            confidence_threshold=conf_thresh,
            target_gcps=max_gcps
        )
        pipeline = LunaTicsPipeline(cfg)
        result = pipeline.run(
            img_src=img_src_raw,
            img_ref=img_ref_raw,
            meta_src=meta_src,
            meta_ref=meta_ref
        )
        st.session_state.pipeline_result = result
        st.toast(f"Luna-tics Run Completed: {result.run_id} (RMSE: {result.metrics.rmse_px} px)", icon="✅")


# TAB 1: MISSION CONTROL
if nav_selection == "1. Mission Control":
    st.markdown("## 🛰️ Mission Control Station")
    st.markdown("""
    **Luna-tics** is a unified, sensor-aware lunar image registration system that combines physical illumination information,
    scale awareness, modality-tolerant structural representation, terrain consistency, robust geometry, and spatial GCP optimization
    to produce reliable and explainable lunar image correspondences.
    """)

    # Core System KPI Status
    res = st.session_state.pipeline_result
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Dataset</div>
            <div class="metric-value" style="font-size: 18px;">{meta_src.sensor} ↔ {meta_ref.sensor}</div>
            <div class="metric-sub">{meta_src.data_source_type}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        val = f"{res.metrics.rmse_px} px" if res else "Awaiting Run"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Measured RMSE</div>
            <div class="metric-value-green">{val}</div>
            <div class="metric-sub">Reprojection Residual</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        val = f"{res.metrics.inlier_ratio_pct}%" if res else "Awaiting Run"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Inlier Ratio</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">USAC-MAGSAC Consensus</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        val = f"{res.metrics.final_gcp_count} pts" if res else "Awaiting Run"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Distributed GCPs</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">Spatial Network</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Problem Statement (SIH 2026 — PS 26166)")
    st.info("""
    **Challenge:** Lunar imagery captured across Chandrayaan-2 sensors (IIRS, TMC-2, OHRC) and lunar basemaps (LRO WAC/NAC)
    exhibits extreme disparities due to Sun azimuth/elevation angles, harsh shadow casting, disparate GSD scales,
    and deceptive repetitive crater morphologies.

    **Luna-tics Core Contribution:** Evidence-Guided Correspondence Fusion ($C_i = w_P P_i + w_S S_i + w_G G_i + w_T T_i + w_D D_i + w_U U_i$)
    which physically evaluates Photometric reliability, Scale consistency, Geometric validity, Crater neighborhood consensus,
    Descriptor distinctiveness, and Spatial coverage utility.
    """)

    if res is None:
        st.button("🚀 Click RUN LUNA-TICS to execute registration on the selected pair", on_click=lambda: None)


# TAB 2: DATA & SENSORS
elif nav_selection == "2. Data & Sensors":
    st.markdown("## 🔍 Sensor & Metadata Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Source: {meta_src.sensor} ({meta_src.instrument_host})")
        if img_src_raw.ndim == 3:
            st.image(np.clip(img_src_raw[0] / (img_src_raw[0].max() + 1e-5), 0, 1), caption=f"Source Band 0 ({meta_src.sensor})", use_container_width=True)
        else:
            st.image(np.clip(img_src_raw / (img_src_raw.max() + 1e-5), 0, 1), caption=f"Source ({meta_src.sensor})", use_container_width=True)

        st.markdown(f"""
        - **Dimensions:** {meta_src.image_dimensions[1]} x {meta_src.image_dimensions[0]} px (Bands: {meta_src.bands})
        - **Ground Sampling Distance (GSD):** {meta_src.gsd} m/px
        - **Solar Incidence Angle:** {meta_src.incidence_angle_deg}°
        - **Emission Angle:** {meta_src.emission_angle_deg}°
        - **Phase Angle:** {meta_src.phase_angle_deg}°
        - **Sun Azimuth / Elevation:** {meta_src.sun_azimuth_deg}° / {meta_src.sun_elevation_deg}°
        - **Data Label:** `{os.path.basename(meta_src.file_path)}`
        """)

    with col2:
        st.markdown(f"### Reference: {meta_ref.sensor} ({meta_ref.instrument_host})")
        st.image(np.clip(img_ref_raw / (img_ref_raw.max() + 1e-5), 0, 1), caption=f"Reference Basemap ({meta_ref.sensor})", use_container_width=True)

        st.markdown(f"""
        - **Dimensions:** {meta_ref.image_dimensions[1]} x {meta_ref.image_dimensions[0]} px
        - **Ground Sampling Distance (GSD):** {meta_ref.gsd} m/px
        - **Data Format:** GeoTIFF / Calibrated Lunar Raster
        - **Data Source Category:** `{meta_ref.data_source_type}`
        """)

    st.markdown("### 📐 Physical & Geometric Relationship Prior")
    scale_prior = meta_ref.get_effective_gsd(100.0) / max(0.01, meta_src.get_effective_gsd(80.0))
    st.write(f"• **Estimated GSD Scale Ratio:** `{scale_prior:.2f}` (Reference is {scale_prior:.2f}x coarser/finer than source)")
    st.write("• **Search Geometry Mode:** Footprint bounding-box coarse localization with multi-scale phase correlation fallback.")


# TAB 3: LIVE PROCESSING TIMELINE
elif nav_selection == "3. Live Processing Timeline":
    st.markdown("## ⏱️ Live Processing Pipeline Stages")
    res = st.session_state.pipeline_result
    if res is None:
        st.warning("No registration run executed yet. Click 'RUN LUNA-TICS' in the sidebar.")
    else:
        st.markdown(f"**Run ID:** `{res.run_id}` | Total Execution Time: **{res.metrics.runtime_seconds} seconds**")
        
        stages = [
            ("1. Data Ingestion & Spectral Handling", f"Loaded {res.source_meta.sensor} ({res.source_meta.bands} bands) & {res.reference_meta.sensor}. Spectral reduction: {res.source_meta.spectral_reduction_method or 'Direct'}"),
            ("2. Radiometric Percentile Normalization", "Cosmic ray outlier rejection (1%-99% clipping) + CLAHE local contrast"),
            ("3. Illumination Reliability Mapping", f"Source mean reliability: {res.stage_statistics['illumination']['source_mean_reliability']} | Physical angles utilized: {res.stage_statistics['illumination']['physics_angles_used']}"),
            ("4. Modality-Tolerant Structural Representation", f"Multi-scale Scharr gradient + Phase congruency edge energy. Mean edge energy: {res.stage_statistics['structural']['source_edge_energy']}"),
            ("5. GSD Scale Estimation & Pyramid Planning", f"Scale ratio: {res.stage_statistics['scale']['scale_ratio']} | Pyramid levels: {res.stage_statistics['scale']['planned_pyramid_levels']}"),
            ("6. Coarse Spatial Alignment", f"Mode: {res.stage_statistics['coarse_alignment']['alignment_mode']} | Est. translation: {res.stage_statistics['coarse_alignment']['estimated_translation']}"),
            ("7. Primary Correspondence Matching (SIFT)", f"Extracted {res.stage_statistics['primary_matching']['source_features']} src / {res.stage_statistics['primary_matching']['reference_features']} ref features. Candidates: {res.stage_statistics['primary_matching']['candidate_matches']}"),
            ("8. Selective Learned Rescue", f"Triggered: {res.stage_statistics['learned_rescue']['rescue_triggered']} | Status: {res.stage_statistics['learned_rescue']['status']}"),
            ("9. Evidence-Guided Confidence Fusion", f"Evaluated 6 factors. Verified matches: {res.stage_statistics['evidence_fusion']['accepted_matches']} / {res.stage_statistics['evidence_fusion']['total_evaluated']} (Mean conf: {res.stage_statistics['evidence_fusion']['mean_confidence']})"),
            ("10. Terrain / Neighborhood Consistency Verification", "Evaluated local k-NN crater topology graph. Spurious isolated crater matches pruned."),
            ("11. Robust USAC-MAGSAC Geometry", f"Solver: {res.stage_statistics['robust_geometry']['solver']} | Model: {res.stage_statistics['robust_geometry']['model_used']} | Inliers: {res.stage_statistics['robust_geometry']['inlier_count']} ({res.stage_statistics['robust_geometry']['inlier_ratio']*100:.1f}%) | RMSE: {res.stage_statistics['robust_geometry']['rmse_px']} px"),
            ("12. Spatially Distributed GCP Optimization", f"Partitioned {res.stage_statistics['gcp_optimization']['total_grid_cells']} cells. Selected {res.stage_statistics['gcp_optimization']['total_gcps']} distributed GCPs (Occupancy: {res.stage_statistics['gcp_optimization']['grid_occupancy_pct']}%, Coverage: {res.stage_statistics['gcp_optimization']['spatial_coverage_pct']}%)"),
            ("13. Sub-pixel NCC Local Refinement", f"Mean displacement shift: {res.stage_statistics['subpixel_refinement'].get('mean_subpixel_shift_px', 0.0)} px (Technique: Quadratic interpolation)"),
            ("14. Final Product Creation & Export", f"Warped product, checkerboard, quiver residuals, CSVs, and HTML report generated in runs/{res.run_id}/")
        ]

        for title, detail in stages:
            st.markdown(f"""
            <div class="timeline-step">
                <strong style="color: #58a6ff;">✓ {title}</strong><br>
                <span style="color: #8b949e;">{detail}</span>
            </div>
            """, unsafe_allow_html=True)


# TAB 4: CORRESPONDENCE ANALYSIS
elif nav_selection == "4. Correspondence Analysis":
    st.markdown("## 🔍 Multi-State Correspondence Analysis")
    res = st.session_state.pipeline_result
    if res is None:
        st.warning("Execute registration to inspect correspondence stages.")
    else:
        state_choice = st.radio(
            "Filter Correspondence Stage:",
            ["[1] All Candidate Matches", "[2] Confidence-Filtered Matches", "[3] USAC-MAGSAC Inliers", "[4] Final Spatially Distributed GCP Network"],
            horizontal=True
        )

        if "[1]" in state_choice:
            active_matches = res.all_candidates
            active_ev = res.confidence_evidences
            label = f"Showing All Initial Candidate Matches ({len(active_matches)} pairs)"
        elif "[2]" in state_choice:
            idxs = [i for i, ev in enumerate(res.confidence_evidences) if ev.is_accepted]
            active_matches = [res.all_candidates[i] for i in idxs]
            active_ev = [res.confidence_evidences[i] for i in idxs]
            label = f"Showing Confidence-Filtered Matches ({len(active_matches)} pairs)"
        elif "[3]" in state_choice:
            idxs = [i for i, inl in enumerate(res.inlier_mask) if inl]
            active_matches = [res.all_candidates[i] for i in idxs]
            active_ev = [res.confidence_evidences[i] for i in idxs]
            label = f"Showing Geometrically Verified Inliers ({len(active_matches)} pairs)"
        else:
            active_matches = res.final_gcps
            active_ev = res.final_gcp_evidences
            label = f"Showing Final Spatially Distributed GCP Network ({len(active_matches)} points)"

        st.markdown(f"**{label}**")
        corr_img = Visualizer.draw_correspondences(res.img_src_struct, res.img_ref_struct, active_matches, active_ev)
        st.image(corr_img, caption="Source (Left) ↔ Reference (Right) Correspondence Field", use_container_width=True)

        st.markdown("### Reprojection Residual Quiver Vector Field")
        st.image(res.quiver_img, caption="Vector Quiver Field: Direction & Magnitude of Residuals (Magnified 8x)", use_container_width=True)


# TAB 5: WHY THIS MATCH? (EXPLAINABILITY)
elif nav_selection == "5. Why This Match? (Explainability)":
    st.markdown("## 🧠 Scientific Explainability: 'Why This Match?'")
    res = st.session_state.pipeline_result
    if res is None:
        st.warning("Please run registration first.")
    else:
        st.markdown("""
        Inspect the explicit multi-evidence scoring breakdown for any individual Ground Control Point,
        or verify why spurious correspondences were systematically rejected.
        """)

        sub_tab1, sub_tab2 = st.tabs(["✓ Accepted GCP Inspector", "✗ Rejected Match Inspector"])

        with sub_tab1:
            if not res.final_gcps:
                st.info("No GCPs available.")
            else:
                gcp_idx = st.selectbox(
                    "Select Ground Control Point to Inspect:",
                    range(len(res.final_gcps)),
                    format_func=lambda i: f"GCP #{res.final_gcps[i].id} — Confidence: {res.final_gcp_evidences[i].final_confidence:.2f} (Src: {res.final_gcps[i].pt_src[0]:.1f}, {res.final_gcps[i].pt_src[1]:.1f})"
                )

                gcp_match = res.final_gcps[gcp_idx]
                gcp_ev = res.final_gcp_evidences[gcp_idx]

                c_patch1, c_patch2 = st.columns(2)
                patch_s, patch_r = Visualizer.extract_patch_pair(res.img_src_struct, res.img_ref_struct, gcp_match, patch_size=64)
                with c_patch1:
                    st.image(patch_s, caption=f"Source Patch Zoom ({res.source_meta.sensor})", use_container_width=True)
                with c_patch2:
                    st.image(patch_r, caption=f"Reference Patch Zoom ({res.reference_meta.sensor})", use_container_width=True)

                st.markdown("#### Evidence Breakdown Table")
                df_evidence = pd.DataFrame([
                    {"Evidence Factor": "Descriptor Distinctiveness (D)", "Score [0,1]": gcp_ev.descriptor_evidence, "Physical Interpretation": "Distinctiveness under SIFT Lowe's ratio"},
                    {"Evidence Factor": "Scale Consistency (S)", "Score [0,1]": gcp_ev.scale_evidence, "Physical Interpretation": "Octave size conforms to physical GSD ratio"},
                    {"Evidence Factor": "Geometric Consistency (G)", "Score [0,1]": gcp_ev.geometry_evidence, "Physical Interpretation": "Sub-pixel residual alignment under USAC-MAGSAC"},
                    {"Evidence Factor": "Terrain Neighborhood (T)", "Score [0,1]": gcp_ev.terrain_evidence, "Physical Interpretation": "Preserves relative distance & orientation with neighboring craters"},
                    {"Evidence Factor": "Photometric Reliability (P)", "Score [0,1]": gcp_ev.photometric_evidence, "Physical Interpretation": "Located outside deep cast shadows & saturated highlights"},
                    {"Evidence Factor": "Spatial Contribution (U)", "Score [0,1]": gcp_ev.spatial_contribution, "Physical Interpretation": "Prevents clustering; fills critical scene grid cell"},
                    {"Evidence Factor": "FINAL CONFIDENCE (C)", "Score [0,1]": gcp_ev.final_confidence, "Physical Interpretation": "Weighted Evidence-Guided Fusion Consensus"}
                ])
                st.table(df_evidence)

        with sub_tab2:
            rejected = [ev for ev in res.confidence_evidences if not ev.is_accepted]
            if not rejected:
                st.success("All candidate matches satisfied confidence criteria.")
            else:
                st.markdown(f"Found **{len(rejected)} rejected correspondences**. Inspect why they were rejected:")
                rej_idx = st.selectbox(
                    "Select Rejected Match:",
                    range(len(rejected)),
                    format_func=lambda i: f"Match #{rejected[i].match_id} — {rejected[i].rejection_reason}"
                )
                rej_ev = rejected[rej_idx]
                st.error(f"**Rejection Verdict:** {rej_ev.rejection_reason}")

                # Find candidate match
                c_m = next((m for m in res.all_candidates if m.id == rej_ev.match_id), None)
                if c_m:
                    p_s, p_r = Visualizer.extract_patch_pair(res.img_src_struct, res.img_ref_struct, c_m, patch_size=64)
                    cp1, cp2 = st.columns(2)
                    with cp1:
                        st.image(p_s, caption="Rejected Source Feature", use_container_width=True)
                    with cp2:
                        st.image(p_r, caption="Mismatched Reference Feature", use_container_width=True)


# TAB 6: REGISTRATION RESULTS
elif nav_selection == "6. Registration Results":
    st.markdown("## 🎯 Final Registration Output")
    res = st.session_state.pipeline_result
    if res is None:
        st.warning("Run registration to inspect alignment products.")
    else:
        st.markdown(f"### Performance Metrics Summary: Run `{res.run_id}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE Error", f"{res.metrics.rmse_px} px")
        m2.metric("Median Residual", f"{res.metrics.median_error_px} px")
        m3.metric("95th Percentile Error", f"{res.metrics.p95_error_px} px")
        m4.metric("Spatial Footprint Coverage", f"{res.metrics.spatial_coverage_pct}%")

        view_mode = st.radio(
            "Visualization Display Mode:",
            ["Checkerboard Interleave", "Before vs After Composite", "Absolute Difference Map"],
            horizontal=True
        )

        if view_mode == "Checkerboard Interleave":
            tiles = st.slider("Checkerboard Grid Tiles", 4, 16, 8, 2)
            chk = Visualizer.create_checkerboard(res.img_ref_struct, res.img_warped, tiles, tiles)
            st.image(chk, caption="Interleaved Checkerboard: Note continuous alignment of crater rims across cell borders", use_container_width=True)
        elif view_mode == "Before vs After Composite":
            c_b, c_a = st.columns(2)
            with c_b:
                st.image(res.overlay_before, caption="BEFORE: Unaligned Composite (Red=Source, Cyan=Reference)", use_container_width=True)
            with c_a:
                st.image(res.overlay_after, caption="AFTER: Registered Composite (Seamless neutral alignment)", use_container_width=True)
        else:
            st.image(res.difference_img, caption="Absolute Difference Map (Inferno colormap; dark indicates zero registration error)", use_container_width=True)


# TAB 7: METHOD BENCHMARK
elif nav_selection == "7. Method Benchmark":
    st.markdown("## 📊 Comparative Method Benchmark")
    st.markdown("""
    Evaluation of **Luna-tics** against standard classical and deep learning registration baselines.
    *Strict Scientific Honesty: Unavailable dependencies are truthfully reported without fabricated statistics.*
    """)

    if st.button("Run Multi-Method Benchmark on Current Dataset", type="primary"):
        with st.spinner("Benchmarking SIFT+RANSAC, SuperPoint+LightGlue, and Luna-tics..."):
            sift_bench = BenchmarkRunner.run_sift_ransac_baseline(img_src_raw, img_ref_raw)
            sp_bench = BenchmarkRunner.run_superpoint_lightglue_baseline(img_src_raw, img_ref_raw)
            
            # Luna-tics stats
            if st.session_state.pipeline_result:
                luna_res = st.session_state.pipeline_result
            else:
                pipe = LunaTicsPipeline()
                luna_res = pipe.run(img_src_raw, img_ref_raw, meta_src, meta_ref)
                st.session_state.pipeline_result = luna_res

            luna_bench = {
                "method": "Luna-tics (Ours)",
                "features": luna_res.metrics.total_features_detected,
                "matches": luna_res.metrics.candidate_matches_count,
                "inliers": luna_res.metrics.inlier_count,
                "inlier_ratio_pct": f"{luna_res.metrics.inlier_ratio_pct}%",
                "rmse_px": f"{luna_res.metrics.rmse_px} px",
                "coverage_pct": f"{luna_res.metrics.spatial_coverage_pct}%",
                "runtime_sec": f"{luna_res.metrics.runtime_seconds} s",
                "status": luna_res.metrics.status
            }

            st.session_state.benchmark_results = [sift_bench, sp_bench, luna_bench]

    if st.session_state.benchmark_results:
        df_bench = pd.DataFrame(st.session_state.benchmark_results)
        st.table(df_bench)

    st.markdown("""
    ### Why Luna-tics Outperforms Classical Baselines:
    1. **Physics-Aware:** Explicitly maps shadow boundaries and grazing-angle photometric instability.
    2. **Scale-Aware:** Employs physical GSD scale priors ($s_{prior} = \\text{GSD}_{ref}/\\text{GSD}_{src}$) to eliminate octave search ambiguity.
    3. **Terrain-Aware:** Rejects deceptive repetitive crater matches via graph neighborhood consistency.
    4. **Spatially-Aware:** Enforces cell-partitioned GCP distribution instead of dense clusters.
    """)


# TAB 8: ABLATION STUDY
elif nav_selection == "8. Ablation Study":
    st.markdown("## 🔬 Empirical Ablation Study")
    st.markdown("""
    Demonstrating the empirical value of each individual component of Luna-tics:
    Illumination Reliability ($P$), Scale Prior ($S$), Terrain Neighborhood ($T$), and Spatial GCP Optimization ($U$).
    """)

    if st.button("Run Full Ablation Suite on Active Dataset", type="primary"):
        with st.spinner("Computing 5 ablation variations..."):
            ablation_records = []
            configs = [
                ("Full Luna-tics", PipelineConfig()),
                ("w/o Illumination Reliability (P=0)", PipelineConfig(disable_illumination=True)),
                ("w/o Scale Consistency (S=0)", PipelineConfig(disable_scale=True)),
                ("w/o Terrain Neighborhood Consistency", PipelineConfig(disable_terrain=True)),
                ("w/o Spatial GCP Optimization", PipelineConfig(disable_gcp_opt=True)),
                ("w/o Learned Rescue", PipelineConfig(disable_learned_rescue=True))
            ]

            for name, cfg in configs:
                pipe = LunaTicsPipeline(cfg)
                res = pipe.run(img_src_raw, img_ref_raw, meta_src, meta_ref)
                ablation_records.append({
                    "Configuration": name,
                    "Inliers": res.metrics.inlier_count,
                    "Inlier Ratio (%)": f"{res.metrics.inlier_ratio_pct}%",
                    "RMSE (px)": f"{res.metrics.rmse_px} px",
                    "Spatial Coverage (%)": f"{res.metrics.spatial_coverage_pct}%",
                    "Uniformity Index": res.metrics.spatial_uniformity_index,
                    "Runtime (s)": f"{res.metrics.runtime_seconds} s"
                })

            st.session_state.ablation_results = ablation_records

    if st.session_state.ablation_results:
        st.table(pd.DataFrame(st.session_state.ablation_results))


# TAB 9: EXPORT & RUN HISTORY
elif nav_selection == "9. Export & Run History":
    st.markdown("## 💾 Export & Reproducible Run Artifacts")
    res = st.session_state.pipeline_result
    if res is None:
        st.warning("No run artifacts available yet. Execute registration first.")
    else:
        st.markdown(f"**Run Identifier:** `{res.run_id}` | Artifact Directory: `{res.run_dir}`")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            with open(os.path.join(res.run_dir, "gcps.csv"), "r") as f:
                st.download_button("📥 Download GCPs CSV", f.read(), file_name="gcps.csv", mime="text/csv")
        with c2:
            with open(os.path.join(res.run_dir, "confidence.csv"), "r") as f:
                st.download_button("📥 Download Confidence CSV", f.read(), file_name="confidence.csv", mime="text/csv")
        with c3:
            with open(os.path.join(res.run_dir, "report.html"), "r") as f:
                st.download_button("📥 Download HTML Report", f.read(), file_name="report.html", mime="text/html")

        st.markdown("### Exported Files in Run Directory")
        if os.path.exists(res.run_dir):
            files = os.listdir(res.run_dir)
            st.code("\n".join(files))
