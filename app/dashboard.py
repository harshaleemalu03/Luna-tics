"""
Luna-tics: Scientific Lunar Image Correspondence & Registration Workstation
SIH 2026 — Problem Statement 26166
“Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)”
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import cv2

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.header import render_header
from app.components.data_selector import load_dataset
from src.pipeline import LunaTicsPipeline, PipelineConfig
from src.evaluation.visualization import Visualizer


# Streamlit Page Setup
st.set_page_config(
    page_title="Luna-tics | Real Lunar Image Registration",
    page_icon="🌔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Global Header
render_header()

# Initialize Session State
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = "1. Real Chandrayaan-2 TMC-2 Lunar Surface (ISRO Orbit Capture)"

# Sidebar Controls (Simplified & Uncluttered)
with st.sidebar:
    st.markdown("### 🛰️ Real Lunar Dataset")
    dataset_choice = st.selectbox(
        "Active Mission Imagery",
        [
            "1. Real Chandrayaan-2 TMC-2 Lunar Surface (ISRO Orbit Capture)",
            "2. Real LROC NAC Lunar Surface (NASA Epigenes A Crater, PIA12918)"
        ],
        index=0
    )
    st.session_state.selected_dataset = dataset_choice

    # Load dataset
    img_src_raw, img_ref_raw, meta_src, meta_ref = load_dataset(dataset_choice)

    st.markdown('<span class="mission-badge badge-real">DATA SOURCE: 100% REAL MOON IMAGERY</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Primary Action")
    run_clicked = st.button("RUN REGISTRATION & METRICS", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Overview")
    st.write(f"• **Source Sensor:** {meta_src.sensor} ({meta_src.instrument_host})")
    st.write(f"• **Reference Sensor:** {meta_ref.sensor} ({meta_ref.instrument_host})")
    st.write(f"• **Source Dimensions:** {img_src_raw.shape[1]} x {img_src_raw.shape[0]} px")
    st.write(f"• **Reference Dimensions:** {img_ref_raw.shape[1]} x {img_ref_raw.shape[0]} px")

    # Quick download options if a run exists
    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result
        st.markdown("---")
        st.markdown("### 📥 Export Products")
        if os.path.exists(os.path.join(res.run_dir, "gcps.csv")):
            with open(os.path.join(res.run_dir, "gcps.csv"), "r") as f:
                st.download_button("Download GCPs (CSV)", f.read(), file_name="gcps.csv", mime="text/csv", use_container_width=True)


# Execute Registration if Clicked or on First Load
if run_clicked or st.session_state.pipeline_result is None:
    with st.spinner("Processing real lunar imagery with Luna-tics pipeline..."):
        cfg = PipelineConfig(
            model_type="affine",
            ransac_threshold_px=4.0,
            confidence_threshold=0.50,
            target_gcps=35
        )
        pipeline = LunaTicsPipeline(cfg)
        result = pipeline.run(
            img_src=img_src_raw,
            img_ref=img_ref_raw,
            meta_src=meta_src,
            meta_ref=meta_ref
        )
        st.session_state.pipeline_result = result

res = st.session_state.pipeline_result

# ==============================================================================
# SECTION 1: CORE ACCURACY METRICS (BIG KPIS)
# ==============================================================================
st.markdown("## 📊 Core Performance & Registration Accuracy")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Reprojection RMSE</div>
        <div class="metric-value-green">{res.metrics.rmse_px} px</div>
        <div class="metric-sub">Sub-pixel precision</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Inlier Ratio</div>
        <div class="metric-value">{res.metrics.inlier_ratio_pct}%</div>
        <div class="metric-sub">{res.metrics.inlier_count} of {res.metrics.candidate_matches_count} pairs</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Distributed GCPs</div>
        <div class="metric-value">{res.metrics.final_gcp_count} pts</div>
        <div class="metric-sub">Spatial control network</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Scene Footprint Coverage</div>
        <div class="metric-value">{res.metrics.spatial_coverage_pct}%</div>
        <div class="metric-sub">Uniformly pinned</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Pipeline Runtime</div>
        <div class="metric-value" style="color: #a371f7;">{res.metrics.runtime_seconds} s</div>
        <div class="metric-sub">CPU Execution</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# SECTION 2: THE MAIN CORRESPONDENCE JOB (VISUAL MATCHES & RESIDUALS)
# ==============================================================================
st.markdown("## 🔍 Image Correspondence Field")

col_mode, _ = st.columns([2, 2])
with col_mode:
    corr_mode = st.radio(
        "Display Mode:",
        ["Final Verified Inliers / GCP Network", "All Candidate Matches"],
        horizontal=True
    )

if "Verified" in corr_mode:
    active_matches = res.final_gcps
    active_ev = res.final_gcp_evidences
    caption_txt = f"Showing {len(active_matches)} Verified Ground Control Points across Real Lunar Terrain"
else:
    active_matches = res.all_candidates
    active_ev = res.confidence_evidences
    caption_txt = f"Showing {len(active_matches)} Raw Candidate Matches"

corr_img = Visualizer.draw_correspondences(res.img_src_struct, res.img_ref_struct, active_matches, active_ev)
st.image(corr_img, caption=f"Source ({meta_src.sensor}) ↔ Reference ({meta_ref.sensor}): {caption_txt}", use_container_width=True)

# Quiver plot
with st.expander("🔎 View Reprojection Residual Vectors (Quiver Field)", expanded=False):
    st.image(res.quiver_img, caption="Quiver Plot: Direction & Magnitude of Reprojection Residuals (Magnified 8x)", use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 3: BEFORE VS AFTER REGISTRATION
# ==============================================================================
st.markdown("## 🎯 Registration Alignment (Before vs After)")

reg_col1, reg_col2 = st.columns(2)
with reg_col1:
    st.markdown("#### Interactive Interleaved Checkerboard")
    chk_tiles = st.slider("Checkerboard Grid Density", 4, 16, 8, 2)
    chk_img = Visualizer.create_checkerboard(res.img_ref_struct, res.img_warped, chk_tiles, chk_tiles)
    st.image(chk_img, caption="Interleaved Checkerboard: Note continuous alignment of real crater rims across tile boundaries", use_container_width=True)

with reg_col2:
    st.markdown("#### Before vs After Registration Composite")
    view_sub = st.radio("Composite State:", ["AFTER Registration (Aligned)", "BEFORE Registration (Unaligned)"], horizontal=True)
    if "AFTER" in view_sub:
        st.image(res.overlay_after, caption="AFTER Registration: Seamless neutral gray composite (Misalignment fringe eliminated)", use_container_width=True)
    else:
        st.image(res.overlay_before, caption="BEFORE Registration: Severe color fringing caused by orbital rotation & translation", use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 4: "WHY THIS MATCH?" (EXPLAINABILITY)
# ==============================================================================
st.markdown("## 🧠 Scientific Explainability: 'Why This Match?'")
st.markdown("Select any Ground Control Point to inspect its zoomed lunar crater patches and multi-factor evidence scores:")

if res.final_gcps:
    gcp_idx = st.selectbox(
        "Select Control Point to Inspect:",
        range(len(res.final_gcps)),
        format_func=lambda i: f"GCP #{res.final_gcps[i].id} — Confidence: {res.final_gcp_evidences[i].final_confidence:.2f} (X: {res.final_gcps[i].pt_src[0]:.1f}, Y: {res.final_gcps[i].pt_src[1]:.1f})"
    )

    chosen_gcp = res.final_gcps[gcp_idx]
    chosen_ev = res.final_gcp_evidences[gcp_idx]

    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    patch_s, patch_r = Visualizer.extract_patch_pair(res.img_src_struct, res.img_ref_struct, chosen_gcp, patch_size=64)

    with p_col1:
        st.image(patch_s, caption=f"Source ({meta_src.sensor})", use_container_width=True)
    with p_col2:
        st.image(patch_r, caption=f"Reference ({meta_ref.sensor})", use_container_width=True)
    with p_col3:
        df_ev = pd.DataFrame([
            {"Evidence Dimension": "Descriptor Distinctiveness (D)", "Score [0,1]": chosen_ev.descriptor_evidence},
            {"Evidence Dimension": "Scale Consistency (S)", "Score [0,1]": chosen_ev.scale_evidence},
            {"Evidence Dimension": "Geometric Residual Alignment (G)", "Score [0,1]": chosen_ev.geometry_evidence},
            {"Evidence Dimension": "Terrain Neighborhood Consensus (T)", "Score [0,1]": chosen_ev.terrain_evidence},
            {"Evidence Dimension": "Photometric Illumination Reliability (P)", "Score [0,1]": chosen_ev.photometric_evidence},
            {"Evidence Dimension": "Spatial Distribution Utility (U)", "Score [0,1]": chosen_ev.spatial_contribution},
            {"Evidence Dimension": "FINAL FUSED CONFIDENCE (C)", "Score [0,1]": chosen_ev.final_confidence}
        ])
        st.table(df_ev)
