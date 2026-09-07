"""
Luna-tics: Mission Header and Workstation CSS Styling
"""

import streamlit as st


def render_header():
    st.markdown("""
    <style>
    /* Dark ISRO / Planetary Science Workstation Theme */
    .stApp {
        background-color: #0b0e14;
        color: #c9d1d9;
    }
    .metric-card {
        background: #151b23;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-value-green {
        font-size: 24px;
        font-weight: 700;
        color: #3fb950;
    }
    .metric-sub {
        font-size: 11px;
        color: #7d8590;
        margin-top: 4px;
    }
    .mission-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background-color: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #1f6feb55;
    }
    .badge-real {
        background-color: #23863622;
        color: #3fb950;
        border: 1px solid #23863655;
    }
    .badge-synthetic {
        background-color: #d2992222;
        color: #d29922;
        border: 1px solid #d2992255;
    }
    .timeline-step {
        padding: 8px 12px;
        border-left: 2px solid #388bfd;
        background: #161b22;
        margin-bottom: 6px;
        border-radius: 0 4px 4px 0;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <h1 style="margin: 0; color: #f0f6fc; font-size: 28px; letter-spacing: -0.02em;">Luna-tics</h1>
            <span class="mission-badge">SIH 2026 • PS 26166</span>
        </div>
        <p style="margin: 0; color: #8b949e; font-size: 14px;">
            <strong>Physics-guided, confidence-driven lunar image registration.</strong>
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align: right; font-size: 12px; color: #8b949e; padding-top: 6px;">
            <div><strong>ISRO Chandrayaan-2</strong></div>
            <div style="color: #58a6ff;">IIRS • TMC-2 • OHRC ↔ LRO</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid #21262d; margin: 12px 0 18px 0;'>", unsafe_allow_html=True)
