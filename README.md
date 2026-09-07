# Luna-tics

### Physics-guided, confidence-driven lunar image registration.
**Smart India Hackathon 2026 — Problem Statement 26166**  
*“Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)”*

---

## 🌟 Core Product Statement

> **Luna-tics** is a unified, sensor-aware lunar image registration system that combines physical illumination information, scale awareness, modality-tolerant structural representation, terrain consistency, robust geometry and spatial GCP optimization to produce reliable and explainable lunar image correspondences.

---

## 🌒 The Challenge of Lunar Optical Registration

Orbital imagery of the Moon captured by ISRO Chandrayaan-2 instruments (**IIRS**, **TMC-2**, **OHRC**) and lunar reference basemaps (**LRO WAC**, **LRO NAC**) presents extreme registration challenges that cause classical feature detectors (SIFT, ORB) and naive deep learning models to fail:

1. **Extreme Solar Illumination Disparities:** The Moon lacks an atmosphere, creating pitch-black cast shadows, blinding specular rims, and sharp contrast reversals as the Sun's azimuth and elevation angles change across orbits.
2. **Extreme Scale Differences:** Ground Sampling Distances (GSD) span three orders of magnitude:
   - **OHRC:** ~0.25 m/pixel (Ultra high-resolution landed hazard assessment)
   - **TMC-2:** ~5.0 m/pixel (Stereo surface topography)
   - **IIRS:** ~80.0 m/pixel (256-band hyperspectral mineralogy)
   - **LRO WAC:** ~100.0 m/pixel (Global monochrome context)
   - **LRO NAC:** ~0.50 m/pixel (Targeted narrow-angle optical)
3. **Repetitive Crater Morphologies:** The lunar regolith is dominated by self-similar circular impact craters and rim structures. A descriptor can easily match a crater rim to an identical crater kilometers away in the wrong location.
4. **Hyperspectral Dimensionality:** IIRS cubes span 256 contiguous spectral bands from 800 nm to 5000 nm, which cannot be blindly processed like RGB images.

---

## 🚀 Central Technical Innovation: Evidence-Guided Correspondence Fusion

Instead of trusting a single local descriptor distance, **Luna-tics** introduces an **Evidence-Guided Correspondence Fusion** decision framework. Every candidate correspondence is evaluated across six independent physical, geometric, and spatial evidence factors:

$$C_i = \frac{w_P \cdot P_i + w_S \cdot S_i + w_G \cdot G_i + w_T \cdot T_i + w_D \cdot D_i + w_U \cdot U_i}{w_P + w_S + w_G + w_T + w_D + w_U}$$

Where $C_i \in [0, 1]$ answers:  
*“Is this correspondence physically plausible, scale-consistent, structurally valid, geometrically verified, and useful for the final control-point network?”*

| Factor | Evidence Dimension | Mathematical & Physical Formulation |
|---|---|---|
| **$P_i$** | **Photometric Reliability** | Lommel-Seeliger scattering law $\frac{\cos(i)}{\cos(i)+\cos(e)}$ modulated by continuous shadow and saturation penalty maps. |
| **$S_i$** | **Scale Consistency** | Octave ratio conformity with physical GSD scale prior: $\exp\left(-\frac{(\ln(s_{obs}) - \ln(s_{prior}))^2}{2\sigma_s^2}\right)$. |
| **$G_i$** | **Geometric Consistency** | Reprojection residual alignment under USAC-MAGSAC transformation. |
| **$T_i$** | **Terrain Neighborhood** | Topological $k$-NN crater consensus: verifies relative distance and angular relationships with surrounding landmarks. |
| **$D_i$** | **Descriptor Distinctiveness** | Lowe ratio test distinctiveness: $1.0 - r_i^2$. |
| **$U_i$** | **Spatial Utility Contribution** | Cell scarcity bonus: rewards correspondences that provide coverage to sparsely populated regions of the scene. |

---

## 🏗️ Complete System Architecture

```
INPUT: Chandrayaan-2 (IIRS / TMC-2 / OHRC) + Lunar Reference (LRO WAC / NAC)
   │
   ├── [Stage 1] Data Ingestion & PDS4 / GeoTIFF Extraction
   │   ├── Parse ISRO PDS4 XML labels (Sun azimuth, elevation, incidence, emission, phase, GSD)
   │   └── Spectral Reduction for IIRS (Informative band selection / PCA / Averaging)
   │
   ├── [Stage 2] Radiometric Normalization
   │   └── 1%-99% percentile clipping (cosmic ray rejection) + CLAHE contrast enhancement
   │
   ├── [Stage 3] Illumination Reliability Mapping
   │   └── Dense [0, 1] reliability map penalizing cast shadows and specular saturation
   │
   ├── [Stage 4] Common Structural Representation
   │   └── Multi-scale directional Scharr gradients + Phase Congruency edge moments
   │
   ├── [Stage 5] Sensor-Aware Scale Estimation & Pyramid Planning
   │   └── Theoretical ratio s_prior = GSD_ref / GSD_src; focused octave level planning
   │
   ├── [Stage 6] Coarse Spatial Alignment
   │   └── Geographic footprint overlap bounding-box search / Fourier phase correlation
   │
   ├── [Stage 7] Primary Correspondence Matching
   │   └── Structural SIFT extraction with bidirectional mutual consistency
   │
   ├── [Stage 8] Selective Learned Rescue
   │   └── Monitors sparse ROIs; selectively triggers SuperPoint + LightGlue
   │
   ├── [Stage 9] Evidence-Guided Confidence Fusion
   │   └── Fuses P, S, G, T, D, U into explainable confidence scores C_i
   │
   ├── [Stage 10] Terrain / Neighborhood Consistency Filter
   │   └── Local graph topology check: rejects deceptive repetitive crater matches
   │
   ├── [Stage 11] Robust Geometry Verification
   │   └── USAC-MAGSAC estimator (Affine vs Homography) with residual tracking
   │
   ├── [Stage 12] Spatially Distributed GCP Optimization
   │   └── Grid-cell partition optimizer: prevents clustering, maximizes scene coverage
   │
   ├── [Stage 13] Local Sub-Pixel Refinement
   │   └── Normalized Cross-Correlation (NCC) with 2D quadratic peak interpolation
   │
   └── [Stage 14] Registration Output & Export
       └── Warped GeoTIFF, Checkerboard, Residual quiver field, CSVs, and HTML Report
```

---

## 📦 Project Structure

```
luna-tics/
├── app/
│   ├── dashboard.py                  # Main Streamlit workstation dashboard
│   └── components/
│       ├── header.py                 # Mission header & dark workstation CSS
│       └── data_selector.py          # Dataset ingestion & metadata loaders
├── src/
│   ├── io/
│   │   ├── metadata.py               # Standardized LunarMetadata dataclass
│   │   ├── pds4.py                   # PDS4 XML label parser & generator
│   │   ├── qub.py                    # IIRS hyperspectral cube reduction
│   │   └── geotiff.py                # Large-data windowed & tiled GeoTIFF I/O
│   ├── preprocessing/
│   │   ├── normalization.py          # Dynamic range percentile clipping & CLAHE
│   │   ├── photometric.py            # Lommel-Seeliger & image illumination maps
│   │   ├── structural.py             # Phase congruency & multi-scale gradients
│   │   └── scale.py                  # GSD scale prior & adaptive pyramid planner
│   ├── matching/
│   │   ├── primary.py                # Modality-tolerant SIFT matching engine
│   │   ├── learned_rescue.py         # Selective recovery with graceful fallback
│   │   └── confidence.py             # 6-factor Evidence-Guided Fusion engine
│   ├── geometry/
│   │   ├── coarse_alignment.py       # Footprint/phase correlation coarse search
│   │   ├── robust_estimation.py      # USAC-MAGSAC affine/homography solver
│   │   └── refinement.py             # Sub-pixel NCC quadratic interpolation
│   ├── gcp/
│   │   ├── terrain_consistency.py    # Crater graph & relative distance filter
│   │   └── spatial_selection.py      # Spatial cell GCP distribution optimizer
│   ├── evaluation/
│   │   ├── metrics.py                # Dynamic RMSE, inlier ratio, uniformity
│   │   ├── benchmark.py              # Comparative multi-method baseline runner
│   │   └── visualization.py          # Checkerboard, quiver plots, patch zooms
│   └── pipeline.py                   # Unified LunaTicsPipeline orchestration engine
├── data/
│   ├── raw/
│   │   ├── chandrayaan2_iirs/        # IIRS PDS4 XML + calibrated hyperspectral cube
│   │   ├── chandrayaan2_tmc2/        # TMC-2 optical strip + PDS4 XML
│   │   └── chandrayaan2_ohrc/        # OHRC high-res sample + PDS4 XML
│   ├── reference/                    # LRO WAC reference mosaic tile
│   └── synthetic/                    # Calibrated synthetic crater test benchmark
├── runs/                             # Reproducible run folders (LUNA-YYYY-MM-DD-XXX)
├── scripts/                          # Dataset generator & maintenance utilities
├── tests/                            # 19 automated pytest unit & integration tests
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Windows, Linux, or macOS

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/team-luna-tics/Luna-tics.git
cd Luna-tics

# Install required dependencies
pip install -r requirements.txt

# Generate calibrated lunar dataset pairs (IIRS, TMC-2, OHRC, and LRO WAC)
python scripts/generate_sample_data.py

# Verify system with automated test suite (19 tests)
pytest -v
```

---

## 💻 Running the Scientific Workstation

Launch the dark-mode mission dashboard:
```bash
streamlit run app/dashboard.py
```

The workstation will launch at `http://localhost:8501`.

---

## ⏱️ Judge Demo Mode (2–3 Minute Presentation Guide)

For SIH 2026 evaluators, Luna-tics includes a streamlined one-click **Judge Demo**:

1. **Step 1:** Select **Chandrayaan-2 IIRS ↔ LRO WAC (Hyperspectral SWIR)** from the sidebar.
2. **Step 2:** Click **⭐ JUDGE DEMO** (or **🚀 RUN LUNA-TICS**).
3. **Step 3:** Open **3. Live Processing Timeline** to show all 14 stages executing with real statistics.
4. **Step 4:** Open **4. Correspondence Analysis** and toggle between:
   - `[1] All Candidates` $\to$ Raw unverified matches.
   - `[4] Final GCP Network` $\to$ Clean, spatially distributed control points.
5. **Step 5:** Open **5. Why This Match?** and select any GCP:
   - Point out the zoomed source and reference patches.
   - Show the dynamic breakdown of all 6 evidence factors ($P, S, G, T, D, U$).
   - Switch to the *Rejected Match Inspector* to show explainable rejection reasons.
6. **Step 6:** Open **6. Registration Results** to demonstrate:
   - The seamless alignment of crater rims across the **Checkerboard Interleave** slider.
   - The Before vs After color overlay.
   - The dynamically computed accuracy metrics: **RMSE < 1.0 px**, **Inlier Ratio > 80%**, **Coverage > 40%**.
7. **Step 7:** Open **9. Export & Run History** to download the official GCP CSV, registered GeoTIFF, and complete HTML report.

---

## 🔬 Scientific Honesty & Integrity Statement

In strict adherence to planetary remote sensing standards:
- **No Fabricated Numbers:** Luna-tics never hardcodes accuracy claims or fake benchmark percentages. All metrics (RMSE, inlier ratio, coverage, uniformity, runtime) are calculated dynamically from actual array math.
- **Explicit Data Categorization:** Real Chandrayaan-2 PDS4 products and synthetic calibration benches are visibly distinguished in the interface using color-coded badges (`REAL ISRO ARCHIVE` vs `DEMO / SYNTHETIC DATA`).
- **Graceful Dependency Handling:** If optional neural weights (SuperPoint / LightGlue) are absent, Luna-tics reports `Unavailable — dependency not installed` rather than inventing mock numbers.

---

## 👥 Team Identity

- **Team Name:** Luna-tics
- **Solution Name:** Luna-tics
- **Tagline:** Physics-guided, confidence-driven lunar image registration.
- **Hackathon:** Smart India Hackathon 2026 (PS 26166)
