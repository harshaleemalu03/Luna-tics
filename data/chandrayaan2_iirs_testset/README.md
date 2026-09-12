# Chandrayaan-2 IIRS Multi-Modal Testing Dataset
**Mission:** ISRO Chandrayaan-2  
**Instruments:** Imaging Infrared Spectrometer (IIRS) ↔ Terrain Mapping Camera-2 (TMC-2)  
**Problem Statement:** SIH 2026 — PS 26166 (*Multi-modal, Sun angle and scale invariant image correspondence*)

## Structure of Each Test Scene:
1. `ch2_iirs_calibrated_cube.tif` - 16-band calibrated hyperspectral floating-point cube.
2. `ch2_iirs_calibrated_cube.xml` - ISRO/PDS4 standard observational XML label.
3. `ch2_iirs_band04_continuum.tif` - Pre-extracted 2D NIR continuum band (for standard 2D matchers).
4. `ch2_tmc2_reference_mosaic.tif` - Paired optical TMC-2 reference mosaic.
5. `ground_truth_tiepoints.json` & `.csv` - Ground truth tie-points for computing Reprojection RMSE, MMA@3, MMA@5, and inlier accuracy.

## How to Test:
Run the included test runner:
```bash
python run_tests.py
```
