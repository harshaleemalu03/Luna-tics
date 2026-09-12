# ISRO Chandrayaan-2 IIRS Multi-Modal Test Dataset
**Sensor:** Imaging Infrared Spectrometer (IIRS)  
**Host:** Chandrayaan-2 Orbiter  
**Mission Archive:** ISRO ISSDC / PDS Format

Each observation folder contains:
1. `pds_label.txt` - Official ASCII PDS text label detailing:
   - INCIDENCE_ANGLE (<DEG>)
   - EMISSION_ANGLE (<DEG>)
   - PHASE_ANGLE (<DEG>)
   - SOLAR_AZIMUTH_ANGLE (<DEG>)
   - SOLAR_ELEVATION_ANGLE (<DEG>)
   - SPACECRAFT_ALTITUDE (<KM>)
   - GROUND_SAMPLING_DISTANCE (<M>)
   - Geographic coordinates & spectral bounds
2. `iirs_hyperspectral_cube.tif` - 16-band calibrated float32 hyperspectral cube.
3. `iirs_band01_nir_0.8um.png / .tif` - Pre-extracted NIR continuum image.
4. `iirs_band08_pyroxene_2.0um.png / .tif` - Pre-extracted SWIR pyroxene band.
5. `iirs_band12_hydroxyl_3.0um.png / .tif` - Pre-extracted MWIR water/OH band.
6. `reference_tmc2_optical.png / .tif` - Paired Chandrayaan-2 TMC-2 optical reference.
7. `reference_tmc2_pds_label.txt` - Reference optical PDS text label.
8. `ground_truth_tiepoints.csv` & `.json` - Ground truth tiepoints for RMSE and MMA evaluation.
