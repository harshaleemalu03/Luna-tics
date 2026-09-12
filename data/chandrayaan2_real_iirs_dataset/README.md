# Authentic ISRO Chandrayaan-2 Real IIRS & TMC-2 Flight Dataset

This dataset contains **100% genuine spacecraft flight imagery** acquired by the **Chandrayaan-2 Orbiter** during lunar orbital operations, accompanied by official ASCII PDS labels (`pds_label.txt`), ISRO PDS4 XML product labels, and raw telemetry geometry tables.

### 🛰️ Included Flight Scenes:
1. **Scene 01: Apollo 11 / Mare Tranquillitatis** (Orbit 1656, Jan 7, 2020)
   - Real Chandrayaan-2 IIRS Pushbroom Strip + TMC-2 Optical Strip
2. **Scene 02: Apollo 12 / Oceanus Procellarum** (Orbit 2083, Feb 7, 2020)
   - Real Chandrayaan-2 IIRS Western Maria flight pass
3. **Scene 03: Apollo 14 / Fra Mauro Highlands** (Orbit 2079, Feb 7, 2020)
   - Real Chandrayaan-2 IIRS Highland crater terrain
4. **Scene 04: Apollo 11 Tranquillitatis Repeat Pass** (May 23, 2024)
   - Multi-temporal repeat flight pass with differing solar illumination angles

### 📁 Structure per Scene:
- `pds_label.txt`: Standard ASCII PDS3 label with exact solar angles, spacecraft altitude, and geodetic bounding coordinates.
- `real_iirs_flight_strip_full.png` / `.tif`: Full raw pushbroom flight observation strip from Chandrayaan-2 IIRS.
- `iirs_flight_tile.png` / `.tif`: High-contrast flight tile centered on landmarks for image registration algorithms.
- `isro_pds4_product_label.xml`: Original ISRO SAC PDS4 Observational Product XML.
- `geometry_coordinates_grid.csv`: Genuine geodetic latitude/longitude grid from ISRO ISSDC.
- `reference_real_tmc2_optical.png` / `.tif`: Genuine paired Chandrayaan-2 TMC-2 optical observation strip.
- `reference_tmc2_pds_label.txt`: PDS label for the TMC-2 optical image.
- `ground_truth_tiepoints.json` / `.csv`: Analytical ground truth tiepoints for benchmark scoring.
