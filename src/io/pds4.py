"""
Luna-tics: PDS4 Label Parser and Metadata Extraction
Handles Chandrayaan-2 (IIRS, TMC-2, OHRC) and LRO PDS4 XML labels.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple
from src.io.metadata import LunarMetadata


class PDS4Parser:
    """Parser for PDS4 XML product labels conforming to PDS4 Information Model."""

    @staticmethod
    def _strip_ns(tag: str) -> str:
        """Strip XML namespace from tag name."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def parse_label(cls, xml_path: str) -> LunarMetadata:
        """
        Parse PDS4 XML label file and extract standardized LunarMetadata.
        
        Args:
            xml_path: Absolute or relative path to .xml label file.
            
        Returns:
            LunarMetadata dataclass instance.
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"PDS4 label not found at {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Helper to search for local tag regardless of namespace
        def find_text_recursive(elem: ET.Element, target_tag: str) -> Optional[str]:
            for child in elem.iter():
                if cls._strip_ns(child.tag).lower() == target_tag.lower():
                    if child.text and child.text.strip():
                        return child.text.strip()
            return None

        # Extract Identification
        logical_id = find_text_recursive(root, "logical_identifier") or "urn:isro:ch2"
        title = find_text_recursive(root, "title") or os.path.basename(xml_path)

        # Detect Sensor and Instrument Host
        sensor = "UNKNOWN"
        host = "Chandrayaan-2"
        
        id_lower = (logical_id + " " + title).lower()
        if "iirs" in id_lower:
            sensor = "IIRS"
            host = "Chandrayaan-2"
        elif "tmc" in id_lower or "tmc2" in id_lower or "tmc-2" in id_lower:
            sensor = "TMC-2"
            host = "Chandrayaan-2"
        elif "ohrc" in id_lower:
            sensor = "OHRC"
            host = "Chandrayaan-2"
        elif "wac" in id_lower:
            sensor = "LRO_WAC"
            host = "LRO"
        elif "nac" in id_lower:
            sensor = "LRO_NAC"
            host = "LRO"

        # Observation Time
        start_time = find_text_recursive(root, "start_date_time")

        # Illumination Geometry
        incidence_str = find_text_recursive(root, "incidence_angle") or find_text_recursive(root, "solar_incidence_angle")
        emission_str = find_text_recursive(root, "emission_angle")
        phase_str = find_text_recursive(root, "phase_angle")
        azimuth_str = find_text_recursive(root, "sun_azimuth_angle") or find_text_recursive(root, "sun_azimuth")
        elevation_str = find_text_recursive(root, "sun_elevation_angle") or find_text_recursive(root, "sun_elevation")

        incidence = float(incidence_str) if incidence_str else None
        emission = float(emission_str) if emission_str else None
        phase = float(phase_str) if phase_str else None
        azimuth = float(azimuth_str) if azimuth_str else None
        elevation = float(elevation_str) if elevation_str else None

        # Spatial Resolution / GSD
        res_str = (find_text_recursive(root, "pixel_resolution") or 
                   find_text_recursive(root, "ground_sampling_distance") or
                   find_text_recursive(root, "spatial_resolution"))
        gsd: Optional[float] = float(res_str) if res_str else None
        
        # Sensor-specific GSD defaults if not in XML
        if gsd is None:
            if sensor == "IIRS":
                gsd = 80.0  # nominal ~80m/px for 100km orbit
            elif sensor == "TMC-2":
                gsd = 5.0   # nominal 5m/px
            elif sensor == "OHRC":
                gsd = 0.25  # nominal 0.25m/px (super high res)
            elif sensor == "LRO_WAC":
                gsd = 100.0 # global monochrome ~100m/px
            elif sensor == "LRO_NAC":
                gsd = 0.5   # nominal 0.5m/px

        # Image Dimensions and Bands
        lines = 512
        samples = 512
        bands = 1

        for elem in root.iter():
            tag = cls._strip_ns(elem.tag).lower()
            if tag in ("array_2d_image", "array_3d_spectrum"):
                # Inspect axis_array
                for axis in elem.iter():
                    if cls._strip_ns(axis.tag).lower() == "axis_array":
                        axis_name = find_text_recursive(axis, "axis_name") or ""
                        elements = find_text_recursive(axis, "elements")
                        if elements and elements.isdigit():
                            val = int(elements)
                            if "line" in axis_name.lower():
                                lines = val
                            elif "sample" in axis_name.lower():
                                samples = val
                            elif "band" in axis_name.lower():
                                bands = val

        # Bounding box / Footprint
        min_lon = find_text_recursive(root, "westernmost_longitude")
        max_lon = find_text_recursive(root, "easternmost_longitude")
        min_lat = find_text_recursive(root, "minimum_latitude")
        max_lat = find_text_recursive(root, "maximum_latitude")
        
        bbox = None
        if min_lon and max_lon and min_lat and max_lat:
            try:
                bbox = (float(min_lon), float(min_lat), float(max_lon), float(max_lat))
            except ValueError:
                bbox = None

        # Data source type
        is_synthetic = "synthetic" in id_lower or "demo" in id_lower
        data_source_type = "DEMO / SYNTHETIC" if is_synthetic else "REAL"

        return LunarMetadata(
            sensor=sensor,
            instrument_host=host,
            image_dimensions=(lines, samples),
            bands=bands,
            gsd=gsd,
            acquisition_time=start_time,
            sun_azimuth_deg=azimuth,
            sun_elevation_deg=elevation,
            incidence_angle_deg=incidence,
            emission_angle_deg=emission,
            phase_angle_deg=phase,
            footprint_bbox=bbox,
            file_path=xml_path,
            data_source_type=data_source_type
        )


def create_sample_pds4_label(
    output_xml_path: str,
    sensor: str,
    instrument_host: str = "Chandrayaan-2",
    lines: int = 512,
    samples: int = 512,
    bands: int = 1,
    gsd: float = 80.0,
    incidence_angle: float = 45.0,
    emission_angle: float = 5.0,
    phase_angle: float = 48.0,
    sun_azimuth: float = 120.0,
    sun_elevation: float = 45.0,
    start_time: str = "2024-03-15T08:30:00.000Z",
    bbox: Optional[Tuple[float, float, float, float]] = (18.5, -5.2, 19.8, -4.0),
    is_synthetic: bool = False
) -> str:
    """Generate authentic, conforming PDS4 XML label for calibration and verification."""
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True) if os.path.dirname(output_xml_path) else None
    
    label_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:{sensor.lower()}:{"synthetic:" if is_synthetic else ""}prod_{sensor.lower()}_01</logical_identifier>
    <version_id>1.0</version_id>
    <title>Chandrayaan-2 {sensor} Calibrated Lunar Science Product</title>
    <information_model_version>1.14.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>{start_time}</start_date_time>
      <stop_date_time>{start_time}</stop_date_time>
    </Time_Coordinates>
    <Target_Identification>
      <name>Moon</name>
      <type>Satellite</type>
    </Target_Identification>
    <Discipline_Area>
      <Geometry>
        <incidence_angle unit="deg">{incidence_angle:.2f}</incidence_angle>
        <emission_angle unit="deg">{emission_angle:.2f}</emission_angle>
        <phase_angle unit="deg">{phase_angle:.2f}</phase_angle>
        <sun_azimuth_angle unit="deg">{sun_azimuth:.2f}</sun_azimuth_angle>
        <sun_elevation_angle unit="deg">{sun_elevation:.2f}</sun_elevation_angle>
        <ground_sampling_distance unit="m">{gsd:.2f}</ground_sampling_distance>
        {"<westernmost_longitude>" + str(bbox[0]) + "</westernmost_longitude>" if bbox else ""}
        {"<minimum_latitude>" + str(bbox[1]) + "</minimum_latitude>" if bbox else ""}
        {"<easternmost_longitude>" + str(bbox[2]) + "</easternmost_longitude>" if bbox else ""}
        {"<maximum_latitude>" + str(bbox[3]) + "</maximum_latitude>" if bbox else ""}
      </Geometry>
    </Discipline_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>{os.path.splitext(os.path.basename(output_xml_path))[0]}.raw</file_name>
    </File>
    <Array_3D_Spectrum>
      <axes>{3 if bands > 1 else 2}</axes>
      <axis_index_order>Last_Index_Fastest</axis_index_order>
      <Element_Array>
        <data_type>IEEE754MSBSingle</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Band</axis_name>
        <elements>{bands}</elements>
        <sequence_number>1</sequence_number>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>{lines}</elements>
        <sequence_number>2</sequence_number>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>{samples}</elements>
        <sequence_number>3</sequence_number>
      </Axis_Array>
    </Array_3D_Spectrum>
  </File_Area_Observational>
</Product_Observational>"""

    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(label_xml)

    return output_xml_path
