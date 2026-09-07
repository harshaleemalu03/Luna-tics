"""
Luna-tics: Metadata Definition and Handling
Physics-guided, confidence-driven lunar image registration.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any
import json


@dataclass
class LunarMetadata:
    """Standardized metadata representation for Lunar Remote Sensing products."""
    sensor: str                             # e.g., "IIRS", "TMC-2", "OHRC", "LRO_WAC", "LRO_NAC"
    instrument_host: str                    # "Chandrayaan-2", "LRO", or "Synthetic"
    image_dimensions: Tuple[int, int]       # (height, width)
    bands: int = 1                          # 1 for pan/multispectral band, >1 for hyperspectral
    gsd: Optional[float] = None             # Ground Sampling Distance in meters/pixel
    acquisition_time: Optional[str] = None
    wavelength_min_um: Optional[float] = None
    wavelength_max_um: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    footprint_bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    crs_info: Optional[str] = None
    file_path: str = ""
    data_source_type: str = "REAL"          # "REAL" or "DEMO / SYNTHETIC"
    spectral_reduction_method: Optional[str] = None
    fallback_flags: Dict[str, str] = field(default_factory=dict)

    def has_illumination_geometry(self) -> bool:
        """Check if physical solar geometry angles are available."""
        return (self.incidence_angle_deg is not None and self.emission_angle_deg is not None) or \
               (self.sun_azimuth_deg is not None and self.sun_elevation_deg is not None)

    def has_spatial_coordinates(self) -> bool:
        """Check if geographic footprint coordinates are available."""
        return self.footprint_bbox is not None

    def get_effective_gsd(self, fallback: float = 1.0) -> float:
        """Return GSD or record explicit fallback."""
        if self.gsd is not None and self.gsd > 0:
            return float(self.gsd)
        self.fallback_flags["gsd"] = f"Estimated fallback: {fallback:.2f} m/pixel"
        return fallback

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        inc = f"{self.incidence_angle_deg:.1f}°" if self.incidence_angle_deg is not None else "N/A"
        em = f"{self.emission_angle_deg:.1f}°" if self.emission_angle_deg is not None else "N/A"
        ph = f"{self.phase_angle_deg:.1f}°" if self.phase_angle_deg is not None else "N/A"
        gsd_str = f"{self.gsd:.2f} m/px" if self.gsd is not None else "Not specified in header"
        
        return (
            f"Sensor: {self.sensor} ({self.instrument_host})\n"
            f"Dimensions: {self.image_dimensions[1]} x {self.image_dimensions[0]} (Bands: {self.bands})\n"
            f"GSD: {gsd_str}\n"
            f"Illumination: Inc={inc}, Em={em}, Phase={ph}\n"
            f"Data Source: {self.data_source_type}"
        )
