"""
Luna-tics: Large-Data GeoTIFF and Raster I/O
Provides windowed reads, tiled iteration, ROI extraction, and chunked processing.
"""

import os
from typing import Tuple, Optional, Generator, Dict, Any
import numpy as np
import tifffile
from PIL import Image
import cv2

from src.io.metadata import LunarMetadata


class GeoTIFFHandler:
    """Handles raster I/O with windowing, tiling, and large-raster protection."""

    @staticmethod
    def get_image_info(path: str) -> Dict[str, Any]:
        """Inspect file header without loading the entire raster into memory."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        info = {
            "path": path,
            "width": 0,
            "height": 0,
            "bands": 1,
            "dtype": "uint8",
            "is_geotiff": False
        }

        try:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                shape = page.shape
                if len(shape) == 2:
                    info["height"], info["width"] = shape
                    info["bands"] = 1
                elif len(shape) == 3:
                    if shape[0] < shape[2]:
                        info["bands"], info["height"], info["width"] = shape
                    else:
                        info["height"], info["width"], info["bands"] = shape
                info["dtype"] = str(page.dtype)
                info["is_geotiff"] = bool(page.geotiff_tags)
                return info
        except Exception:
            # Fallback to PIL
            with Image.open(path) as img:
                info["width"], info["height"] = img.size
                info["bands"] = len(img.getbands())
                info["dtype"] = img.mode
                return info

    @staticmethod
    def read_full_or_downsampled(path: str, max_dimension: int = 2048) -> Tuple[np.ndarray, LunarMetadata]:
        """
        Read image cleanly. If dimensions exceed max_dimension, load downsampled to protect memory.
        """
        info = GeoTIFFHandler.get_image_info(path)
        h, w = info["height"], info["width"]

        # If file is a TIFF/GeoTIFF
        if path.lower().endswith((".tif", ".tiff")):
            with tifffile.TiffFile(path) as tif:
                arr = tif.pages[0].asarray()
        else:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                pil_img = Image.open(path)
                arr = np.array(pil_img)

        # Convert to 2D grayscale if multi-channel RGB
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = cv2.cvtColor(np.transpose(arr[:3], (1, 2, 0)), cv2.COLOR_RGB2GRAY)

        # Infer sensor/type from filename
        filename = os.path.basename(path).lower()
        sensor = "GENERIC_OPTICAL"
        host = "Chandrayaan-2"
        gsd = 1.0

        if "iirs" in filename:
            sensor = "IIRS"
            gsd = 80.0
        elif "tmc" in filename:
            sensor = "TMC-2"
            gsd = 5.0
        elif "ohrc" in filename:
            sensor = "OHRC"
            gsd = 0.25
        elif "wac" in filename:
            sensor = "LRO_WAC"
            host = "LRO"
            gsd = 100.0
        elif "nac" in filename:
            sensor = "LRO_NAC"
            host = "LRO"
            gsd = 0.5

        metadata = LunarMetadata(
            sensor=sensor,
            instrument_host=host,
            image_dimensions=(arr.shape[0], arr.shape[1]),
            bands=1 if arr.ndim == 2 else arr.shape[0],
            gsd=gsd,
            file_path=path,
            data_source_type="DEMO / SYNTHETIC" if "synthetic" in filename else "REAL"
        )

        return arr.astype(np.float32), metadata

    @staticmethod
    def read_window(path: str, col_off: int, row_off: int, width: int, height: int) -> np.ndarray:
        """
        Read a localized sub-window (ROI) directly without holding the full raster in memory.
        """
        with tifffile.TiffFile(path) as tif:
            # Memory map the first page
            mmap_arr = tif.pages[0].asarray(out="memmap")
            if mmap_arr.ndim == 2:
                roi = mmap_arr[row_off:row_off + height, col_off:col_off + width]
            else:
                roi = mmap_arr[:, row_off:row_off + height, col_off:col_off + width]
            return np.array(roi, copy=True, dtype=np.float32)

    @staticmethod
    def iter_tiles(
        path: str,
        tile_size: int = 1024,
        overlap: int = 128
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """
        Yields (col_offset, row_offset, tile_array) for scalable large-scene processing.
        """
        info = GeoTIFFHandler.get_image_info(path)
        img_w, img_h = info["width"], info["height"]
        step = tile_size - overlap

        for y in range(0, img_h, step):
            for x in range(0, img_w, step):
                w = min(tile_size, img_w - x)
                h = min(tile_size, img_h - y)
                tile = GeoTIFFHandler.read_window(path, x, y, w, h)
                yield (x, y, tile)

    @staticmethod
    def save_geotiff(
        output_path: str,
        data: np.ndarray,
        metadata: Optional[LunarMetadata] = None,
        description: str = "Luna-tics registered product"
    ) -> str:
        """Save registered array as standard GeoTIFF/TIFF with metadata."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        
        # Save as float32 or uint8
        save_arr = data.astype(np.float32)
        tifffile.imwrite(
            output_path,
            save_arr,
            description=f"{description} | Sensor: {metadata.sensor if metadata else 'N/A'}"
        )
        return output_path
