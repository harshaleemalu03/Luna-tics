"""
Luna-tics: Hyperspectral Cube Handling (IIRS QUB / BSQ / Multi-band Arrays)
Supports band selection, spectral averaging, and PCA reduction.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
from sklearn.decomposition import PCA


class HyperspectralProcessor:
    """Processes hyperspectral cubes into optimized 2D representations for registration."""

    @staticmethod
    def reduce_spectral_bands(
        cube: np.ndarray,
        method: str = "informative_selection",
        selected_bands: Optional[List[int]] = None,
        n_components: int = 1
    ) -> Tuple[np.ndarray, dict]:
        """
        Reduce 3D hyperspectral cube (Bands, Height, Width) or (Height, Width, Bands)
        to a 2D structural image suitable for feature matching.

        Args:
            cube: 3D numpy array.
            method: Reduction method - 'informative_selection', 'band_averaging', 'pca', or 'first_band'.
            selected_bands: Indices of bands to use. If None, chooses default lunar diagnostic bands.
            n_components: Number of PCA components if method == 'pca'.

        Returns:
            Tuple of (reduced_2d_image, reduction_metadata_dict)
        """
        if cube.ndim == 2:
            # Already 2D
            return cube.astype(np.float32), {
                "bands_available": 1,
                "bands_used": [0],
                "reduction_method": "none_already_single_band"
            }

        # Normalize shape to (Bands, Height, Width)
        if cube.shape[0] > cube.shape[2] and cube.shape[2] <= 512:
            # Likely (Height, Width, Bands)
            cube = np.transpose(cube, (2, 0, 1))

        num_bands, height, width = cube.shape

        if method == "informative_selection":
            # Select diagnostic lunar mineral absorption window (e.g. 1000nm band, avoiding thermal noise)
            if selected_bands is None or len(selected_bands) == 0:
                # Choose ~1/4 to 1/2 of spectral range if 256 bands (e.g., band 40 ~ 1000nm for pyroxene/olivine)
                idx = min(40, num_bands - 1)
                selected_bands = [idx]
            
            valid_indices = [b for b in selected_bands if 0 <= b < num_bands]
            if not valid_indices:
                valid_indices = [0]
                
            if len(valid_indices) == 1:
                reduced = cube[valid_indices[0]].astype(np.float32)
            else:
                reduced = np.mean(cube[valid_indices], axis=0).astype(np.float32)

            meta = {
                "bands_available": num_bands,
                "bands_used": valid_indices,
                "reduction_method": "informative_selection"
            }

        elif method == "band_averaging":
            # Average across selected or all non-noisy bands
            if selected_bands is not None and len(selected_bands) > 0:
                valid_indices = [b for b in selected_bands if 0 <= b < num_bands]
            else:
                valid_indices = list(range(num_bands))

            reduced = np.mean(cube[valid_indices], axis=0).astype(np.float32)
            meta = {
                "bands_available": num_bands,
                "bands_used": valid_indices,
                "reduction_method": "band_averaging"
            }

        elif method == "pca":
            # PCA across spectral axis: captures dominant spatial variation
            # Reshape (Bands, H, W) -> (H * W, Bands)
            reshaped = cube.reshape(num_bands, height * width).T
            # Replace NaNs or Infs
            reshaped = np.nan_to_num(reshaped, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(reshaped)
            
            # Use first principal component
            reduced = transformed[:, 0].reshape(height, width).astype(np.float32)
            
            # Orient component so mean correlation with average band is positive
            mean_band = np.mean(cube, axis=0)
            if np.corrcoef(reduced.flatten(), mean_band.flatten())[0, 1] < 0:
                reduced = -reduced

            meta = {
                "bands_available": num_bands,
                "bands_used": list(range(num_bands)),
                "reduction_method": f"pca_pc1 (explained_variance={pca.explained_variance_ratio_[0]:.3f})"
            }
        else:
            reduced = cube[0].astype(np.float32)
            meta = {
                "bands_available": num_bands,
                "bands_used": [0],
                "reduction_method": "fallback_first_band"
            }

        return reduced, meta
