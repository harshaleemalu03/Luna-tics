"""
Luna-tics: Diagnostic and Scientific Remote Sensing Visualization Suite
Generates before/after overlays, checkerboard grids, difference maps, quiver vectors, and match evidence cards.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.matching.primary import CandidateMatch
from src.matching.confidence import MatchEvidence


class Visualizer:
    """Generates publication-quality scientific registration visualizations."""

    @staticmethod
    def warp_image_to_reference(
        img_src_u8: np.ndarray,
        img_ref_u8: np.ndarray,
        H_matrix: np.ndarray
    ) -> np.ndarray:
        """Warp source image onto reference coordinate frame using transformation matrix."""
        h_r, w_r = img_ref_u8.shape[:2]
        if H_matrix.shape == (2, 3):
            warped = cv2.warpAffine(img_src_u8, H_matrix, (w_r, h_r), flags=cv2.INTER_LINEAR)
        elif H_matrix.shape == (3, 3):
            warped = cv2.warpPerspective(img_src_u8, H_matrix, (w_r, h_r), flags=cv2.INTER_LINEAR)
        else:
            warped = np.zeros_like(img_ref_u8)
        return warped

    @staticmethod
    def create_checkerboard(
        img_ref_u8: np.ndarray,
        img_warped_u8: np.ndarray,
        num_tiles_x: int = 8,
        num_tiles_y: int = 8
    ) -> np.ndarray:
        """
        Generate interleaved checkerboard comparison image.
        Aligned features seamlessly transition across cell boundaries.
        """
        h, w = img_ref_u8.shape[:2]
        tile_w = max(1, w // num_tiles_x)
        tile_h = max(1, h // num_tiles_y)

        checkerboard = np.copy(img_ref_u8)
        for y_idx in range(num_tiles_y):
            for x_idx in range(num_tiles_x):
                # Checkerboard condition: alternate cells
                if (x_idx + y_idx) % 2 == 1:
                    y0 = y_idx * tile_h
                    y1 = min(h, (y_idx + 1) * tile_h)
                    x0 = x_idx * tile_w
                    x1 = min(w, (x_idx + 1) * tile_w)
                    checkerboard[y0:y1, x0:x1] = img_warped_u8[y0:y1, x0:x1]

        # Draw thin divider grid lines for visual clarity
        checker_rgb = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2RGB) if checkerboard.ndim == 2 else checkerboard.copy()
        for x_idx in range(1, num_tiles_x):
            x = x_idx * tile_w
            cv2.line(checker_rgb, (x, 0), (x, h), (0, 255, 255), 1)
        for y_idx in range(1, num_tiles_y):
            y = y_idx * tile_h
            cv2.line(checker_rgb, (0, y), (w, y), (0, 255, 255), 1)

        return checker_rgb

    @staticmethod
    def create_difference_image(
        img_ref_u8: np.ndarray,
        img_warped_u8: np.ndarray
    ) -> np.ndarray:
        """
        Compute absolute radiometric difference map with colormap.
        Masks out non-overlapping zero-fill borders.
        """
        overlap_mask = (img_warped_u8 > 0)
        diff = np.abs(img_ref_u8.astype(np.float32) - img_warped_u8.astype(np.float32))
        diff[~overlap_mask] = 0.0

        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
        colored_diff = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
        colored_diff[~overlap_mask] = [20, 20, 25]  # dark background for outside bounds
        return colored_diff

    @staticmethod
    def create_side_by_side_overlay(
        img_src_u8: np.ndarray,
        img_warped_u8: np.ndarray,
        img_ref_u8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        - before_blend: False-color composite of unaligned Source (Red) + Reference (Cyan/Green)
        - after_blend: False-color composite of Warped Source (Red) + Reference (Cyan/Green)
        Aligned regions appear gray/neutral, misaligned regions show colored fringe.
        """
        h, w = img_ref_u8.shape[:2]
        src_resized = cv2.resize(img_src_u8, (w, h))

        # Before alignment: Red = Source, Green/Blue = Reference
        before = np.zeros((h, w, 3), dtype=np.uint8)
        before[:, :, 0] = src_resized
        before[:, :, 1] = img_ref_u8
        before[:, :, 2] = img_ref_u8

        # After alignment: Red = Warped Source, Green/Blue = Reference
        after = np.zeros((h, w, 3), dtype=np.uint8)
        after[:, :, 0] = img_warped_u8
        after[:, :, 1] = img_ref_u8
        after[:, :, 2] = img_ref_u8

        return before, after

    @staticmethod
    def draw_correspondences(
        img_src_u8: np.ndarray,
        img_ref_u8: np.ndarray,
        matches: List[CandidateMatch],
        evidences: Optional[List[MatchEvidence]] = None,
        max_draw: int = 150
    ) -> np.ndarray:
        """
        Draw side-by-side correspondence lines connecting source to reference.
        Color-coded by confidence (Green = high, Yellow = medium, Red = low).
        """
        h_s, w_s = img_src_u8.shape[:2]
        h_r, w_r = img_ref_u8.shape[:2]
        max_h = max(h_s, h_r)

        canvas = np.zeros((max_h, w_s + w_r, 3), dtype=np.uint8)
        s_rgb = cv2.cvtColor(img_src_u8, cv2.COLOR_GRAY2RGB) if img_src_u8.ndim == 2 else img_src_u8
        r_rgb = cv2.cvtColor(img_ref_u8, cv2.COLOR_GRAY2RGB) if img_ref_u8.ndim == 2 else img_ref_u8

        canvas[:h_s, :w_s] = s_rgb
        canvas[:h_r, w_s:w_s + w_r] = r_rgb

        step = max(1, len(matches) // max_draw)
        for i in range(0, len(matches), step):
            m = matches[i]
            xs, ys = int(round(m.pt_src[0])), int(round(m.pt_src[1]))
            xr, yr = int(round(m.pt_ref[0])) + w_s, int(round(m.pt_ref[1]))

            # Color by confidence
            if evidences and i < len(evidences):
                conf = evidences[i].final_confidence
                if conf >= 0.75:
                    color = (50, 255, 50)   # Green
                elif conf >= 0.50:
                    color = (50, 220, 255)  # Yellow
                else:
                    color = (50, 50, 255)   # Red
            else:
                color = (0, 255, 200)

            cv2.circle(canvas, (xs, ys), 4, color, -1)
            cv2.circle(canvas, (xr, yr), 4, color, -1)
            cv2.line(canvas, (xs, ys), (xr, yr), color, 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def draw_quiver_residuals(
        img_ref_u8: np.ndarray,
        matches: List[CandidateMatch],
        residuals: np.ndarray,
        H_matrix: np.ndarray,
        scale_factor: float = 8.0
    ) -> np.ndarray:
        """
        Render quiver plot of registration residual vectors overlaid on reference image.
        Magnifies residual displacement vectors by scale_factor for visual clarity.
        """
        h_r, w_r = img_ref_u8.shape[:2]
        canvas = cv2.cvtColor(img_ref_u8, cv2.COLOR_GRAY2RGB) if img_ref_u8.ndim == 2 else img_ref_u8.copy()

        for i, m in enumerate(matches):
            pt_s = np.array([m.pt_src[0], m.pt_src[1], 1.0], dtype=np.float32)
            if H_matrix.shape == (2, 3):
                pt_mapped = H_matrix @ pt_s
            else:
                proj = H_matrix @ pt_s
                pt_mapped = proj[:2] / max(1e-6, proj[2])

            xr, yr = m.pt_ref
            dx = (pt_mapped[0] - xr) * scale_factor
            dy = (pt_mapped[1] - yr) * scale_factor

            err = residuals[i] if i < len(residuals) else 0.0
            color = (0, 255, 50) if err < 2.0 else (0, 165, 255) if err < 5.0 else (0, 0, 255)

            p0 = (int(round(xr)), int(round(yr)))
            p1 = (int(round(xr + dx)), int(round(yr + dy)))

            cv2.circle(canvas, p0, 3, color, -1)
            cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.3)

        return canvas

    @staticmethod
    def extract_patch_pair(
        img_src_u8: np.ndarray,
        img_ref_u8: np.ndarray,
        match: CandidateMatch,
        patch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract zoomed image patches around corresponding points."""
        half = patch_size // 2
        h_s, w_s = img_src_u8.shape[:2]
        h_r, w_r = img_ref_u8.shape[:2]

        xs, ys = int(round(match.pt_src[0])), int(round(match.pt_src[1]))
        xr, yr = int(round(match.pt_ref[0])), int(round(match.pt_ref[1]))

        # Helper to crop with padding
        def crop_with_pad(img, x, y, h, w):
            x0, x1 = x - half, x + half
            y0, y1 = y - half, y + half

            pad_l = max(0, -x0)
            pad_r = max(0, x1 - w)
            pad_t = max(0, -y0)
            pad_b = max(0, y1 - h)

            crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if pad_l > 0 or pad_r > 0 or pad_t > 0 or pad_b > 0:
                crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            return crop

        patch_s = crop_with_pad(img_src_u8, xs, ys, h_s, w_s)
        patch_r = crop_with_pad(img_ref_u8, xr, yr, h_r, w_r)

        # Mark center with subtle crosshair
        patch_s_rgb = cv2.cvtColor(patch_s, cv2.COLOR_GRAY2RGB) if patch_s.ndim == 2 else patch_s.copy()
        patch_r_rgb = cv2.cvtColor(patch_r, cv2.COLOR_GRAY2RGB) if patch_r.ndim == 2 else patch_r.copy()

        cv2.drawMarker(patch_s_rgb, (half, half), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        cv2.drawMarker(patch_r_rgb, (half, half), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

        return patch_s_rgb, patch_r_rgb
