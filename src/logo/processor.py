import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from svgpathtools import svg2paths

logger = logging.getLogger(__name__)

class LogoProcessor:
    """Parses SVGs, flattens geometry, normalizes coordinates, and rasterizes into evenly distributed point clouds."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.target_width = 300
        self.target_height = 340
        
    def process_logo(self, svg_path: Path, output_json: Path, point_count: int, scale_factor: float, padding: float, debug_img_name: str) -> None:
        logger.info(f"Processing logo: {svg_path.name}")
        
        if not svg_path.exists():
            logger.error(f"Missing SVG file: {svg_path}")
            return
            
        try:
            # Add conversion flags to support SVG basic shapes (rect, circle, polygon, etc)
            paths, attributes = svg2paths(
                str(svg_path),
                convert_lines_to_paths=True,
                convert_polylines_to_paths=True,
                convert_polygons_to_paths=True,
                convert_circles_to_paths=True,
                convert_ellipses_to_paths=True,
                convert_rectangles_to_paths=True
            )
        except Exception as e:
            logger.error(f"Failed to parse SVG {svg_path.name}: {e}")
            paths = []
            
        final_points = []
        if not paths:
            logger.warning(f"No valid paths found in {svg_path.name}. Generating fallback uniform distribution.")
            # Generate a fallback uniform distribution so the pipeline doesn't crash
            np.random.seed(42)
            for i in range(point_count):
                final_points.append({
                    "id": i,
                    "x": round(self.target_width / 2 + np.random.uniform(-100, 100), 2),
                    "y": round(self.target_height / 2 + np.random.uniform(-100, 100), 2)
                })
        else:
            # Sample paths to polygons
            polygons = []
            for path in paths:
                # We must break paths into continuous subpaths to preserve SVG holes correctly.
                for subpath in path.continuous_subpaths():
                    poly = []
                    for segment in subpath:
                        seg_len = segment.length()
                        # Adaptive sampling: drastically increase density for curved segments
                        if type(segment).__name__ in ['CubicBezier', 'QuadraticBezier', 'Arc']:
                            num_samples = max(10, int(seg_len / 0.15))
                        else:
                            num_samples = max(2, int(seg_len / 1.0))
                            
                        for i in range(num_samples):
                            try:
                                c = segment.point(i / float(num_samples - 1))
                                poly.append([c.real, c.imag])
                            except Exception:
                                pass
                    if poly:
                        polygons.append(np.array(poly, dtype=np.float32))
                    
            if not polygons:
                logger.warning(f"No polygons could be extracted from {svg_path.name}")
                return
            
            # Calculate bounding box of all raw geometry
            all_pts = np.vstack(polygons)
            min_x, min_y = np.min(all_pts, axis=0)
            max_x, max_y = np.max(all_pts, axis=0)
            
            orig_w = max(1.0, float(max_x - min_x))
            orig_h = max(1.0, float(max_y - min_y))
            
            # Calculate uniform scale to fit target bounds with padding
            target_w = self.target_width - 2 * padding
            target_h = self.target_height - 2 * padding
            
            scale = min(target_w / orig_w, target_h / orig_h) * scale_factor
            
            # Center translation
            scaled_w = orig_w * scale
            scaled_h = orig_h * scale
            tx = (self.target_width - scaled_w) / 2.0 - min_x * scale
            ty = (self.target_height - scaled_h) / 2.0 - min_y * scale
        
            # Transform polygons
            scaled_polys = []
            for p in polygons:
                sp = p.copy()
                sp[:, 0] = sp[:, 0] * scale + tx
                sp[:, 1] = sp[:, 1] * scale + ty
                scaled_polys.append(sp.astype(np.int32))
                
            # Rasterize geometry to get uniformly filled area points
            mask = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
            
            # cv2.fillPoly correctly handles holes (even-odd rule) when multiple intersecting polygons are passed.
            cv2.fillPoly(mask, scaled_polys, 255)
            cv2.polylines(mask, scaled_polys, isClosed=True, color=255, thickness=1)
            
            y_idx, x_idx = np.where(mask == 255)
            available_points = list(zip(x_idx, y_idx))
            
            if not available_points:
                logger.warning(f"Rasterization yielded 0 points for {svg_path.name}")
                return
                
            # Sample exactly point_count dots from the mask
            logger.info(f"Geometry yielded {len(available_points)} valid pixels. Resampling to target {point_count} points...")
            
            # If we need more points than available pixels, we sample with replacement
            replace = len(available_points) < point_count
            sampled_indices = np.random.choice(len(available_points), size=point_count, replace=replace)
            
            # Add slight sub-pixel jitter to prevent strict grid alignment artifacts during floating-point morphing
            np.random.seed(42)
            for i, idx in enumerate(sampled_indices):
                px, py = available_points[idx]
                jx = float(px + np.random.uniform(-0.5, 0.5))
                jy = float(py + np.random.uniform(-0.5, 0.5))
                
                final_points.append({
                    "id": i,
                    "x": round(jx, 2),
                    "y": round(jy, 2)
                })
            
        # Export JSON
        logger.info(f"Exporting {len(final_points)} normalized points to {output_json}")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(final_points, f, indent=2)
            
        # Debug Image
        debug_path = self.debug_dir / debug_img_name
        debug_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Draw the points faintly
        for pt in final_points:
            ix, iy = int(pt["x"]), int(pt["y"])
            if 0 <= ix < self.target_width and 0 <= iy < self.target_height:
                # Add brightness accumulation for density visualization
                curr = debug_img[iy, ix]
                debug_img[iy, ix] = [min(255, curr[0] + 50), min(255, curr[1] + 50), min(255, curr[2] + 50)]
                
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"Saved debug point cloud to {debug_path}")

