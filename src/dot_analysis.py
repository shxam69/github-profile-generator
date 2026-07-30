import json
import logging
import math
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image
from scipy.ndimage import convolve
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Dot:
    id: int
    x: int
    y: int
    brightness: int
    distance_from_center: float
    normalized_x: float
    normalized_y: float
    is_edge: bool
    neighbors: int
    cluster_id: Optional[int]
    band_id: Optional[int]
    animation_group: Optional[int]

class DotAnalyzer:
    """Analyzes dithered portrait dots and generates spatial/cluster metadata."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze(self, dither_path: Path, grayscale_path: Path, output_json: Path) -> None:
        """
        Extracts foreground pixels as Dot objects and calculates 
        spatial features, edges, density, and connected components.
        """
        logger.info("Starting dot analysis pipeline...")
        
        if not dither_path.exists():
            raise FileNotFoundError(f"Missing dither image: {dither_path}")
        if not grayscale_path.exists():
            raise FileNotFoundError(f"Missing grayscale image: {grayscale_path}")
            
        # Load images
        logger.info(f"Loading {dither_path.name} and {grayscale_path.name}...")
        dither_img = Image.open(dither_path).convert("L")
        binary_arr = (np.array(dither_img) > 127).astype(np.uint8)
        
        gray_img = Image.open(grayscale_path).convert("L")
        gray_arr = np.array(gray_img)
        
        height, width = binary_arr.shape
        center_x, center_y = width / 2.0, height / 2.0
        
        # 1. Compute neighbors (3x3 kernel)
        logger.info("Computing local neighbors (3x3 window)...")
        kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
        # convolve counts the center pixel too, so we subtract binary_arr
        neighbors_arr = convolve(binary_arr, kernel_3x3, mode='constant', cval=0) - binary_arr
        
        # 2. Compute local density (11x11 kernel)
        logger.info("Computing local density (11x11 window)...")
        kernel_11x11 = np.ones((11, 11), dtype=np.uint8)
        density_arr = convolve(binary_arr, kernel_11x11, mode='constant', cval=0)
        
        # Visualize density
        max_density = density_arr.max()
        if max_density > 0:
            density_vis = (density_arr / max_density * 255).astype(np.uint8)
        else:
            density_vis = np.zeros_like(density_arr, dtype=np.uint8)
            
        density_color = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
        density_color[binary_arr == 0] = [0, 0, 0]  # Mask out background
        cv2.imwrite(str(self.debug_dir / "11_density.png"), density_color)
        logger.info("Saved debug output: 11_density.png")
        
        # 3. Identify edges
        logger.info("Identifying edge dots...")
        # A foreground dot is on the edge if it has less than 8 neighbors
        is_edge_arr = (neighbors_arr < 8) & (binary_arr == 1)
        
        # Visualize edges
        edge_vis = np.zeros((height, width, 3), dtype=np.uint8)
        edge_vis[is_edge_arr] = [0, 0, 255] # Red edges (BGR format for cv2)
        edge_vis[(binary_arr == 1) & ~is_edge_arr] = [255, 255, 255] # White internal dots
        cv2.imwrite(str(self.debug_dir / "12_edges.png"), edge_vis)
        logger.info("Saved debug output: 12_edges.png")
        
        # 4. Build connected clusters
        logger.info("Building connected clusters...")
        num_labels, labels = cv2.connectedComponents(binary_arr, connectivity=8)
        
        # Visualize clusters
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] # Background is black
        cluster_vis = colors[labels]
        cv2.imwrite(str(self.debug_dir / "10_clusters.png"), cluster_vis)
        logger.info("Saved debug output: 10_clusters.png")
        
        # 5. Build Dots metadata
        logger.info("Extracting foreground pixels and constructing Dot objects...")
        dots = []
        dot_id = 0
        
        ys, xs = np.where(binary_arr == 1)
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            
            dot = Dot(
                id=dot_id,
                x=x,
                y=y,
                brightness=int(gray_arr[y, x]),
                distance_from_center=float(math.hypot(x - center_x, y - center_y)),
                normalized_x=float(x / width),
                normalized_y=float(y / height),
                is_edge=bool(is_edge_arr[y, x]),
                neighbors=int(neighbors_arr[y, x]),
                cluster_id=int(labels[y, x]),
                band_id=None,
                animation_group=None
            )
            dots.append(dot)
            dot_id += 1
            
        logger.info(f"Successfully analyzed {len(dots)} dots. Exporting to JSON...")
        
        # 6. Export to JSON
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump([asdict(d) for d in dots], f, indent=2)
            
        logger.info(f"Sprint 4 Pipeline completed. Metadata saved to {output_json}")
