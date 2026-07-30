import json
import logging
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class IntroShimmerGenerator:
    """Generates evenly distributed shimmer groups to avoid wipe-like reveals."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_evenness_score(self, groups: Dict[int, List[int]], node_coords: Dict[int, Tuple[int, int]], global_cx: float, global_cy: float) -> float:
        """
        Calculates the maximum centroid deviation across all groups.
        Lower is better (more evenly distributed).
        """
        max_deviation = 0.0
        
        for g_id, dots in groups.items():
            if not dots:
                continue
                
            cx = sum(node_coords[d][0] for d in dots) / len(dots)
            cy = sum(node_coords[d][1] for d in dots) / len(dots)
            
            deviation = math.hypot(cx - global_cx, cy - global_cy)
            if deviation > max_deviation:
                max_deviation = deviation
                
        return max_deviation

    def generate(self, graph_json_path: Path, output_json: Path, num_groups: int, threshold: float, max_attempts: int) -> None:
        logger.info(f"Loading graph dots from {graph_json_path}")
        
        if not graph_json_path.exists():
            raise FileNotFoundError(f"Missing graph JSON file: {graph_json_path}")
            
        with open(graph_json_path, 'r') as f:
            graph_data = json.load(f)
            
        nodes = graph_data["nodes"]
        dot_ids = [n["id"] for n in nodes]
        node_coords = {n["id"]: (n["x"], n["y"]) for n in nodes}
        
        if not dot_ids:
            logger.warning("No dots found in graph. Skipping intro shimmer generation.")
            return
            
        # Compute global centroid
        global_cx = sum(x for x, y in node_coords.values()) / len(node_coords)
        global_cy = sum(y for x, y in node_coords.values()) / len(node_coords)
        
        # Determine canvas size
        max_x = max(x for x, y in node_coords.values())
        max_y = max(y for x, y in node_coords.values())
        width, height = max_x + 1, max_y + 1
        
        # Partition dots ensuring even distribution
        best_groups: Dict[int, List[int]] = {}
        best_score = float('inf')
        
        logger.info(f"Generating {num_groups} intro groups (Target evenness <= {threshold}px)...")
        
        for attempt in range(max_attempts):
            # Shuffle randomly to distribute assignments globally
            random.shuffle(dot_ids)
            groups = {i: [] for i in range(num_groups)}
            
            for i, d in enumerate(dot_ids):
                groups[i % num_groups].append(d)
                
            score = self._compute_evenness_score(groups, node_coords, global_cx, global_cy)
            
            if score < best_score:
                best_score = score
                best_groups = groups
                
            if score <= threshold:
                logger.info(f"Found acceptable group assignment on attempt {attempt + 1}. Score: {score:.2f}px")
                break
                
        if best_score > threshold:
            logger.warning(f"Could not reach target evenness threshold. Best score achieved: {best_score:.2f}px")
            
        # Format the export payload
        logger.info("Exporting intro groups...")
        output_payload = []
        for g_id, dots in best_groups.items():
            output_payload.append({
                "group_id": g_id,
                "dot_ids": dots,
                "delay": round(g_id * 0.05, 2),  # Staggered delay mapping
                "duration": 2.0
            })
            
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(output_payload, f, indent=2)
            
        # 3 Debug Visualizations
        logger.info("Generating Sprint 7 debug visualizations...")
        
        # 16_intro_groups.png (Color code by group)
        groups_img = np.zeros((height, width, 3), dtype=np.uint8)
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(num_groups, 3), dtype=np.uint8)
        
        for g_id, dots in best_groups.items():
            for d in dots:
                x, y = node_coords[d]
                groups_img[y, x] = colors[g_id]
        cv2.imwrite(str(self.debug_dir / "16_intro_groups.png"), groups_img)
        
        # 17_intro_heatmap.png (Heatmap of delay/group assignments)
        heatmap_img = np.zeros((height, width), dtype=np.uint8)
        for g_id, dots in best_groups.items():
            for d in dots:
                x, y = node_coords[d]
                heatmap_img[y, x] = int((g_id / max(1, num_groups - 1)) * 255)
        
        heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        # Apply mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for d in dot_ids:
            x, y = node_coords[d]
            mask[y, x] = 1
        heatmap_colored[mask == 0] = [0, 0, 0]
        
        cv2.imwrite(str(self.debug_dir / "17_intro_heatmap.png"), heatmap_colored)
        
        # 18_evenness.png (Plot global centroid and all group centroids to visually prove tightness)
        evenness_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw all dots faintly
        for d in dot_ids:
            x, y = node_coords[d]
            evenness_img[y, x] = [50, 50, 50]
            
        # Draw group centroids as red dots
        for g_id, dots in best_groups.items():
            if dots:
                cx = int(sum(node_coords[d][0] for d in dots) / len(dots))
                cy = int(sum(node_coords[d][1] for d in dots) / len(dots))
                cv2.circle(evenness_img, (cx, cy), radius=2, color=(0, 0, 255), thickness=-1)
                
        # Draw global centroid as bright green cross
        cv2.drawMarker(evenness_img, (int(global_cx), int(global_cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        cv2.imwrite(str(self.debug_dir / "18_evenness.png"), evenness_img)
        
        logger.info("Sprint 7 Pipeline completed successfully.")
