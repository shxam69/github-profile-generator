import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class TravellerPathEngine:
    """Builds globally optimal continuous paths for traveller dots to morph between logos."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.canvas_w = 300
        self.canvas_h = 340
        
    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
        
    def _count_collisions(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Estimates crossing paths in O(N^2). Since N=900, this is ~400k operations (very fast)."""
        collisions = 0
        n = len(p1)
        for i in range(n):
            for j in range(i + 1, n):
                if self._intersect(p1[i], p2[i], p1[j], p2[j]):
                    collisions += 1
        return collisions

    def build_paths(self, graph_json: Path, logo_jsons: List[Path], output_json: Path, traveller_count: int) -> None:
        logger.info(f"Building traveller paths for {traveller_count} dots...")
        
        # Load Portrait
        with open(graph_json, 'r') as f:
            portrait_nodes = json.load(f)["nodes"]
            
        portrait_coords = np.array([[n["x"], n["y"]] for n in portrait_nodes])
        portrait_ids = np.array([n["id"] for n in portrait_nodes])
        
        # Load Logos
        logos_coords = []
        logos_ids = []
        for lpath in logo_jsons:
            with open(lpath, 'r') as f:
                logo_pts = json.load(f)
            logos_coords.append(np.array([[p["x"], p["y"]] for p in logo_pts]))
            logos_ids.append(np.array([p["id"] for p in logo_pts]))
            
        # Select Travellers Deterministically
        np.random.seed(42)
        traveller_indices = np.random.choice(len(portrait_nodes), size=traveller_count, replace=False)
        curr_coords = portrait_coords[traveller_indices]
        orig_coords = curr_coords.copy()
        
        # Base JSON structure
        paths = []
        for i in range(traveller_count):
            paths.append({
                "traveller_id": i,
                "portrait_dot": int(portrait_ids[traveller_indices[i]])
            })
            
        total_distances = np.zeros(traveller_count)
        unused_points = []
        total_collisions = 0
        
        # Optimize paths between logos
        for step, (l_coords, l_ids) in enumerate(zip(logos_coords, logos_ids)):
            logger.info(f"Computing globally optimal assignment for Logo {step + 1}...")
            
            # Cost matrix C[i, j] = dist(traveller_i, logo_point_j)
            cost_matrix = cdist(curr_coords, l_coords, metric='euclidean')
            
            # Hungarian algorithm ensures no overlapping destinations and minimizes total global cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            target_coords = l_coords[col_ind]
            
            # Collision estimation for this transit
            col_est = self._count_collisions(curr_coords, target_coords)
            total_collisions += col_est
            logger.info(f"Phase {step} -> {step+1}: Estimated {col_est} path crosses.")
            
            # Unused points
            used_set = set(col_ind)
            unused_logo_pts = len(l_coords) - len(used_set)
            unused_points.append(unused_logo_pts)
            
            # Update paths
            for i, r in enumerate(row_ind):
                c = col_ind[i]
                paths[r][f"logo{step + 1}_point"] = int(l_ids[c])
                total_distances[r] += cost_matrix[r, c]
                
            curr_coords = target_coords
            
        # Return Journey
        logger.info("Computing return journey bounds...")
        for i in range(traveller_count):
            # Distance back to initial portrait position
            dist = math.hypot(curr_coords[i][0] - orig_coords[i][0], curr_coords[i][1] - orig_coords[i][1])
            total_distances[i] += dist
            paths[i]["return_dot"] = paths[i]["portrait_dot"]
            
        # Compute Statistics
        stats = {
            "average_path_length": float(round(np.mean(total_distances), 2)),
            "max_path_length": float(round(np.max(total_distances), 2)),
            "min_path_length": float(round(np.min(total_distances), 2)),
            "travel_variance": float(round(np.var(total_distances), 2)),
            "collision_estimate": total_collisions,
            "unused_logo_points": sum(unused_points)
        }
        
        stats_path = self.debug_dir / "27_path_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Path Stats: Avg Dist = {stats['average_path_length']}, Max Dist = {stats['max_path_length']}, Collisions = {stats['collision_estimate']}")
            
        # Export JSON
        logger.info(f"Exporting complete traveller paths to {output_json}")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(paths, f, indent=2)
            
        # Generate Debug Visualizations
        logger.info("Generating Sprint 10 debug visualizations...")
        
        # 25_traveller_portrait.png
        img25 = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        for c in portrait_coords:
            img25[int(c[1]), int(c[0])] = [30, 30, 30]
        for c in orig_coords:
            cv2.circle(img25, (int(c[0]), int(c[1])), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.imwrite(str(self.debug_dir / "25_traveller_portrait.png"), img25)
        
        # 26_logo_assignments.png (Show Logo 1 used vs unused)
        img26 = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        logo1_used_indices = [p["logo1_point"] for p in paths]
        used_set_l1 = set(logo1_used_indices)
        for pt in logos_coords[0]:
            img26[int(pt[1]), int(pt[0])] = [50, 50, 50]
        for p in paths:
            # find coord
            idx = p["logo1_point"]
            # Just simple linear search for debug since array might not match id exactly, 
            # actually we mapped ids directly so let's lookup coords by ID safely.
            pass # skipping exact lookup for brevity, will rely on known indices since ID = index for logos usually.
            
        # Safe coordinate lookup for Logo 1
        l1_coord_map = {int(i): c for i, c in zip(logos_ids[0], logos_coords[0])}
        for idx in used_set_l1:
            if idx in l1_coord_map:
                c = l1_coord_map[idx]
                cv2.circle(img26, (int(c[0]), int(c[1])), radius=1, color=(255, 255, 0), thickness=-1)
        cv2.imwrite(str(self.debug_dir / "26_logo_assignments.png"), img26)
        
        # 28_path_preview.png (Draw lines between Phase 0 and Phase 1)
        img28 = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        # Using addWeighted to create transparency effect
        overlay = img28.copy()
        for i in range(traveller_count):
            p1 = (int(orig_coords[i][0]), int(orig_coords[i][1]))
            l1_id = paths[i]["logo1_point"]
            c2 = l1_coord_map[l1_id]
            p2 = (int(c2[0]), int(c2[1]))
            cv2.line(overlay, p1, p2, color=(0, 100, 255), thickness=1)
        cv2.addWeighted(overlay, 0.4, img28, 0.6, 0, img28)
        
        # Draw end points
        for i in range(traveller_count):
            p1 = (int(orig_coords[i][0]), int(orig_coords[i][1]))
            cv2.circle(img28, p1, 1, (0, 255, 0), -1)
            
        cv2.imwrite(str(self.debug_dir / "28_path_preview.png"), img28)
        
        logger.info("Sprint 10 Pipeline completed successfully.")
