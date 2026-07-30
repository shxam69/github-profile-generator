import json
import logging
import math
import random
import heapq
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class DriftBandGenerator:
    """Generates organic, balanced drift bands across the dot graph using Noisy Balanced Dijkstra Region Growing."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, graph_json_path: Path, output_json: Path, num_bands: int, noise_level: float, balance_weight: float) -> None:
        logger.info(f"Loading graph dots from {graph_json_path}")
        
        if not graph_json_path.exists():
            raise FileNotFoundError(f"Missing graph JSON file: {graph_json_path}")
            
        with open(graph_json_path, 'r') as f:
            graph_data = json.load(f)
            
        nodes = graph_data["nodes"]
        dot_ids = [n["id"] for n in nodes]
        
        if not dot_ids:
            logger.warning("No dots found in graph. Skipping drift band generation.")
            return
            
        # Parse graph structures
        adjacency: Dict[int, List[int]] = {n["id"]: n["neighbors"] for n in nodes}
        node_coords: Dict[int, Tuple[int, int]] = {n["id"]: (n["x"], n["y"]) for n in nodes}
        
        width = max(x for x, y in node_coords.values()) + 1
        height = max(y for x, y in node_coords.values()) + 1
        
        # 1. Select initial seeds (using simple spatial grids to ensure decent spacing, or random)
        logger.info(f"Selecting {num_bands} seed nodes for bands...")
        # A simple furthest point heuristic or random works well. 
        # For simplicity and organic looks, we randomly shuffle and take 94 spaced out seeds.
        # Even better: K-Means initialization on coords to get exactly `num_bands` well-spaced seeds.
        coords_array = np.array([node_coords[d] for d in dot_ids], dtype=np.float32)
        # We can just use cv2.kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(coords_array, num_bands, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        seed_ids = []
        # Snap centers to nearest dot
        for center in centers:
            cx, cy = center
            closest_dot = min(dot_ids, key=lambda d: (node_coords[d][0] - cx)**2 + (node_coords[d][1] - cy)**2)
            seed_ids.append(closest_dot)
            
        # 2. Add Positional Noise to Coordinates
        logger.info(f"Applying positional noise (sigma={noise_level})...")
        noisy_coords = {}
        random.seed(42) # Reproducible noise
        for d in dot_ids:
            x, y = node_coords[d]
            noisy_coords[d] = (x + random.gauss(0, noise_level), y + random.gauss(0, noise_level))
            
        # 3. Multi-source Dijkstra Region Growing
        logger.info(f"Growing bands via Noisy Balanced Dijkstra (balance_weight={balance_weight})...")
        pq = []
        band_assignments: Dict[int, int] = {}
        band_sizes = {i: 0 for i in range(num_bands)}
        
        # Priority Queue holds: (cost, band_id, current_node_id)
        for b_id, s_id in enumerate(seed_ids):
            heapq.heappush(pq, (0.0, b_id, s_id))
            
        while pq:
            cost, b_id, curr = heapq.heappop(pq)
            
            if curr in band_assignments:
                continue
                
            # Assign
            band_assignments[curr] = b_id
            band_sizes[b_id] += 1
            
            # Expand
            for neighbor in adjacency[curr]:
                if neighbor not in band_assignments:
                    # Edge cost based on NOISY coordinates
                    dx = noisy_coords[curr][0] - noisy_coords[neighbor][0]
                    dy = noisy_coords[curr][1] - noisy_coords[neighbor][1]
                    dist = math.hypot(dx, dy)
                    
                    # Balance penalty slows down rapidly growing bands
                    penalty = band_sizes[b_id] * balance_weight
                    
                    new_cost = cost + dist + penalty
                    heapq.heappush(pq, (new_cost, b_id, neighbor))
                    
        # Ensure all nodes are assigned (handle disconnected components)
        unassigned = [d for d in dot_ids if d not in band_assignments]
        if unassigned:
            logger.warning(f"Found {len(unassigned)} unassigned dots (disconnected components). Resolving via spatial proximity...")
            for u in unassigned:
                ux, uy = node_coords[u]
                best_dist = float('inf')
                best_band = 0
                for a_id, b_id in band_assignments.items():
                    ax, ay = node_coords[a_id]
                    dist = (ux - ax)**2 + (uy - ay)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_band = b_id
                band_assignments[u] = best_band
                band_sizes[best_band] += 1
                    
        # Group final dots by band
        bands: Dict[int, List[int]] = {i: [] for i in range(num_bands)}
        for d, b_id in band_assignments.items():
            bands[b_id].append(d)
            
        # 4. Compute Statistics
        logger.info("Computing band statistics...")
        sizes = [len(dots) for dots in bands.values()]
        avg_size = float(np.mean(sizes))
        min_size = min(sizes)
        max_size = max(sizes)
        
        # Compactness and Smoothness
        internal_edges = 0
        boundary_edges = 0
        total_edges = 0
        boundary_nodes = set()
        
        for curr in dot_ids:
            curr_band = band_assignments[curr]
            is_boundary = False
            for neighbor in adjacency[curr]:
                total_edges += 1
                if band_assignments[neighbor] == curr_band:
                    internal_edges += 1
                else:
                    boundary_edges += 1
                    is_boundary = True
            if is_boundary:
                boundary_nodes.add(curr)
                
        # Compactness: % of edges that are purely internal
        compactness = internal_edges / max(1, total_edges)
        
        # Smoothness proxy: lower boundary nodes relative to total nodes means smoother, larger regions
        # If boundary is extremely fractal/jagged, there are many boundary nodes.
        # Normalized inverse metric.
        boundary_smoothness = 1.0 - (len(boundary_nodes) / len(dot_ids))
        
        stats = {
            "average_size": round(avg_size, 2),
            "min_size": min_size,
            "max_size": max_size,
            "compactness": round(compactness, 4),
            "boundary_smoothness": round(boundary_smoothness, 4)
        }
        
        logger.info(f"Band Stats: Avg={stats['average_size']}, Min={stats['min_size']}, Max={stats['max_size']}, Comp={stats['compactness']}")
        
        stats_path = self.debug_dir / "21_band_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # 5. Export JSON Payload
        logger.info(f"Exporting drift bands to {output_json}...")
        output_payload = []
        for b_id, dots in bands.items():
            output_payload.append({
                "band_id": b_id,
                "dot_ids": dots
            })
            
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(output_payload, f, indent=2)
            
        # 6. Debug Visualizations
        logger.info("Generating Sprint 8 debug visualizations...")
        
        # 19_band_map.png
        map_img = np.zeros((height, width, 3), dtype=np.uint8)
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(num_bands, 3), dtype=np.uint8)
        
        for curr, b_id in band_assignments.items():
            x, y = node_coords[curr]
            map_img[y, x] = colors[b_id]
        cv2.imwrite(str(self.debug_dir / "19_band_map.png"), map_img)
        
        # 20_band_boundaries.png
        bounds_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background faint
        for curr in dot_ids:
            x, y = node_coords[curr]
            bounds_img[y, x] = [30, 30, 30]
            
        # Highlight boundaries in bright cyan
        for curr in boundary_nodes:
            x, y = node_coords[curr]
            bounds_img[y, x] = [255, 255, 0] # Cyan in BGR
            
        cv2.imwrite(str(self.debug_dir / "20_band_boundaries.png"), bounds_img)
        
        logger.info("Sprint 8 Pipeline completed successfully.")
