import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    id: int
    neighbors: List[int]
    degree: int
    cluster: int
    edge_distance: int
    density: float
    graph_depth: int
    x: int
    y: int

class GraphBuilder:
    """Builds a connected graph representation from isolated Dot objects."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def build(self, dots_json_path: Path, output_graph_path: Path) -> None:
        """Reads dots.json, connects 8-way neighbors, computes graph metadata, and saves JSON."""
        logger.info(f"Loading dots from {dots_json_path}")
        
        if not dots_json_path.exists():
            raise FileNotFoundError(f"Missing dots JSON file: {dots_json_path}")
            
        with open(dots_json_path, 'r') as f:
            dots_data = json.load(f)
            
        logger.info(f"Loaded {len(dots_data)} dots. Initializing node maps...")
        
        # Build spatial map to find neighbors via coordinates
        spatial_map: Dict[Tuple[int, int], int] = {}
        node_coords: Dict[int, Tuple[int, int]] = {}
        
        for d in dots_data:
            node_id = d["id"]
            x, y = d["x"], d["y"]
            spatial_map[(x, y)] = node_id
            node_coords[node_id] = (x, y)
            
        # Determine canvas size for debug visualizations
        max_x = max([x for x, y in node_coords.values()]) if node_coords else 0
        max_y = max([y for x, y in node_coords.values()]) if node_coords else 0
        width = max_x + 1
        height = max_y + 1
            
        # 1. Connect neighbors (8-connectivity)
        logger.info("Building 8-connectivity adjacency lists...")
        adjacency_list: Dict[int, List[int]] = {n: [] for n in node_coords}
        
        for node_id, (x, y) in node_coords.items():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in spatial_map:
                        neighbor_id = spatial_map[(nx, ny)]
                        adjacency_list[node_id].append(neighbor_id)
                        
        # 2. Compute Clusters (Connected Components via BFS)
        logger.info("Traversing connected components...")
        clusters: Dict[int, int] = {}
        cluster_id = 0
        
        for node_id in node_coords:
            if node_id not in clusters:
                q = deque([node_id])
                clusters[node_id] = cluster_id
                
                while q:
                    curr = q.popleft()
                    for neighbor in adjacency_list[curr]:
                        if neighbor not in clusters:
                            clusters[neighbor] = cluster_id
                            q.append(neighbor)
                cluster_id += 1
                
        # 3. Compute Edge Distance (Multi-source BFS from edges)
        logger.info("Computing edge distances...")
        edge_distance: Dict[int, int] = {}
        q_edges = deque()
        
        # Any node with degree < 8 is an edge node
        for node_id, neighbors in adjacency_list.items():
            if len(neighbors) < 8:
                edge_distance[node_id] = 0
                q_edges.append(node_id)
                
        while q_edges:
            curr = q_edges.popleft()
            curr_dist = edge_distance[curr]
            
            for neighbor in adjacency_list[curr]:
                if neighbor not in edge_distance:
                    edge_distance[neighbor] = curr_dist + 1
                    q_edges.append(neighbor)
                    
        # 4. Compute Graph Depth (BFS from cluster centroids)
        logger.info("Computing graph depth from cluster centroids...")
        graph_depth: Dict[int, int] = {}
        cluster_nodes = {i: [] for i in range(cluster_id)}
        
        for node_id, cid in clusters.items():
            cluster_nodes[cid].append(node_id)
            
        q_depth = deque()
        for cid, nodes in cluster_nodes.items():
            if not nodes:
                continue
                
            cx = sum(node_coords[n][0] for n in nodes) / len(nodes)
            cy = sum(node_coords[n][1] for n in nodes) / len(nodes)
            
            root_node = min(nodes, key=lambda n: (node_coords[n][0] - cx)**2 + (node_coords[n][1] - cy)**2)
            
            graph_depth[root_node] = 0
            q_depth.append(root_node)
            
        while q_depth:
            curr = q_depth.popleft()
            curr_depth = graph_depth[curr]
            for neighbor in adjacency_list[curr]:
                if neighbor not in graph_depth:
                    graph_depth[neighbor] = curr_depth + 1
                    q_depth.append(neighbor)
                    
        # 5. Build Final Node Objects
        logger.info("Constructing final GraphNode representations...")
        graph_nodes = []
        for d in dots_data:
            n_id = d["id"]
            neighbors = adjacency_list[n_id]
            degree = len(neighbors)
            density = round(degree / 8.0, 2)
            
            gn = GraphNode(
                id=n_id,
                neighbors=neighbors,
                degree=degree,
                cluster=clusters[n_id],
                edge_distance=edge_distance.get(n_id, 0),
                density=density,
                graph_depth=graph_depth.get(n_id, 0),
                x=d["x"],
                y=d["y"]
            )
            graph_nodes.append(gn)
            
        # 6. Export to JSON
        logger.info(f"Exporting graph metadata to {output_graph_path}...")
        output_data = {
            "nodes": [
                {
                    "id": n.id,
                    "x": n.x,
                    "y": n.y,
                    "neighbors": n.neighbors,
                    "degree": n.degree,
                    "cluster": n.cluster,
                    "edge_distance": n.edge_distance,
                    "density": n.density,
                    "graph_depth": n.graph_depth
                } for n in graph_nodes
            ]
        }
        
        output_graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_graph_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        # 7. Generate Debug Visualizations
        logger.info("Generating graph visualization debug outputs...")
        
        # 13_graph_overlay.png
        overlay_img = np.zeros((height, width, 3), dtype=np.uint8)
        for n in graph_nodes:
            # White for nodes, maybe color edges if we wanted but since it's 8-connected adjacent pixels 
            # drawing the pixels effectively draws the graph
            overlay_img[n.y, n.x] = [200, 200, 200]
        cv2.imwrite(str(self.debug_dir / "13_graph_overlay.png"), overlay_img)
        
        # 14_degree_heatmap.png
        degree_img = np.zeros((height, width), dtype=np.uint8)
        for n in graph_nodes:
            degree_img[n.y, n.x] = int((n.degree / 8.0) * 255)
        degree_colored = cv2.applyColorMap(degree_img, cv2.COLORMAP_MAGMA)
        degree_colored[degree_img == 0] = [0, 0, 0] # Background transparent-ish
        cv2.imwrite(str(self.debug_dir / "14_degree_heatmap.png"), degree_colored)
        
        # 15_graph_components.png
        comp_img = np.zeros((height, width, 3), dtype=np.uint8)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(cluster_id + 1, 3), dtype=np.uint8)
        for n in graph_nodes:
            comp_img[n.y, n.x] = colors[n.cluster]
        cv2.imwrite(str(self.debug_dir / "15_graph_components.png"), comp_img)
        
        logger.info("Sprint 5 Graph Engine completed successfully.")
