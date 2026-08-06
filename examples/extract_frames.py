import json
import sys
from pathlib import Path

# Determine the project root directory
project_root = Path(__file__).resolve().parent.parent

# Add 'src' directory to python path
sys.path.append(str(project_root / "src"))

import cv2
import numpy as np
from renderer.frame_generator import FrameGenerator

def main():
    timeline = project_root / "output/compiled_timeline.json"
    graph = project_root / "output/dot_graph.json"
    logos = [
        project_root / "output/logo1_points.json",
        project_root / "output/logo2_points.json",
        project_root / "output/logo3_points.json"
    ]
    fg = FrameGenerator(timeline, graph, logos)
    
    times = [
        (12.0, project_root / "debug/final_logo1_frame.png"), 
        (18.0, project_root / "debug/final_logo2_frame.png"), 
        (24.0, project_root / "debug/final_logo3_frame.png")
    ]
             
    for t, out_path in times:
        print(f"Generating frame at t={t}")
        frame_data = fg.get_frame(t)
        
        img = np.zeros((340, 300, 3), dtype=np.uint8)
        for d in frame_data:
            x, y, op = int(d[0]), int(d[1]), d[2]
            if op > 0.05 and 0 <= x < 300 and 0 <= y < 340:
                # Add points as white
                img[y, x] = [int(255*op), int(255*op), int(255*op)]
                
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
