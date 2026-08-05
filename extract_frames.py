import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path("src").resolve()))

import cv2
import numpy as np
from src.renderer.frame_generator import FrameGenerator

def main():
    timeline = Path("output/compiled_timeline.json")
    graph = Path("output/dot_graph.json")
    logos = [Path("output/logo1_points.json"), Path("output/logo2_points.json"), Path("output/logo3_points.json")]
    fg = FrameGenerator(timeline, graph, logos)
    
    times = [(12.0, "debug/final_logo1_frame.png"), 
             (18.0, "debug/final_logo2_frame.png"), 
             (24.0, "debug/final_logo3_frame.png")]
             
    for t, out_path in times:
        print(f"Generating frame at t={t}")
        frame_data = fg.get_frame(t)
        
        img = np.zeros((340, 300, 3), dtype=np.uint8)
        for d in frame_data:
            x, y, op = int(d[0]), int(d[1]), d[2]
            if op > 0.05 and 0 <= x < 300 and 0 <= y < 340:
                # Add points as white
                img[y, x] = [int(255*op), int(255*op), int(255*op)]
                
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
