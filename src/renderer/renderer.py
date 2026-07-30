import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2

from renderer.frame_generator import FrameGenerator
from renderer.svg_renderer import SVGRenderer

logger = logging.getLogger(__name__)

class Renderer:
    """Orchestrates FrameGenerator and SVGRenderer, providing the high-level API."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.w = 300
        self.h = 340
        
    def render(self, timeline_json: Path, graph_json: Path, logos_jsons: List[Path], output_svg: Path) -> None:
        logger.info("Initializing Render Engine...")
        
        # 1. Initialize Frame Generator (the python simulation engine)
        frame_gen = FrameGenerator(timeline_json, graph_json, logos_jsons)
        
        # 2. Render Debug Frames
        logger.info("Generating debug PNG frames from simulation engine...")
        
        frames = [
            (0.0, "31_frame_000.png"),
            (8.0, "32_frame_middle.png"),
            (18.0, "33_frame_final.png")
        ]
        
        for t, fname in frames:
            # Query state at exact time
            state = frame_gen.get_frame(t)
            
            # Rasterize
            img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            for d in state:
                x, y, opacity, scale = d
                ix, iy = int(x), int(y)
                if 0 <= ix < self.w and 0 <= iy < self.h and opacity > 0:
                    c = int(opacity * 255)
                    img[iy, ix] = [c, c, c]
                    
            cv2.imwrite(str(self.debug_dir / fname), img)
            logger.info(f"Generated frame at t={t}s -> {fname}")
            
        # 3. Export Render Statistics
        stats = {
            "total_dots": frame_gen.num_dots,
            "total_events": len(frame_gen.events),
            "simulated_duration": 20.0
        }
        with open(self.debug_dir / "34_render_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        # 4. Generate Animated SVG via SMIL mapping
        svg_renderer = SVGRenderer(frame_gen, output_svg)
        svg_renderer.generate()
        
        logger.info("Sprint 12 Pipeline completed successfully.")
