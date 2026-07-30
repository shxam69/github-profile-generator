import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from renderer.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)

class SVGRenderer:
    """Generates a standalone SMIL-animated SVG."""
    
    def __init__(self, frame_gen: FrameGenerator, output_svg: Path):
        self.frame_gen = frame_gen
        self.output_svg = output_svg
        self.total_dur = 20.0
        
    def generate(self) -> None:
        logger.info(f"Generating SMIL SVG to {self.output_svg}...")
        
        # Determine all unique keyframe timestamps needed for the global timeline
        # To bake in the easing curves as linear segments in SMIL, we sample every event densely.
        time_set = set([0.0])
        for events in self.frame_gen.dot_events.values():
            for e in events:
                start = e["start_time"]
                end = e["end_time"]
                # Sample 5 points per event to approximate easing curves natively in SVG
                for i in range(6):
                    t = start + (end - start) * (i / 5.0)
                    time_set.add(round(t, 2))
                    
        key_times = sorted(list(time_set))
        self.total_dur = max(key_times)
        
        logger.info(f"Global timeline contains {len(key_times)} distinct keyframes. Total duration: {self.total_dur}s")
        
        # Precompute state at all key_times for all dots
        # frames is shape (len(key_times), 14248, 4)
        frames = []
        for t in key_times:
            frames.append(self.frame_gen.get_frame(t))
            
        frames = np.array(frames) # (T, N, 4)
        
        # Write SVG
        self.output_svg.parent.mkdir(parents=True, exist_ok=True)
        
        w = 300
        h = 340
        
        with open(self.output_svg, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #0d1117;">\n')
            
            for d_id in range(self.frame_gen.num_dots):
                # Extract history for this dot
                dot_history = frames[:, d_id, :]
                xs = dot_history[:, 0]
                ys = dot_history[:, 1]
                ops = dot_history[:, 2]
                
                # Check if it ever changes
                x_changes = np.any(xs != xs[0])
                y_changes = np.any(ys != ys[0])
                op_changes = np.any(ops != ops[0])
                
                cx_str = str(round(xs[0], 1))
                cy_str = str(round(ys[0], 1))
                op_str = str(round(ops[0], 2))
                
                # Format keyTimes (SMIL expects 0 to 1, strictly increasing)
                k_times = []
                for idx, t in enumerate(key_times):
                    if idx == 0:
                        k_times.append("0")
                    elif idx == len(key_times) - 1:
                        k_times.append("1")
                    else:
                        k_times.append(f"{t / self.total_dur:.5f}")
                smil_keyTimes = ";".join(k_times)
                
                # We can optimize strings: if a dot never moves, don't write an animate tag for cx/cy
                animate_tags = ""
                if op_changes:
                    v_str = ";".join([str(round(v, 2)) for v in ops])
                    animate_tags += f'<animate attributeName="opacity" values="{v_str}" keyTimes="{smil_keyTimes}" dur="{self.total_dur}s" repeatCount="indefinite" />'
                    
                if x_changes:
                    v_str = ";".join([str(round(v, 1)) for v in xs])
                    animate_tags += f'<animate attributeName="cx" values="{v_str}" keyTimes="{smil_keyTimes}" dur="{self.total_dur}s" repeatCount="indefinite" />'
                    
                if y_changes:
                    v_str = ";".join([str(round(v, 1)) for v in ys])
                    animate_tags += f'<animate attributeName="cy" values="{v_str}" keyTimes="{smil_keyTimes}" dur="{self.total_dur}s" repeatCount="indefinite" />'
                
                if animate_tags:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}">{animate_tags}</circle>\n')
                else:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}" />\n')
                    
            f.write('</svg>\n')
            
        logger.info(f"SVG completely written. File size: {self.output_svg.stat().st_size / 1024 / 1024:.2f} MB")
