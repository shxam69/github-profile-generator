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
                # Target 30 frames per second for smooth easing approximation
                fps = 15.0
                duration = end - start
                num_samples = max(6, int(duration * fps))
                for i in range(num_samples + 1):
                    t = start + duration * (i / float(num_samples))
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
            
            # Format keyTimes globally for lookup
            global_k_times = []
            for idx, t in enumerate(key_times):
                if idx == 0:
                    global_k_times.append("0")
                elif idx == len(key_times) - 1:
                    global_k_times.append("1")
                else:
                    global_k_times.append(f"{t / self.total_dur:.5f}")
                    
            def optimize_track(values, decimals):
                rounded = [round(v, decimals) for v in values]
                if all(v == rounded[0] for v in rounded):
                    return None, None
                
                keep = []
                N = len(rounded)
                for i in range(N):
                    if i == 0 or i == N - 1:
                        keep.append(i)
                    elif rounded[i] != rounded[i-1] or rounded[i] != rounded[i+1]:
                        keep.append(i)
                        
                keep = sorted(list(set(keep)))
                        
                v_str = ";".join([f"{rounded[i]:g}" for i in keep])
                t_str = ";".join([global_k_times[i] for i in keep])
                return v_str, t_str

            for d_id in range(self.frame_gen.num_dots):
                # Extract history for this dot
                dot_history = frames[:, d_id, :]
                xs = dot_history[:, 0]
                ys = dot_history[:, 1]
                ops = dot_history[:, 2]
                
                cx_str = f"{round(xs[0], 1):g}"
                cy_str = f"{round(ys[0], 1):g}"
                op_str = f"{round(ops[0], 2):g}"
                
                # Optimize strings: if a dot never moves, don't write an animate tag for cx/cy
                animate_tags = ""
                
                op_vals, op_times = optimize_track(ops, 2)
                if op_vals:
                    animate_tags += f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    
                x_vals, x_times = optimize_track(xs, 1)
                if x_vals:
                    animate_tags += f'    <animate attributeName="cx" values="{x_vals}" keyTimes="{x_times}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    
                y_vals, y_times = optimize_track(ys, 1)
                if y_vals:
                    animate_tags += f'    <animate attributeName="cy" values="{y_vals}" keyTimes="{y_times}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                
                if animate_tags:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}">\n{animate_tags}  </circle>\n')
                else:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}" />\n')
                    
            f.write('</svg>\n')
            
        logger.info(f"SVG completely written. File size: {self.output_svg.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Validate the SVG XML structure to catch truncation or malformed tags
        logger.info("Validating SVG XML structure...")
        try:
            import xml.etree.ElementTree as ET
            ET.parse(self.output_svg)
            logger.info("SVG Validation passed. Document is well-formed.")
        except Exception as e:
            logger.error(f"SVG Validation failed! Document is malformed: {e}")
            raise RuntimeError(f"Generated SVG is invalid: {e}")
