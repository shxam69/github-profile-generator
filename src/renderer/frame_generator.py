import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np

from renderer.interpolator import Interpolator

logger = logging.getLogger(__name__)

class FrameGenerator:
    """Computes instantaneous state of all particles at any arbitrary timestamp t."""
    
    def __init__(self, timeline_json: Path, graph_json: Path, logos_jsons: List[Path]):
        self.num_dots = 14248
        
        with open(graph_json, 'r') as f:
            graph_data = json.load(f)["nodes"]
            
        # Map initial coordinates
        self.orig_coords = {n["id"]: (n["x"], n["y"]) for n in graph_data}
        
        # Map logo coordinates
        self.logo_coords = {}
        for idx, lp in enumerate(logos_jsons, 1):
            if lp.exists():
                with open(lp, 'r') as f:
                    pts = json.load(f)
                    self.logo_coords[idx] = {p["id"]: (p["x"], p["y"]) for p in pts}
                    
        # Pre-process events per dot for fast seeking
        with open(timeline_json, 'r') as f:
            self.events = json.load(f)
            
        # Structure: dot_id -> list of chronologically sorted events affecting it
        self.dot_events: Dict[int, List[Dict]] = {i: [] for i in self.orig_coords.keys()}
        
        for e in self.events:
            for d_id in e["affected_ids"]:
                self.dot_events[d_id].append(e)
                
        # State Cache
        # Since we might scrub, keeping the last computed state isn't strictly necessary for a direct query,
        # but we can cache base states. For now, querying is extremely fast anyway (binary search or linear scan of ~4 events).
        
    def get_frame(self, t: float) -> np.ndarray:
        """
        Returns a numpy array of shape (num_dots, 4) representing [x, y, opacity, scale] for all dots at time t.
        """
        state = np.zeros((self.num_dots, 4), dtype=np.float32)
        
        for d_id, (ox, oy) in self.orig_coords.items():
            events = self.dot_events[d_id]
            
            # Default state (before any events)
            curr_x, curr_y = ox, oy
            curr_opacity = 0.0
            curr_scale = 1.0
            
            # Find the active event or the state after all past events
            for e in events:
                if t < e["start_time"]:
                    break # Future event, stop processing
                    
                etype = e["event_type"]
                start = e["start_time"]
                end = e["end_time"]
                meta = e.get("metadata", {})
                easing = e.get("easing", "linear")
                
                # If event is already finished by time t
                if t >= end:
                    progress = 1.0
                else:
                    progress = (t - start) / max(0.001, end - start)
                    
                # Apply event logic
                if etype == "intro_reveal":
                    curr_opacity = Interpolator.interpolate(0.0, 1.0, progress, easing)
                elif etype == "portrait_hold":
                    pass # stays fully visible and at portrait
                elif etype == "drift_start":
                    pass
                elif etype == "drift_band":
                    curr_opacity = Interpolator.interpolate(1.0, 0.0, progress, easing)
                elif etype == "traveller_depart":
                    target = meta["target_point"]
                    tx, ty = self.logo_coords[1][target]
                    curr_x = Interpolator.interpolate(ox, tx, progress, easing)
                    curr_y = Interpolator.interpolate(oy, ty, progress, easing)
                elif etype == "logo_hold":
                    logo_idx = meta["logo"]
                    # wait, if hold, it stays at target. It's already there from previous transition.
                elif etype == "logo_transition":
                    from_pt = meta["from_point"]
                    to_pt = meta["target_point"]
                    from_logo = meta["from_logo"]
                    target_logo = meta["target_logo"]
                    
                    fx, fy = self.logo_coords[from_logo][from_pt]
                    tx, ty = self.logo_coords[target_logo][to_pt]
                    
                    curr_x = Interpolator.interpolate(fx, tx, progress, easing)
                    curr_y = Interpolator.interpolate(fy, ty, progress, easing)
                elif etype == "traveller_return":
                    from_pt = meta["from_point"]
                    to_pt = meta["target_point"]
                    from_logo = meta.get("from_logo", 3)
                    
                    fx, fy = self.logo_coords[from_logo][from_pt]
                    tx, ty = self.orig_coords[to_pt]
                    
                    curr_x = Interpolator.interpolate(fx, tx, progress, easing)
                    curr_y = Interpolator.interpolate(fy, ty, progress, easing)
                elif etype == "portrait_restore":
                    curr_opacity = Interpolator.interpolate(0.0, 1.0, progress, easing)
                    
            state[d_id] = [curr_x, curr_y, curr_opacity, curr_scale]
            
        return state
