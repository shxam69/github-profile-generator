import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class AnimationEvent:
    event_id: str
    event_type: str
    start_time: float
    end_time: float
    affected_ids: List[int]
    easing: str
    metadata: Dict[str, Any]

class AnimationCompiler:
    """Compiles individual animation datasets into a unified chronology."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[AnimationEvent] = []
        
    def _add_event(self, event_type: str, start: float, end: float, affected_ids: List[int], easing: str = "linear", metadata: Dict = None) -> None:
        if metadata is None:
            metadata = {}
        event_id = f"evt_{len(self.events)}_{event_type}"
        self.events.append(AnimationEvent(
            event_id=event_id,
            event_type=event_type,
            start_time=round(start, 3),
            end_time=round(end, 3),
            affected_ids=affected_ids,
            easing=easing,
            metadata=metadata
        ))
        
    def compile(self, intro_json: Path, drift_json: Path, travellers_json: Path, graph_json: Path, output_json: Path) -> None:
        logger.info("Starting animation timeline compilation...")
        
        # Load Datasets
        with open(intro_json, 'r') as f:
            intro_groups = json.load(f)
            
        with open(drift_json, 'r') as f:
            drift_bands = json.load(f)
            
        with open(travellers_json, 'r') as f:
            traveller_paths = json.load(f)
            
        with open(graph_json, 'r') as f:
            graph_data = json.load(f)
            all_dot_ids = [n["id"] for n in graph_data["nodes"]]
            
        # 1. Intro Reveal (0.0s to ~3.0s)
        logger.info("Compiling intro_reveal events...")
        max_intro_end = 0.0
        for g in intro_groups:
            start = float(g.get("delay", 0.0))
            end = start + float(g.get("duration", 2.0))
            max_intro_end = max(max_intro_end, end)
            self._add_event("intro_reveal", start, end, g["dot_ids"], easing="easeOutCubic")
            
        # 2. Portrait Hold
        logger.info("Compiling portrait_hold events...")
        hold_start = max_intro_end
        hold_end = hold_start + 3.0  # Increased dwell time
        self._add_event("portrait_hold", hold_start, hold_end, all_dot_ids, easing="easeInOutSine")
        
        # Determine travellers vs background dots
        traveller_ids = [t["portrait_dot"] for t in traveller_paths]
        traveller_set = set(traveller_ids)
        background_dot_ids = [d for d in all_dot_ids if d not in traveller_set]
        
        # 3. Drift Start (background dots dissolve)
        logger.info("Compiling drift_start and drift_band events...")
        drift_start_time = hold_end
        
        # Global drift start trigger
        self._add_event("drift_start", drift_start_time, drift_start_time + 0.1, background_dot_ids, easing="step")
        
        # Stagger bands dissolving
        max_drift_end = 0.0
        for i, band_info in enumerate(drift_bands):
            band_id = band_info["band_id"]
            dot_ids = band_info["dot_ids"]
            # Stagger delays organically
            delay = (i % 10) * 0.15  # Slightly wider stagger
            start = drift_start_time + delay
            end = start + 2.5  # Increased duration, reduces peak velocity
            max_drift_end = max(max_drift_end, end)
            # Remove travellers from drift band so they don't dissolve
            b_dots = [d for d in dot_ids if d not in traveller_set]
            if b_dots:
                self._add_event("drift_band", start, end, b_dots, easing="easeInOutCubic", metadata={"band_id": band_id})
                
        # 4. Traveller Depart (Portrait -> Logo 1)
        logger.info("Compiling traveller depart events...")
        traveller_depart_start = drift_start_time
        traveller_depart_end = traveller_depart_start + 3.0  # Increased duration
        
        for t in traveller_paths:
            # Replaced easeInOutExpo with easeInOutCubic for smoother cinematic trajectory
            self._add_event("traveller_depart", traveller_depart_start, traveller_depart_end, [t["portrait_dot"]], easing="easeInOutCubic", metadata={
                "target_point": t["logo1_point"]
            })
            
        # 5. Logo Hold (Logo 1)
        logo1_hold_start = traveller_depart_end
        logo1_hold_end = logo1_hold_start + 3.5  # Increased dwell time
        self._add_event("logo_hold", logo1_hold_start, logo1_hold_end, traveller_ids, easing="easeInOutSine", metadata={"logo": 1})
        
        # 6. Logo Transition (Logo 1 -> Logo 2)
        logo1_trans_start = logo1_hold_end
        logo1_trans_end = logo1_trans_start + 2.5  # Increased duration
        for t in traveller_paths:
            self._add_event("logo_transition", logo1_trans_start, logo1_trans_end, [t["portrait_dot"]], easing="easeInOutCubic", metadata={
                "from_point": t["logo1_point"],
                "target_point": t["logo2_point"],
                "from_logo": 1,
                "target_logo": 2
            })
            
        # 7. Logo Hold (Logo 2)
        logo2_hold_start = logo1_trans_end
        logo2_hold_end = logo2_hold_start + 3.5  # Increased dwell time
        self._add_event("logo_hold", logo2_hold_start, logo2_hold_end, traveller_ids, easing="easeInOutSine", metadata={"logo": 2})
        
        # 8. Logo Transition (Logo 2 -> Logo 3)
        logo2_trans_start = logo2_hold_end
        logo2_trans_end = logo2_trans_start + 2.5  # Increased duration
        for t in traveller_paths:
            self._add_event("logo_transition", logo2_trans_start, logo2_trans_end, [t["portrait_dot"]], easing="easeInOutCubic", metadata={
                "from_point": t["logo2_point"],
                "target_point": t["logo3_point"],
                "from_logo": 2,
                "target_logo": 3
            })
            
        # 9. Logo Hold (Logo 3)
        logo3_hold_start = logo2_trans_end
        logo3_hold_end = logo3_hold_start + 3.5  # Increased dwell time
        self._add_event("logo_hold", logo3_hold_start, logo3_hold_end, traveller_ids, easing="easeInOutSine", metadata={"logo": 3})
        
        # 10. Traveller Return (Logo 3 -> Portrait)
        return_start = logo3_hold_end
        return_end = return_start + 3.0  # Increased duration
        for t in traveller_paths:
            self._add_event("traveller_return", return_start, return_end, [t["portrait_dot"]], easing="easeInOutCubic", metadata={
                "from_point": t["logo3_point"],
                "target_point": t["return_dot"],
                "from_logo": 3
            })
            
        # 11. Portrait Restore (Drift bands returning)
        restore_start = return_start
        for i, band_info in enumerate(drift_bands):
            band_id = band_info["band_id"]
            dot_ids = band_info["dot_ids"]
            delay = (i % 10) * 0.15
            start = restore_start + delay
            end = start + 2.5
            b_dots = [d for d in dot_ids if d not in traveller_set]
            if b_dots:
                self._add_event("portrait_restore", start, end, b_dots, easing="easeInOutCubic", metadata={"band_id": band_id})
                
        # Sort chronologically
        self.events.sort(key=lambda e: e.start_time)
        
        # Export compiled timeline
        logger.info(f"Exporting {len(self.events)} compiled events to {output_json}")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        events_dicts = [asdict(e) for e in self.events]
        with open(output_json, 'w') as f:
            json.dump(events_dicts, f, indent=2)
            
        # Generate Debug Outputs
        logger.info("Generating Sprint 11 debug visualizations...")
        
        # Event Summary
        summary = {}
        for e in self.events:
            t = e.event_type
            if t not in summary:
                summary[t] = {"count": 0, "min_time": 999.0, "max_time": 0.0}
            summary[t]["count"] += 1
            summary[t]["min_time"] = min(summary[t]["min_time"], e.start_time)
            summary[t]["max_time"] = max(summary[t]["max_time"], e.end_time)
            
        with open(self.debug_dir / "30_event_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Timeline Visualization JSON (Gantt style array)
        gantt = []
        for t, data in summary.items():
            gantt.append({
                "phase": t,
                "start": data["min_time"],
                "end": data["max_time"],
                "event_count": data["count"]
            })
            
        with open(self.debug_dir / "29_timeline_visualization.json", 'w') as f:
            json.dump(gantt, f, indent=2)
            
        logger.info("Sprint 11 Pipeline completed successfully.")
