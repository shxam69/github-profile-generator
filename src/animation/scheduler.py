import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List
from .timeline import AnimationTimeline

logger = logging.getLogger(__name__)

class AnimationScheduler:
    """Manages the scheduling, timing, and orchestration of multiple animation timelines."""
    
    def __init__(self) -> None:
        self.timelines: List[AnimationTimeline] = []
        
    def add_timeline(self, timeline: AnimationTimeline) -> None:
        """Registers a timeline with the scheduler."""
        timeline.calculate_duration()
        self.timelines.append(timeline)
        logger.info(f"Added timeline '{timeline.name}' with duration {timeline.duration}s")
        
    def export_debug(self, output_path: Path) -> None:
        """Exports the entire scheduled animation structure to JSON for debugging and validation."""
        logger.info(f"Exporting animation scheduler structure to {output_path}")
        
        # Serialize the timelines using dataclasses asdict
        data = [asdict(t) for t in self.timelines]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info("Scheduler debug export completed.")
