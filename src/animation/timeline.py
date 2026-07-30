from dataclasses import dataclass, field
from typing import List
from .group import AnimationGroup

@dataclass
class AnimationTimeline:
    """A master timeline sequence containing multiple animation groups."""
    name: str
    duration: float = 0.0  # Total duration of the timeline (seconds)
    groups: List[AnimationGroup] = field(default_factory=list)
    
    def calculate_duration(self) -> float:
        """Dynamically calculates the total duration based on all sub-groups."""
        max_duration = 0.0
        for group in self.groups:
            # If repeat is infinite (-1), the timeline conceptually runs forever, 
            # but we calculate duration based on finite bounds if we can.
            repeat_count = group.repeat if group.repeat > 0 else 1
            group_end = group.start_time + group.delay + (group.duration * repeat_count)
            if group_end > max_duration:
                max_duration = group_end
        self.duration = max_duration
        return self.duration
