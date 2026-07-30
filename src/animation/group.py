from dataclasses import dataclass, field
from typing import List, Optional
from .keyframe import AnimationKeyframe

@dataclass
class AnimationGroup:
    """A logical grouping of keyframes targeting a specific set of dot IDs."""
    name: str
    target_ids: List[int] = field(default_factory=list)
    keyframes: List[AnimationKeyframe] = field(default_factory=list)
    
    # Timing configurations
    start_time: float = 0.0      # Absolute start time in the timeline (seconds)
    duration: float = 1.0        # Duration of one loop (seconds)
    delay: float = 0.0           # Initial delay before starting (seconds)
    
    # Playback configurations
    repeat: int = 1              # Number of times to repeat (0/1 = play once, -1 = infinite)
    reverse: bool = False        # If True, play backwards or yo-yo if repeating
    easing: str = "linear"       # Group-level easing override
