from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AnimationState:
    """Tracks the current runtime state of an animation sequence."""
    is_playing: bool = False
    current_time: float = 0.0    # Absolute time in seconds since animation started
    progress: float = 0.0        # Normalized progress (0.0 to 1.0)
    current_frame: int = 0
    fps: int = 60
    
    # Global state overrides or properties that might be influenced dynamically
    global_properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.global_properties is None:
            self.global_properties = {}
