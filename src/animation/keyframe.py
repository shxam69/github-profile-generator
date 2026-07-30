from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AnimationKeyframe:
    """Defines a specific state of animation properties at a given normalized time."""
    time: float  # Normalized time from 0.0 to 1.0 within the group's duration
    properties: Dict[str, Any] = field(default_factory=dict)
    easing: str = "linear"  # e.g., 'linear', 'ease-in', 'ease-out', 'ease-in-out', 'spring'
    interpolation: str = "lerp"  # e.g., 'lerp', 'slerp', 'step'
