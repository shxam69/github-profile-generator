from typing import Any
from renderer.easing import EASING_FUNCTIONS

class Interpolator:
    @staticmethod
    def interpolate(start_val: float, end_val: float, progress: float, easing_name: str = "linear") -> float:
        """Interpolates between two values given a progress [0, 1] and an easing function."""
        progress = max(0.0, min(1.0, progress))
        
        easing_fn = EASING_FUNCTIONS.get(easing_name, EASING_FUNCTIONS["linear"])
        eased_progress = easing_fn(progress)
        
        return start_val + (end_val - start_val) * eased_progress
