import math
from typing import Callable

def linear(t: float) -> float:
    return t

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return t * (2 - t)

def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

def ease_out_cubic(t: float) -> float:
    return (t - 1) ** 3 + 1

def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1

def ease_in_out_expo(t: float) -> float:
    if t == 0 or t == 1:
        return t
    if t < 0.5:
        return 0.5 * math.pow(2, (20 * t) - 10)
    return -0.5 * math.pow(2, (-20 * t) + 10) + 1

def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2

def step(t: float) -> float:
    return 1.0 if t >= 1.0 else 0.0

EASING_FUNCTIONS = {
    "linear": linear,
    "easeInQuad": ease_in_quad,
    "easeOutQuad": ease_out_quad,
    "easeInOutQuad": ease_in_out_quad,
    "easeOutCubic": ease_out_cubic,
    "easeInOutCubic": ease_in_out_cubic,
    "easeInOutExpo": ease_in_out_expo,
    "easeInOutSine": ease_in_out_sine,
    "step": step
}
