import logging
import math
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from renderer.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)

# Maps the "easing" label written into compiled_timeline.json by the compiler
# onto SMIL keySplines control points (cubic-bezier form, same convention as
# CSS transition-timing-function). Unrecognized/absent easing and flat
# (non-moving) segments fall back to the identity/linear spline.
EASING_TO_SPLINE = {
    "easeInOutCubic": "0.85 0 0.15 1",  # Snappy, high-acceleration cinematic easing (Task 4)
    "easeInOutSine": "0.37 0 0.63 1",
    "easeInOutQuad": "0.45 0 0.55 1",
    "easeOutCubic": "0.25 1 0.5 1",      # Snappy and smooth ease-out for intro shimmer (Task 4)
    "linear": "0 0 1 1",
}
DEFAULT_SPLINE = "0 0 1 1"


class SVGRenderer:
    """Generates a standalone SMIL-animated SVG."""

    # ── adaptive temporal sampling thresholds ─────────────────────────────
    # Per-event start->end travel distance (HR-space px, same units as
    # FrameGenerator positions) below which no interior samples are added.
    SHORT_MOVE_PX = 15.0
    # Above this distance the move gets 2 interior samples (thirds) instead
    # of 1 (midpoint).
    LONG_MOVE_PX = 60.0
    # After adding candidate interior samples, drop any whose x/y value is
    # within this tolerance (px) of the straight-line interpolation between
    # its two surviving neighbours.
    SIMPLIFY_TOL_PX = 0.3

    def __init__(self, frame_gen: FrameGenerator, output_svg: Path):
        self.frame_gen = frame_gen
        self.output_svg = output_svg
        self.total_dur = 20.0

    def generate(self) -> None:
        logger.info(f"Generating SMIL SVG to {self.output_svg}...")

        # ── Pass 1: boundary-only global grid ──────────────────────────────
        boundary_time_set = set([0.0])
        for events in self.frame_gen.dot_events.values():
            for e in events:
                boundary_time_set.add(round(e["start_time"], 2))
                boundary_time_set.add(round(e["end_time"], 2))

        boundary_key_times = sorted(boundary_time_set)
        self.total_dur = max(boundary_key_times) if boundary_key_times else self.total_dur

        boundary_frames = np.array([self.frame_gen.get_frame(t) for t in boundary_key_times])  # (T0, N, 4)
        bt_index = {t: i for i, t in enumerate(boundary_key_times)}

        # ── Pass 2: adaptive resampling ─────────────────────────────────────
        time_set = set(boundary_key_times)
        per_dot_extra_times: Dict[int, set] = {d_id: set() for d_id in self.frame_gen.dot_events}

        for d_id, events in self.frame_gen.dot_events.items():
            for e in events:
                t0 = round(e["start_time"], 2)
                t1 = round(e["end_time"], 2)
                if t0 == t1 or t0 not in bt_index or t1 not in bt_index:
                    continue

                p0 = boundary_frames[bt_index[t0], d_id, :2]
                p1 = boundary_frames[bt_index[t1], d_id, :2]
                dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))

                if dist < self.SHORT_MOVE_PX:
                    fracs: List[float] = []
                elif dist < self.LONG_MOVE_PX:
                    fracs = [0.5]
                else:
                    fracs = [1.0 / 3.0, 2.0 / 3.0]

                for frac in fracs:
                    t_extra = round(t0 + (t1 - t0) * frac, 2)
                    if t0 < t_extra < t1:
                        time_set.add(t_extra)
                        per_dot_extra_times[d_id].add(t_extra)

        key_times = sorted(time_set)
        logger.info(
            f"Global timeline contains {len(boundary_key_times)} event-boundary keyframes "
            f"+ {len(key_times) - len(boundary_key_times)} adaptive interior samples "
            f"= {len(key_times)} total. Total duration: {self.total_dur}s"
        )

        # Precompute state at every key_time
        frames = np.array([self.frame_gen.get_frame(t) for t in key_times])  # (T, N, 4)

        # Determine traveler dots (Task 3 / Task 5 / Task 1)
        travelers = set()
        for events in self.frame_gen.dot_events.values():
            for e in events:
                if e["event_type"] in ("traveller_depart", "logo_transition", "traveller_return"):
                    travelers.update(e["affected_ids"])

        self.output_svg.parent.mkdir(parents=True, exist_ok=True)
        w, h = 300, 340

        with open(self.output_svg, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #0d1117;">\n')

            global_k_times = []
            for idx, t in enumerate(key_times):
                if idx == 0:
                    global_k_times.append("0")
                elif idx == len(key_times) - 1:
                    global_k_times.append("1")
                else:
                    global_k_times.append(f"{t / self.total_dur:.5f}")

            def build_track(d_id: int, values: np.ndarray, decimals: int, is_position: bool = False):
                dot_events = self.frame_gen.dot_events[d_id]
                dot_boundary_times = {0.0, self.total_dur}
                for e in dot_events:
                    dot_boundary_times.add(round(e["start_time"], 2))
                    dot_boundary_times.add(round(e["end_time"], 2))
                if is_position:
                    dot_boundary_times |= per_dot_extra_times.get(d_id, set())

                keep = [i for i, t in enumerate(key_times) if t in dot_boundary_times]
                if len(keep) < 2:
                    return None, None, None

                rounded = [round(values[i], decimals) for i in keep]
                if all(v == rounded[0] for v in rounded):
                    return None, None, None

                # ── redundant-keyframe removal ──────────────────────────────
                if is_position and len(keep) > 2:
                    kept_t = [key_times[i] for i in keep]
                    simplified = [0]
                    i = 1
                    while i < len(keep) - 1:
                        t_prev = kept_t[simplified[-1]]
                        v_prev = rounded[simplified[-1]]
                        t_next = kept_t[i + 1]
                        v_next = rounded[i + 1]
                        span = t_next - t_prev
                        if span > 1e-6:
                            predicted = v_prev + (v_next - v_prev) * (kept_t[i] - t_prev) / span
                            if abs(rounded[i] - predicted) < self.SIMPLIFY_TOL_PX:
                                i += 1
                                continue
                        simplified.append(i)
                        i += 1
                    simplified.append(len(keep) - 1)
                    keep = [keep[j] for j in simplified]
                    rounded = [rounded[j] for j in simplified]

                v_str = ";".join(f"{v:g}" for v in rounded)
                t_str = ";".join(global_k_times[i] for i in keep)

                kept_times = [key_times[i] for i in keep]
                splines = []
                for a, b in zip(kept_times[:-1], kept_times[1:]):
                    matched_easing = None
                    for e in dot_events:
                        if e["start_time"] - 0.01 <= a and b <= e["end_time"] + 0.01:
                            matched_easing = e.get("easing", "linear")
                            break
                    splines.append(EASING_TO_SPLINE.get(matched_easing, DEFAULT_SPLINE))

                return v_str, t_str, ";".join(splines)

            # Group identical opacity animations to optimize the final DOM (Task 1)
            # Map: (op_vals, op_times, op_splines) -> list of (cx_str, cy_str)
            opacity_groups = {}
            static_backgrounds = []
            traveler_elements = []

            for d_id in range(self.frame_gen.num_dots):
                dot_history = frames[:, d_id, :]
                xs = dot_history[:, 0]
                ys = dot_history[:, 1]
                ops = dot_history[:, 2]

                cx_str = f"{round(xs[0], 1):g}"
                cy_str = f"{round(ys[0], 1):g}"
                op_str = f"{round(ops[0], 2):g}"

                if d_id in travelers:
                    animate_tags = ""
                    op_vals, op_times, op_splines = build_track(d_id, ops, 2, is_position=False)
                    if op_vals:
                        animate_tags += (
                            f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" '
                            f'calcMode="spline" keySplines="{op_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                        )

                    x_vals, x_times, x_splines = build_track(d_id, xs, 1, is_position=True)
                    if x_vals:
                        animate_tags += (
                            f'    <animate attributeName="cx" values="{x_vals}" keyTimes="{x_times}" '
                            f'calcMode="spline" keySplines="{x_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                        )

                    y_vals, y_times, y_splines = build_track(d_id, ys, 1, is_position=True)
                    if y_vals:
                        animate_tags += (
                            f'    <animate attributeName="cy" values="{y_vals}" keyTimes="{y_times}" '
                            f'calcMode="spline" keySplines="{y_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                        )

                    # Smooth color morphing system (Task 3)
                    # Timeline: Portrait -> Dragon -> NX -> </> -> Portrait
                    col_times = [0.0, 9.45, 11.45, 15.45, 17.45, 21.45, 23.45, 27.45, 29.45, self.total_dur]
                    col_vals = ["#E6EDF3", "#E6EDF3", "#DC2626", "#DC2626", "#8B5CF6", "#8B5CF6", "#06B6D4", "#06B6D4", "#E6EDF3", "#E6EDF3"]
                    col_t_str = ";".join(f"{t / self.total_dur:.5f}" for t in col_times)
                    col_v_str = ";".join(col_vals)
                    col_splines = ";".join([
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1"
                    ])
                    animate_tags += (
                        f'    <animate attributeName="fill" values="{col_v_str}" keyTimes="{col_t_str}" '
                        f'calcMode="spline" keySplines="{col_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    )
                    # Smooth radius system for Dragon (Task 2)
                    r_times = [0.0, 9.45, 11.45, 15.45, 17.45, self.total_dur]
                    r_vals = [1.0, 1.0, 1.15, 1.15, 1.0, 1.0]
                    r_t_str = ";".join(f"{t / self.total_dur:.5f}" for t in r_times)
                    r_v_str = ";".join(f"{v:g}" for v in r_vals)
                    r_splines = ";".join([
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1",
                        "0.85 0 0.15 1",
                        "0.37 0 0.63 1"
                    ])
                    animate_tags += (
                        f'    <animate attributeName="r" values="{r_v_str}" keyTimes="{r_t_str}" '
                        f'calcMode="spline" keySplines="{r_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    )
                    traveler_elements.append((cx_str, cy_str, op_str, animate_tags))
                else:
                    op_vals, op_times, op_splines = build_track(d_id, ops, 2, is_position=False)
                    if op_vals:
                        key = (op_vals, op_times, op_splines)
                        if key not in opacity_groups:
                            opacity_groups[key] = []
                        opacity_groups[key].append((cx_str, cy_str))
                    else:
                        static_backgrounds.append((cx_str, cy_str, op_str))

            # Write opacity groups (Task 1) with subtle, organic drift (Task 5)
            for idx, ((op_vals, op_times, op_splines), dots) in enumerate(opacity_groups.items()):
                first_op = op_vals.split(";")[0]

                # Subtle organic drift (translation) applied once on the group element (Task 5 / Task 6)
                drift_amp = 0.45
                drift_dur = round(15.0 + (idx % 7) * 2.1, 1)  # slow, asynchronous cycles
                dx1 = round(drift_amp * math.sin(idx * 1.7), 2)
                dy1 = round(drift_amp * math.cos(idx * 1.3), 2)
                dx2 = round(drift_amp * math.sin(idx * 1.7 + 2.0), 2)
                dy2 = round(drift_amp * math.cos(idx * 1.3 + 2.0), 2)
                dx3 = round(drift_amp * math.sin(idx * 1.7 + 4.0), 2)
                dy3 = round(drift_amp * math.cos(idx * 1.3 + 4.0), 2)
                drift_vals = f"0 0; {dx1} {dy1}; {dx2} {dy2}; {dx3} {dy3}; 0 0"

                f.write(f'  <g opacity="{first_op}">\n')
                f.write(
                    f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" '
                    f'calcMode="spline" keySplines="{op_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                )
                f.write(
                    f'    <animateTransform attributeName="transform" type="translate" values="{drift_vals}" '
                    f'dur="{drift_dur}s" repeatCount="indefinite" />\n'
                )
                for cx, cy in dots:
                    f.write(f'    <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" />\n')
                f.write('  </g>\n')

            # Write static backgrounds
            for cx, cy, op in static_backgrounds:
                f.write(f'  <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" opacity="{op}" />\n')

            # Write traveler dots individually
            for cx, cy, op, tags in traveler_elements:
                f.write(f'  <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" opacity="{op}">\n{tags}  </circle>\n')

            f.write('</svg>\n')

        logger.info(f"SVG completely written. File size: {self.output_svg.stat().st_size / 1024 / 1024:.2f} MB")

        logger.info("Validating SVG XML structure...")
        try:
            import xml.etree.ElementTree as ET
            ET.parse(self.output_svg)
            logger.info("SVG Validation passed. Document is well-formed.")
        except Exception as e:
            logger.error(f"SVG Validation failed! Document is malformed: {e}")
            raise RuntimeError(f"Generated SVG is invalid: {e}")
