import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from renderer.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)

# Maps the "easing" label written into compiled_timeline.json by the compiler
# onto SMIL keySplines control points (cubic-bezier form, same convention as
# CSS transition-timing-function). Unrecognized/absent easing and flat
# (non-moving) segments fall back to the identity/linear spline -- safe for a
# flat segment since the interpolated value doesn't change regardless of the
# curve shape applied to it.
EASING_TO_SPLINE = {
    "easeInOutCubic": "0.65 0 0.35 1",
    "easeInOutSine": "0.37 0 0.63 1",
    "easeInOutQuad": "0.45 0 0.55 1",
    "linear": "0 0 1 1",
}
DEFAULT_SPLINE = "0 0 1 1"


class SVGRenderer:
    """Generates a standalone SMIL-animated SVG."""

    def __init__(self, frame_gen: FrameGenerator, output_svg: Path):
        self.frame_gen = frame_gen
        self.output_svg = output_svg
        self.total_dur = 20.0

    def generate(self) -> None:
        logger.info(f"Generating SMIL SVG to {self.output_svg}...")

        # Global time grid = event BOUNDARIES only (start_time, end_time) --
        # not 6 approximation samples per event. Every event contributes
        # exactly 2 candidate times regardless of duration or how many other
        # events happen to overlap it, so grid density no longer spikes
        # wherever many travellers' transitions happen to cluster. The eased
        # shape between two boundary times is reconstructed natively by the
        # browser via calcMode="spline" below, not approximated with extra
        # intermediate samples.
        time_set = set([0.0])
        for events in self.frame_gen.dot_events.values():
            for e in events:
                time_set.add(round(e["start_time"], 2))
                time_set.add(round(e["end_time"], 2))

        key_times = sorted(time_set)
        self.total_dur = max(key_times) if key_times else self.total_dur

        logger.info(f"Global timeline contains {len(key_times)} distinct event-boundary keyframes. Total duration: {self.total_dur}s")

        # Precompute state at all key_times for all dots (same mechanism as
        # before -- FrameGenerator.get_frame is untouched).
        frames = []
        for t in key_times:
            frames.append(self.frame_gen.get_frame(t))
        frames = np.array(frames)  # (T, N, 4)

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

            def build_track(d_id: int, values: np.ndarray, decimals: int):
                """Keep only the timestamps that are actual event boundaries
                for THIS dot (not every global grid point whose rounded value
                differs from its predecessor). Each kept-to-kept interval gets
                its own keySplines entry, taken from whichever event owns
                that interval."""
                dot_events = self.frame_gen.dot_events[d_id]
                dot_boundary_times = {0.0, self.total_dur}
                for e in dot_events:
                    dot_boundary_times.add(round(e["start_time"], 2))
                    dot_boundary_times.add(round(e["end_time"], 2))

                keep = [i for i, t in enumerate(key_times) if t in dot_boundary_times]
                if len(keep) < 2:
                    return None, None, None

                rounded = [round(values[i], decimals) for i in keep]
                if all(v == rounded[0] for v in rounded):
                    return None, None, None

                v_str = ";".join(f"{v:g}" for v in rounded)
                t_str = ";".join(global_k_times[i] for i in keep)

                kept_times = [key_times[i] for i in keep]
                splines = []
                for a, b in zip(kept_times[:-1], kept_times[1:]):
                    matched_easing = None
                    for e in dot_events:
                        if abs(e["start_time"] - a) < 0.01 and abs(e["end_time"] - b) < 0.01:
                            matched_easing = e.get("easing", "linear")
                            break
                    splines.append(EASING_TO_SPLINE.get(matched_easing, DEFAULT_SPLINE))

                return v_str, t_str, ";".join(splines)

            for d_id in range(self.frame_gen.num_dots):
                dot_history = frames[:, d_id, :]
                xs = dot_history[:, 0]
                ys = dot_history[:, 1]
                ops = dot_history[:, 2]

                cx_str = f"{round(xs[0], 1):g}"
                cy_str = f"{round(ys[0], 1):g}"
                op_str = f"{round(ops[0], 2):g}"

                animate_tags = ""

                op_vals, op_times, op_splines = build_track(d_id, ops, 2)
                if op_vals:
                    animate_tags += (
                        f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" '
                        f'calcMode="spline" keySplines="{op_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    )

                x_vals, x_times, x_splines = build_track(d_id, xs, 1)
                if x_vals:
                    animate_tags += (
                        f'    <animate attributeName="cx" values="{x_vals}" keyTimes="{x_times}" '
                        f'calcMode="spline" keySplines="{x_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    )

                y_vals, y_times, y_splines = build_track(d_id, ys, 1)
                if y_vals:
                    animate_tags += (
                        f'    <animate attributeName="cy" values="{y_vals}" keyTimes="{y_times}" '
                        f'calcMode="spline" keySplines="{y_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                    )

                if animate_tags:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}">\n{animate_tags}  </circle>\n')
                else:
                    f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="#c9d1d9" opacity="{op_str}" />\n')

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
