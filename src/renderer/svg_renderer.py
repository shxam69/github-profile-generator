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
        self.total_dur = max(boundary_key_times) if boundary_key_times else 20.0

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
        w, h = 920, 400

        with open(self.output_svg, 'w', encoding='utf-8') as f:
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

            # Define gradients, filter and clip paths
            f.write('  <defs>\n')
            f.write('    <!-- Purple/Cyan Glow Filter for terminal border -->\n')
            f.write('    <filter id="terminal-glow" x="-10%" y="-10%" width="120%" height="120%">\n')
            f.write('      <feGaussianBlur stdDeviation="4" result="blur" />\n')
            f.write('      <feComponentTransfer in="blur" result="glow">\n')
            f.write('        <feFuncA type="linear" slope="0.3"/>\n')
            f.write('      </feComponentTransfer>\n')
            f.write('      <feMerge>\n')
            f.write('        <feMergeNode in="glow" />\n')
            f.write('        <feMergeNode in="SourceGraphic" />\n')
            f.write('      </feMerge>\n')
            f.write('    </filter>\n')
            f.write('    <!-- Purple/Cyan Neon Gradient for borders and rings -->\n')
            f.write('    <linearGradient id="neon-glow-grad" x1="0%" y1="0%" x2="100%" y2="100%">\n')
            f.write('      <stop offset="0%" stop-color="#a855f7" />\n')
            f.write('      <stop offset="100%" stop-color="#06b6d4" />\n')
            f.write('    </linearGradient>\n')
            f.write('    <!-- PROFILE_ENGINE Clip Path matching the viewport below header line -->\n')
            f.write('    <clipPath id="profileClip">\n')
            f.write('      <rect x="25" y="88" width="320" height="282" />\n')
            f.write('    </clipPath>\n')
            f.write('  </defs>\n')

            # Glow behind the main outer card container
            f.write('  <!-- Neon Glow Border -->\n')
            f.write('  <rect x="10" y="10" width="900" height="380" rx="10" ry="10" fill="none" stroke="url(#neon-glow-grad)" stroke-width="2" filter="url(#terminal-glow)" />\n')
            
            # Solid Application Container
            f.write('  <!-- Main Application Container -->\n')
            f.write('  <rect x="10" y="10" width="900" height="380" rx="10" ry="10" fill="#0d1117" stroke="#30363d" stroke-width="1.2" />\n')
            
            # Application Header (VSCode / macOS top bar)
            f.write('  <!-- Application Header -->\n')
            f.write('  <path d="M 10 20 A 10 10 0 0 1 20 10 L 900 10 A 10 10 0 0 1 910 20 L 910 46 L 10 46 Z" fill="#161b22" />\n')
            f.write('  <line x1="10" y1="46" x2="910" y2="46" stroke="#30363d" stroke-width="1.2" />\n')
            
            # macOS Window Controls (using rounded rects for validator compliance)
            f.write('  <!-- macOS style window controls -->\n')
            f.write('  <rect x="25" y="23" width="8" height="8" rx="4" ry="4" fill="#ff5f56" />\n')
            f.write('  <rect x="37" y="23" width="8" height="8" rx="4" ry="4" fill="#ffbd2e" />\n')
            f.write('  <rect x="49" y="23" width="8" height="8" rx="4" ry="4" fill="#27c93f" />\n')
            
            # Centered Header Title & Subtitle
            f.write('  <text x="460" y="27" fill="#ffffff" font-family="monospace" font-size="10.5" font-weight="bold" text-anchor="middle">SHYAM.DEV</text>\n')
            f.write('  <text x="460" y="38" fill="#8b949e" font-family="monospace" font-size="8" text-anchor="middle">Developer Dashboard</text>\n')
            
            # Right Status Indicator
            f.write('  <!-- Status indicator -->\n')
            f.write('  <rect x="844" y="24" width="6" height="6" rx="3" ry="3" fill="#27c93f" />\n')
            f.write('  <text x="856" y="30" fill="#8b949e" font-family="monospace" font-size="8.5" font-weight="bold">ACTIVE</text>\n')
            
            # ── Left Animation Panel (Inner Frame) ─────────────────────────────
            f.write('  <!-- Left Animation Panel -->\n')
            f.write('  <rect x="25" y="60" width="320" height="310" rx="8" ry="8" fill="#090d13" stroke="#21262d" stroke-width="1.2" />\n')
            f.write('  <text x="37" y="78" fill="#8b949e" font-family="monospace" font-size="8.5" font-weight="bold">PROFILE_ENGINE</text>\n')
            f.write('  <line x1="25" y1="88" x2="345" y2="88" stroke="#21262d" stroke-width="1" />\n')

            # ── Particle Animation Viewport Fitting (Task 6) ──────────────────
            # Wrap in clip path and center original 300x340 coordinates at 0.74 scale
            f.write('  <g clip-path="url(#profileClip)">\n')
            f.write('    <g transform="translate(74, 103.2) scale(0.74)">\n')

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

                f.write(f'      <g opacity="{first_op}">\n')
                f.write(
                    f'        <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" '
                    f'calcMode="spline" keySplines="{op_splines}" dur="{self.total_dur}s" repeatCount="indefinite" />\n'
                )
                f.write(
                    f'        <animateTransform attributeName="transform" type="translate" values="{drift_vals}" '
                    f'dur="{drift_dur}s" repeatCount="indefinite" />\n'
                )
                for cx, cy in dots:
                    f.write(f'        <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" />\n')
                f.write('      </g>\n')

            # Write static backgrounds
            for cx, cy, op in static_backgrounds:
                f.write(f'      <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" opacity="{op}" />\n')

            # Write traveler dots individually
            for cx, cy, op, tags in traveler_elements:
                f.write(f'      <circle cx="{cx}" cy="{cy}" r="1" fill="#E6EDF3" opacity="{op}">\n{tags}      </circle>\n')

            f.write('    </g>\n')
            f.write('  </g>\n')

            # ── Vertical Divider between Left and Right Panels ──────────────────
            f.write('  <!-- Column Divider -->\n')
            f.write('  <line x1="365" y1="46" x2="365" y2="390" stroke="#30363d" stroke-width="1" />\n')
            
            # ── Right Dashboard Panel (Monospace Text) ─────────────────────────
            f.write('  <g font-family="monospace" font-size="9.5" xml:space="preserve">\n')
            
            # Sub-Column 1: Left Info Column
            # IDENTITY Section
            f.write('    <text x="395" y="80" fill="#06b6d4" font-weight="bold">IDENTITY</text>\n')
            f.write('    <line x1="395" y1="84" x2="630" y2="84" stroke="#30363d" stroke-width="1" />\n')
            
            f.write('    <text x="395" y="98" fill="#8b949e">Name</text>\n')
            f.write('    <text x="475" y="98" fill="#ffffff">SHYAM A</text>\n')
            
            f.write('    <text x="395" y="113" fill="#8b949e">Role</text>\n')
            f.write('    <text x="475" y="113" fill="#ffffff">Software Engineer</text>\n')
            
            f.write('    <text x="395" y="128" fill="#8b949e">Status</text>\n')
            f.write('    <text x="475" y="128"><tspan fill="#27c93f">● </tspan><tspan fill="#ffffff">Open to Work</tspan></text>\n')
            
            f.write('    <text x="395" y="143" fill="#8b949e">Location</text>\n')
            f.write('    <text x="475" y="143" fill="#ffffff">Chennai, India</text>\n')

            # EDUCATION Section
            f.write('    <text x="395" y="178" fill="#06b6d4" font-weight="bold">EDUCATION</text>\n')
            f.write('    <line x1="395" y1="182" x2="630" y2="182" stroke="#30363d" stroke-width="1" />\n')
            
            f.write('    <text x="395" y="196" fill="#8b949e">Degree</text>\n')
            f.write('    <text x="475" y="196" fill="#ffffff">B.Tech CSE</text>\n')
            
            f.write('    <text x="395" y="211" fill="#8b949e">College</text>\n')
            f.write('    <text x="475" y="211" fill="#ffffff">SRM University</text>\n')
            
            f.write('    <text x="395" y="226" fill="#8b949e">Grad</text>\n')
            f.write('    <text x="475" y="226" fill="#ffffff">2024</text>\n')

            # CURRENT FOCUS Section
            f.write('    <text x="395" y="261" fill="#06b6d4" font-weight="bold">CURRENT FOCUS</text>\n')
            f.write('    <line x1="395" y1="265" x2="630" y2="265" stroke="#30363d" stroke-width="1" />\n')
            
            f.write('    <text x="395" y="279" fill="#8b949e">AI / ML</text>\n')
            f.write('    <text x="475" y="279" fill="#ffffff">LLMs • RAG • Agents</text>\n')
            
            f.write('    <text x="395" y="294" fill="#8b949e">Backend</text>\n')
            f.write('    <text x="475" y="294" fill="#ffffff">Microservices • APIs</text>\n')
            
            f.write('    <text x="395" y="309" fill="#8b949e">Cloud</text>\n')
            f.write('    <text x="475" y="309" fill="#ffffff">AWS • Docker • K8s</text>\n')

            # Sub-Column 2: Right Tech/Contact Column
            # TECH STACK Section
            f.write('    <text x="660" y="80" fill="#06b6d4" font-weight="bold">TECH STACK</text>\n')
            f.write('    <line x1="660" y1="84" x2="895" y2="84" stroke="#30363d" stroke-width="1" />\n')
            
            f.write('    <text x="660" y="98" fill="#8b949e">Languages</text>\n')
            f.write('    <text x="750" y="98" fill="#ffffff">Python • JS • Java</text>\n')
            
            f.write('    <text x="660" y="113" fill="#8b949e">Frameworks</text>\n')
            f.write('    <text x="750" y="113" fill="#ffffff">React • FastAPI</text>\n')
            
            f.write('    <text x="660" y="128" fill="#8b949e">Databases</text>\n')
            f.write('    <text x="750" y="128" fill="#ffffff">Postgres • Redis</text>\n')
            
            f.write('    <text x="660" y="143" fill="#8b949e">Cloud</text>\n')
            f.write('    <text x="750" y="143" fill="#ffffff">AWS • GCP • Vercel</text>\n')
            
            f.write('    <text x="660" y="158" fill="#8b949e">Tools</text>\n')
            f.write('    <text x="750" y="158" fill="#ffffff">Git • Docker • K8s</text>\n')

            # CONTACT Section
            f.write('    <text x="660" y="196" fill="#06b6d4" font-weight="bold">CONTACT</text>\n')
            f.write('    <line x1="660" y1="200" x2="895" y2="200" stroke="#30363d" stroke-width="1" />\n')
            
            f.write('    <text x="660" y="214" fill="#8b949e">Portfolio</text>\n')
            f.write('    <text x="750" y="214" fill="#ffffff">shyam.dev</text>\n')
            
            f.write('    <text x="660" y="229" fill="#8b949e">GitHub</text>\n')
            f.write('    <text x="750" y="229" fill="#ffffff">github/shxam69</text>\n')
            
            f.write('    <text x="660" y="244" fill="#8b949e">LinkedIn</text>\n')
            f.write('    <text x="750" y="244" fill="#ffffff">linkedin/shxam</text>\n')
            
            f.write('    <text x="660" y="259" fill="#8b949e">Email</text>\n')
            f.write('    <text x="750" y="259" fill="#ffffff">shyam@example.com</text>\n')

            # Bottom Command line with Blinking Cursor
            f.write('    <text x="395" y="355" fill="#58a6ff">$ <tspan fill="#06b6d4">█</tspan><animate attributeName="opacity" values="1;0;1" dur="0.8s" repeatCount="indefinite" /></text>\n')
            
            f.write('  </g>\n')

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


