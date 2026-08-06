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

        # ── Loop transition smoothing (Task 6) ──────────────────────────────
        # Smoothly blend the final 0.5s of opacity to match the first frame (0.0s)
        fade_start_t = self.total_dur - 0.5
        fade_start_idx = 0
        for idx, t in enumerate(key_times):
            if t <= fade_start_t:
                fade_start_idx = idx
        fade_start_actual_t = key_times[fade_start_idx]
        if self.total_dur - fade_start_actual_t > 0.01:
            for idx in range(fade_start_idx + 1, len(key_times)):
                t = key_times[idx]
                factor = (self.total_dur - t) / (self.total_dur - fade_start_actual_t)
                for d_id in range(self.frame_gen.num_dots):
                    op_start = frames[fade_start_idx, d_id, 2]
                    op_end = frames[0, d_id, 2]
                    frames[idx, d_id, 2] = op_start * factor + op_end * (1.0 - factor)



        # Determine traveler dots (Task 3 / Task 5 / Task 1)
        travelers = set()
        for events in self.frame_gen.dot_events.values():
            for e in events:
                if e["event_type"] in ("traveller_depart", "logo_transition", "traveller_return"):
                    travelers.update(e["affected_ids"])

        self.output_svg.parent.mkdir(parents=True, exist_ok=True)
        w, h = 920, 400

        with open(self.output_svg, 'w', encoding='utf-8') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #0A101F;">\n')

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
                    # Smooth radius system for traveler morphs (Task 2)
                    r_times = [
                        0.0,
                        7.45, 8.45, 9.45,
                        13.45, 14.45, 15.45,
                        19.45, 20.45, 21.45,
                        25.45, 26.45, 27.45,
                        self.total_dur
                    ]
                    r_vals = [
                        1.0,
                        1.0, 1.15, 1.0,
                        1.0, 1.15, 1.0,
                        1.0, 1.15, 1.0,
                        1.0, 1.15, 1.0,
                        1.0
                    ]
                    r_t_str = ";".join(f"{t / self.total_dur:.5f}" for t in r_times)
                    r_v_str = ";".join(f"{v:g}" for v in r_vals)
                    r_splines = ";".join(["0.37 0 0.63 1"] * (len(r_times) - 1))
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
            f.write('    <!-- Subtle Cyan Glow Filter for particles density and soft bloom -->\n')
            f.write('    <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">\n')
            f.write('      <feGaussianBlur stdDeviation="1.2" result="blur" />\n')
            f.write('      <feComponentTransfer in="blur" result="glow">\n')
            f.write('        <feFuncA type="linear" slope="0.6"/>\n')
            f.write('      </feComponentTransfer>\n')
            f.write('      <feMerge>\n')
            f.write('        <feMergeNode in="glow" />\n')
            f.write('        <feMergeNode in="SourceGraphic" />\n')
            f.write('      </feMerge>\n')
            f.write('    </filter>\n')
            f.write('    <!-- Soft Drop Shadow for premium depth -->\n')
            f.write('    <filter id="card-shadow" x="-10%" y="-10%" width="120%" height="120%">\n')
            f.write('      <feDropShadow dx="0" dy="8" stdDeviation="16" flood-color="#000000" flood-opacity="0.6"/>\n')
            f.write('    </filter>\n')
            f.write('    <!-- Subtle Title Glow Pulse Filter -->\n')
            f.write('    <filter id="title-glow" x="-20%" y="-20%" width="140%" height="140%">\n')
            f.write('      <feGaussianBlur stdDeviation="2" result="blur" />\n')
            f.write('      <feComponentTransfer in="blur" result="glow">\n')
            f.write('        <feFuncA type="linear" slope="0.5"/>\n')
            f.write('      </feComponentTransfer>\n')
            f.write('      <feMerge>\n')
            f.write('        <feMergeNode in="glow" />\n')
            f.write('        <feMergeNode in="SourceGraphic" />\n')
            f.write('      </feMerge>\n')
            f.write('    </filter>\n')
            f.write('    <!-- Purple/Cyan Neon Gradient for borders and rings -->\n')
            f.write('    <linearGradient id="neon-glow-grad" x1="0%" y1="0%" x2="100%" y2="100%">\n')
            f.write('      <stop offset="0%" stop-color="#A78BFA" />\n')
            f.write('      <stop offset="100%" stop-color="#22D3EE" />\n')
            f.write('    </linearGradient>\n')
            f.write('    <!-- Premium Translucent Card Background Gradient -->\n')
            f.write('    <linearGradient id="card-bg" x1="0%" y1="0%" x2="100%" y2="100%">\n')
            f.write('      <stop offset="0%" stop-color="#0A101F" />\n')
            f.write('      <stop offset="100%" stop-color="#050810" />\n')
            f.write('    </linearGradient>\n')
            scenes = [
                # Scene 1: Software Engineer
                {
                    "start": 0.0,
                    "end": 7.45,
                    "clear": 9.45,
                    "sections": {
                        "identity": [
                            ("name", "SHYAM A", 475, 98, 60),
                            ("role", "Software Engineer", 475, 113, 110),
                            ("status", "Active Coding", 475, 128, 95, "#10B981"),
                            ("location", "Chennai, India", 475, 143, 95)
                        ],
                        "specialization": [
                            ("core", "Systems Design", 475, 196, 95),
                            ("backend", "Java / Python", 475, 211, 95),
                            ("frontend", "React / TypeScript", 475, 226, 120),
                            ("nextgen", "AI Systems", 475, 241, 75)
                        ],
                        "techstack": [
                            ("lang", "Java • Python • JS", 750, 98, 120),
                            ("backend", "Spring Boot • FastAPI", 750, 113, 145),
                            ("frontend", "React • Tailwind", 750, 128, 110),
                            ("db", "Postgres • Redis", 750, 143, 110),
                            ("devops", "Docker • AWS • Git", 750, 158, 125)
                        ],
                        "building": [
                            ("ai", "ProjectForge AI", 750, 214, 110),
                            ("svg", "Particle Morph Engine", 750, 229, 140),
                            ("platform", "VYROX", 750, 244, 50)
                        ],
                        "terminal": [
                            ("cmd", "$ whoami", 395, 283, 70, "#A78BFA"),
                            ("out", "SHYAM A", 395, 298, 60)
                        ]
                    }
                },
                # Scene 2: Java Full Stack
                {
                    "start": 9.45,
                    "end": 13.45,
                    "clear": 15.45,
                    "sections": {
                        "identity": [
                            ("name", "SHYAM A", 475, 98, 60),
                            ("role", "Java Full Stack", 475, 113, 105),
                            ("status", "Open to Work", 475, 128, 95, "#10B981"),
                            ("location", "Remote / India", 475, 143, 95)
                        ],
                        "specialization": [
                            ("core", "OOP Architecture", 475, 196, 110),
                            ("backend", "Microservices", 475, 211, 95),
                            ("frontend", "Next.js / Angular", 475, 226, 115),
                            ("nextgen", "Cloud Native", 475, 241, 90)
                        ],
                        "techstack": [
                            ("lang", "Java • SQL • TypeScript", 750, 98, 150),
                            ("backend", "Spring Cloud • Hibernate", 750, 113, 160),
                            ("frontend", "React • HTML5 / CSS3", 750, 128, 145),
                            ("db", "MySQL • MongoDB", 750, 143, 110),
                            ("devops", "K8s • Jenkins • Maven", 750, 158, 140)
                        ],
                        "building": [
                            ("ai", "Secured Banking API", 750, 214, 135),
                            ("svg", "Event Bus Broker", 750, 229, 120),
                            ("platform", "Micro-Frontend Hub", 750, 244, 130)
                        ],
                        "terminal": [
                            ("cmd", "$ stack", 395, 283, 60, "#A78BFA"),
                            ("out", "Java, Spring Boot, React", 395, 298, 160)
                        ]
                    }
                },
                # Scene 3: AI Engineer
                {
                    "start": 15.45,
                    "end": 19.45,
                    "clear": 21.45,
                    "sections": {
                        "identity": [
                            ("name", "SHYAM A", 475, 98, 60),
                            ("role", "AI Engineer", 475, 113, 85),
                            ("status", "Researching", 475, 128, 85, "#F59E0B"),
                            ("location", "Labs / Chennai", 475, 143, 95)
                        ],
                        "specialization": [
                            ("core", "Deep Learning", 475, 196, 95),
                            ("backend", "FastAPI / PyTorch", 475, 211, 115),
                            ("frontend", "Streamlit / Gradio", 475, 226, 120),
                            ("nextgen", "Agentic Workflows", 475, 241, 120)
                        ],
                        "techstack": [
                            ("lang", "Python • Mojo • CUDA", 750, 98, 135),
                            ("backend", "FastAPI • LangChain", 750, 113, 130),
                            ("frontend", "React • WebGPU", 750, 128, 95),
                            ("db", "Chroma • pgvector", 750, 143, 120),
                            ("devops", "HuggingFace • RunPod", 750, 158, 135)
                        ],
                        "building": [
                            ("ai", "Fine-tuned LLM", 750, 214, 100),
                            ("svg", "Semantic Search DB", 750, 229, 125),
                            ("platform", "Autonomous Coder", 750, 244, 120)
                        ],
                        "terminal": [
                            ("cmd", "$ building", 395, 283, 85, "#A78BFA"),
                            ("out", "ProjectForge AI, Morph Engine", 395, 298, 180)
                        ]
                    }
                },
                # Scene 4: Open Source Builder
                {
                    "start": 21.45,
                    "end": 25.45,
                    "clear": 27.45,
                    "sections": {
                        "identity": [
                            ("name", "SHYAM A", 475, 98, 60),
                            ("role", "Open Source Builder", 475, 113, 130),
                            ("status", "Collaborating", 475, 128, 95, "#A78BFA"),
                            ("location", "GitHub / Chennai", 475, 143, 110)
                        ],
                        "specialization": [
                            ("core", "Package Dev", 475, 196, 85),
                            ("backend", "Go / Node.js", 475, 211, 85),
                            ("frontend", "Vanilla CSS / SVG", 475, 226, 120),
                            ("nextgen", "Edge Computing", 475, 241, 100)
                        ],
                        "techstack": [
                            ("lang", "JavaScript • Rust • Go", 750, 98, 150),
                            ("backend", "Express • GinGonic", 750, 113, 125),
                            ("frontend", "Svelte • Web Components", 750, 128, 155),
                            ("db", "SQLite • Redis", 750, 143, 95),
                            ("devops", "GitHub Actions • Vercel", 750, 158, 160)
                        ],
                        "building": [
                            ("ai", "Particle Morph SVG", 750, 214, 125),
                            ("svg", "Smil Animation Pack", 750, 229, 135),
                            ("platform", "Markdown Compiler", 750, 244, 125)
                        ],
                        "terminal": [
                            ("cmd", "$ status", 395, 283, 70, "#A78BFA"),
                            ("out", "Available", 395, 298, 65, "#10B981")
                        ]
                    }
                }
            ]

            # Write clipPaths for all scenes and rows
            for s_idx, scene in enumerate(scenes, 1):
                start = scene["start"]
                end = scene["end"]
                clear = scene["clear"]
                
                if s_idx == 1:
                    sec_timings = [
                        (0.2, 1.2),
                        (1.4, 2.4),
                        (2.6, 3.6),
                        (3.8, 4.8),
                        (5.0, 6.5)
                    ]
                else:
                    sec_timings = [
                        (start + 0.1, start + 0.7),
                        (start + 0.7, start + 1.3),
                        (start + 1.3, start + 1.9),
                        (start + 1.9, start + 2.5),
                        (start + 2.5, start + 3.2)
                    ]
                
                sections_list = ["identity", "specialization", "techstack", "building", "terminal"]
                for sec_name, (sec_start, sec_end) in zip(sections_list, sec_timings):
                    rows = scene["sections"][sec_name]
                    n_rows = len(rows)
                    row_dur = (sec_end - sec_start) / n_rows
                    
                    for r_idx, row in enumerate(rows):
                        row_name = row[0]
                        row_val = row[1]
                        row_x = row[2]
                        row_y = row[3]
                        row_w = row[4]
                        
                        r_start = sec_start + r_idx * row_dur
                        r_end = r_start + row_dur * 0.85
                        
                        clip_id = f"clip-s{s_idx}-{sec_name}-{row_name}"
                        
                        k0 = f"{0.0:.5f}"
                        k1 = f"{r_start / 31.3:.5f}"
                        k2 = f"{r_end / 31.3:.5f}"
                        k3 = f"{end / 31.3:.5f}"
                        k4 = f"{clear / 31.3:.5f}"
                        k5 = f"{1.0:.5f}"
                        
                        f.write(f'    <clipPath id="{clip_id}">\n')
                        f.write(f'      <rect x="{row_x}" y="{row_y - 9}" height="15" width="0">\n')
                        f.write(f'        <animate attributeName="width" values="0;0;{row_w};{row_w};0;0" ')
                        f.write(f'keyTimes="{k0};{k1};{k2};{k3};{k4};{k5}" dur="31.3s" repeatCount="indefinite" />\n')
                        f.write(f'      </rect>\n')
                        f.write(f'    </clipPath>\n')

            f.write('    <!-- PROFILE_ENGINE Clip Path matching the viewport below header line -->\n')
            f.write('    <clipPath id="profileClip">\n')
            f.write('      <rect x="25" y="88" width="320" height="282" />\n')
            f.write('    </clipPath>\n')
            f.write('  </defs>\n')

            # Glow behind the main outer card container
            f.write('  <!-- Neon Glow Border -->\n')
            f.write('  <rect x="10" y="10" width="900" height="380" rx="10" ry="10" fill="none" stroke="url(#neon-glow-grad)" stroke-width="2" filter="url(#terminal-glow)" />\n')
            
            # Solid Application Container (using premium drop shadow & glass-like border)
            f.write('  <!-- Main Application Container -->\n')
            f.write('  <rect x="10" y="10" width="900" height="380" rx="10" ry="10" fill="url(#card-bg)" stroke="#1E293B" stroke-width="1.2" filter="url(#card-shadow)" />\n')
            
            # Application Header (VSCode / macOS top bar)
            f.write('  <!-- Application Header -->\n')
            f.write('  <path d="M 10 20 A 10 10 0 0 1 20 10 L 900 10 A 10 10 0 0 1 910 20 L 910 46 L 10 46 Z" fill="#0C1424" />\n')
            f.write('  <line x1="10" y1="46" x2="910" y2="46" stroke="#1E293B" stroke-width="1.2" />\n')
            
            # macOS Window Controls (using rounded rects for validator compliance)
            f.write('  <!-- macOS style window controls -->\n')
            f.write('  <rect x="25" y="23" width="8" height="8" rx="4" ry="4" fill="#ff5f56" />\n')
            f.write('  <rect x="37" y="23" width="8" height="8" rx="4" ry="4" fill="#ffbd2e" />\n')
            f.write('  <rect x="49" y="23" width="8" height="8" rx="4" ry="4" fill="#27c93f" />\n')
            
            # Centered Header Title & Subtitle with title glow pulse
            f.write('  <!-- Header Title (with pulsing glow layer) -->\n')
            f.write('  <text x="460" y="24" fill="#22D3EE" font-family="monospace" font-size="13" font-weight="bold" text-anchor="middle" filter="url(#title-glow)">\n')
            f.write('    <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" />\n')
            f.write('    SHYAM.DEV\n')
            f.write('  </text>\n')
            f.write('  <text x="460" y="24" fill="#F8FAFC" font-family="monospace" font-size="13" font-weight="bold" text-anchor="middle">SHYAM.DEV</text>\n')
            f.write('  <text x="460" y="39" fill="#94A3B8" font-family="monospace" font-size="8" text-anchor="middle">Developer Dashboard</text>\n')
            
            # Right Status Indicator with soft pulse animation
            f.write('  <!-- Status indicator -->\n')
            f.write('  <g>\n')
            f.write('    <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite" />\n')
            f.write('    <rect x="844" y="24" width="6" height="6" rx="3" ry="3" fill="#10B981" />\n')
            f.write('    <text x="856" y="30" fill="#10B981" font-family="monospace" font-size="8.5" font-weight="bold">ACTIVE</text>\n')
            f.write('  </g>\n')
            
            # ── Left Animation Panel (Inner Frame with glass backdrop look) ─────────────────────────────
            f.write('  <!-- Left Animation Panel -->\n')
            f.write('  <rect x="25" y="60" width="320" height="310" rx="8" ry="8" fill="#03071280" stroke="#1E293B" stroke-width="1.2" />\n')
            f.write('  <text x="37" y="78" fill="#22D3EE" font-family="monospace" font-size="8.5" font-weight="bold">PARTICLE ENGINE</text>\n')
            f.write('  <line x1="25" y1="88" x2="345" y2="88" stroke="#1E293B" stroke-width="1" />\n')

            # ── Particle Animation Viewport Fitting ──────────────────
            # Wrap in clip path and center original 300x340 coordinates at 0.885 scale
            f.write('  <g clip-path="url(#profileClip)">\n')
            # Outer wrapper for translation (6s) and scale (7s) breathing effect
            f.write('    <g>\n')
            f.write('      <animateTransform attributeName="transform" type="translate" values="0,0; 0,2; 0,0" dur="6s" repeatCount="indefinite" additive="sum" />\n')
            f.write('      <g transform="translate(185, 220)">\n')
            f.write('        <animateTransform attributeName="transform" type="scale" values="1; 1.015; 1" dur="7s" repeatCount="indefinite" additive="sum" />\n')
            f.write('        <g transform="translate(-185, -220)">\n')
            # The particle rendering group with a cyan glow filter for apparent density and soft bloom
            f.write('          <g transform="translate(52.25, 54.7) scale(0.885)" filter="url(#cyan-glow)">\n')

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

            # Close the nested breathing and rendering groups
            f.write('          </g>\n')
            f.write('        </g>\n')
            f.write('      </g>\n')
            f.write('    </g>\n')
            f.write('  </g>\n')

            # ── Vertical Divider between Left and Right Panels ──────────────────
            f.write('  <!-- Column Divider -->\n')
            f.write('  <line x1="365" y1="46" x2="365" y2="390" stroke="#1E293B" stroke-width="1.2" />\n')
            
            # ── Right Dashboard Panel (Monospace Text) ─────────────────────────
            f.write('  <g font-family="monospace" font-size="9.5" xml:space="preserve">\n')
            
            # Static Labels (These remain completely static while values type)
            # Column 1 Labels
            f.write('    <!-- Static Labels Column 1 -->\n')
            f.write('    <text x="395" y="80" fill="#22D3EE" font-weight="bold">IDENTITY</text>\n')
            f.write('    <line x1="395" y1="84" x2="630" y2="84" stroke="#1E293B" stroke-width="1" />\n')
            f.write('    <text x="395" y="98" fill="#94A3B8">Name</text>\n')
            f.write('    <text x="395" y="113" fill="#94A3B8">Role</text>\n')
            f.write('    <text x="395" y="128" fill="#94A3B8">Status</text>\n')
            f.write('    <text x="395" y="143" fill="#94A3B8">Location</text>\n')
            
            f.write('    <text x="395" y="178" fill="#22D3EE" font-weight="bold">SPECIALIZATION</text>\n')
            f.write('    <line x1="395" y1="182" x2="630" y2="182" stroke="#1E293B" stroke-width="1" />\n')
            f.write('    <text x="395" y="196" fill="#94A3B8">Core</text>\n')
            f.write('    <text x="395" y="211" fill="#94A3B8">Backend</text>\n')
            f.write('    <text x="395" y="226" fill="#94A3B8">Frontend</text>\n')
            f.write('    <text x="395" y="241" fill="#94A3B8">Next Gen</text>\n')
            
            f.write('    <text x="395" y="265" fill="#22D3EE" font-weight="bold">TERMINAL</text>\n')
            f.write('    <line x1="395" y1="269" x2="630" y2="269" stroke="#1E293B" stroke-width="1" />\n')
            
            # Column 2 Labels
            f.write('    <!-- Static Labels Column 2 -->\n')
            f.write('    <text x="660" y="80" fill="#22D3EE" font-weight="bold">TECH STACK</text>\n')
            f.write('    <line x1="660" y1="84" x2="895" y2="84" stroke="#1E293B" stroke-width="1" />\n')
            f.write('    <text x="660" y="98" fill="#94A3B8">Languages</text>\n')
            f.write('    <text x="660" y="113" fill="#94A3B8">Backend</text>\n')
            f.write('    <text x="660" y="128" fill="#94A3B8">Frontend</text>\n')
            f.write('    <text x="660" y="143" fill="#94A3B8">Databases</text>\n')
            f.write('    <text x="660" y="158" fill="#94A3B8">DevOps</text>\n')
            
            f.write('    <text x="660" y="196" fill="#22D3EE" font-weight="bold">CURRENTLY BUILDING</text>\n')
            f.write('    <line x1="660" y1="200" x2="895" y2="200" stroke="#1E293B" stroke-width="1" />\n')
            f.write('    <text x="660" y="214" fill="#94A3B8">AI Agent</text>\n')
            f.write('    <text x="660" y="229" fill="#94A3B8">SVG Engine</text>\n')
            f.write('    <text x="660" y="244" fill="#94A3B8">Platform</text>\n')

            # Write the scene text elements and their cursors
            for s_idx, scene in enumerate(scenes, 1):
                start = scene["start"]
                end = scene["end"]
                clear = scene["clear"]
                
                if s_idx == 1:
                    sec_timings = [
                        (0.2, 1.2),
                        (1.4, 2.4),
                        (2.6, 3.6),
                        (3.8, 4.8),
                        (5.0, 6.5)
                    ]
                else:
                    sec_timings = [
                        (start + 0.1, start + 0.7),
                        (start + 0.7, start + 1.3),
                        (start + 1.3, start + 1.9),
                        (start + 1.9, start + 2.5),
                        (start + 2.5, start + 3.2)
                    ]
                
                sections_list = ["identity", "specialization", "techstack", "building", "terminal"]
                
                k0 = f"{0.0:.5f}"
                k_start = f"{start / 31.3:.5f}"
                k_clear = f"{clear / 31.3:.5f}"
                k5 = f"{1.0:.5f}"
                
                f.write(f'    <!-- SCENE {s_idx} VALUES -->\n')
                f.write(f'    <g>\n')
                f.write(f'      <animate attributeName="visibility" values="hidden;visible;hidden" keyTimes="{k0};{k_start};{k_clear}" dur="31.3s" repeatCount="indefinite" />\n')
                
                for sec_name, (sec_start, sec_end) in zip(sections_list, sec_timings):
                    rows = scene["sections"][sec_name]
                    n_rows = len(rows)
                    row_dur = (sec_end - sec_start) / n_rows
                    
                    for r_idx, row in enumerate(rows):
                        row_name = row[0]
                        row_val = row[1]
                        row_x = row[2]
                        row_y = row[3]
                        row_w = row[4]
                        row_color = row[5] if len(row) > 5 else "#F8FAFC"
                        
                        r_start = sec_start + r_idx * row_dur
                        r_end = r_start + row_dur * 0.85
                        
                        clip_id = f"clip-s{s_idx}-{sec_name}-{row_name}"
                        
                        if row_name == "status":
                            f.write(f'      <text x="{row_x}" y="{row_y}" clip-path="url(#{clip_id})"><tspan fill="{row_color}">● </tspan><tspan fill="#F8FAFC">{row_val}</tspan></text>\n')
                        else:
                            f.write(f'      <text x="{row_x}" y="{row_y}" fill="{row_color}" clip-path="url(#{clip_id})">{row_val}</text>\n')
                        
                        # Blinking typing cursor
                        c0 = f"{0.0:.5f}"
                        c1 = f"{r_start / 31.3:.5f}"
                        c2 = f"{r_end / 31.3:.5f}"
                        c3 = f"{1.0:.5f}"
                        
                        f.write(f'      <g>\n')
                        f.write(f'        <animate attributeName="visibility" values="hidden;visible;hidden" keyTimes="{c0};{c1};{c2};{c3}" dur="31.3s" repeatCount="indefinite" />\n')
                        f.write(f'        <rect x="{row_x}" y="{row_y-8}" width="5" height="10" fill="#22D3EE">\n')
                        f.write(f'          <animateTransform attributeName="transform" type="translate" values="0,0; {row_w},0; {row_w},0" keyTimes="{c0};{c1};{c2}" dur="31.3s" repeatCount="indefinite" />\n')
                        f.write(f'          <animate attributeName="opacity" values="1;0;1" dur="0.65s" repeatCount="indefinite" />\n')
                        f.write(f'        </rect>\n')
                        f.write(f'      </g>\n')
                
                f.write(f'    </g>\n')
            
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


