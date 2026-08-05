"""
Renderer Test Harness
=====================
Standalone script that generates progressively complex SVG test files
to isolate browser rendering failures in the SMIL animation pipeline.

This script:
- Reads FROZEN pipeline data (compiled_timeline.json, dot_graph.json, logo*_points.json)
- Replicates the EXACT XML generation logic from svg_renderer.py
- Writes independent test SVGs to output/renderer_tests/
- Reports statistics for each generated file

It does NOT modify any production code or pipeline components.

Usage:
    cd src
    python -m renderer.renderer_test_harness
"""

import logging
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from renderer.frame_generator import FrameGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("renderer_test_harness")

# ─── Constants (matching svg_renderer.py exactly) ───
W = 300
H = 340
FILL = "#c9d1d9"
BG = "#0d1117"


class TestConfig:
    """Configuration for a single test stage."""
    def __init__(
        self,
        name: str,
        description: str,
        num_circles: int,
        enable_opacity: bool = False,
        enable_cx: bool = False,
        enable_cy: bool = False,
        max_keyframes: Optional[int] = None,
    ):
        self.name = name
        self.description = description
        self.num_circles = num_circles
        self.enable_opacity = enable_opacity
        self.enable_cx = enable_cx
        self.enable_cy = enable_cy
        self.max_keyframes = max_keyframes  # None = use all keyframes from production


# ─── Test Matrix ───
STAGES = [
    # Scale stages
    TestConfig("stage1_100_static", "100 static circles, no animation", 100),
    TestConfig("stage2_14248_static", "14248 static circles, no animation", 14248),
    TestConfig("stage3_100_opacity", "100 circles, opacity animation only", 100, enable_opacity=True),
    TestConfig("stage4_100_full_anim", "100 circles, opacity+cx+cy animation", 100, enable_opacity=True, enable_cx=True, enable_cy=True),
    TestConfig("stage5_1000_full_anim", "1000 fully animated circles", 1000, enable_opacity=True, enable_cx=True, enable_cy=True),
    TestConfig("stage6_5000_full_anim", "5000 fully animated circles", 5000, enable_opacity=True, enable_cx=True, enable_cy=True),
    TestConfig("stage7_10000_full_anim", "10000 fully animated circles", 10000, enable_opacity=True, enable_cx=True, enable_cy=True),
    TestConfig("stage8_14248_full_anim", "14248 fully animated circles (production scale)", 14248, enable_opacity=True, enable_cx=True, enable_cy=True),
    # Keyframe density experiments
    TestConfig("kf_a_100_5kf", "100 circles, ~5 keyframes per animate", 100, enable_opacity=True, enable_cx=True, enable_cy=True, max_keyframes=5),
    TestConfig("kf_b_100_20kf", "100 circles, ~20 keyframes per animate", 100, enable_opacity=True, enable_cx=True, enable_cy=True, max_keyframes=20),
    TestConfig("kf_c_100_100kf", "100 circles, ~100 keyframes per animate", 100, enable_opacity=True, enable_cx=True, enable_cy=True, max_keyframes=100),
]


def build_frame_generator(base_dir: Path) -> FrameGenerator:
    """Initialize FrameGenerator from frozen pipeline data."""
    timeline_json = base_dir / "output" / "compiled_timeline.json"
    graph_json = base_dir / "output" / "dot_graph.json"
    logo_jsons = [
        base_dir / "output" / "logo1_points.json",
        base_dir / "output" / "logo2_points.json",
        base_dir / "output" / "logo3_points.json",
    ]

    for p in [timeline_json, graph_json] + logo_jsons:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen data not found: {p}")

    logger.info("Initializing FrameGenerator from frozen pipeline data...")
    return FrameGenerator(timeline_json, graph_json, logo_jsons)


def compute_global_timeline(frame_gen: FrameGenerator) -> Tuple[List[float], float]:
    """
    Replicate svg_renderer.py lines 21-34 EXACTLY.
    Returns (sorted key_times list, total_dur).
    """
    time_set = set([0.0])
    for events in frame_gen.dot_events.values():
        for e in events:
            start = e["start_time"]
            end = e["end_time"]
            # Target 30 frames per second for smooth easing approximation
            fps = 15.0
            duration = end - start
            num_samples = max(6, int(duration * fps))
            for i in range(num_samples + 1):
                t = start + duration * (i / float(num_samples))
                time_set.add(round(t, 2))

    key_times = sorted(list(time_set))
    total_dur = max(key_times)
    return key_times, total_dur


def compute_global_ktimes_strings(key_times: List[float], total_dur: float) -> List[str]:
    """
    Replicate svg_renderer.py lines 56-63 EXACTLY.
    Returns list of formatted keyTimes strings.
    """
    global_k_times = []
    for idx, t in enumerate(key_times):
        if idx == 0:
            global_k_times.append("0")
        elif idx == len(key_times) - 1:
            global_k_times.append("1")
        else:
            global_k_times.append(f"{t / total_dur:.5f}")
    return global_k_times


def optimize_track(values, decimals, global_k_times):
    """
    Replicate svg_renderer.py lines 65-82 EXACTLY.
    Returns (values_str, keytimes_str) or (None, None).
    """
    rounded = [round(v, decimals) for v in values]
    if all(v == rounded[0] for v in rounded):
        return None, None

    keep = []
    N = len(rounded)
    for i in range(N):
        if i == 0 or i == N - 1:
            keep.append(i)
        elif rounded[i] != rounded[i-1] or rounded[i] != rounded[i+1]:
            keep.append(i)

    keep = sorted(list(set(keep)))

    v_str = ";".join([f"{rounded[i]:g}" for i in keep])
    t_str = ";".join([global_k_times[i] for i in keep])
    return v_str, t_str


def limit_keyframes(v_str: Optional[str], t_str: Optional[str], max_kf: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Subsample a values/keyTimes pair to at most max_kf entries.
    Always keeps the first and last entry.
    """
    if v_str is None or t_str is None:
        return None, None

    vals = v_str.split(";")
    times = t_str.split(";")
    n = len(vals)

    if n <= max_kf:
        return v_str, t_str

    # Always include first and last; evenly sample the rest
    indices = [0]
    if max_kf > 2:
        step = (n - 1) / (max_kf - 1)
        for i in range(1, max_kf - 1):
            indices.append(int(round(i * step)))
    indices.append(n - 1)
    indices = sorted(set(indices))

    new_vals = ";".join([vals[i] for i in indices])
    new_times = ";".join([times[i] for i in indices])
    return new_vals, new_times


def select_dot_ids(frame_gen: FrameGenerator, count: int) -> List[int]:
    """
    Select dot IDs for testing.
    For count < total, select a representative spread:
    - Dots that have events (animated dots)
    - Dots that don't have events (static dots)
    This ensures we test both animated and static circles.
    """
    all_ids = sorted(frame_gen.orig_coords.keys())
    total = len(all_ids)

    if count >= total:
        return all_ids

    # Find dots with and without animation events
    animated_ids = [d_id for d_id in all_ids if len(frame_gen.dot_events.get(d_id, [])) > 0]
    static_ids = [d_id for d_id in all_ids if len(frame_gen.dot_events.get(d_id, [])) == 0]

    # Aim for roughly 80% animated, 20% static (matching production ratio)
    n_animated = min(int(count * 0.8), len(animated_ids))
    n_static = min(count - n_animated, len(static_ids))
    # If not enough static, fill with more animated
    if n_animated + n_static < count:
        n_animated = min(count - n_static, len(animated_ids))

    # Evenly sample from each pool
    selected = []
    if n_animated > 0 and len(animated_ids) > 0:
        step = max(1, len(animated_ids) // n_animated)
        selected.extend(animated_ids[::step][:n_animated])
    if n_static > 0 and len(static_ids) > 0:
        step = max(1, len(static_ids) // n_static)
        selected.extend(static_ids[::step][:n_static])

    # If still short (shouldn't happen normally), pad from all_ids
    if len(selected) < count:
        remaining = [d for d in all_ids if d not in set(selected)]
        selected.extend(remaining[:count - len(selected)])

    return sorted(selected[:count])


def generate_test_svg(
    config: TestConfig,
    frame_gen: FrameGenerator,
    key_times: List[float],
    total_dur: float,
    global_k_times: List[str],
    frames: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Generate a single test SVG file.
    Returns a statistics dict.
    """
    out_path = output_dir / f"{config.name}.svg"
    logger.info(f"--- Generating: {config.name} ({config.description}) ---")

    # Select which dots to include
    dot_ids = select_dot_ids(frame_gen, config.num_circles)
    actual_count = len(dot_ids)
    logger.info(f"  Selected {actual_count} dots for this test")

    circle_count = 0
    animate_count = 0
    total_keyframes = 0
    animate_kf_counts = []

    with open(out_path, 'w') as f:
        # SVG header — EXACT match to svg_renderer.py line 53
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background-color: {BG};">\n')

        for d_id in dot_ids:
            # Extract history for this dot — matches svg_renderer.py lines 86-89
            dot_history = frames[:, d_id, :]
            xs = dot_history[:, 0]
            ys = dot_history[:, 1]
            ops = dot_history[:, 2]

            cx_str = f"{round(xs[0], 1):g}"
            cy_str = f"{round(ys[0], 1):g}"
            # For static stages (no opacity animation), override opacity to 1
            # so circles are visible. At t=0 all dots have opacity=0 (pre-intro_reveal),
            # which is correct for animated stages where <animate> handles the transition.
            if config.enable_opacity:
                op_str = f"{round(ops[0], 2):g}"
            else:
                op_str = "1"

            animate_tags = ""

            # Opacity animation
            if config.enable_opacity:
                op_vals, op_times = optimize_track(ops, 2, global_k_times)
                if config.max_keyframes is not None and op_vals is not None:
                    op_vals, op_times = limit_keyframes(op_vals, op_times, config.max_keyframes)
                if op_vals:
                    animate_tags += f'    <animate attributeName="opacity" values="{op_vals}" keyTimes="{op_times}" dur="{total_dur}s" repeatCount="indefinite" />\n'
                    kf_count = len(op_vals.split(";"))
                    animate_count += 1
                    total_keyframes += kf_count
                    animate_kf_counts.append(kf_count)

            # CX animation
            if config.enable_cx:
                x_vals, x_times = optimize_track(xs, 1, global_k_times)
                if config.max_keyframes is not None and x_vals is not None:
                    x_vals, x_times = limit_keyframes(x_vals, x_times, config.max_keyframes)
                if x_vals:
                    animate_tags += f'    <animate attributeName="cx" values="{x_vals}" keyTimes="{x_times}" dur="{total_dur}s" repeatCount="indefinite" />\n'
                    kf_count = len(x_vals.split(";"))
                    animate_count += 1
                    total_keyframes += kf_count
                    animate_kf_counts.append(kf_count)

            # CY animation
            if config.enable_cy:
                y_vals, y_times = optimize_track(ys, 1, global_k_times)
                if config.max_keyframes is not None and y_vals is not None:
                    y_vals, y_times = limit_keyframes(y_vals, y_times, config.max_keyframes)
                if y_vals:
                    animate_tags += f'    <animate attributeName="cy" values="{y_vals}" keyTimes="{y_times}" dur="{total_dur}s" repeatCount="indefinite" />\n'
                    kf_count = len(y_vals.split(";"))
                    animate_count += 1
                    total_keyframes += kf_count
                    animate_kf_counts.append(kf_count)

            # Write circle — EXACT match to svg_renderer.py lines 110-113
            if animate_tags:
                f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="{FILL}" opacity="{op_str}">\n{animate_tags}  </circle>\n')
            else:
                f.write(f'  <circle cx="{cx_str}" cy="{cy_str}" r="1" fill="{FILL}" opacity="{op_str}" />\n')

            circle_count += 1

        f.write('</svg>\n')

    # File size
    file_size = out_path.stat().st_size
    file_size_kb = file_size / 1024
    file_size_mb = file_size / (1024 * 1024)

    # Average keyframes
    avg_kf = round(sum(animate_kf_counts) / len(animate_kf_counts), 1) if animate_kf_counts else 0

    # XML validation
    xml_valid = True
    xml_error = None
    try:
        ET.parse(out_path)
    except Exception as e:
        xml_valid = False
        xml_error = str(e)

    stats = {
        "name": config.name,
        "description": config.description,
        "file": str(out_path),
        "file_size_bytes": file_size,
        "file_size_display": f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_kb:.1f} KB",
        "circle_count": circle_count,
        "animate_count": animate_count,
        "total_dom_nodes": circle_count + animate_count + 1,  # +1 for <svg>
        "total_keyframes": total_keyframes,
        "avg_keyframes_per_animate": avg_kf,
        "xml_valid": xml_valid,
        "xml_error": xml_error,
        "browser_result": "PENDING",
    }

    logger.info(f"  Written: {stats['file_size_display']}, {circle_count} circles, {animate_count} animates, avg {avg_kf} kf/animate, XML: {'PASS' if xml_valid else 'FAIL'}")
    return stats


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "output" / "renderer_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RENDERER TEST HARNESS")
    logger.info("=" * 70)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # 1. Initialize FrameGenerator (read-only, frozen data)
    frame_gen = build_frame_generator(base_dir)
    logger.info(f"FrameGenerator initialized: {frame_gen.num_dots} dots, {len(frame_gen.events)} events")

    # 2. Compute global timeline (replicating svg_renderer.py)
    logger.info("Computing global timeline...")
    key_times, total_dur = compute_global_timeline(frame_gen)
    global_k_times = compute_global_ktimes_strings(key_times, total_dur)
    logger.info(f"Global timeline: {len(key_times)} keyframes, duration {total_dur}s")

    # 3. Precompute all frames (replicating svg_renderer.py lines 40-44)
    logger.info(f"Precomputing state at {len(key_times)} timestamps for {frame_gen.num_dots} dots...")
    frames = []
    for i, t in enumerate(key_times):
        if i % 50 == 0:
            logger.info(f"  Computing frame {i}/{len(key_times)}...")
        frames.append(frame_gen.get_frame(t))
    frames = np.array(frames)  # (T, N, 4)
    logger.info(f"Precomputation complete. Shape: {frames.shape}")

    # 4. Generate all test SVGs
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING TEST SVGs")
    logger.info("=" * 70)

    all_stats = []
    for config in STAGES:
        stats = generate_test_svg(
            config, frame_gen, key_times, total_dur, global_k_times, frames, output_dir
        )
        all_stats.append(stats)

    # 5. Write summary report
    report_path = output_dir / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nReport written to: {report_path}")

    # 6. Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'Stage':<30} {'Size':>10} {'Circles':>8} {'Animates':>9} {'DOM':>7} {'Avg KF':>7} {'XML':>5}")
    logger.info("-" * 80)
    for s in all_stats:
        logger.info(
            f"{s['name']:<30} {s['file_size_display']:>10} {s['circle_count']:>8} "
            f"{s['animate_count']:>9} {s['total_dom_nodes']:>7} {s['avg_keyframes_per_animate']:>7} "
            f"{'PASS' if s['xml_valid'] else 'FAIL':>5}"
        )
    logger.info("-" * 80)
    logger.info("")
    logger.info("Next step: Open each SVG in Edge, Chrome, and Firefox.")
    logger.info("Record which stages render and animate correctly.")
    logger.info("The first failing stage identifies the browser's threshold.")


if __name__ == "__main__":
    main()
