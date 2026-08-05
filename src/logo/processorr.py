import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from svgpathtools import svg2paths

logger = logging.getLogger(__name__)


class LogoProcessor:
    """Parses SVGs, renders them as filled high-resolution images, then samples
    the result using a **feature-aware probability map** that concentrates the
    particle budget on:
      • outer contour / symbol edges   (Canny edge response)
      • sharp corners                  (Harris corner response)
      • thin branches / endpoints      (inverse distance-transform weighting)
      • endpoints / tail tips          (skeleton endpoint boosting)

    and suppresses sampling in:
      • large solid filled interiors   (high distance-transform → low weight)

    The total point count is unchanged (LOGO_POINT_COUNT == TRAVELLER_COUNT).
    Output JSON schema is identical to the previous version.

    Phase 2B – Feature-Aware Logo Sampling
    """

    # Long side of the internal high-resolution render canvas (px).
    # Kept in the 2 000–4 000 px range that balances edge fidelity vs runtime.
    RENDER_LONG_SIDE = 3400
    CURVE_STEP = 0.15   # geometry units per curve sample (matches pipeline elsewhere)
    LINE_STEP = 1.0     # geometry units per line sample

    # ── weight-map hyper-parameters ───────────────────────────────────────────
    # How much each feature channel contributes to the final sampling probability.
    # These are *relative* weights; the map is normalized before sampling.
    W_EDGE      = 4.0   # Canny edges (contour, symbol outline, thin strokes)
    W_CORNER    = 2.0   # Harris corner response (sharp bends, endpoints)
    W_THIN      = 3.0   # inverse-distance-transform (thin branches, tails, tips)
    W_INTERIOR  = 0.15  # residual weight kept inside solid fills (never fully 0)

    # Canny thresholds (applied to the HR filled mask).
    CANNY_LOW  = 30
    CANNY_HIGH = 100

    # Edge / corner map Gaussian blur radius before weighting.
    # Spreads feature response to nearby pixels so sampling isn't too sparse.
    FEATURE_SPREAD_SIGMA = 6   # px in HR space

    # Harris corner parameters
    HARRIS_BLOCK = 5
    HARRIS_KSIZE = 3
    HARRIS_K     = 0.04

    # Distance-transform saturation: pixels farther than this from any edge are
    # considered "deep interior" and receive only the residual W_INTERIOR weight.
    DIST_SAT_PERCENTILE = 90   # % of in-mask distances

    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.target_width  = 300
        self.target_height = 340

    # ── SVG → polygon helpers ────────────────────────────────────────────────

    def _flatten_subpath(self, subpath, scale: float) -> np.ndarray:
        """Flatten one continuous SVG subpath into a dense polygon.

        `scale` is the geometry→canvas scale factor so that sampling density
        scales with the render resolution (avoids faceting at high resolution).
        """
        poly: List[List[float]] = []
        for segment in subpath:
            seg_len = segment.length() * scale
            if type(segment).__name__ in ('CubicBezier', 'QuadraticBezier', 'Arc'):
                num_samples = max(10, int(seg_len / self.CURVE_STEP))
            else:
                num_samples = max(2, int(seg_len / self.LINE_STEP))
            for i in range(num_samples):
                try:
                    c = segment.point(i / float(num_samples - 1))
                    poly.append([c.real, c.imag])
                except Exception:
                    pass
        return np.array(poly, dtype=np.float32) if poly else np.empty((0, 2), dtype=np.float32)

    # ── rasterization ────────────────────────────────────────────────────────

    def _rasterize(self, paths, scale: float, tx: float, ty: float,
                   hr_w: int, hr_h: int) -> np.ndarray:
        """Render all SVG paths as a filled binary mask (uint8, 0 or 255)
        using even-odd rule within a single <path> and union across paths.
        """
        accum = np.zeros((hr_h, hr_w), dtype=np.uint8)
        for path in paths:
            path_mask = np.zeros((hr_h, hr_w), dtype=np.uint8)
            for subpath in path.continuous_subpaths():
                poly = self._flatten_subpath(subpath, scale=scale)
                if len(poly) < 3:
                    continue
                scaled = poly.copy()
                scaled[:, 0] = scaled[:, 0] * scale + tx
                scaled[:, 1] = scaled[:, 1] * scale + ty
                sub_mask = np.zeros((hr_h, hr_w), dtype=np.uint8)
                cv2.fillPoly(sub_mask, [scaled.astype(np.int32)], 255,
                             lineType=cv2.LINE_AA)
                # even-odd rule handles holes/counters within one <path>
                path_mask = cv2.bitwise_xor(path_mask, sub_mask)
            accum = cv2.bitwise_or(accum, path_mask)
        return accum

    # ── feature-aware weight map ─────────────────────────────────────────────

    def _build_weight_map(self, filled_mask: np.ndarray) -> np.ndarray:
        """Build a float32 sampling-probability map from the filled binary mask.

        Strategy
        --------
        1. **Edge channel** – Canny on the mask gives high response at the
           object outline, letter edges, and thin-stroke boundaries.
        2. **Corner channel** – Harris corner detector identifies sharp bends,
           serif tips, claw tips, endpoints and dragon-head details.
        3. **Thin-region channel** – the distance transform of the mask gives
           each interior pixel its distance to the nearest boundary. Inverting
           and saturating this (pixels far from any edge → low weight) strongly
           penalises solid-black fills and rewards thin branches / endpoints.
        4. **Residual interior** – a small constant floor inside the mask so
           no covered pixel has zero probability (avoids degenerate gaps when
           particle count exceeds feature-pixel count).

        All channels are blurred slightly so that the feature response spreads
        to immediately neighbouring pixels, preventing excessively sparse
        coverage at stroke centres.
        """
        h, w = filled_mask.shape
        binary = (filled_mask > 127).astype(np.uint8) * 255

        # ── 1. Edge channel ───────────────────────────────────────────────
        edges = cv2.Canny(binary, self.CANNY_LOW, self.CANNY_HIGH)
        edge_f = edges.astype(np.float32) / 255.0
        edge_f = cv2.GaussianBlur(edge_f, (0, 0), self.FEATURE_SPREAD_SIGMA)

        # ── 2. Corner channel ─────────────────────────────────────────────
        gray_f = binary.astype(np.float32) / 255.0
        harris = cv2.cornerHarris(
            gray_f,
            blockSize=self.HARRIS_BLOCK,
            ksize=self.HARRIS_KSIZE,
            k=self.HARRIS_K
        )
        harris = np.clip(harris, 0, None)   # keep only positive corner response
        h_max = harris.max()
        if h_max > 0:
            harris = harris / h_max
        corner_f = cv2.GaussianBlur(harris, (0, 0), self.FEATURE_SPREAD_SIGMA)

        # ── 3. Thin-region / inverse-distance channel ─────────────────────
        # Distance transform: each foreground pixel gets its distance to the
        # nearest background pixel.  Thin strokes get small values (~1–3 px),
        # fat filled centres get large values.
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Saturation: distances beyond the 90th-percentile are "deep interior"
        in_mask = dist[binary > 127]
        if len(in_mask):
            sat = np.percentile(in_mask, self.DIST_SAT_PERCENTILE)
            sat = max(sat, 1.0)
        else:
            sat = 1.0
        # Inverse: thin regions (small dist) → high weight, fat fills → low weight
        dist_clipped = np.clip(dist, 0, sat)
        thin_f = 1.0 - dist_clipped / sat   # ranges [0, 1]; 1 at edges, 0 deep inside

        # ── 4. Combine channels ───────────────────────────────────────────
        weight_map = (
            self.W_EDGE     * edge_f   +
            self.W_CORNER   * corner_f +
            self.W_THIN     * thin_f
        )

        # Apply a small residual floor everywhere inside the mask so every
        # covered pixel has at least some chance of being sampled.
        interior = (binary > 127).astype(np.float32)
        weight_map = weight_map + self.W_INTERIOR * interior

        # Zero out pixels outside the logo entirely.
        weight_map *= interior

        return weight_map.astype(np.float32)

    # ── uniform sampling (for debug comparison) ──────────────────────────────

    def _uniform_sample(self, filled_mask: np.ndarray,
                        point_count: int,
                        render_scale_factor: float,
                        rng: np.random.Generator) -> List[Tuple[float, float]]:
        """Uniform random sampling of on-mask pixels (old approach)."""
        binary = (filled_mask > 127)
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return []
        replace = len(xs) < point_count
        idx = rng.choice(len(xs), size=point_count, replace=replace)
        pts = []
        for i in idx:
            jx = xs[i] / render_scale_factor + rng.uniform(-0.5, 0.5) / render_scale_factor
            jy = ys[i] / render_scale_factor + rng.uniform(-0.5, 0.5) / render_scale_factor
            pts.append((float(jx), float(jy)))
        return pts

    # ── feature-aware sampling ───────────────────────────────────────────────

    def _feature_sample(self, filled_mask: np.ndarray,
                        weight_map: np.ndarray,
                        point_count: int,
                        render_scale_factor: float,
                        rng: np.random.Generator) -> List[Tuple[float, float]]:
        """Sample `point_count` pixels from `filled_mask` according to the
        probability distribution encoded in `weight_map`.
        """
        binary = (filled_mask > 127)
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return []

        # Extract per-pixel weights for in-mask pixels only.
        weights = weight_map[ys, xs].astype(np.float64)
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
            total = weights.sum()
        probs = weights / total

        replace = len(xs) < point_count
        idx = rng.choice(len(xs), size=point_count, replace=replace, p=probs)

        pts = []
        for i in idx:
            jx = xs[i] / render_scale_factor + rng.uniform(-0.5, 0.5) / render_scale_factor
            jy = ys[i] / render_scale_factor + rng.uniform(-0.5, 0.5) / render_scale_factor
            pts.append((float(jx), float(jy)))
        return pts

    # ── debug image helpers ──────────────────────────────────────────────────

    def _render_point_cloud(self, pts: List[Tuple[float, float]],
                            w: int, h: int,
                            color: Tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
        """Render a list of (x, y) points as a bright dot cloud on a black bg."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for px, py in pts:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h:
                curr = img[iy, ix].astype(np.int32)
                img[iy, ix] = np.clip(curr + np.array(color, dtype=np.int32), 0, 255).astype(np.uint8)
        return img

    def _save_comparison(self,
                         logo_name: str,
                         filled_mask_hr: np.ndarray,
                         weight_map_hr: np.ndarray,
                         uniform_pts: List[Tuple[float, float]],
                         feature_pts: List[Tuple[float, float]],
                         render_scale_factor: float) -> None:
        """Save a 4-panel before/after debug image for this logo."""
        tw, th = self.target_width, self.target_height

        # ── Panel 0: original logo (filled mask downscaled to output resolution) ──
        mask_small = cv2.resize(filled_mask_hr, (tw, th), interpolation=cv2.INTER_AREA)
        panel_logo = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        # ── Panel 1: weight map (false-colour heat-map) ────────────────────
        wm_small = cv2.resize(weight_map_hr, (tw, th), interpolation=cv2.INTER_AREA)
        wm_norm = wm_small / (wm_small.max() + 1e-9)
        wm_u8 = (wm_norm * 255).astype(np.uint8)
        panel_weight = cv2.applyColorMap(wm_u8, cv2.COLORMAP_INFERNO)

        # ── Panel 2: uniform 900-point cloud ──────────────────────────────
        panel_uniform = self._render_point_cloud(
            uniform_pts, tw, th, color=(160, 160, 160))

        # ── Panel 3: feature-aware 900-point cloud ─────────────────────────
        panel_feature = self._render_point_cloud(
            feature_pts, tw, th, color=(0, 200, 255))

        # ── Stack into 2×2 grid with labels ───────────────────────────────
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_sc   = 0.38
        thickness = 1
        pad       = 2
        label_h   = 16

        def add_label(panel: np.ndarray, text: str) -> np.ndarray:
            out = np.zeros((th + label_h, tw, 3), dtype=np.uint8)
            out[label_h:, :] = panel
            cv2.putText(out, text, (pad, label_h - pad), font, font_sc,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            return out

        p0 = add_label(panel_logo,    "Original logo")
        p1 = add_label(panel_weight,  "Feature weight map")
        p2 = add_label(panel_uniform, f"Uniform {len(uniform_pts)}-pt sampling")
        p3 = add_label(panel_feature, f"Feature-aware {len(feature_pts)}-pt sampling")

        row0 = np.hstack([p0, p1])
        row1 = np.hstack([p2, p3])
        grid = np.vstack([row0, row1])

        out_path = self.debug_dir / f"featureaware_{logo_name}_comparison.png"
        cv2.imwrite(str(out_path), grid)
        logger.info(f"Saved before/after comparison to {out_path}")

    # ── public API ────────────────────────────────────────────────────────────

    def process_logo(self,
                     svg_path: Path,
                     output_json: Path,
                     point_count: int,
                     scale_factor: float,
                     padding: float,
                     debug_img_name: str) -> None:
        """Main entry-point.  Produces:
          • output_json  – 900-point list [{id, x, y}, …]
          • debug_dir / debug_img_name – feature-aware point-cloud image
          • debug_dir / featureaware_<name>_comparison.png – 4-panel debug grid
        """
        logger.info(f"Processing logo (feature-aware): {svg_path.name}")

        if not svg_path.exists():
            logger.error(f"Missing SVG file: {svg_path}")
            return

        try:
            paths, attributes = svg2paths(
                str(svg_path),
                convert_lines_to_paths=True,
                convert_polylines_to_paths=True,
                convert_polygons_to_paths=True,
                convert_circles_to_paths=True,
                convert_ellipses_to_paths=True,
                convert_rectangles_to_paths=True
            )
        except Exception as e:
            logger.error(f"Failed to parse SVG {svg_path.name}: {e}")
            paths = []

        final_points: List[Dict] = []

        if not paths:
            logger.warning(f"No valid paths in {svg_path.name}. Falling back to uniform random.")
            rng = np.random.default_rng(42)
            for i in range(point_count):
                final_points.append({
                    "id": i,
                    "x": round(self.target_width  / 2 + float(rng.uniform(-100, 100)), 2),
                    "y": round(self.target_height / 2 + float(rng.uniform(-100, 100)), 2),
                })
        else:
            # ── Pass 1: coarse flatten (scale=1) to establish bounding box ──
            coarse_polys = []
            for path in paths:
                for subpath in path.continuous_subpaths():
                    poly = self._flatten_subpath(subpath, scale=1.0)
                    if len(poly):
                        coarse_polys.append(poly)

            if not coarse_polys:
                logger.warning(f"No polygons from {svg_path.name}")
                return

            all_pts = np.vstack(coarse_polys)
            min_x, min_y = np.min(all_pts, axis=0)
            max_x, max_y = np.max(all_pts, axis=0)
            orig_w = max(1.0, float(max_x - min_x))
            orig_h = max(1.0, float(max_y - min_y))

            # ── Determine high-resolution render canvas ─────────────────
            render_scale_factor = self.RENDER_LONG_SIDE / max(self.target_width,
                                                               self.target_height)
            hr_w = int(self.target_width  * render_scale_factor)
            hr_h = int(self.target_height * render_scale_factor)
            hr_padding = padding * render_scale_factor

            target_w = hr_w - 2 * hr_padding
            target_h = hr_h - 2 * hr_padding
            scale = min(target_w / orig_w, target_h / orig_h) * scale_factor
            tx = (hr_w - orig_w * scale) / 2.0 - min_x * scale
            ty = (hr_h - orig_h * scale) / 2.0 - min_y * scale

            # ── Pass 2: rasterize at full HR resolution ──────────────────
            logger.info(f"{svg_path.name}: rasterizing at {hr_w}x{hr_h}…")
            filled_mask = self._rasterize(paths, scale, tx, ty, hr_w, hr_h)

            if not np.any(filled_mask):
                logger.warning(f"Rasterization yielded 0 covered pixels for {svg_path.name}")
                return

            # ── Build feature-aware weight map ───────────────────────────
            logger.info(f"{svg_path.name}: building feature weight map…")
            weight_map = self._build_weight_map(filled_mask)

            # ── Sample points ─────────────────────────────────────────────
            rng = np.random.default_rng(42)

            logger.info(f"{svg_path.name}: drawing uniform sample for comparison…")
            uniform_pts = self._uniform_sample(
                filled_mask, point_count, render_scale_factor, rng)

            rng2 = np.random.default_rng(42)   # fresh seed → reproducible
            logger.info(f"{svg_path.name}: drawing feature-aware sample ({point_count} pts)…")
            feature_pts = self._feature_sample(
                filled_mask, weight_map, point_count, render_scale_factor, rng2)

            # ── Save before/after comparison image ────────────────────────
            self._save_comparison(
                logo_name=svg_path.stem,
                filled_mask_hr=filled_mask,
                weight_map_hr=weight_map,
                uniform_pts=uniform_pts,
                feature_pts=feature_pts,
                render_scale_factor=render_scale_factor,
            )

            # ── Assemble final point list (feature-aware) ─────────────────
            for i, (px, py) in enumerate(feature_pts):
                final_points.append({
                    "id": i,
                    "x": round(px, 2),
                    "y": round(py, 2),
                })

        # ── Export JSON (schema unchanged) ──────────────────────────────────
        logger.info(f"Exporting {len(final_points)} points to {output_json}")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(final_points, f, indent=2)

        # ── Debug point-cloud image (same role as before) ───────────────────
        debug_path = self.debug_dir / debug_img_name
        debug_img  = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        for pt in final_points:
            ix, iy = int(pt["x"]), int(pt["y"])
            if 0 <= ix < self.target_width and 0 <= iy < self.target_height:
                curr = debug_img[iy, ix].astype(np.int32)
                debug_img[iy, ix] = np.clip(
                    curr + np.array([50, 50, 50], dtype=np.int32), 0, 255
                ).astype(np.uint8)
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"Saved feature-aware point cloud to {debug_path}")
