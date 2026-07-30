import logging
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class SvgBuilder:
    """Class to build an optimized SVG from a 1-bit monochrome pixel array."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def build(self, binary_arr: np.ndarray, output_path: Path, inversion: bool, scale: float) -> None:
        """
        Converts the 1-bit array into an SVG path.
        
        Args:
            binary_arr: 2D numpy array of 1s and 0s.
            output_path: Where to save the final .svg file.
            inversion: If True, draws the 1s (light pixels) using white fill. 
                       If False, draws the 0s (dark pixels) using black fill.
            scale: Scalefactor for the final SVG width/height.
        """
        logger.info("Building optimized SVG from dithered mask...")
        height, width = binary_arr.shape
        
        # Determine which value we are drawing as the "foreground"
        # Since the image was black background, inversion=True means we draw the white pixels.
        target_val = 1 if inversion else 0
        
        path_data = []
        
        # Merge adjacent pixels into SVG path runs instead of creating one rectangle per pixel
        logger.info("Merging adjacent pixels into SVG path runs...")
        for y in range(height):
            start_x = -1
            for x in range(width):
                if binary_arr[y, x] == target_val:
                    if start_x == -1:
                        start_x = x
                else:
                    if start_x != -1:
                        rect_w = x - start_x
                        # Draw a filled rectangle of 1px height
                        path_data.append(f"M{start_x},{y} h{rect_w} v1 h-{rect_w} Z")
                        start_x = -1
            
            # Handle case where a run goes all the way to the end of the row
            if start_x != -1:
                rect_w = width - start_x
                path_data.append(f"M{start_x},{y} h{rect_w} v1 h-{rect_w} Z")
                
        d_string = " ".join(path_data)
        
        svg_width = int(width * scale)
        svg_height = int(height * scale)
        fill_color = "#ffffff" if inversion else "#000000"
        
        # Use shape-rendering="crispEdges" to prevent anti-aliasing blurring between pixel paths
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{svg_width}" height="{svg_height}" shape-rendering="crispEdges">
    <path d="{d_string}" fill="{fill_color}"/>
</svg>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(svg_content)
        logger.info(f"Monochrome SVG portrait saved successfully to {output_path}")
        
        # Save preview png directly from the binary array using PIL (avoids cairosvg OS dependency)
        logger.info("Generating SVG preview PNG using PIL...")
        try:
            preview_arr = np.zeros((height, width, 4), dtype=np.uint8)
            if inversion:
                preview_arr[binary_arr == target_val] = [255, 255, 255, 255]
            else:
                preview_arr[binary_arr == target_val] = [0, 0, 0, 255]
                
            preview_img = Image.fromarray(preview_arr, mode="RGBA")
            
            if scale > 1.0:
                preview_img = preview_img.resize((svg_width, svg_height), Image.Resampling.NEAREST)
                
            preview_img.save(self.debug_dir / "09_svg_preview.png")
            logger.info("Saved debug output: 09_svg_preview.png")
        except Exception as e:
            logger.error(f"Failed to generate SVG preview PNG: {e}", exc_info=True)
