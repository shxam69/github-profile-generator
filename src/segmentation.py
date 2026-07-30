import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from scipy import ndimage

logger = logging.getLogger(__name__)

class BackgroundSegmenter:
    """A class to handle background segmentation of a portrait."""
    
    def __init__(self, debug_dir: Path) -> None:
        """Initializes the segmenter and ensures the debug directory exists."""
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_debug(self, img_array: np.ndarray, filename: str, is_binary: bool = False) -> None:
        """Helper to save intermediate numpy arrays as images."""
        if is_binary:
            img_array = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_array).save(self.debug_dir / filename)
        logger.info(f"Saved debug output: {filename}")
        
    def segment(self, input_path: Path, closing_kernel_size: int, feather_radius: int) -> None:
        """
        Executes the segmentation pipeline.
        
        Args:
            input_path: Path to the processed soft image.
            closing_kernel_size: Kernel size for morphological closing.
            feather_radius: Blur radius for feathering edges.
        """
        logger.info(f"Starting background segmentation on {input_path}")
        
        if not input_path.exists():
            logger.error("Input file not found.")
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # 1. Load processed image
        logger.info("Loading processed image for segmentation...")
        img_pil = Image.open(input_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 2. Detect foreground (GrabCut)
        logger.info("Detecting foreground using GrabCut algorithm...")
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        h, w = img_cv.shape[:2]
        # Rect for GrabCut: (x, y, width, height) - we leave a 5% margin on sides and top, 0% on bottom
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - margin_y)
        
        cv2.grabCut(img_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # 3. Produce a binary mask
        logger.info("Producing binary mask from foreground detection...")
        # In grabcut, 0=bg, 2=pr_bg, 1=fg, 3=pr_fg. So we want 1 and 3 as foreground
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        self._save_debug(binary_mask, "01_binary_mask.png", is_binary=True)
        
        # 4. Apply morphological closing
        logger.info(f"Applying morphological closing (kernel size: {closing_kernel_size})...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        self._save_debug(closed_mask, "02_closed_mask.png", is_binary=True)
        
        # 5. Fill holes
        logger.info("Filling holes in the segmentation mask...")
        filled_mask = ndimage.binary_fill_holes(closed_mask).astype(np.uint8)
        self._save_debug(filled_mask, "03_filled_mask.png", is_binary=True)
        
        # 6. Keep only largest connected component
        logger.info("Isolating largest connected component...")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
        
        largest_label = 0
        max_area = 0
        # Start at 1 to skip background (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                largest_label = i
                
        if largest_label > 0:
            largest_component = (labels == largest_label).astype(np.uint8)
        else:
            logger.warning("No foreground component found. Using empty mask.")
            largest_component = np.zeros_like(filled_mask)
            
        self._save_debug(largest_component, "04_largest_component.png", is_binary=True)
        
        # 7. Feather edges slightly
        logger.info(f"Feathering mask edges (radius: {feather_radius})...")
        mask_pil = Image.fromarray(largest_component * 255).convert("L")
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather_radius))
            
        # 8. Save final subject outputs
        logger.info("Generating final compositions...")
        
        # Subject only (transparent background)
        subject_only = img_pil.copy()
        subject_only.putalpha(mask_pil)
        subject_only.save(self.debug_dir / "05_subject_only.png")
        logger.info("Saved debug output: 05_subject_only.png")
        
        # Subject on black
        black_bg = Image.new("RGB", img_pil.size, (0, 0, 0))
        subject_on_black = Image.composite(img_pil, black_bg, mask_pil)
        subject_on_black.save(self.debug_dir / "06_subject_on_black.png")
        logger.info("Saved debug output: 06_subject_on_black.png")
        
        logger.info("Sprint 2 Pipeline completed successfully.")