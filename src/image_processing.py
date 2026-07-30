import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Tuple, Optional, Any
from skimage.feature import Cascade
from skimage import data

logger = logging.getLogger(__name__)

class ImageProcessor:
    """A class to handle image processing for profile banners."""
    
    def __init__(self) -> None:
        """Initializes the ImageProcessor with a Haar cascade for face detection."""
        try:
            self.face_cascade = Cascade(data.lbp_frontal_face_cascade_filename())
        except Exception as e:
            logger.error(f"Failed to load cascade for face detection: {e}")
            raise RuntimeError("Cascade could not be loaded.") from e

    def _crop_head_and_shoulders(self, img_pil: Image.Image) -> Tuple[Optional[Image.Image], Any]:
        """
        Detects a face in the image and crops it to a head-and-shoulders portrait.
        
        Args:
            img_pil: PIL Image object.
            
        Returns:
            Tuple of (cropped PIL Image or None, raw faces data list)
        """
        # Convert to numpy array for skimage
        img_np = np.array(img_pil)
        
        # Detect faces
        faces = self.face_cascade.detect_multi_scale(
            img_np, 
            scale_factor=1.2, 
            step_ratio=1, 
            min_size=(30, 30), 
            max_size=(1000, 1000)
        )
        
        if not faces:
            return None, faces
            
        # Get the largest face (assumed to be the main subject)
        faces_sorted = sorted(faces, key=lambda x: x['width'] * x['height'], reverse=True)
        face = faces_sorted[0]
        
        # r = row (y), c = col (x)
        y, x = face['r'], face['c']
        w, h = face['width'], face['height']
        
        # Calculate padding for a typical head-and-shoulders portrait
        top_padding = int(h * 0.5)
        bottom_padding = int(h * 1.5)
        side_padding = int(w * 0.8)
        
        # Define crop coordinates
        width, height = img_pil.size
        crop_y1 = max(0, y - top_padding)
        crop_y2 = min(height, y + h + bottom_padding)
        crop_x1 = max(0, x - side_padding)
        crop_x2 = min(width, x + w + side_padding)
        
        # Crop the PIL image
        return img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2)), faces

    def process_pipeline(
        self, 
        input_path: Path, 
        output_path: Path, 
        target_size: Tuple[int, int],
        autocontrast_cutoff: int,
        contrast_factor: float,
        unsharp_radius: int,
        unsharp_percent: int,
        unsharp_threshold: int = 3,
        debug_dir: Optional[Path] = None
    ) -> None:
        """
        Executes the image processing pipeline: auto-crop, resize, and enhancements.
        """
        logger.info(f"Starting image processing pipeline for {input_path}")
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # 0. Load image
        try:
            original_img = Image.open(input_path).convert("RGB")
            logger.info(f"Loaded image {input_path.name} with dimensions: {original_img.size[0]}x{original_img.size[1]} pixels.")
            if debug_dir:
                original_img.save(debug_dir / "original.png")
        except Exception as e:
            logger.error(f"Failed to read image: {input_path}. Error: {e}")
            raise

        # 1. Crop to head and shoulders
        logger.info("Attempting auto-crop to head and shoulders...")
        cropped_img, detected_faces = self._crop_head_and_shoulders(original_img)
        
        if detected_faces:
            logger.info(f"Face detector verified: Found {len(detected_faces)} face(s).")
            logger.info(f"Largest face detected at {detected_faces[0]}.")
        else:
            logger.warning("No face detected by the cascade classifier.")
            logger.warning("Reason: The classifier might not have recognized any face patterns due to angle, lighting, or the image being a non-human subject.")
            logger.warning("Fallback: Proceeding with the original full image instead of a cropped portrait.")
        
        if cropped_img is None:
            cropped_img = original_img.copy()

        if debug_dir:
            cropped_img.save(debug_dir / "cropped.png")

        # 2. Resize to exact dimensions
        logger.info(f"Resizing image to {target_size[0]}x{target_size[1]}")
        # ImageOps.fit maintains aspect ratio by cropping the excess
        resized_img = ImageOps.fit(cropped_img, target_size, method=Image.Resampling.LANCZOS)
        
        if debug_dir:
            resized_img.save(debug_dir / "resized.png")

        # 3. Apply Auto Contrast
        logger.info(f"Applying Auto Contrast (cutoff={autocontrast_cutoff})")
        autocontrast_img = ImageOps.autocontrast(resized_img, cutoff=autocontrast_cutoff)
        if debug_dir:
            autocontrast_img.save(debug_dir / "autocontrast.png")
        
        # 4. Apply Contrast enhancement
        logger.info(f"Applying Contrast enhancement (x{contrast_factor})")
        enhancer = ImageEnhance.Contrast(autocontrast_img)
        contrast_img = enhancer.enhance(contrast_factor)
        if debug_dir:
            contrast_img.save(debug_dir / "contrast.png")
        
        # 5. Apply Unsharp Mask
        logger.info(f"Applying Unsharp Mask (radius={unsharp_radius}, percent={unsharp_percent})")
        sharpened_img = contrast_img.filter(
            ImageFilter.UnsharpMask(
                radius=unsharp_radius, 
                percent=unsharp_percent, 
                threshold=unsharp_threshold
            )
        )
        if debug_dir:
            sharpened_img.save(debug_dir / "sharpened.png")

        # 6. Save final output
        logger.info(f"Saving final processed image to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure it's not a placeholder
        sharpened_img.save(output_path)
        logger.info("Image processing pipeline completed successfully.")

