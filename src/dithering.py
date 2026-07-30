import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

class Ditherer:
    """Class to apply Floyd-Steinberg dithering with serpentine scanning."""
    
    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def _floyd_steinberg(self, arr: np.ndarray, threshold: int) -> np.ndarray:
        """Applies Floyd-Steinberg error diffusion with serpentine scanning to a float array."""
        arr = arr.copy()
        height, width = arr.shape
        binary_arr = np.zeros_like(arr, dtype=np.uint8)
        
        for y in range(height):
            # Serpentine scan direction
            if y % 2 == 0:
                x_range = range(width)
                direction = 1
            else:
                x_range = range(width - 1, -1, -1)
                direction = -1
                
            for x in x_range:
                old_pixel = arr[y, x]
                new_pixel = 255 if old_pixel >= threshold else 0
                binary_arr[y, x] = 1 if new_pixel == 255 else 0
                error = old_pixel - new_pixel
                
                if direction == 1:
                    if x + 1 < width:
                        arr[y, x + 1] += error * 7.0 / 16.0
                    if y + 1 < height:
                        if x - 1 >= 0:
                            arr[y + 1, x - 1] += error * 3.0 / 16.0
                        arr[y + 1, x] += error * 5.0 / 16.0
                        if x + 1 < width:
                            arr[y + 1, x + 1] += error * 1.0 / 16.0
                else:
                    if x - 1 >= 0:
                        arr[y, x - 1] += error * 7.0 / 16.0
                    if y + 1 < height:
                        if x + 1 < width:
                            arr[y + 1, x + 1] += error * 3.0 / 16.0
                        arr[y + 1, x] += error * 5.0 / 16.0
                        if x - 1 >= 0:
                            arr[y + 1, x - 1] += error * 1.0 / 16.0
                            
        return binary_arr
        
    def process(self, input_path: Path, target_size: tuple[int, int], threshold: int, gamma: float = 1.0, clahe_clip: float = 2.0, clahe_grid: tuple[int, int] = (8, 8)) -> dict[str, np.ndarray]:
        """
        Reads an image, grayscales it, resizes it, and applies 
        various preprocessing techniques before Floyd-Steinberg dithering.
        
        Returns:
            Dictionary of arrays for 'global', 'gamma', and 'adaptive' techniques.
        """
        logger.info(f"Starting dithering process on {input_path}")
        
        if not input_path.exists():
            logger.error("Input file not found.")
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Load image and convert to grayscale
        logger.info("Converting image to grayscale and resizing...")
        img = Image.open(input_path).convert("L")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img.save(self.debug_dir / "07_grayscale.png")
        
        base_arr = np.array(img, dtype=float)
        
        results = {}
        
        # 1. Global (Original)
        logger.info("Applying standard Floyd-Steinberg dithering...")
        global_binary = self._floyd_steinberg(base_arr, threshold)
        Image.fromarray(global_binary * 255, mode="L").save(self.debug_dir / "08_dither_global.png")
        results["global"] = global_binary
        
        # 2. Gamma Corrected
        logger.info(f"Applying Gamma correction ({gamma}) + dithering...")
        # Scale to 0-1, apply gamma, scale back
        gamma_arr = 255.0 * ((base_arr / 255.0) ** (1.0 / gamma))
        gamma_binary = self._floyd_steinberg(gamma_arr, threshold)
        Image.fromarray(gamma_binary * 255, mode="L").save(self.debug_dir / "08_dither_gamma.png")
        results["gamma"] = gamma_binary
        
        # 3. Adaptive (CLAHE)
        logger.info(f"Applying CLAHE (clip={clahe_clip}, grid={clahe_grid}) + dithering...")
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        adaptive_uint8 = clahe.apply(base_arr.astype(np.uint8))
        adaptive_arr = adaptive_uint8.astype(float)
        adaptive_binary = self._floyd_steinberg(adaptive_arr, threshold)
        Image.fromarray(adaptive_binary * 255, mode="L").save(self.debug_dir / "08_dither_adaptive.png")
        results["adaptive"] = adaptive_binary
        
        return results
