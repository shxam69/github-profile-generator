from pathlib import Path
from typing import Tuple, List

class Config:
    """Configuration settings for the Profile Banner Generator."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    INPUT_DIR: Path = BASE_DIR / "input"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    DEBUG_DIR: Path = BASE_DIR / "debug"

    INPUT_PHOTO: Path = INPUT_DIR / "photo.png"
    OUTPUT_PHOTO_SOFT: Path = OUTPUT_DIR / "processed_soft.png"
    OUTPUT_PHOTO_STRONG: Path = OUTPUT_DIR / "processed_strong.png"

    # Image Size
    TARGET_WIDTH: int = 300
    TARGET_HEIGHT: int = 340
    TARGET_SIZE: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)

    # Enhancement Parameters
    AUTOCONTRAST_CUTOFF: int = 1
    CONTRAST_FACTOR: float = 1.3
    
    # Soft Sharpening
    UNSHARP_RADIUS_SOFT: int = 2
    UNSHARP_PERCENT_SOFT: int = 90
    
    # Strong Sharpening
    UNSHARP_RADIUS_STRONG: int = 3
    UNSHARP_PERCENT_STRONG: int = 140
    
    UNSHARP_THRESHOLD: int = 3
    # Sprint 2: Segmentation Parameters
    CLOSING_KERNEL_SIZE: int = 7
    FEATHER_RADIUS: int = 2
    
    # Sprint 3: SVG and Dithering Parameters
    DITHER_THRESHOLD: int = 127
    SVG_INVERSION: bool = True
    SVG_SCALE: float = 1.0
    OUTPUT_SVG: Path = OUTPUT_DIR / "portrait.svg"
    
    # Preprocessing
    GAMMA_VALUE: float = 1.2
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    
    # Sprint 4: Dot Analysis Parameters
    OUTPUT_DOTS_JSON: Path = OUTPUT_DIR / "dots.json"
    
    # Sprint 5: Graph Engine Parameters
    OUTPUT_GRAPH_JSON: Path = OUTPUT_DIR / "dot_graph.json"
    
    # Sprint 6: Animation Framework Parameters
    ANIMATION_DEBUG_JSON: Path = DEBUG_DIR / "animation_timeline.json"
    
    # Sprint 7: Intro Shimmer Parameters
    INTRO_NUM_GROUPS: int = 60
    INTRO_EVENNESS_THRESHOLD: float = 12.0  # Max pixel deviation allowed for group centroid
    INTRO_MAX_ATTEMPTS: int = 100
    OUTPUT_INTRO_GROUPS_JSON: Path = OUTPUT_DIR / "intro_groups.json"
    
    # Sprint 8: Drift Band Parameters
    DRIFT_NUM_BANDS: int = 94
    DRIFT_POS_NOISE: float = 8.0
    DRIFT_BALANCE_WEIGHT: float = 0.3
    OUTPUT_DRIFT_BANDS_JSON: Path = OUTPUT_DIR / "drift_bands.json"
    
    # Sprint 9: Logo Processing Parameters
    LOGO_POINT_COUNT: int = 900  # Must exactly match TRAVELLER_COUNT to ensure uniform sparse sampling
    LOGO_SCALE: float = 1.0
    LOGO_PADDING: float = 10.0
    INPUT_LOGOS: List[Path] = [
        INPUT_DIR / "logo1.svg",
        INPUT_DIR / "logo2.svg",
        INPUT_DIR / "logo3.svg"
    ]
    
    # Sprint 10: Traveller Path Parameters
    TRAVELLER_COUNT: int = 900
    OUTPUT_TRAVELLERS_JSON: Path = OUTPUT_DIR / "traveller_paths.json"
    
    # Sprint 11: Animation Compiler
    OUTPUT_TIMELINE_JSON: Path = OUTPUT_DIR / "compiled_timeline.json"
