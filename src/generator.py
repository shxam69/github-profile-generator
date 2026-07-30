import logging
import sys
from pathlib import Path

# Add src to the Python path if executed from another directory
sys.path.append(str(Path(__file__).resolve().parent))

from config import Config
from image_processing import ImageProcessor

def setup_logging() -> None:
    """Configures the logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main() -> None:
    """Main execution function for the image processing pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Profile Banner Generator (Sprint 1)")
    
    try:
        processor = ImageProcessor()
        
        # Soft Sharpening
        logger.info("Generating soft processed image...")
        processor.process_pipeline(
            input_path=Config.INPUT_PHOTO,
            output_path=Config.OUTPUT_PHOTO_SOFT,
            target_size=Config.TARGET_SIZE,
            autocontrast_cutoff=Config.AUTOCONTRAST_CUTOFF,
            contrast_factor=Config.CONTRAST_FACTOR,
            unsharp_radius=Config.UNSHARP_RADIUS_SOFT,
            unsharp_percent=Config.UNSHARP_PERCENT_SOFT,
            unsharp_threshold=Config.UNSHARP_THRESHOLD,
            debug_dir=Config.DEBUG_DIR
        )
        
        # Strong Sharpening
        logger.info("Generating strong processed image...")
        processor.process_pipeline(
            input_path=Config.INPUT_PHOTO,
            output_path=Config.OUTPUT_PHOTO_STRONG,
            target_size=Config.TARGET_SIZE,
            autocontrast_cutoff=Config.AUTOCONTRAST_CUTOFF,
            contrast_factor=Config.CONTRAST_FACTOR,
            unsharp_radius=Config.UNSHARP_RADIUS_STRONG,
            unsharp_percent=Config.UNSHARP_PERCENT_STRONG,
            unsharp_threshold=Config.UNSHARP_THRESHOLD,
            debug_dir=None  # Only save debug images once to avoid overwriting
        )
        logger.info("Sprint 1 Pipeline executed successfully for both configurations.")
        
        # Sprint 2: Background Segmentation
        from segmentation import BackgroundSegmenter
        logger.info("Starting Sprint 2: Background Segmentation...")
        segmenter = BackgroundSegmenter(debug_dir=Config.DEBUG_DIR)
        segmenter.segment(
            input_path=Config.OUTPUT_PHOTO_SOFT,
            closing_kernel_size=Config.CLOSING_KERNEL_SIZE,
            feather_radius=Config.FEATHER_RADIUS
        )
        logger.info("Sprint 2 Pipeline executed successfully.")
        
        # Sprint 3: Floyd-Steinberg SVG Portrait
        from dithering import Ditherer
        from svg_builder import SvgBuilder
        logger.info("Starting Sprint 3: SVG Generation...")
        ditherer = Ditherer(debug_dir=Config.DEBUG_DIR)
        # We use debug/06_subject_on_black.png as specified
        subject_on_black_path = Config.DEBUG_DIR / "06_subject_on_black.png"
        
        binary_arrs = ditherer.process(
            input_path=subject_on_black_path,
            target_size=Config.TARGET_SIZE,
            threshold=Config.DITHER_THRESHOLD,
            gamma=Config.GAMMA_VALUE,
            clahe_clip=Config.CLAHE_CLIP_LIMIT,
            clahe_grid=Config.CLAHE_GRID_SIZE
        )
        
        # We build the final SVG using the adaptive preprocessing method 
        # as it typically produces the most detailed dithered portrait.
        svg_builder = SvgBuilder(debug_dir=Config.DEBUG_DIR)
        svg_builder.build(
            binary_arr=binary_arrs["adaptive"],
            output_path=Config.OUTPUT_SVG,
            inversion=Config.SVG_INVERSION,
            scale=Config.SVG_SCALE
        )
        logger.info("Sprint 3 Pipeline executed successfully.")
        
        # Sprint 4: Dot Analysis Engine
        from dot_analysis import DotAnalyzer
        logger.info("Starting Sprint 4: Dot Analysis...")
        
        analyzer = DotAnalyzer(debug_dir=Config.DEBUG_DIR)
        analyzer.analyze(
            dither_path=Config.DEBUG_DIR / "08_dither_adaptive.png",
            grayscale_path=Config.DEBUG_DIR / "07_grayscale.png",
            output_json=Config.OUTPUT_DOTS_JSON
        )
        logger.info("Sprint 4 Pipeline executed successfully.")
        
        # Sprint 5: Graph Engine
        from graph_builder import GraphBuilder
        logger.info("Starting Sprint 5: Graph Engine...")
        
        graph_builder = GraphBuilder(debug_dir=Config.DEBUG_DIR)
        graph_builder.build(
            dots_json_path=Config.OUTPUT_DOTS_JSON,
            output_graph_path=Config.OUTPUT_GRAPH_JSON
        )
        logger.info("Sprint 5 Pipeline executed successfully.")
        
        # Sprint 6: Animation Framework Initialization
        from animation import AnimationScheduler, AnimationTimeline, AnimationGroup, AnimationKeyframe
        logger.info("Starting Sprint 6: Animation Framework...")
        
        scheduler = AnimationScheduler()
        
        # Build Master Timeline with Placeholders
        master_timeline = AnimationTimeline(name="MasterSequence")
        
        placeholders = [
            "Intro",
            "Hold",
            "Drift",
            "Logo1",
            "Transition",
            "Logo2",
            "Transition",
            "Logo3",
            "Return"
        ]
        
        current_start = 0.0
        for p in placeholders:
            # Create a dummy group for each placeholder phase
            group = AnimationGroup(
                name=p,
                start_time=current_start,
                duration=2.0,
                delay=0.0,
                target_ids=[],
                keyframes=[
                    AnimationKeyframe(time=0.0, properties={"opacity": 0.0}),
                    AnimationKeyframe(time=1.0, properties={"opacity": 1.0})
                ]
            )
            master_timeline.groups.append(group)
            current_start += 2.0
            
        scheduler.add_timeline(master_timeline)
        scheduler.export_debug(Config.ANIMATION_DEBUG_JSON)
        logger.info("Sprint 6 Pipeline executed successfully.")
        
        # Sprint 7: Intro Shimmer Generator
        from animation.intro import IntroShimmerGenerator
        logger.info("Starting Sprint 7: Intro Shimmer Generator...")
        
        intro_gen = IntroShimmerGenerator(debug_dir=Config.DEBUG_DIR)
        intro_gen.generate(
            graph_json_path=Config.OUTPUT_GRAPH_JSON,
            output_json=Config.OUTPUT_INTRO_GROUPS_JSON,
            num_groups=Config.INTRO_NUM_GROUPS,
            threshold=Config.INTRO_EVENNESS_THRESHOLD,
            max_attempts=Config.INTRO_MAX_ATTEMPTS
        )
        logger.info("Sprint 7 Pipeline executed successfully.")
        
        # Sprint 8: Drift Band Generator
        from animation.drift_bands import DriftBandGenerator
        logger.info("Starting Sprint 8: Drift Band Generator...")
        
        drift_gen = DriftBandGenerator(debug_dir=Config.DEBUG_DIR)
        drift_gen.generate(
            graph_json_path=Config.OUTPUT_GRAPH_JSON,
            output_json=Config.OUTPUT_DRIFT_BANDS_JSON,
            num_bands=Config.DRIFT_NUM_BANDS,
            noise_level=Config.DRIFT_POS_NOISE,
            balance_weight=Config.DRIFT_BALANCE_WEIGHT
        )
        logger.info("Sprint 8 Pipeline executed successfully.")
        
        # Sprint 9: Logo Processing Engine
        from logo.processor import LogoProcessor
        logger.info("Starting Sprint 9: Logo Processing Engine...")
        
        logo_processor = LogoProcessor(debug_dir=Config.DEBUG_DIR)
        
        for idx, svg_path in enumerate(Config.INPUT_LOGOS, 1):
            output_json = Config.OUTPUT_DIR / f"logo{idx}_points.json"
            debug_img_name = f"21_logo{idx}_points.png" # Fixed to matching numbering like 22_logo1_points.png
            debug_img_name = f"{21+idx}_logo{idx}_points.png" 
            
            logo_processor.process_logo(
                svg_path=svg_path,
                output_json=output_json,
                point_count=Config.LOGO_POINT_COUNT,
                scale_factor=Config.LOGO_SCALE,
                padding=Config.LOGO_PADDING,
                debug_img_name=debug_img_name
            )
            
        logger.info("Sprint 9 Pipeline executed successfully.")
        
        # Sprint 10: Traveller Path Engine
        from animation.traveller import TravellerPathEngine
        logger.info("Starting Sprint 10: Traveller Path Engine...")
        
        traveller_engine = TravellerPathEngine(debug_dir=Config.DEBUG_DIR)
        
        logo_jsons = [
            Config.OUTPUT_DIR / "logo1_points.json",
            Config.OUTPUT_DIR / "logo2_points.json",
            Config.OUTPUT_DIR / "logo3_points.json"
        ]
        
        traveller_engine.build_paths(
            graph_json=Config.OUTPUT_GRAPH_JSON,
            logo_jsons=logo_jsons,
            output_json=Config.OUTPUT_TRAVELLERS_JSON,
            traveller_count=Config.TRAVELLER_COUNT
        )
        
        logger.info("Sprint 10 Pipeline executed successfully.")
        
        # Sprint 11: Animation Compiler
        from animation.compiler import AnimationCompiler
        logger.info("Starting Sprint 11: Animation Compiler...")
        
        compiler = AnimationCompiler(debug_dir=Config.DEBUG_DIR)
        
        compiler.compile(
            intro_json=Config.OUTPUT_INTRO_GROUPS_JSON,
            drift_json=Config.OUTPUT_DRIFT_BANDS_JSON,
            travellers_json=Config.OUTPUT_TRAVELLERS_JSON,
            graph_json=Config.OUTPUT_GRAPH_JSON,
            output_json=Config.OUTPUT_TIMELINE_JSON
        )
        
        logger.info("Sprint 11 Pipeline executed successfully.")
        
        # Sprint 12: Multi-Target Render Engine
        from renderer.renderer import Renderer
        logger.info("Starting Sprint 12: Multi-Target Render Engine...")
        
        engine = Renderer(debug_dir=Config.DEBUG_DIR)
        
        engine.render(
            timeline_json=Config.OUTPUT_TIMELINE_JSON,
            graph_json=Config.OUTPUT_GRAPH_JSON,
            logos_jsons=logo_jsons,
            output_svg=Config.OUTPUT_DIR / "animated_profile.svg"
        )
        
        logger.info("Sprint 12 Pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
