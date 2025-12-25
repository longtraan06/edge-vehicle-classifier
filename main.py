"""Main entry point for Vehicle Detection & Tracking System."""

import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_settings
from core.pipeline import VehicleTrackingPipeline
from services.video_service import VideoWriterAsync, CameraOverlay, get_youtube_stream_url
from utils import setup_logging, get_logger


def main():
    """Main application entry point."""
    
    # Load configuration
    try:
        settings = get_settings()
    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure config/settings.yaml exists")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Setup logging
    setup_logging(
        log_dir=settings.logging.log_dir,
        log_level=settings.logging.level,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
        log_format=settings.logging.format
    )
    
    logger = get_logger(__name__)
    logger.info("="*70)
    logger.info(" VEHICLE DETECTION & TRACKING SYSTEM")
    logger.info(" Professional Edition with Modular Architecture")
    logger.info("="*70)
    
    # Handle YouTube URLs
    camera_source = settings.camera.source
    if "youtube.com" in camera_source or "youtu.be" in camera_source:
        logger.info(f"Detected YouTube URL: {camera_source}")
        stream_url = get_youtube_stream_url(camera_source)
        if stream_url:
            camera_source = stream_url
            logger.info("YouTube stream URL extracted successfully")
        else:
            logger.error("Failed to extract YouTube stream URL")
            return 1
    
    # Initialize video capture
    logger.info(f"Opening camera: {camera_source}")
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        logger.error("Failed to open camera source")
        return 1
    
    # Get camera FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = settings.camera.fps
    logger.info(f"Camera FPS: {video_fps:.2f}")
    
    # Initialize pipeline
    pipeline = VehicleTrackingPipeline(settings)
    pipeline.initialize()
    
    try:
        # Web configuration (if enabled)
        if settings.web.enabled:
            config = pipeline.run_web_configuration(cap)
            pipeline.overlay = CameraOverlay(
                config['roi'],
                config['tripwires'],
                frame_shape=None
            )
        else:
            logger.warning("Web interface disabled, using empty ROI/tripwires")
            pipeline.overlay = CameraOverlay(
                roi=None,
                tripwires=[],
                frame_shape=None
            )
        
        # Initialize video writer (if enabled)
        if settings.output.save_video:
            # Read first frame to get dimensions
            ret, first_frame = cap.read()
            if ret:
                if (settings.camera.processing_width and 
                    first_frame.shape[1] > settings.camera.processing_width):
                    h, w = first_frame.shape[:2]
                    ratio = settings.camera.processing_width / w
                    new_h = int(h * ratio)
                    first_frame = cv2.resize(
                        first_frame,
                        (settings.camera.processing_width, new_h)
                    )
                
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*settings.output.video_codec)
                pipeline.video_writer = VideoWriterAsync(
                    output_path=str(settings.get_absolute_path(
                        settings.output.video_output_path
                    )),
                    fourcc=fourcc,
                    fps=settings.output.video_fps,
                    frame_size=(w, h)
                )
                
                # Put first frame back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Main processing loop
        logger.info("Starting main processing loop...")
        logger.info("Press 'q' to quit (if UI enabled)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                is_live_stream = cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1
                if is_live_stream:
                    logger.warning("Lost connection, reconnecting in 5s...")
                    cap.release()
                    import time
                    time.sleep(5)
                    cap = cv2.VideoCapture(camera_source)
                    continue
                else:
                    logger.info("End of video file reached")
                    break
            
            # Resize if needed
            if (settings.camera.processing_width and 
                frame.shape[1] > settings.camera.processing_width):
                h, w = frame.shape[:2]
                ratio = settings.camera.processing_width / w
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (settings.camera.processing_width, new_h))
            
            # Determine if detection should run
            run_detection = (pipeline.frame_counter % 
                           settings.tracking.detection_interval == 0)
            
            # Process frame
            frame = pipeline.process_frame(frame, run_detection)
            
            # Save video
            if pipeline.video_writer:
                pipeline.video_writer.write(frame)
            
            # Display (if enabled)
            if settings.output.enable_ui:
                cv2.imshow("Vehicle Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break
            
            pipeline.frame_counter += 1
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        cap.release()
        
        if settings.output.enable_ui:
            cv2.destroyAllWindows()
        
        pipeline.cleanup()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
