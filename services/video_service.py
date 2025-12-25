"""Video processing services."""

import cv2
import numpy as np
import queue
from pathlib import Path
from threading import Thread
from typing import Optional, List, Tuple

from utils.logger import LoggerMixin


class VideoWriterAsync(LoggerMixin):
    """
    Async video writer to prevent blocking main loop.
    
    Uses a separate thread to write frames to disk.
    """
    
    def __init__(
        self,
        output_path: str,
        fourcc: int,
        fps: float,
        frame_size: Tuple[int, int]
    ):
        """
        Initialize async video writer.
        
        Args:
            output_path: Output video file path
            fourcc: Video codec fourcc
            fps: Frames per second
            frame_size: Frame size (width, height)
        """
        self.output_path = output_path
        self.q = queue.Queue()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            raise IOError(f"Failed to open video writer: {output_path}")
        
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        
        self.logger.info(f"Video will be saved to: {output_path}")
    
    def _run(self) -> None:
        """Writer thread main loop."""
        while True:
            frame = self.q.get()
            if frame is None:  # Stop signal
                break
            self.writer.write(frame)
        
        self.writer.release()
        self.logger.info("Video writer released")
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame (non-blocking).
        
        Args:
            frame: Frame to write
        """
        self.q.put(frame)
    
    def release(self) -> None:
        """Release video writer and wait for thread to finish."""
        self.q.put(None)  # Send stop signal
        self.thread.join()
        self.logger.info(f"Video saved: {self.output_path}")


class CameraOverlay(LoggerMixin):
    """
    Cached overlay for ROI and tripwires.
    
    Pre-renders ROI and tripwires to an overlay image for fast frame composition.
    """
    
    def __init__(
        self,
        roi: np.ndarray,
        tripwires: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        frame_shape: Optional[Tuple[int, int, int]] = None
    ):
        """
        Initialize camera overlay.
        
        Args:
            roi: ROI polygon points
            tripwires: List of tripwire lines [(start, end), ...]
            frame_shape: Frame shape (h, w, c) for pre-rendering
        """
        self.roi = roi
        self.tripwires = tripwires
        self.overlay = None
        
        if frame_shape is not None:
            self.create_overlay(frame_shape)
    
    def create_overlay(self, frame_shape: Tuple[int, int, int]) -> None:
        """
        Create overlay image.
        
        Args:
            frame_shape: Frame shape (h, w, c)
        """
        self.overlay = np.zeros(frame_shape, dtype=np.uint8)
        
        # Draw ROI
        if self.roi.size > 0:
            cv2.polylines(
                self.overlay,
                [self.roi],
                isClosed=True,
                color=(255, 255, 0),  # Yellow
                thickness=2
            )
        
        # Draw tripwires
        for line in self.tripwires:
            cv2.line(
                self.overlay,
                line[0],
                line[1],
                color=(0, 0, 255),  # Red
                thickness=2
            )
        
        self.logger.debug(
            f"Overlay created: {len(self.roi)} ROI points, "
            f"{len(self.tripwires)} tripwires"
        )
    
    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply overlay to frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Frame with overlay
        """
        if self.overlay is None:
            self.create_overlay(frame.shape)
        
        return cv2.add(frame, self.overlay)


def get_youtube_stream_url(youtube_url: str) -> Optional[str]:
    """
    Extract direct stream URL from YouTube.
    
    Args:
        youtube_url: YouTube video URL
    
    Returns:
        Direct stream URL or None if failed
    """
    import yt_dlp
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info(f"Extracting YouTube stream URL from: {youtube_url}")
    
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            if 'url' in info:
                stream_url = info['url']
                logger.info(f"Found stream URL: {stream_url[:70]}...")
                return stream_url
            else:
                # Try to find HLS stream
                for f in info.get('formats', []):
                    if f.get('url') and 'm3u8' in f.get('url', ''):
                        stream_url = f['url']
                        logger.info(f"Found HLS stream: {stream_url[:70]}...")
                        return stream_url
        
        logger.warning(f"No suitable stream found for {youtube_url}")
        return None
    
    except Exception as e:
        logger.error(f"Error extracting YouTube stream: {e}")
        return None
