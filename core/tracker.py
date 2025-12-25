"""Kalman Filter based tracking system."""

import time
import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from typing import Dict, List, Optional, Any

from utils.logger import LoggerMixin


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bounding box to z-space format for Kalman filter.
    
    Format: [center_x, center_y, area, aspect_ratio]
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Z-space representation [cx, cy, a, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + 0.5 * w
    cy = bbox[1] + 0.5 * h
    a = w * h
    r = w / float(h)
    return np.array([cx, cy, a, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Convert z-space format back to bounding box format.
    
    Args:
        x: State vector [cx, cy, a, r, ...]
    
    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    if np.any(np.isnan(x)):
        return np.array([[0, 0, 0, 0]])

    s = x[2]
    r = x[3]
    s = max(0, s)
    r = max(1e-4, r)

    w = np.sqrt(s * r)
    h = s / w if w > 1e-4 else 0

    if np.isnan(w) or np.isnan(h) or np.isinf(w) or np.isinf(h):
        return np.array([[0, 0, 0, 0]])

    return np.array([
        x[0] - 0.5 * w,
        x[1] - 0.5 * h,
        x[0] + 0.5 * w,
        x[1] + 0.5 * h
    ]).reshape((1, 4))


class KalmanBoxTracker(LoggerMixin):
    """
    Kalman Filter based tracker for bounding boxes.
    
    Features:
    - 7-state Kalman filter (x, y, area, aspect_ratio, vx, vy, v_area)
    - Voting-based classification with confidence tracking
    - Frame history with quality scores
    - Classification finalization logic
    """
    
    count = 0  # Global tracker ID counter
    
    def __init__(self, bbox: np.ndarray, max_history_size: int = 15):
        """
        Initialize Kalman box tracker.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            max_history_size: Maximum frames to keep in history
        """
        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        # Tracker properties
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # Classification properties
        self.class_name: Optional[str] = None
        self.confidence: float = 0.0
        self.color: Optional[tuple] = None
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.class_confidences: Dict[str, List[float]] = defaultdict(list)
        self.has_been_classified: bool = False
        
        # Frame history
        self.frame_history: List[Dict[str, Any]] = []
        self.max_history_size = max_history_size
    
    def update(self, bbox: np.ndarray) -> None:
        """
        Update tracker with new detection.
        
        Args:
            bbox: New bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self) -> np.ndarray:
        """
        Predict next state using Kalman filter.
        
        Returns:
            Predicted bounding box
        """
        self.kf.predict()
        
        # Ensure area doesn't go negative
        if self.kf.x[2] < 0:
            self.kf.x[2] = 0
        
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def increment_time_since_update(self) -> None:
        """Increment time since last update (called on detection frames)."""
        self.time_since_update += 1
    
    def get_state(self) -> np.ndarray:
        """
        Get current tracker state (bounding box).
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)
    
    def save_frame_to_history(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        quality_score: float,
        all_trackers_snapshot: Optional[List[Dict]] = None
    ) -> None:
        """
        Save frame to history (keeps only best frames).
        
        Args:
            frame: Frame image
            box: Bounding box of this tracker
            quality_score: Quality score of this frame
            all_trackers_snapshot: Snapshot of all trackers at this moment
        """
        timestamp = time.time()
        self.frame_history.append({
            'frame': frame.copy(),
            'box': box.copy(),
            'quality_score': quality_score,
            'timestamp': timestamp,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'all_trackers_snapshot': all_trackers_snapshot
        })
        
        # Keep only best frames
        if len(self.frame_history) > self.max_history_size:
            self.frame_history.sort(key=lambda x: x['quality_score'], reverse=True)
            self.frame_history = self.frame_history[:self.max_history_size]
    
    def finalize_classification(self) -> Optional[Dict[str, Any]]:
        """
        Finalize vehicle classification based on voting.
        
        Returns:
            Dictionary with finalized data or None if not enough data
        """
        if not self.class_counts:
            return None
        
        # Majority voting for final class
        final_class = max(self.class_counts, key=self.class_counts.get)
        
        # Get best frame (only from frames with final class)
        best_frame_data = None
        if self.frame_history:
            frames_of_final_class = [
                f for f in self.frame_history 
                if f.get('class_name') == final_class
            ]
            
            if frames_of_final_class:
                best_frame_data = max(
                    frames_of_final_class,
                    key=lambda x: x['quality_score']
                )
                self.logger.debug(
                    f"Tracker {self.id}: Final class='{final_class}', "
                    f"Best frame quality={best_frame_data['quality_score']:.2f}"
                )
            else:
                self.logger.warning(
                    f"Tracker {self.id}: No frames with final class '{final_class}'"
                )
        
        # Get confidence from best frame
        final_confidence = (
            best_frame_data.get('confidence', self.confidence)
            if best_frame_data else self.confidence
        )
        
        return {
            'id': self.id,
            'final_class': final_class,
            'class_details': dict(self.class_counts),
            'class_confidences': {
                k: list(v) for k, v in self.class_confidences.items()
            },
            'last_box': self.get_state()[0].astype(int),
            'confidence': final_confidence,
            'best_frame_data': best_frame_data
        }
    
    def __repr__(self) -> str:
        """String representation of tracker."""
        return (
            f"KalmanBoxTracker(id={self.id}, class={self.class_name}, "
            f"hits={self.hits}, age={self.age})"
        )
