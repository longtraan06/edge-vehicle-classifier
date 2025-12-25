"""Main processing pipeline for vehicle tracking."""

import cv2
import time
import queue
import numpy as np
from threading import Thread, Event
from typing import List, Optional

from config import Settings
from core import YOLOv8Detector, KalmanBoxTracker
from services import VideoWriterAsync, CameraOverlay, save_vehicle_metadata
from services.alert_service import (
    AlertNotifier,
    APIAlertService,
    TelegramAlertService
)
from utils import (
    get_logger,
    calculate_quality_score,
    associate_detections_to_trackers,
    check_bbox_line_intersection
)
from web import app as web_app, run_web_server

logger = get_logger(__name__)


class VehicleTrackingPipeline:
    """Main pipeline for vehicle tracking system."""
    
    def __init__(self, settings: Settings):
        """
        Initialize tracking pipeline.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.detector: Optional[YOLOv8Detector] = None
        self.trackers: List[KalmanBoxTracker] = []
        self.overlay: Optional[CameraOverlay] = None
        self.video_writer: Optional[VideoWriterAsync] = None
        self.alert_notifier: Optional[AlertNotifier] = None
        
        self.frame_counter = 0
        self.all_finalized_vehicles = []
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.last_fps_print = time.time()
    
    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("="*70)
        logger.info(" VEHICLE TRACKING SYSTEM - Professional Edition")
        logger.info(" WITH WEB INTERFACE FOR ROI/TRIPWIRE CONFIGURATION")
        logger.info("="*70)
        
        # Initialize detector
        logger.info("Loading YOLOv8 NCNN detector...")
        self.detector = YOLOv8Detector(
            param_path=str(self.settings.get_absolute_path(self.settings.model.param_path)),
            bin_path=str(self.settings.get_absolute_path(self.settings.model.bin_path)),
            input_size=self.settings.model.input_size,
            conf_threshold=self.settings.model.conf_threshold,
            iou_threshold=self.settings.model.iou_threshold,
            per_class_conf=self.settings.model.per_class_thresholds,
            allowed_classes=['car', 'motorbike', 'person', 'truck']
        )
        logger.info("Detector loaded successfully")
        
        # Initialize alert services
        if self.settings.alert.camera_id and self.settings.alert.ai_module_id:
            services = []
            
            # API alert service
            api_service = APIAlertService(
                api_url=self.settings.alert.api_url,
                camera_id=self.settings.alert.camera_id,
                ai_module_id=self.settings.alert.ai_module_id,
                zone_id=self.settings.alert.zone_id,
                skip_person=self.settings.alert.skip_person_alerts
            )
            services.append(api_service)
            
            # Telegram alert service (if configured)
            if (self.settings.alert.telegram_bot_token and 
                self.settings.alert.telegram_chat_id):
                telegram_service = TelegramAlertService(
                    bot_token=self.settings.alert.telegram_bot_token,
                    chat_id=self.settings.alert.telegram_chat_id,
                    skip_person=self.settings.alert.skip_person_alerts
                )
                services.append(telegram_service)
            
            # Start alert notifier
            notification_queue = queue.Queue(
                maxsize=self.settings.alert.queue_maxsize
            )
            stop_event = Event()
            self.alert_notifier = AlertNotifier(
                services=services,
                notification_queue=notification_queue,
                stop_event=stop_event
            )
        else:
            logger.warning(
                "No alert configuration found, alerts will not be sent"
            )
    
    def run_web_configuration(self, cap: cv2.VideoCapture) -> dict:
        """
        Run web interface for configuration.
        
        Args:
            cap: Video capture object
        
        Returns:
            Configuration dictionary with ROI and tripwires
        """
        # Start Flask web server in separate thread
        flask_thread = Thread(
            target=run_web_server,
            args=(self.settings.web.host, self.settings.web.port),
            daemon=True
        )
        flask_thread.start()
        
        logger.info(
            f"Web interface: http://{self.settings.web.host}:"
            f"{self.settings.web.port}"
        )
        logger.info("Please configure ROI and tripwires, then click 'Start System'")
        
        # Feed frames to web interface
        config_ready_event = web_app.get_config_ready_event()
        frame_skip_counter = 0
        
        while not config_ready_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            
            # Skip frames to reduce CPU
            frame_skip_counter += 1
            if frame_skip_counter < 5:
                time.sleep(0.1)
                continue
            frame_skip_counter = 0
            
            # Resize if needed
            if self.settings.camera.processing_width and \
               frame.shape[1] > self.settings.camera.processing_width:
                h, w = frame.shape[:2]
                ratio = self.settings.camera.processing_width / w
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (self.settings.camera.processing_width, new_h))
            
            # Update web frame
            web_app.set_web_frame(frame)
            time.sleep(0.5)
        
        # Get configuration
        config = web_app.get_camera_configuration()
        if config is None:
            raise ValueError("No configuration received from web interface")
        
        logger.info("Configuration complete! Starting tracking...")
        return config
    
    def process_frame(
        self,
        frame: np.ndarray,
        run_detection: bool
    ) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            run_detection: Whether to run detection on this frame
        
        Returns:
            Processed frame with visualizations
        """
        # Apply overlay
        if self.overlay:
            frame = self.overlay.apply_to_frame(frame)
        
        # Predict all trackers
        predicted_boxes = [tracker.predict()[0] for tracker in self.trackers]
        
        # Detection (every N frames)
        if run_detection:
            self._process_detections(frame, predicted_boxes)
        
        # Tripwire check & finalization
        self._check_tripwires(frame, predicted_boxes)
        
        # Draw trackers
        self._draw_trackers(frame)
        
        # Update FPS
        self._update_fps()
        
        # Draw FPS on frame (if UI enabled)
        if self.settings.output.enable_ui:
            self._draw_stats(frame)
        
        return frame
    
    def _process_detections(
        self,
        frame: np.ndarray,
        predicted_boxes: List[np.ndarray]
    ) -> None:
        """Process detections and update trackers."""
        # Increment time_since_update for all trackers
        for tracker in self.trackers:
            tracker.increment_time_since_update()
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # ROI filtering
        if self.overlay and self.overlay.roi.size > 0:
            filtered_detections = []
            for det in detections:
                box = det.box
                center_point = (int((box[0] + box[2]) / 2), int(box[3]))
                if cv2.pointPolygonTest(self.overlay.roi, center_point, False) >= 0:
                    filtered_detections.append(det)
            detections = filtered_detections
        
        # IoU matching
        detection_boxes = [d.box for d in detections]
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            np.array(predicted_boxes),
            np.array(detection_boxes),
            iou_threshold=self.settings.tracking.iou_threshold
        )
        
        # Update matched trackers
        for match in matches:
            tracker_idx, detection_idx = match[0], match[1]
            det = detections[detection_idx]
            tracker = self.trackers[tracker_idx]
            
            tracker.update(det.box)
            tracker.confidence = det.confidence
            tracker.class_name = det.class_name
            
            if not tracker.has_been_classified:
                tracker.color = self.detector.get_color(det.class_id)
            
            # Voting-based classification
            tracker.class_counts[det.class_name] += 1
            tracker.class_confidences[det.class_name].append(float(det.confidence))
            
            # Save frame to history
            all_trackers_snapshot = self._create_trackers_snapshot()
            quality_score = calculate_quality_score(frame, det.box, det.confidence)
            tracker.save_frame_to_history(
                frame, det.box, quality_score, all_trackers_snapshot
            )
        
        # Create new trackers
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_tracker = KalmanBoxTracker(
                det.box,
                max_history_size=self.settings.tracking.max_history_size
            )
            new_tracker.confidence = det.confidence
            new_tracker.class_name = det.class_name
            new_tracker.color = self.detector.get_color(det.class_id)
            new_tracker.class_counts[det.class_name] += 1
            new_tracker.class_confidences[det.class_name].append(float(det.confidence))
            self.trackers.append(new_tracker)
        
        # Remove old trackers
        self.trackers = [
            t for i, t in enumerate(self.trackers)
            if not (i in unmatched_trks and 
                    t.time_since_update > self.settings.tracking.max_age)
        ]
    
    def _create_trackers_snapshot(self) -> List[dict]:
        """Create snapshot of all current trackers."""
        snapshot = []
        for t in self.trackers:
            if t.hits >= self.settings.tracking.min_hits_to_display:
                box = t.get_state()[0]
                if not np.any(np.isnan(box)):
                    snapshot.append({
                        'id': t.id,
                        'box': box.astype(int).copy(),
                        'class_name': t.class_name,
                        'confidence': t.confidence,
                        'color': t.color if t.color else (0, 255, 0)
                    })
        return snapshot
    
    def _check_tripwires(
        self,
        frame: np.ndarray,
        predicted_boxes: List[np.ndarray]
    ) -> None:
        """Check tripwire crossings and finalize vehicles."""
        if not self.overlay or not self.overlay.tripwires:
            return
        
        for i, tracker in enumerate(self.trackers):
            if (not tracker.has_been_classified and
                tracker.hits >= self.settings.tracking.min_hits_for_classification):
                
                current_box = (predicted_boxes[i] if i < len(predicted_boxes)
                             else tracker.get_state()[0])
                
                for line in self.overlay.tripwires:
                    if check_bbox_line_intersection(current_box, line[0], line[1]):
                        final_data = tracker.finalize_classification()
                        
                        if final_data:
                            # Save metadata
                            save_vehicle_metadata(
                                final_data,
                                metadata_file=str(self.settings.get_absolute_path(
                                    self.settings.output.metadata_file
                                ))
                            )
                            
                            # Add current frame context
                            final_data['frame'] = frame.copy()
                            final_data['all_trackers'] = self._create_trackers_snapshot()
                            
                            self.all_finalized_vehicles.append(final_data)
                            
                            # Send alert
                            if self.alert_notifier:
                                try:
                                    self.alert_notifier.notification_queue.put(
                                        final_data,
                                        block=False
                                    )
                                except queue.Full:
                                    logger.warning("Alert queue full, dropping alert")
                        
                        tracker.has_been_classified = True
                        tracker.color = (0, 255, 255)  # Yellow
                        break
    
    def _draw_trackers(self, frame: np.ndarray) -> None:
        """Draw tracker bounding boxes on frame."""
        for tracker in self.trackers:
            if tracker.hits >= self.settings.tracking.min_hits_to_display:
                box = tracker.get_state()[0]
                if not np.any(np.isnan(box)):
                    x1, y1, x2, y2 = box.astype(int)
                    label = (f"ID {tracker.id}: {tracker.class_name} "
                           f"({tracker.confidence:.2f})")
                    color = tracker.color if tracker.color else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.fps_frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (time.time() - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        # Print stats periodically
        if time.time() - self.last_fps_print >= 1.0:
            logger.info(
                f"FPS: {self.current_fps:.2f} | Trackers: {len(self.trackers)} | "
                f"Finalized: {len(self.all_finalized_vehicles)} | "
                f"Frame: {self.frame_counter}"
            )
            self.last_fps_print = time.time()
    
    def _draw_stats(self, frame: np.ndarray) -> None:
        """Draw statistics on frame."""
        cv2.putText(
            frame, f"FPS: {self.current_fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Trackers: {len(self.trackers)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            frame, f"Finalized: {len(self.all_finalized_vehicles)}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        if self.video_writer:
            self.video_writer.release()
        
        if self.alert_notifier:
            self.alert_notifier.stop()
        
        logger.info("="*70)
        logger.info(" SUMMARY")
        logger.info("="*70)
        logger.info(f"Total finalized vehicles: {len(self.all_finalized_vehicles)}")
        logger.info(f"Total frames processed: {self.frame_counter}")
        logger.info("System stopped successfully")
