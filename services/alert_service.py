"""Alert notification services."""

import cv2
import json
import os
import queue
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from threading import Thread, Event
from typing import Dict, Any, Optional

from utils.logger import LoggerMixin


class AlertService(ABC, LoggerMixin):
    """Abstract base class for alert services."""
    
    @abstractmethod
    def send_alert(self, vehicle_data: Dict[str, Any]) -> bool:
        """
        Send alert for a vehicle.
        
        Args:
            vehicle_data: Vehicle data from tracker finalization
        
        Returns:
            True if alert sent successfully, False otherwise
        """
        pass


class APIAlertService(AlertService):
    """Alert service using HTTP API."""
    
    def __init__(
        self,
        api_url: str,
        camera_id: str,
        ai_module_id: str,
        zone_id: str = "",
        skip_person: bool = True
    ):
        """
        Initialize API alert service.
        
        Args:
            api_url: API endpoint URL
            camera_id: Camera ID
            ai_module_id: AI module ID
            zone_id: Zone ID (optional)
            skip_person: Skip alerts for 'person' class
        """
        self.api_url = api_url
        self.camera_id = camera_id
        self.ai_module_id = ai_module_id
        self.zone_id = zone_id
        self.skip_person = skip_person
    
    def send_alert(self, vehicle_data: Dict[str, Any]) -> bool:
        """Send alert to API endpoint."""
        try:
            vehicle_id = vehicle_data['id']
            vehicle_class = vehicle_data['final_class']
            confidence = vehicle_data.get('confidence', 0.0)
            
            # Skip person alerts if configured
            if self.skip_person and vehicle_class == 'person':
                self.logger.info(f"Skipping alert for person (ID: {vehicle_id})")
                return True
            
            # Prepare frame image
            frame, last_box = self._prepare_alert_image(vehicle_data)
            if frame is None:
                self.logger.error("No frame available for alert")
                return False
            
            # Draw bounding boxes
            self._draw_alert_boxes(frame, vehicle_data, last_box)
            
            # Encode image
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                self.logger.error("Failed to encode image")
                return False
            
            image_bytes = encoded_image.tobytes()
            
            # Prepare request
            tz_utc7 = timezone(timedelta(hours=7))
            timestamp = datetime.now(tz_utc7).strftime("%Y-%m-%d %H:%M:%S")
            title = f"{vehicle_class} - {timestamp}"
            
            data = {
                'camera_id': self.camera_id,
                'ai_module_id': self.ai_module_id,
                'title': title,
                'status': 'auto',
                'zone_id': self.zone_id
            }
            
            files = {
                'img_file': ('vehicle.jpg', image_bytes, 'image/jpeg')
            }
            
            # Send request
            response = requests.post(
                self.api_url,
                data=data,
                files=files,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(
                    f"Alert sent successfully for ID {vehicle_id}: {result}"
                )
                return True
            else:
                self.logger.error(
                    f"Alert failed (status {response.status_code}): "
                    f"{response.text}"
                )
                return False
        
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}", exc_info=True)
            return False
    
    def _prepare_alert_image(
        self,
        vehicle_data: Dict[str, Any]
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Prepare frame and bounding box for alert image."""
        best_frame_data = vehicle_data.get('best_frame_data')
        
        if best_frame_data:
            frame = best_frame_data['frame'].copy()
            last_box = best_frame_data['box']
            self.logger.debug(
                f"Using best frame (quality: "
                f"{best_frame_data['quality_score']:.2f})"
            )
            return frame, last_box
        
        # Fallback to current frame
        frame = vehicle_data.get('frame')
        last_box = vehicle_data.get('last_box')
        
        if frame is not None:
            self.logger.warning("Using current frame (no best frame available)")
        
        return frame, last_box
    
    def _draw_alert_boxes(
        self,
        frame: Any,
        vehicle_data: Dict[str, Any],
        main_box: Any
    ) -> None:
        """Draw bounding boxes on alert image."""
        vehicle_id = vehicle_data['id']
        vehicle_class = vehicle_data['final_class']
        confidence = vehicle_data.get('confidence', 0.0)
        
        # Get trackers snapshot
        best_frame_data = vehicle_data.get('best_frame_data')
        if best_frame_data:
            all_trackers = best_frame_data.get('all_trackers_snapshot', [])
        else:
            all_trackers = vehicle_data.get('all_trackers', [])
        
        # Draw other trackers first (skip person)
        for tracker_info in all_trackers:
            if tracker_info['id'] == vehicle_id:
                continue
            
            if tracker_info.get('class_name') == 'person':
                continue
            
            box = tracker_info['box']
            x1, y1, x2, y2 = box
            color = tracker_info.get('color', (0, 255, 0))
            class_name = tracker_info.get('class_name', 'unknown')
            conf = tracker_info.get('confidence', 0.0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {tracker_info['id']}: {class_name} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw main vehicle box (yellow, larger)
        if main_box is not None:
            x1, y1, x2, y2 = main_box
            color = (0, 255, 255)  # Yellow
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"ID {vehicle_id}: {vehicle_class} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


class TelegramAlertService(AlertService):
    """Alert service using Telegram bot."""
    
    def __init__(self, bot_token: str, chat_id: str, skip_person: bool = True):
        """
        Initialize Telegram alert service.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat or group ID
            skip_person: Skip alerts for 'person' class
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.skip_person = skip_person
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    def send_alert(self, vehicle_data: Dict[str, Any]) -> bool:
        """Send alert via Telegram."""
        try:
            vehicle_id = vehicle_data['id']
            vehicle_class = vehicle_data['final_class']
            
            # Skip person alerts if configured
            if self.skip_person and vehicle_class == 'person':
                self.logger.info(
                    f"Skipping Telegram alert for person (ID: {vehicle_id})"
                )
                return True
            
            # Prepare image
            frame, _ = self._prepare_image(vehicle_data)
            if frame is None:
                return False
            
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                self.logger.error("Failed to encode image")
                return False
            
            image_bytes = encoded_image.tobytes()
            
            # Prepare caption
            tz_utc7 = timezone(timedelta(hours=7))
            timestamp = datetime.now(tz_utc7).strftime("%Y-%m-%d %H:%M:%S")
            
            caption = (
                f"**Phương tiện được nhận dạng**\n\n"
                f"**Loại xe:** {vehicle_class}\n"
                f"**Thời gian:** {timestamp}"
            )
            
            # Send request
            files = {'photo': ('vehicle.jpg', image_bytes, 'image/jpeg')}
            payload = {
                'chat_id': self.chat_id,
                'caption': caption,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(
                self.api_url,
                files=files,
                data=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(
                    f"Telegram alert sent successfully for ID {vehicle_id}"
                )
                return True
            else:
                self.logger.error(
                    f"Telegram alert failed: {response.text}"
                )
                return False
        
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}", exc_info=True)
            return False
    
    def _prepare_image(self, vehicle_data: Dict[str, Any]) -> tuple:
        """Prepare image for Telegram alert."""
        best_frame_data = vehicle_data.get('best_frame_data')
        
        if best_frame_data:
            return best_frame_data['frame'].copy(), True
        
        frame = vehicle_data.get('frame')
        return frame, frame is not None


class AlertNotifier(LoggerMixin):
    """
    Alert notifier with queue-based processing.
    
    Runs in a separate thread to send alerts without blocking main loop.
    """
    
    def __init__(
        self,
        services: list[AlertService],
        notification_queue: queue.Queue,
        stop_event: Event
    ):
        """
        Initialize alert notifier.
        
        Args:
            services: List of alert services to use
            notification_queue: Queue for vehicle data
            stop_event: Event to signal stop
        """
        self.services = services
        self.notification_queue = notification_queue
        self.stop_event = stop_event
        
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        
        self.logger.info(
            f"AlertNotifier started with {len(services)} services"
        )
    
    def _run(self) -> None:
        """Main processing loop."""
        while not self.stop_event.is_set():
            try:
                vehicle_data = self.notification_queue.get(timeout=1)
                self._send_notifications(vehicle_data)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
        
        self.logger.info("AlertNotifier stopped")
    
    def _send_notifications(self, vehicle_data: Dict[str, Any]) -> None:
        """Send notifications using all configured services."""
        vehicle_id = vehicle_data.get('id', 'unknown')
        
        for service in self.services:
            try:
                service.send_alert(vehicle_data)
            except Exception as e:
                self.logger.error(
                    f"Error sending alert via {service.__class__.__name__}: {e}",
                    exc_info=True
                )
    
    def stop(self, timeout: float = 3.0) -> None:
        """
        Stop alert notifier.
        
        Args:
            timeout: Maximum time to wait for thread to finish
        """
        self.stop_event.set()
        self.thread.join(timeout=timeout)


def save_vehicle_metadata(
    vehicle_data: Dict[str, Any],
    metadata_file: str = "data/logs/vehicle_logs.json"
) -> None:
    """
    Save vehicle metadata to JSON file.
    
    Args:
        vehicle_data: Vehicle data from tracker finalization
        metadata_file: Path to JSON file
    """
    from pathlib import Path
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    
    try:
        # Ensure directory exists
        Path(metadata_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        tz_utc7 = timezone(timedelta(hours=7))
        timestamp = datetime.now(tz_utc7).strftime("%Y-%m-%d %H:%M:%S")
        
        metadata = {
            "tracker_id": vehicle_data['id'],
            "final_class": vehicle_data['final_class'],
            "final_confidence": float(vehicle_data.get('confidence', 0.0)),
            "class_history": {
                "votes": dict(vehicle_data.get('class_details', {})),
                "confidences": vehicle_data.get('class_confidences', {})
            },
            "total_votes": sum(vehicle_data.get('class_details', {}).values()),
            "best_frame_quality": float(
                vehicle_data.get('best_frame_data', {}).get('quality_score', 0.0)
                if vehicle_data.get('best_frame_data') else 0.0
            ),
            "timestamp": timestamp,
            "bbox_final": (
                vehicle_data.get('last_box', []).tolist()
                if hasattr(vehicle_data.get('last_box', []), 'tolist')
                else list(vehicle_data.get('last_box', []))
            ),
        }
        
        # Read existing data
        all_vehicles = []
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_vehicles = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted file {metadata_file}, creating new one")
                all_vehicles = []
        
        # Append new data
        all_vehicles.append(metadata)
        
        # Write back
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_vehicles, f, ensure_ascii=False, indent=2)
        
        logger.info(
            f"Saved metadata for Tracker ID {vehicle_data['id']} "
            f"({vehicle_data['final_class']}) to {metadata_file}"
        )
    
    except Exception as e:
        logger.error(f"Error saving metadata: {e}", exc_info=True)
