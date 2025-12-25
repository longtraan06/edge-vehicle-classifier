"""Flask routes for web interface."""

import time
import cv2
import numpy as np
from threading import Thread
from flask import render_template, Response, request, jsonify

from utils.logger import get_logger
from . import app as web_app

logger = get_logger(__name__)


def register_routes(app):
    """Register all routes to Flask app."""
    
    @app.route('/')
    def index():
        """Main configuration page."""
        return render_template('config_single.html')
    
    @app.route('/video_feed')
    def video_feed():
        """Video stream endpoint."""
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    @app.route('/save_config', methods=['POST'])
    @app.route('/save_config/0', methods=['POST'])
    def save_config():
        """Save ROI and tripwire configuration."""
        logger.info("Received config save request from web")
        data = request.json
        
        # Get current frame dimensions
        web_frame = web_app.get_web_frame()
        if web_frame is None:
            logger.warning("No frame available for scaling")
            return jsonify({
                "status": "error",
                "message": "No frame available"
            }), 400
        
        frame_h, frame_w = web_frame.shape[:2]
        display_width = data.get('displayWidth', 800)
        scale = frame_w / display_width
        
        logger.debug(
            f"Frame size: {frame_w}x{frame_h}, "
            f"Display width: {display_width}, Scale: {scale}"
        )
        
        # Scale ROI points
        scaled_roi = []
        if data.get('roi'):
            scaled_roi = [
                (int(p['x'] * scale), int(p['y'] * scale))
                for p in data['roi']
            ]
            logger.debug(f"ROI points: {len(scaled_roi)}")
        
        # Scale tripwires
        scaled_tripwires = []
        if data.get('tripwires'):
            for line in data['tripwires']:
                start_point = (
                    int(line['start']['x'] * scale),
                    int(line['start']['y'] * scale)
                )
                end_point = (
                    int(line['end']['x'] * scale),
                    int(line['end']['y'] * scale)
                )
                scaled_tripwires.append((start_point, end_point))
            logger.debug(f"Tripwires: {len(scaled_tripwires)}")
        
        # Save configuration
        config = {
            'roi': np.array([scaled_roi], dtype=np.int32),
            'tripwires': scaled_tripwires
        }
        web_app.set_camera_configuration(config)
        
        logger.info("✅ ROI and Tripwires configuration saved successfully!")
        logger.info("You can now click 'Start System' to begin tracking")
        
        return jsonify({
            "status": "success",
            "message": "Configuration saved!"
        })
    
    @app.route('/start_system', methods=['POST'])
    def start_system():
        """Start the tracking system."""
        logger.info("Received start signal from web")
        
        # Signal that configuration is ready
        config_ready_event = web_app.get_config_ready_event()
        config_ready_event.set()
        
        # Schedule Flask shutdown
        def delayed_shutdown():
            time.sleep(2)
            logger.info("Shutting down Flask web server...")
            flask_shutdown_event = web_app.get_flask_shutdown_event()
            flask_shutdown_event.set()
            logger.info("Flask server stopped (web interface closed)")
        
        Thread(target=delayed_shutdown, daemon=True).start()
        
        return jsonify({
            "status": "starting",
            "message": "System is starting. Web interface will close in 2 seconds..."
        })


def generate_frames():
    """
    Generator for video streaming.
    
    Yields:
        JPEG frames for multipart response
    """
    flask_shutdown_event = web_app.get_flask_shutdown_event()
    
    while not flask_shutdown_event.is_set():
        frame = web_app.get_web_frame()
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        flag, encoded_image = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encoded_image) + 
               b'\r\n')
        
        time.sleep(0.2)  # 5 FPS for web stream
