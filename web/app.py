"""Flask application factory."""

import time
import threading
from flask import Flask

from utils.logger import get_logger

logger = get_logger(__name__)


# Global state
_camera_configuration = None
_config_ready_event = threading.Event()
_flask_shutdown_event = threading.Event()
_web_frame = None
_web_frame_lock = threading.Lock()


def get_camera_configuration():
    """Get current camera configuration."""
    return _camera_configuration


def set_camera_configuration(config):
    """Set camera configuration."""
    global _camera_configuration
    _camera_configuration = config


def get_config_ready_event():
    """Get configuration ready event."""
    return _config_ready_event


def get_flask_shutdown_event():
    """Get Flask shutdown event."""
    return _flask_shutdown_event


def get_web_frame():
    """Get current web frame (thread-safe)."""
    with _web_frame_lock:
        return _web_frame.copy() if _web_frame is not None else None


def set_web_frame(frame):
    """Set web frame (thread-safe)."""
    global _web_frame
    with _web_frame_lock:
        _web_frame = frame


def create_app():
    """
    Create and configure Flask application.
    
    Returns:
        Flask app instance
    """
    app = Flask(__name__, template_folder='../templates')
    
    # Import routes after app creation to avoid circular imports
    from .routes import register_routes
    register_routes(app)
    
    logger.info("Flask app created successfully")
    
    return app


def run_web_server(host='0.0.0.0', port=5000):
    """
    Run Flask web server.
    
    Args:
        host: Host address
        port: Port number
    """
    app = create_app()
    logger.info(f"Flask server starting at http://{host}:{port}")
    logger.info("Video stream will run at ~5 FPS to save CPU")
    
    app.run(
        host=host,
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
