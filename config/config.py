"""Configuration management using Pydantic for validation."""

import os
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field, validator
import yaml


class CameraConfig(BaseModel):
    """Camera configuration."""
    source: str = Field(..., description="Camera source URL or device ID")
    processing_width: int = Field(640, ge=320, le=1920, description="Frame width for processing")
    fps: float = Field(25.0, gt=0, le=120, description="Target FPS")


class ModelConfig(BaseModel):
    """AI Model configuration."""
    param_path: str = Field(..., description="Path to NCNN param file")
    bin_path: str = Field(..., description="Path to NCNN bin file")
    input_size: int = Field(640, description="Model input size")
    conf_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Default confidence threshold")
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0, description="IOU threshold for NMS")
    per_class_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'car': 0.73,
            'motorbike': 0.63,
            'person': 0.15,
            'truck': 0.3
        },
        description="Per-class confidence thresholds"
    )


class TrackingConfig(BaseModel):
    """Tracking configuration."""
    detection_interval: int = Field(3, ge=1, le=10, description="Detection every N frames")
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0, description="IOU threshold for tracking")
    max_age: int = Field(10, ge=1, le=100, description="Max frames to keep tracker without detection")
    min_hits_to_display: int = Field(2, ge=1, le=10, description="Min hits before displaying tracker")
    min_hits_for_classification: int = Field(5, ge=1, le=20, description="Min hits before finalizing classification")
    max_history_size: int = Field(15, ge=5, le=50, description="Max frames to keep in history")


class AlertConfig(BaseModel):
    """Alert notification configuration."""
    api_url: str = Field(
        "https://api-gw.autoprocai.com/smartaihub/push_alert",
        description="API endpoint for alerts"
    )
    camera_id: Optional[str] = Field(None, description="Camera ID for API")
    ai_module_id: Optional[str] = Field(None, description="AI Module ID for API")
    zone_id: str = Field("", description="Zone ID for API")
    
    # Telegram configuration
    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram chat/group ID")
    
    # Alert queue settings
    queue_maxsize: int = Field(20, ge=1, le=100, description="Max alert queue size")
    skip_person_alerts: bool = Field(True, description="Skip alerts for 'person' class")


class WebConfig(BaseModel):
    """Web interface configuration."""
    enabled: bool = Field(True, description="Enable web interface for configuration")
    host: str = Field("0.0.0.0", description="Flask host")
    port: int = Field(5000, ge=1024, le=65535, description="Flask port")
    stream_fps: int = Field(5, ge=1, le=30, description="Web stream FPS")


class OutputConfig(BaseModel):
    """Output configuration."""
    save_video: bool = Field(False, description="Save output video")
    video_output_path: str = Field("output/videos/output.avi", description="Output video path")
    video_fps: float = Field(20.0, gt=0, le=60, description="Output video FPS")
    video_codec: str = Field("XVID", description="Video codec fourcc")
    
    metadata_file: str = Field("data/logs/vehicle_logs.json", description="Vehicle metadata JSON file")
    
    enable_ui: bool = Field(False, description="Enable OpenCV UI window")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    log_dir: str = Field("logs", description="Log directory")
    max_bytes: int = Field(10485760, description="Max log file size (10MB)")
    backup_count: int = Field(5, description="Number of backup log files")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )


class Settings(BaseModel):
    """Main application settings."""
    camera: CameraConfig
    model: ModelConfig
    tracking: TrackingConfig
    alert: AlertConfig
    web: WebConfig
    output: OutputConfig
    logging: LoggingConfig
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @validator('project_root', pre=True, always=True)
    def set_project_root(cls, v):
        """Ensure project root is a Path object."""
        return Path(v) if not isinstance(v, Path) else v
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path from project root."""
        return self.project_root / relative_path
    
    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_assignment = True


def load_settings_from_yaml(yaml_path: str = None) -> Settings:
    """
    Load settings from YAML file.
    
    Args:
        yaml_path: Path to YAML config file. If None, looks for settings.yaml in config dir.
    
    Returns:
        Settings object
    """
    if yaml_path is None:
        config_dir = Path(__file__).parent
        yaml_path = config_dir / "settings.yaml"
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Settings(**config_dict)


def load_settings_from_env() -> Settings:
    """
    Load settings from environment variables.
    Useful for Docker/production deployments.
    
    Returns:
        Settings object
    """
    config_dict = {
        'camera': {
            'source': os.getenv('CAMERA_SOURCE', ''),
            'processing_width': int(os.getenv('PROCESSING_WIDTH', '640')),
            'fps': float(os.getenv('CAMERA_FPS', '25.0')),
        },
        'model': {
            'param_path': os.getenv('MODEL_PARAM_PATH', 'models/model_best_data/model.ncnn.param'),
            'bin_path': os.getenv('MODEL_BIN_PATH', 'models/model_best_data/model.ncnn.bin'),
            'conf_threshold': float(os.getenv('CONF_THRESHOLD', '0.3')),
        },
        'tracking': {
            'detection_interval': int(os.getenv('DETECTION_INTERVAL', '3')),
            'iou_threshold': float(os.getenv('IOU_THRESHOLD', '0.3')),
            'max_age': int(os.getenv('MAX_AGE', '10')),
            'min_hits_to_display': int(os.getenv('MIN_HITS_DISPLAY', '2')),
            'min_hits_for_classification': int(os.getenv('MIN_HITS_CLASSIFICATION', '5')),
        },
        'alert': {
            'camera_id': os.getenv('CAMERA_ID'),
            'ai_module_id': os.getenv('AI_MODULE_ID'),
            'zone_id': os.getenv('ZONE_ID', ''),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        },
        'web': {
            'enabled': os.getenv('WEB_ENABLED', 'true').lower() == 'true',
            'host': os.getenv('WEB_HOST', '0.0.0.0'),
            'port': int(os.getenv('WEB_PORT', '5000')),
        },
        'output': {
            'save_video': os.getenv('SAVE_VIDEO', 'false').lower() == 'true',
            'video_output_path': os.getenv('VIDEO_OUTPUT_PATH', 'output/videos/output.avi'),
            'enable_ui': os.getenv('ENABLE_UI', 'false').lower() == 'true',
        },
        'logging': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_dir': os.getenv('LOG_DIR', 'logs'),
        }
    }
    
    return Settings(**config_dict)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None, use_env: bool = False) -> Settings:
    """
    Get or create settings instance (Singleton pattern).
    
    Args:
        config_path: Path to YAML config file
        use_env: If True, load from environment variables instead of YAML
    
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None:
        if use_env:
            _settings = load_settings_from_env()
        else:
            _settings = load_settings_from_yaml(config_path)
    
    return _settings


def reset_settings():
    """Reset settings instance (useful for testing)."""
    global _settings
    _settings = None
