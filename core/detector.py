"""YOLOv8 NCNN Detector for vehicle detection."""

import cv2
import numpy as np
import ncnn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from utils.logger import LoggerMixin


@dataclass
class Detection:
    """Detection result data class."""
    box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YOLOv8Detector(LoggerMixin):
    """
    YOLOv8 NCNN Detector optimized for ARM CPU (Jetson Nano).
    
    Features:
    - FP16 arithmetic for speed
    - Winograd & SGEMM convolution optimization
    - Per-class confidence thresholds
    - Configurable allowed classes
    """
    
    def __init__(
        self,
        param_path: str,
        bin_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        per_class_conf: Optional[Dict[str, float]] = None,
        allowed_classes: Optional[List[str]] = None,
        num_threads: int = 5
    ):
        """
        Initialize YOLOv8 NCNN detector.
        
        Args:
            param_path: Path to .param file
            bin_path: Path to .bin file
            input_size: Model input size (default: 640)
            conf_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
            per_class_conf: Per-class confidence thresholds
            allowed_classes: List of allowed class names (None = all)
            num_threads: Number of CPU threads
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.per_class_conf = per_class_conf or {}
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        
        # NCNN network
        self.net = ncnn.Net()
        self._configure_ncnn_optimizations(num_threads)
        
        # Load model
        self.logger.info(f"Loading NCNN model from {param_path}")
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # Class names and colors
        self.class_names = ['car', 'motorbike', 'person', 'truck']
        np.random.seed(112006)
        self.colors = [
            tuple(np.random.randint(0, 255, 3).tolist()) 
            for _ in range(len(self.class_names))
        ]
        
        self.logger.info(f"YOLOv8 Detector initialized with {len(self.class_names)} classes")
    
    def _configure_ncnn_optimizations(self, num_threads: int) -> None:
        """Configure NCNN optimizations for ARM CPU."""
        self.logger.info("Applying NCNN optimizations for ARM CPU...")
        self.net.opt.use_vulkan_compute = False  # No GPU
        self.net.opt.use_winograd_convolution = True
        self.net.opt.use_sgemm_convolution = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.use_packing_layout = True
        self.net.opt.num_threads = num_threads
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocess image for YOLO input.
        
        Args:
            img: Input image (BGR format)
        
        Returns:
            Tuple of (padded_image, scale, pad_h, pad_w)
        """
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded, scale, pad_h, pad_w
    
    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_h: int,
        pad_w: int,
        orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Postprocess YOLO output.
        
        Args:
            output: Raw model output
            scale: Image scale factor
            pad_h: Padding height
            pad_w: Padding width
            orig_shape: Original image shape (h, w)
        
        Returns:
            List of Detection objects
        """
        predictions = output[0].T
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if boxes.shape[0] == 0:
            return []

        # Convert from center format to corner format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Remove padding and scale back to original size
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_shape[1])
        y1 = np.clip(y1, 0, orig_shape[0])
        x2 = np.clip(x2, 0, orig_shape[1])
        y2 = np.clip(y2, 0, orig_shape[0])
        
        # NMS
        boxes_for_nms = np.column_stack([x1, y1, x2 - x1, y2 - y1]).astype(int)
        keep_indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
    
        if isinstance(keep_indices, np.ndarray):
            keep_indices = keep_indices.flatten()
        else:
            return []

        # Build detection results
        results = []
        for idx in keep_indices:
            class_id = class_ids[idx]
            
            # Validate class_id
            if class_id < 0 or class_id >= len(self.class_names):
                self.logger.warning(f"Invalid class_id {class_id}, skipping")
                continue
            
            class_name = self.class_names[class_id]
            
            # Filter by allowed classes
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue
            
            # Apply per-class threshold
            final_threshold = self.per_class_conf.get(class_name, self.conf_threshold)
            if confidences[idx] < final_threshold:
                continue
            
            box = np.array([x1[idx], y1[idx], x2[idx], y2[idx]])
            results.append(Detection(
                box=box.astype(int),
                confidence=float(confidences[idx]),
                class_id=int(class_id),
                class_name=class_name
            ))
        
        return results
    
    def detect(self, img: np.ndarray) -> List[Detection]:
        """
        Run detection on an image.
        
        Args:
            img: Input image (BGR format)
        
        Returns:
            List of Detection objects
        """
        orig_shape = img.shape[:2]
        
        # Preprocess
        preprocessed, scale, pad_h, pad_w = self.preprocess(img)
        
        # Create NCNN Mat
        mat_in = ncnn.Mat.from_pixels(
            preprocessed,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.input_size,
            self.input_size
        )
        
        # Normalize
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])
        
        # Inference
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        
        # Convert output to numpy
        output = np.array(mat_out)
        
        # Reshape if needed
        if len(output.shape) == 1:
            output = output.reshape(84, -1)[np.newaxis, :, :]
        elif len(output.shape) == 2:
            output = output[np.newaxis, :, :]
        
        # Postprocess
        detections = self.postprocess(output, scale, pad_h, pad_w, orig_shape)
        
        return detections
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID."""
        if 0 <= class_id < len(self.colors):
            return self.colors[class_id]
        return (0, 255, 0)  # Default green
