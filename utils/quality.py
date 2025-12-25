"""Quality score calculation for selecting best frames."""

import numpy as np


def quick_quality_check(confidence: float, current_best_score: float) -> bool:
    """
    Quick check if detection could potentially beat current best score.
    Used to skip expensive calculations when not needed.
    
    Args:
        confidence: Detection confidence
        current_best_score: Current best quality score
    
    Returns:
        True if detection might beat current best, False otherwise
    """
    # Maximum possible score calculation (confidence * 1.5 + max area score + max centrality)
    max_possible_score = confidence * 1.5 + 1.5
    return max_possible_score > current_best_score


def calculate_quality_score(frame: np.ndarray, box: np.ndarray, confidence: float) -> float:
    """
    Calculate quality score for a detection frame.
    
    Quality score combines:
    - Detection confidence (weight: 1.5)
    - Normalized box area (weight: 0.5)
    - Centrality (distance from frame center, weight: 1.0)
    
    Args:
        frame: Input frame
        box: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence
    
    Returns:
        Quality score (higher is better)
    """
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return 0.0

    # Confidence component (weight: 1.5)
    score = confidence * 1.5

    # Box area component (weight: 0.5)
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame.shape[0] * frame.shape[1]
    normalized_area = box_area / frame_area
    score += normalized_area * 0.5

    # Centrality component (weight: 1.0)
    frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2
    box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    dist_from_center = np.sqrt((frame_center_x - box_center_x)**2 + (frame_center_y - box_center_y)**2)
    max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
    centrality_score = 1.0 - (dist_from_center / max_dist)
    score += centrality_score * 1.0

    return score
