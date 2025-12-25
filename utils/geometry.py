"""Geometric calculations for detection and tracking."""

import numpy as np
from typing import Tuple, List


def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
    """
    Check if point q lies on line segment pr.
    
    Args:
        p: First point of line segment
        q: Point to check
        r: Second point of line segment
    
    Returns:
        True if q is on segment pr
    """
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


def orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
    """
    Find orientation of ordered triplet (p, q, r).
    
    Args:
        p: First point
        q: Second point
        r: Third point
    
    Returns:
        0 if collinear, 1 if clockwise, 2 if counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def line_intersects(
    p1: Tuple[float, float],
    q1: Tuple[float, float],
    p2: Tuple[float, float],
    q2: Tuple[float, float]
) -> bool:
    """
    Check if line segment p1q1 and p2q2 intersect.
    
    Args:
        p1: Start point of first line
        q1: End point of first line
        p2: Start point of second line
        q2: End point of second line
    
    Returns:
        True if lines intersect
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def check_bbox_line_intersection(
    bbox: np.ndarray,
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float]
) -> bool:
    """
    Check if bounding box intersects with a line.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        line_p1: Start point of line
        line_p2: End point of line
    
    Returns:
        True if bbox intersects with line
    """
    x1, y1, x2, y2 = bbox
    bbox_p1 = (x1, y1)
    bbox_p2 = (x2, y1)
    bbox_p3 = (x2, y2)
    bbox_p4 = (x1, y2)

    # Check all 4 edges of bbox
    if line_intersects(line_p1, line_p2, bbox_p1, bbox_p2):
        return True
    if line_intersects(line_p1, line_p2, bbox_p2, bbox_p3):
        return True
    if line_intersects(line_p1, line_p2, bbox_p3, bbox_p4):
        return True
    if line_intersects(line_p1, line_p2, bbox_p4, bbox_p1):
        return True
    
    # Check if line start point is inside bbox
    if (x1 <= line_p1[0] <= x2 and y1 <= line_p1[1] <= y2):
        return True
        
    return False


def calculate_iou_matrix(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
    """
    Calculate IoU (Intersection over Union) matrix between two sets of boxes.
    
    Args:
        boxesA: Array of boxes shape (N, 4) in format [x1, y1, x2, y2]
        boxesB: Array of boxes shape (M, 4) in format [x1, y1, x2, y2]
    
    Returns:
        IoU matrix of shape (N, M)
    """
    if boxesA.size == 0 or boxesB.size == 0:
        return np.empty((boxesA.shape[0], boxesB.shape[0]))
    
    boxesA = np.expand_dims(boxesA, axis=1)
    boxesB = np.expand_dims(boxesB, axis=0)
    
    xA = np.maximum(boxesA[..., 0], boxesB[..., 0])
    yA = np.maximum(boxesA[..., 1], boxesB[..., 1])
    xB = np.minimum(boxesA[..., 2], boxesB[..., 2])
    yB = np.minimum(boxesA[..., 3], boxesB[..., 3])
    
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxAArea = (boxesA[..., 2] - boxesA[..., 0]) * (boxesA[..., 3] - boxesA[..., 1])
    boxBArea = (boxesB[..., 2] - boxesB[..., 0]) * (boxesB[..., 3] - boxesB[..., 1])
    
    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / unionArea
    iou[unionArea == 0] = 0
    
    return iou


def associate_detections_to_trackers(
    tracked_boxes: np.ndarray,
    detections_boxes: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to tracked objects using Hungarian algorithm.
    
    Args:
        tracked_boxes: Array of tracked boxes (N, 4)
        detections_boxes: Array of detected boxes (M, 4)
        iou_threshold: Minimum IoU for valid match
    
    Returns:
        Tuple of (matches, unmatched_detections, unmatched_trackers)
        - matches: Array of matched pairs shape (K, 2) where K <= min(N, M)
        - unmatched_detections: Indices of unmatched detections
        - unmatched_trackers: Indices of unmatched trackers
    """
    from scipy.optimize import linear_sum_assignment
    
    tracked_boxes = np.asarray(tracked_boxes)
    detections_boxes = np.asarray(detections_boxes)

    if tracked_boxes.size == 0 or detections_boxes.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections_boxes)),
            np.arange(len(tracked_boxes)),
        )

    iou_matrix = calculate_iou_matrix(tracked_boxes, detections_boxes)
    cost_matrix = 1 - iou_matrix
    cost_matrix[np.isnan(cost_matrix)] = 1.0
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_trackers_indices = set()
    matched_detections_indices = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])
            matched_trackers_indices.add(r)
            matched_detections_indices.add(c)
    
    all_trackers_indices = set(range(tracked_boxes.shape[0]))
    all_detections_indices = set(range(detections_boxes.shape[0]))
    
    unmatched_trackers = np.array(list(all_trackers_indices - matched_trackers_indices))
    unmatched_detections = np.array(list(all_detections_indices - matched_detections_indices))

    return np.array(matches, dtype=int), unmatched_detections, unmatched_trackers
