"""57 — Face detection: from raw outputs to boxes

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.
"""

import numpy as np
from typing import Tuple, List


def decode_detections(
    raw_outputs: np.ndarray, score_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw detector outputs of shape (N, 15) into boxes, landmarks, and scores.

    Args:
        raw_outputs: A 2D NumPy array of shape (N, 15). Each row contains:
                     [x, y, w, h, lm1_x, lm1_y, ..., lm5_x, lm5_y, score]
        score_threshold: Minimum confidence score to keep a detection.

    Returns:
        A tuple of three NumPy arrays:
        - boxes: shape (M, 4) containing [x, y, w, h] of filtered detections.
        - landmarks: shape (M, 5, 2) containing the 5 landmark coordinates.
        - scores: shape (M,) containing the confidence scores.
    """
    raise NotImplementedError("your code here")


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: 1D NumPy array of shape (4,) representing [x, y, w, h].
        box2: 1D NumPy array of shape (4,) representing [x, y, w, h].

    Returns:
        The IoU value as a float in the range [0.0, 1.0].
    """
    raise NotImplementedError("your code here")


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes: NumPy array of shape (N, 4) representing [x, y, w, h].
        scores: NumPy array of shape (N,) representing confidence scores.
        iou_threshold: Overlap threshold above which boxes are suppressed.

    Returns:
        A list of integer indices of the boxes to keep, sorted by score descending.
    """
    raise NotImplementedError("your code here")


def scale_boxes(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Scale bounding boxes and clip them to image boundaries.

    Args:
        boxes: NumPy array of shape (N, 4) representing [x, y, w, h].
        scale_x: Scale factor for the horizontal axis.
        scale_y: Scale factor for the vertical axis.
        max_width: Maximum valid width boundary (original image width).
        max_height: Maximum valid height boundary (original image height).

    Returns:
        NumPy array of shape (N, 4) containing the scaled and clipped [x, y, w, h] boxes.
    """
    raise NotImplementedError("your code here")


def crop_regions(image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    """Crop bounding box regions from an image.

    Coordinates must be rounded to the nearest integer. Out-of-bounds coordinates
    must be safely clipped to the image boundaries [0, height] and [0, width].

    Args:
        image: NumPy array of shape (H, W, C) or (H, W).
        boxes: NumPy array of shape (N, 4) representing [x, y, w, h].

    Returns:
        A list of cropped NumPy arrays corresponding to each bounding box.
    """
    raise NotImplementedError("your code here")
