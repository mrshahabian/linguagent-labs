"""57 — Face detection: from raw outputs to boxes

Reference solutions for the exercises.
"""

import numpy as np
from typing import Tuple, List


def decode_detections(
    raw_outputs: np.ndarray, score_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = raw_outputs.dtype
    if len(raw_outputs) == 0:
        return (
            np.empty((0, 4), dtype=dtype),
            np.empty((0, 5, 2), dtype=dtype),
            np.empty((0,), dtype=dtype),
        )

    scores = raw_outputs[:, 14]
    keep = scores >= score_threshold
    filtered = raw_outputs[keep]

    if len(filtered) == 0:
        return (
            np.empty((0, 4), dtype=dtype),
            np.empty((0, 5, 2), dtype=dtype),
            np.empty((0,), dtype=dtype),
        )

    boxes = filtered[:, 0:4]
    landmarks = filtered[:, 4:14].reshape(-1, 5, 2)
    filtered_scores = filtered[:, 14]

    return boxes, landmarks, filtered_scores


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(int(idx))

        if len(order) == 1:
            break

        remaining_boxes = boxes[order[1:]]
        current_box = boxes[idx]

        # Vectorized IoU calculation for the remaining boxes
        x1_1, y1_1, w1, h1 = current_box
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2s = remaining_boxes[:, 0]
        y1_2s = remaining_boxes[:, 1]
        w2s = remaining_boxes[:, 2]
        h2s = remaining_boxes[:, 3]
        x2_2s = x1_2s + w2s
        y2_2s = y1_2s + h2s

        inter_x1 = np.maximum(x1_1, x1_2s)
        inter_y1 = np.maximum(y1_1, y1_2s)
        inter_x2 = np.minimum(x2_1, x2_2s)
        inter_y2 = np.minimum(y2_1, y2_2s)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        area2s = w2s * h2s
        union_area = area1 + area2s - inter_area

        ious = np.zeros_like(inter_area)
        valid = union_area > 0
        ious[valid] = inter_area[valid] / union_area[valid]

        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def scale_boxes(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=boxes.dtype)

    x1 = np.clip(boxes[:, 0] * scale_x, 0, max_width)
    y1 = np.clip(boxes[:, 1] * scale_y, 0, max_height)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2]) * scale_x, 0, max_width)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3]) * scale_y, 0, max_height)

    w = x2 - x1
    h = y2 - y1

    return np.column_stack((x1, y1, w, h))


def crop_regions(image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    crops = []
    if len(boxes) == 0:
        return crops

    rounded = np.round(boxes).astype(int)
    h_img, w_img = image.shape[:2]

    for box in rounded:
        x, y, w, h = box
        x1 = max(0, min(x, w_img))
        y1 = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))
        crops.append(image[y1:y2, x1:x2])

    return crops
