"""Reference answers for 36 - grid and anchor-based object detection (YOLO)."""

import torch


def encode_grid_targets(boxes, class_ids, grid_size, img_size, num_classes):
    """Encode ground-truth boxes into a (grid_size, grid_size, 5+num_classes) tensor.

    YOLO's central idea: divide the image into a grid_size x grid_size
    grid: whichever cell contains a box's CENTER is the one cell
    "responsible" for predicting it - not every cell that overlaps it.
    That cell's slot becomes [objectness=1, x, y, w, h, one-hot class],
    where x, y are the center's offset WITHIN the cell (0 to 1, not
    absolute pixels) and w, h are the box's width/height as a fraction of
    the whole image. Cells with no assigned box are left at all zero,
    which is exactly what makes objectness=0 the right "no object here"
    signal for the loss to train against.

    boxes: (N, 4) in (x1, y1, x2, y2) absolute pixel coordinates.
    class_ids: (N,) integer class index per box.
    """
    target = torch.zeros(grid_size, grid_size, 5 + num_classes)
    cell_size = img_size / grid_size

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        col = min(int(center_x / cell_size), grid_size - 1)
        row = min(int(center_y / cell_size), grid_size - 1)

        x_offset = (center_x - col * cell_size) / cell_size
        y_offset = (center_y - row * cell_size) / cell_size
        width = (x2 - x1) / img_size
        height = (y2 - y1) / img_size

        target[row, col, 0] = 1.0
        target[row, col, 1] = x_offset
        target[row, col, 2] = y_offset
        target[row, col, 3] = width
        target[row, col, 4] = height
        target[row, col, 5 + int(class_id)] = 1.0

    return target


def decode_grid_predictions(grid_output, grid_size, img_size, conf_threshold):
    """Reverse encode_grid_targets: grid tensor -> list of (box, confidence, class).

    Only cells with objectness above conf_threshold produce a detection -
    most of a grid is "no object here" for any real image, which is
    exactly the class imbalance yolo_loss_components has to account for.
    Converts each surviving cell's relative (x, y, w, h) back to absolute
    (x1, y1, x2, y2) pixel coordinates, and reports the class with the
    highest score in that cell.

    Returns a list of (box_tensor, confidence, class_id) tuples, one per
    detection, in row-major cell order.
    """
    cell_size = img_size / grid_size
    detections = []

    for row in range(grid_size):
        for col in range(grid_size):
            cell = grid_output[row, col]
            confidence = cell[0].item()
            if confidence <= conf_threshold:
                continue

            x_offset, y_offset, width, height = cell[1:5]
            center_x = (col + x_offset) * cell_size
            center_y = (row + y_offset) * cell_size
            box_width = width * img_size
            box_height = height * img_size

            box = torch.tensor([
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ])
            class_id = int(torch.argmax(cell[5:]))
            detections.append((box, confidence, class_id))

    return detections


def best_anchor_for_box(box_wh, anchor_whs):
    """Pick the anchor whose SHAPE best matches box_wh, ignoring position.

    YOLOv2/v3 assign a ground-truth box to whichever of several predefined
    anchor shapes (typical width/height ratios, chosen up front by
    clustering the training set's boxes) it resembles most - not by IoU at
    the box's actual location, but by IoU as if both the box and every
    anchor were centered at the same point. This decouples "what shape is
    this object" from "where is it", so an anchor tuned for tall, narrow
    boxes (a person) doesn't compete unfairly with one tuned for wide,
    short boxes (a car) just because of where either happens to sit in the
    image.
    """
    box_w, box_h = box_wh
    box_area = box_w * box_h
    best_index, best_iou = 0, -1.0

    for index, (anchor_w, anchor_h) in enumerate(anchor_whs):
        inter_w = min(box_w, anchor_w)
        inter_h = min(box_h, anchor_h)
        intersection = inter_w * inter_h
        anchor_area = anchor_w * anchor_h
        union = box_area + anchor_area - intersection
        iou = intersection / union if union > 0 else 0.0
        if iou > best_iou:
            best_index, best_iou = index, iou

    return best_index


def yolo_loss_components(pred, target, lambda_coord=5.0, lambda_noobj=0.5):
    """The three weighted pieces of the YOLO loss, summed.

    A grid is mostly empty - most cells have no object in them - so a
    naive sum-of-squared-errors loss would be dominated by driving every
    empty cell's confidence to exactly zero, drowning out the (rarer, more
    important) signal from cells that DO contain an object. Two
    hyperparameters fix the imbalance in opposite directions:

      lambda_coord (>1): amplify the box-coordinate loss, since getting a
      real object's box right matters more than the raw squared-error
      scale would suggest.

      lambda_noobj (<1): shrink the confidence loss for cells with no
      object, so their sheer numerical majority doesn't dominate training.

    Both masks come from target's own objectness channel - only cells
    where target[..., 0] == 1 contribute coordinate and class loss; every
    cell contributes objectness loss, split into the responsible and
    not-responsible masks.
    """
    obj_mask = target[..., 0] == 1
    noobj_mask = ~obj_mask

    coord_loss = lambda_coord * torch.sum(
        (pred[obj_mask][:, 1:5] - target[obj_mask][:, 1:5]) ** 2
    )
    obj_loss = torch.sum((pred[obj_mask][:, 0] - target[obj_mask][:, 0]) ** 2)
    noobj_loss = lambda_noobj * torch.sum(
        (pred[noobj_mask][:, 0] - target[noobj_mask][:, 0]) ** 2
    )
    class_loss = torch.sum((pred[obj_mask][:, 5:] - target[obj_mask][:, 5:]) ** 2)

    return {
        "coord": coord_loss,
        "obj": obj_loss,
        "noobj": noobj_loss,
        "class": class_loss,
        "total": coord_loss + obj_loss + noobj_loss + class_loss,
    }


def _iou_xyxy(box_a, box_b):
    """Single-pair IoU, (x1, y1, x2, y2) format - the same formula as
    module 27's box_iou, not a batched (N, M) version, since greedy
    matching below only ever needs one prediction against one ground
    truth at a time."""
    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_detections_to_ground_truth(detections, ground_truth_boxes, iou_threshold=0.5):
    """Greedily match predictions to ground truth for a mAP-style evaluation.

    Detections are matched HIGHEST CONFIDENCE FIRST, and each ground-truth
    box can be claimed by at most one detection - a second detection that
    also overlaps an already-claimed box is a false positive, even at
    high IoU, because in a real detector it would be a duplicate the
    model should have suppressed itself (this is why NMS runs before
    evaluation, not just before display). This is the step module 32's
    average_precision doesn't include on its own: turning raw
    (box, confidence) detections into the (confidence, is_correct) pairs
    average_precision actually consumes.

    detections: list of (box, confidence, class_id), any order.
    Returns (confidences, labels) - labels[i] is 1 if detections[i] (after
    sorting by confidence) was matched to an unclaimed ground-truth box
    above iou_threshold, else 0.
    """
    ordered = sorted(detections, key=lambda d: d[1], reverse=True)
    claimed = [False] * len(ground_truth_boxes)
    confidences, labels = [], []

    for box, confidence, _class_id in ordered:
        best_index, best_iou = -1, iou_threshold
        for gt_index, gt_box in enumerate(ground_truth_boxes):
            if claimed[gt_index]:
                continue
            iou = _iou_xyxy(box, gt_box)
            if iou > best_iou:
                best_index, best_iou = gt_index, iou

        confidences.append(confidence)
        if best_index >= 0:
            claimed[best_index] = True
            labels.append(1)
        else:
            labels.append(0)

    return confidences, labels
