"""36 - grid and anchor-based object detection (YOLO)

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 36            test your work
    ../run 36 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")


def _iou_xyxy(box_a, box_b):
    """Single-pair IoU, (x1, y1, x2, y2) format - the same formula as
    module 27's box_iou, not a batched (N, M) version, since greedy
    matching below only ever needs one prediction against one ground
    truth at a time."""
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")
