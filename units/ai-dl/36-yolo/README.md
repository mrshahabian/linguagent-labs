# 36 — Grid and anchor-based object detection (YOLO)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/36-yolo/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/ai-dl/yolo?utm_source=github&utm_medium=labs)

*~55 minutes · 20 tests*

Module 27 gives you the primitives — IoU, NMS. This module is the mechanics
specific to a one-shot, grid-based detector: turning ground-truth boxes into
a training target a convolutional grid can predict, decoding that grid back
into boxes, matching a box to the anchor shape it resembles, the loss
weighting that keeps a mostly-empty grid from collapsing to "predict
nothing everywhere," and the greedy matching step that turns raw detections
into the (confidence, correct) pairs module 32's `average_precision`
actually needs.

Work in `exercises.py`. Check your work with `python -m labkit check 36`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/ai-dl/36-yolo/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `encode_grid_targets` | the function | Turning ground-truth boxes into a trainable grid target |
| `decode_grid_predictions` | the function | Turning a model's raw grid output back into boxes |
| `best_anchor_for_box` | the function | YOLOv2/v3's shape-based anchor assignment |
| `yolo_loss_components` | the function | The loss-weighting that fixes YOLO's class imbalance |
| `match_detections_to_ground_truth` | the function | Feeding real detections into module 32's `average_precision` |

---

## 1. One cell is responsible for each box

```python
col = min(int(center_x / cell_size), grid_size - 1)
row = min(int(center_y / cell_size), grid_size - 1)
target[row, col, 0] = 1.0                 # objectness
target[row, col, 1:5] = x_offset, y_offset, width, height
target[row, col, 5 + class_id] = 1.0
```

YOLO's central move: divide the image into a `grid_size x grid_size`
grid, and whichever cell contains a box's **center** — not every cell it
overlaps — is the one cell responsible for predicting it. That cell's
slot holds `[objectness, x, y, w, h, one-hot class]`, where `x, y` are the
center's offset **within the cell** (0 to 1), and `w, h` are the box's
size as a fraction of the whole image. Every other cell stays all zero,
which doubles as exactly the right training signal for "no object here."

```python
>>> target = encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
>>> target[..., 0].sum()          # one cell marked responsible per box
tensor(2.)
```

**In `exercises.py`:** `encode_grid_targets`.

**Test it:** `python3 -m unittest test_exercises.TestEncodeGridTargets -v`

## 2. Decoding: only confident cells produce a detection

```python
confidence = cell[0].item()
if confidence <= conf_threshold:
    continue
center_x = (col + x_offset) * cell_size
...
class_id = int(torch.argmax(cell[5:]))
```

The reverse of encoding: for every cell above `conf_threshold`, convert
the relative `(x, y, w, h)` back to absolute pixel coordinates and read
off the highest-scoring class. Most cells in a real image have no
object, so most cells never produce a detection at all — that sparsity
is exactly what the loss in §4 has to be weighted against, or training
collapses to predicting "no object" everywhere, which is *usually*
right and therefore a tempting local minimum.

```python
>>> detections = decode_grid_predictions(target, grid_size=7, img_size=280, conf_threshold=0.5)
>>> detections[0]
(tensor([10., 10., 50., 90.]), 1.0, 1)
```

**In `exercises.py`:** `decode_grid_predictions`.

**Test it:** `python3 -m unittest test_exercises.TestDecodeGridPredictions -v`

## 3. Anchor assignment: match by shape, not position

```python
inter_w = min(box_w, anchor_w)
inter_h = min(box_h, anchor_h)
iou = intersection / (box_area + anchor_area - intersection)
```

YOLOv2/v3 predict boxes relative to a handful of predefined **anchor**
shapes (typical width/height pairs, usually chosen by clustering the
training set's own boxes) instead of predicting raw width and height
directly. Assignment picks whichever anchor's *shape* the ground-truth
box resembles most — computed as if both were centered at the same
point, so only their width and height matter, never where either
actually sits in the image. This decouples "what shape is this object"
from "where is it": an anchor tuned for tall, narrow boxes doesn't
compete unfairly against one tuned for wide, short boxes just because of
where either happens to be positioned in a particular training image.

```python
>>> anchors = [(10, 10), (50, 20), (20, 60)]
>>> best_anchor_for_box((40, 16), anchors)   # comparable scale + ratio to anchor 1
1
```

**In `exercises.py`:** `best_anchor_for_box`.

**Test it:** `python3 -m unittest test_exercises.TestBestAnchorForBox -v`

## 4. The loss weighting that keeps training from collapsing

```python
coord_loss = lambda_coord * sum((pred[obj][:, 1:5] - target[obj][:, 1:5]) ** 2)
obj_loss = sum((pred[obj][:, 0] - target[obj][:, 0]) ** 2)
noobj_loss = lambda_noobj * sum((pred[noobj][:, 0] - target[noobj][:, 0]) ** 2)
class_loss = sum((pred[obj][:, 5:] - target[obj][:, 5:]) ** 2)
```

A grid is mostly "no object" cells — for any realistic image, empty
cells vastly outnumber cells with a real box in them. A plain
sum-of-squared-errors loss would be dominated by driving every empty
cell's confidence to exactly zero, drowning out the rarer, more
important signal from cells that *do* contain something. Two
hyperparameters fix this in opposite directions: `lambda_coord > 1`
amplifies the box-coordinate loss (getting a real object's box right
matters more than raw squared error alone would weight it), and
`lambda_noobj < 1` shrinks the confidence loss for empty cells, so their
sheer numerical majority doesn't dominate the gradient.

```python
>>> losses = yolo_loss_components(pred, target, lambda_coord=5.0, lambda_noobj=0.5)
>>> losses.keys()
dict_keys(['coord', 'obj', 'noobj', 'class', 'total'])
```

**In `exercises.py`:** `yolo_loss_components`.

**Test it:** `python3 -m unittest test_exercises.TestYoloLossComponents -v`

## 5. From detections to `average_precision`'s inputs

```python
ordered = sorted(detections, key=lambda d: d[1], reverse=True)
for box, confidence, _class_id in ordered:
    # match against the best UNCLAIMED ground-truth box above iou_threshold
```

Module 32's `average_precision` takes `(confidences, labels)` — it
doesn't know what a bounding box is. Getting from raw detections to
those inputs is a real step of its own: sort by confidence, then greedily
match each detection to the best still-unclaimed ground-truth box above
an IoU threshold. **Each ground-truth box can be claimed once.** A second
detection overlapping an already-claimed box is a false positive even at
high IoU — in a working detector that duplicate should have been
suppressed by NMS before evaluation ever runs, so counting it as correct
here would hide a real problem.

```python
>>> confidences, labels = match_detections_to_ground_truth(detections, gt_boxes, iou_threshold=0.5)
>>> average_precision(confidences, labels)   # from module 32
```

**In `exercises.py`:** `match_detections_to_ground_truth`.

**Test it:** `python3 -m unittest test_exercises.TestMatchDetectionsToGroundTruth -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `encode_grid_targets` | One responsible cell per box; relative coordinates |
| `decode_grid_predictions` | Confidence thresholding; most cells predict nothing |
| `best_anchor_for_box` | Shape-only IoU for anchor assignment |
| `yolo_loss_components` | `lambda_coord`/`lambda_noobj` against class imbalance |
| `match_detections_to_ground_truth` | Greedy IoU matching feeding a real mAP computation |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **Why does only the cell containing a box's *center* get marked
   responsible, instead of every cell the box overlaps?**
   <details><summary>Answer</summary>If every overlapping cell were
   responsible, multiple cells would predict the same object, and the
   model would need some other mechanism to avoid multiple redundant
   detections for one box. Assigning exactly one cell (by center) makes
   the training target unambiguous — each object has exactly one cell
   whose prediction it is.</details>

2. **Why does anchor assignment compute IoU as if the box and every
   anchor were centered at the same point, instead of using their real
   positions?**
   <details><summary>Answer</summary>Anchor assignment is answering "what
   shape does this box look like," not "where is it" — the grid cell
   already encodes position. Comparing real positions would make the
   match depend on where the object happens to sit in a particular
   image, which has nothing to do with which anchor shape best fits
   it.</details>

3. **What would happen to training if `lambda_noobj` were 1.0 instead of
   a small fraction?**
   <details><summary>Answer</summary>Empty cells vastly outnumber cells
   with a real object, so the objectness loss from empty cells would
   dominate the total gradient. The model would be pushed hard toward
   "predict no object everywhere," a solution that's usually correct and
   therefore an easy local minimum — genuine objects would get a
   comparatively weak training signal.</details>

4. **Why does `match_detections_to_ground_truth` process detections in
   confidence order and let each ground-truth box be claimed only
   once?**
   <details><summary>Answer</summary>Confidence order means the model's
   most-trusted predictions get first claim on a ground-truth box, which
   matches how a real system would present or act on its results. Only
   allowing one claim per ground-truth box makes a duplicate detection
   of the same object count as a false positive — exactly the failure
   mode NMS exists to prevent before evaluation, so evaluation should
   still catch it if NMS didn't.</details>

5. **`best_anchor_for_box`'s IoU calculation ignores the box's aspect
   ratio if the box is much smaller than every anchor. What's the
   underlying reason?**
   <details><summary>Answer</summary>The overlap is capped by
   <code>min(box_w, anchor_w)</code> and <code>min(box_h, anchor_h)</code>
   — for a box much smaller than every anchor, that intersection is close
   to the box's own area regardless of which anchor it's compared
   against, so the union (and therefore the IoU) ends up dominated by
   each anchor's own size rather than by how well the shapes actually
   match. In practice this means the metric favors similarly-SIZED
   anchors, not just similarly-shaped ones, when the box is
   small.</details>

## Interview questions this module answers

- *Walk me through how YOLO turns a set of ground-truth boxes into
  something a grid of predictions can be trained against.*
- *Why does YOLO need anchor boxes, and how is a box assigned to one?*
- *This detector's loss collapses to predicting no objects — why, and
  how do you fix it?*
- *How do you turn a set of raw detections into a precision-recall curve
  or an mAP number?*
