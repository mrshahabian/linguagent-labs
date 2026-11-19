# 57 — Face detection: from raw outputs to boxes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/57-detection-postprocessing/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/detection-postprocessing?utm_source=github&utm_medium=labs)

*~45 minutes · 26 tests · Intermediate*

In computer vision, deep learning models rarely output clean, ready-to-use bounding boxes directly. Instead, modern face detectors like OpenCV's YuNet or RetinaFace output raw, multi-dimensional tensors containing coordinates, confidence scores, and facial landmark coordinates. These raw outputs are often scaled to a fixed model input size (e.g., 320x320 pixels) rather than the original image's dimensions, and they typically contain hundreds of highly overlapping candidate boxes for a single face.

To use these detections in downstream applications—such as face recognition, privacy blurring, or emotion classification—you must build a post-processing pipeline. This pipeline must decode the raw arrays, filter out low-confidence predictions, eliminate duplicate boxes using Non-Maximum Suppression (NMS), scale the coordinates back to the original image dimensions, and safely crop the regions of interest without crashing on boundary edge cases.

## What you'll build

Five functions in `exercises.py` that form a complete face detection post-processing pipeline:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `decode_detections` | Filtering raw model outputs and splitting them into boxes, landmarks, and scores | Parsing raw output tensors from ONNX/OpenCV models |
| `iou` | Calculating Intersection over Union for two bounding boxes | Evaluating overlap for suppression or object tracking |
| `non_max_suppression` | Eliminating redundant overlapping boxes based on scores and IoU | Standard post-processing step for all anchor-based/anchor-free object detectors |
| `scale_boxes` | Scaling and clipping box coordinates back to original image dimensions | Mapping model-space coordinates back to the source image |
| `crop_regions` | Extracting sub-images from rounded integer coordinates | Preparing face chips for alignment, embedding extraction, or classification |

---

## 1. Decoding Raw Model Outputs

```python
# Example of raw output row format (15 elements):
# [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rmc_x, rmc_y, lmc_x, lmc_y, score]
>>> import numpy as np
>>> raw = np.array([[10.0, 15.0, 50.0, 60.0, 20.0, 25.0, 40.0, 25.0, 30.0, 35.0, 22.0, 55.0, 38.0, 55.0, 0.95]])
>>> boxes, landmarks, scores = decode_detections(raw, score_threshold=0.5)
>>> boxes
array([[10., 15., 50., 60.]])
>>> landmarks
array([[[20., 25.],
        [40., 25.],
        [30., 35.],
        [22., 55.],
        [38., 55.]]])
```

Deep learning face detectors like YuNet return a 2D array of shape `(N, 15)` for `N` candidate detections. Each row contains:
- Bounding box top-left corner and size: `x, y, w, h` (indices 0 to 3)
- Five facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner): five `(x, y)` pairs (indices 4 to 13)
- Detection confidence score: `score` (index 14)

Your first task is to filter out any candidate detections whose confidence score is strictly below a given threshold, and separate the remaining data into three distinct, clean NumPy arrays.

**The Pitfall:** When reshaping the flat landmark coordinates (10 values) into a `(5, 2)` shape, it is easy to accidentally transpose the coordinates or group them incorrectly. Ensure that the first landmark pair corresponds to columns 4 and 5, the second to 6 and 7, and so on. Additionally, if no detections pass the threshold, your function must return empty arrays with the correct dimensions (e.g., shape `(0, 4)` for boxes).

**Test it:** `python3 -m unittest test_exercises.TestDecodeDetections -v`

---

## 2. Intersection over Union (IoU)

```python
>>> box_a = np.array([10, 10, 50, 50])
>>> box_b = np.array([30, 10, 50, 50])
>>> iou(box_a, box_b)
0.42857142857142855
```

Intersection over Union (IoU) measures the overlap between two bounding boxes. It is calculated as the area of the intersection divided by the area of the union of the two boxes:

$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

Where the Union area is defined as:

$$\text{Area of Union} = \text{Area}_A + \text{Area}_B - \text{Area of Intersection}$$

**The Pitfall:** A common mistake is assuming that the box coordinates are represented as `[x1, y1, x2, y2]` (top-left, bottom-right). In this module, boxes are represented as `[x, y, w, h]` (top-left, width, height). You must convert these to corner coordinates to calculate the intersection boundaries, and handle the edge case where the union area is zero to avoid a `ZeroDivisionError`.

**Test it:** `python3 -m unittest test_exercises.TestIoU -v`

---

## 3. Non-Maximum Suppression (NMS)

```python
>>> boxes = np.array([[10, 10, 50, 50], [12, 12, 48, 48], [100, 100, 30, 30]])
>>> scores = np.array([0.9, 0.85, 0.95])
>>> non_max_suppression(boxes, scores, iou_threshold=0.5)
[2, 0]
```

Because face detectors run dense sliding windows or anchor evaluations across an image, they often produce multiple highly overlapping boxes for the same face. Non-Maximum Suppression (NMS) is a greedy algorithm that filters out redundant boxes:
1. Sort the candidate boxes by their confidence scores in descending order.
2. Select the box with the highest score and keep it.
3. Calculate the IoU of this selected box with all other remaining boxes.
4. Suppress (discard) any remaining boxes that have an IoU greater than the `iou_threshold` with the selected box.
5. Repeat the process for the remaining boxes until no boxes are left.

**The Pitfall:** Modifying a list or array while iterating over it in Python leads to skipped elements and index-out-of-bounds bugs. Instead, maintain an array of active indices and filter them using NumPy boolean indexing or integer indexing at each step.

**Test it:** `python3 -m unittest test_exercises.TestNMS -v`

---

## 4. Scaling and Clipping Boxes

```python
>>> boxes = np.array([[10.0, 20.0, 100.0, 100.0]])
>>> scale_boxes(boxes, scale_x=2.0, scale_y=1.5, max_width=180, max_height=140)
array([[ 20.,  30., 160., 110.]])
```

Detectors are usually run on resized images (e.g., scaling a 1920x1080 image down to 320x320). Once boxes are detected, their coordinates must be scaled back to the original image dimensions. Furthermore, scaled boxes can sometimes extend slightly outside the image boundaries due to numerical rounding or model inaccuracies. You must clip the coordinates so they lie strictly within the image boundaries `[0, max_width]` and `[0, max_height]`.

**The Pitfall:** Scaling the width and height directly and then clipping them independently of the top-left corner coordinates is mathematically incorrect. For example, if a box starts at `x = 150` with `w = 50` on an image of width `180`, scaling might place the right edge at `220`. If you clip the width to `180`, the box still goes out of bounds. The correct approach is to scale the absolute corner coordinates `x1` and `x2`, clip both to `[0, max_width]`, and then compute the new width as `x2 - x1`.

**Test it:** `python3 -m unittest test_exercises.TestScaleBoxes -v`

---

## 5. Cropping Regions of Interest

```python
>>> image = np.arange(100).reshape(10, 10)
>>> boxes = np.array([[2.1, 3.9, 3.0, 4.0]])
>>> crops = crop_regions(image, boxes)
>>> crops[0].shape
(4, 3)
```

Once you have scaled, clipped, and suppressed bounding boxes, you often need to extract the corresponding sub-images (crops) from the original image array. Slicing a NumPy array requires integer indices. Since bounding box coordinates are floating-point numbers, you must round them to the nearest integer and cast them to integers before slicing.

**The Pitfall:** Floating-point coordinates like `2.9` must be rounded to `3`, not truncated to `2` (which is what standard integer casting `int(2.9)` does). Use `np.round()` to round to the nearest integer before casting. Additionally, ensure that your slicing handles empty boxes or boxes that lie on the exact boundary edge without raising index errors.

**Test it:** `python3 -m unittest test_exercises.TestCropRegions -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `decode_detections` | Parsing flat multi-class coordinates and scores |
| `iou` | Geometric coordinate calculations and overlap ratios |
| `non_max_suppression` | Greedy filtering algorithms and sorting by confidence |
| `scale_boxes` | Coordinate space transformations and boundary clipping |
| `crop_regions` | Floating-point rounding and safe NumPy array slicing |

## Check your understanding

1. **Why do we need to reshape the landmarks slice to `(-1, 5, 2)` instead of leaving it flat?**
   <details><summary>Answer</summary>Leaving it flat makes it difficult to perform spatial transformations (like scaling or rotation) on individual facial features. Reshaping to <code>(M, 5, 2)</code> groups the landmarks as 5 distinct 2D points <code>(x, y)</code> per face, which matches standard geometric representation.</details>

2. **What happens to the IoU calculation if one of the bounding boxes has a width or height of zero?**
   <details><summary>Answer</summary>Its area will be zero, meaning the intersection area will also be zero. The union area will equal the area of the other box. If both boxes have zero area, the union area is zero, which would cause a division-by-zero error unless explicitly handled by returning <code>0.0</code>.</details>

3. **Why does NMS sort scores in descending order instead of ascending?**
   <details><summary>Answer</summary>We want to keep the most confident detections and suppress highly overlapping, less confident detections. Sorting in descending order ensures that the highest-scoring box is processed first and acts as the anchor for suppressing duplicates.</details>

4. **Why must we clip corner coordinates `(x1, y1, x2, y2)` instead of just clipping `(x, y, w, h)`?**
   <details><summary>Answer</summary>Clipping <code>w</code> and <code>h</code> independently of <code>x</code> and <code>y</code> doesn't guarantee that the box stays within the image boundaries. For example, a box at <code>x = 90</code> with <code>w = 20</code> on a 100-pixel wide image extends to <code>110</code>. Slicing there would exceed the boundary. Clipping corners first ensures both the start and end coordinates are valid, and the resulting width <code>x2 - x1</code> is automatically correct.</details>

5. **What is the difference between `np.round(x).astype(int)` and `x.astype(int)`?**
   <details><summary>Answer</summary><code>x.astype(int)</code> truncates the decimal part (rounds toward zero), so <code>2.9</code> becomes <code>2</code>. <code>np.round(x).astype(int)</code> rounds to the nearest integer, so <code>2.9</code> correctly becomes <code>3</code>, which is much more accurate for pixel coordinates.</details>

## Interview questions this module answers

- *How does Non-Maximum Suppression (NMS) work, and why is it necessary in object detection pipelines?*
- *How do you calculate Intersection over Union (IoU) for two bounding boxes?*
- *What are the common pitfalls when scaling bounding boxes from model-space back to image-space?*
- *How do you safely crop regions of interest from an image using floating-point bounding box coordinates?*
