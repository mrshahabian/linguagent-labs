"""Tests for 57 — Face detection: from raw outputs to boxes."""

import importlib
import os
import unittest
import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestDecodeDetections(unittest.TestCase):
    def test_empty_raw_outputs(self):
        raw = np.empty((0, 15), dtype=np.float32)
        boxes, landmarks, scores = m.decode_detections(raw)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(landmarks.shape, (0, 5, 2))
        self.assertEqual(scores.shape, (0,))

    def test_filtering_by_threshold(self):
        # Two detections: one above threshold (0.9), one below (0.4)
        raw = np.array([
            [10.0, 20.0, 30.0, 40.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0.9],
            [50.0, 60.0, 70.0, 80.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0.4]
        ], dtype=np.float32)
        
        boxes, landmarks, scores = m.decode_detections(raw, score_threshold=0.5)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(scores, [0.9], rtol=1e-5)
        
        # Check landmark reshaping
        expected_landmarks = np.array([[[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]]])
        np.testing.assert_allclose(landmarks, expected_landmarks)

    def test_no_detections_pass_threshold(self):
        raw = np.array([
            [10.0, 20.0, 30.0, 40.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0.3]
        ], dtype=np.float32)
        boxes, landmarks, scores = m.decode_detections(raw, score_threshold=0.5)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(landmarks.shape, (0, 5, 2))
        self.assertEqual(scores.shape, (0,))

    def test_multiple_detections_pass(self):
        raw = np.array([
            [10.0, 20.0, 30.0, 40.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0.9],
            [50.0, 60.0, 70.0, 80.0, 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 0.8],
            [100.0, 110.0, 120.0, 130.0, 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 0.2]
        ], dtype=np.float32)
        boxes, landmarks, scores = m.decode_detections(raw, score_threshold=0.5)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(boxes, [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
        np.testing.assert_allclose(scores, [0.9, 0.8], rtol=1e-5)
        self.assertEqual(landmarks.shape, (2, 5, 2))

    def test_all_detections_pass(self):
        raw = np.array([
            [10.0, 20.0, 30.0, 40.0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0.9],
            [50.0, 60.0, 70.0, 80.0, 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 0.95]
        ], dtype=np.float32)
        boxes, landmarks, scores = m.decode_detections(raw, score_threshold=0.1)
        self.assertEqual(len(boxes), 2)


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        box = np.array([10.0, 10.0, 50.0, 50.0])
        self.assertAlmostEqual(m.iou(box, box), 1.0)

    def test_no_overlap(self):
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([20.0, 20.0, 10.0, 10.0])
        self.assertAlmostEqual(m.iou(box1, box2), 0.0)

    def test_partial_overlap(self):
        box1 = np.array([10.0, 10.0, 20.0, 20.0]) # Area = 400
        box2 = np.array([20.0, 10.0, 20.0, 20.0]) # Area = 400
        # Intersection: x in [20, 30], y in [10, 30] -> Area = 10 * 20 = 200
        # Union: 400 + 400 - 200 = 600
        # IoU = 200 / 600 = 1/3
        self.assertAlmostEqual(m.iou(box1, box2), 1.0 / 3.0)

    def test_zero_area_box(self):
        box1 = np.array([10.0, 10.0, 0.0, 20.0])
        box2 = np.array([10.0, 10.0, 20.0, 20.0])
        self.assertAlmostEqual(m.iou(box1, box2), 0.0)

    def test_one_box_inside_another(self):
        box1 = np.array([10.0, 10.0, 50.0, 50.0]) # Area = 2500
        box2 = np.array([20.0, 20.0, 10.0, 10.0]) # Area = 100, completely inside
        # Intersection = 100
        # Union = 2500 + 100 - 100 = 2500
        # IoU = 100 / 2500 = 0.04
        self.assertAlmostEqual(m.iou(box1, box2), 0.04)

    def test_partial_overlap_on_edges(self):
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([9.0, 9.0, 10.0, 10.0])
        # Intersection: x in [9, 10], y in [9, 10] -> Area = 1
        # Union: 100 + 100 - 1 = 199
        # IoU = 1 / 199
        self.assertAlmostEqual(m.iou(box1, box2), 1.0 / 199.0)


class TestNMS(unittest.TestCase):
    def test_empty_nms(self):
        boxes = np.empty((0, 4))
        scores = np.empty((0,))
        self.assertEqual(m.non_max_suppression(boxes, scores, 0.5), [])

    def test_single_box(self):
        boxes = np.array([[10., 10., 20., 20.]])
        scores = np.array([0.9])
        self.assertEqual(m.non_max_suppression(boxes, scores, 0.5), [0])

    def test_suppression_of_overlapping_boxes(self):
        boxes = np.array([
            [10.0, 10.0, 20.0, 20.0],  # Box 0 (Score 0.9)
            [12.0, 12.0, 18.0, 18.0],  # Box 1 (Score 0.8) - highly overlapping with 0
            [100.0, 100.0, 20.0, 20.0] # Box 2 (Score 0.75) - non-overlapping
        ])
        scores = np.array([0.9, 0.8, 0.75])
        keep = m.non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [0, 2])

    def test_sorting_order_respected(self):
        boxes = np.array([
            [10.0, 10.0, 20.0, 20.0],  # Box 0 (Score 0.7)
            [12.0, 12.0, 18.0, 18.0],  # Box 1 (Score 0.95) - higher score!
        ])
        scores = np.array([0.7, 0.95])
        # Box 1 should be kept, Box 0 suppressed
        keep = m.non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [1])

    def test_all_suppressed_except_one(self):
        boxes = np.array([
            [10.0, 10.0, 20.0, 20.0],
            [11.0, 11.0, 20.0, 20.0],
            [12.0, 12.0, 20.0, 20.0],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep = m.non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [0])

    def test_no_suppression_if_no_overlap(self):
        boxes = np.array([
            [10.0, 10.0, 10.0, 10.0],
            [30.0, 30.0, 10.0, 10.0],
            [50.0, 50.0, 10.0, 10.0],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep = m.non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [0, 1, 2])


class TestScaleBoxes(unittest.TestCase):
    def test_basic_scaling(self):
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        scaled = m.scale_boxes(boxes, scale_x=2.0, scale_y=3.0, max_width=200, max_height=200)
        # x1 = 20, y1 = 60, x2 = (10+30)*2 = 80, y2 = (20+40)*3 = 180
        # w = 60, h = 120
        np.testing.assert_allclose(scaled[0], [20.0, 60.0, 60.0, 120.0])

    def test_clipping_at_boundaries(self):
        boxes = np.array([[50.0, 50.0, 100.0, 100.0]])
        # Scale factors would push right edge to 150*2 = 300, bottom to 150*2 = 300
        # Max width/height are 200
        scaled = m.scale_boxes(boxes, scale_x=2.0, scale_y=2.0, max_width=200, max_height=200)
        # x1 = 100, y1 = 100
        # x2 = clip(300, 0, 200) = 200
        # y2 = clip(300, 0, 200) = 200
        # w = 200 - 100 = 100, h = 200 - 100 = 100
        np.testing.assert_allclose(scaled[0], [100.0, 100.0, 100.0, 100.0])

    def test_empty_boxes_scaling(self):
        boxes = np.empty((0, 4))
        scaled = m.scale_boxes(boxes, 2.0, 2.0, 100, 100)
        self.assertEqual(scaled.shape, (0, 4))

    def test_scaling_with_zero_coordinates(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scaled = m.scale_boxes(boxes, scale_x=1.5, scale_y=1.5, max_width=100, max_height=100)
        np.testing.assert_allclose(scaled[0], [0.0, 0.0, 15.0, 15.0])


class TestCropRegions(unittest.TestCase):
    def test_standard_crop(self):
        # 5x5 image
        image = np.arange(25).reshape(5, 5)
        boxes = np.array([[1.0, 1.0, 3.0, 2.0]])
        crops = m.crop_regions(image, boxes)
        self.assertEqual(len(crops), 1)
        # Expected crop: rows 1 to 3 (exclusive), cols 1 to 4 (exclusive)
        expected = image[1:3, 1:4]
        np.testing.assert_array_equal(crops[0], expected)

    def test_rounding_coordinates(self):
        image = np.arange(25).reshape(5, 5)
        # Coordinates requiring rounding: 0.9 -> 1, 1.1 -> 1, 2.8 -> 3, 1.9 -> 2
        boxes = np.array([[0.9, 1.1, 1.9, 0.8]]) # rounded: [1, 1, 2, 1]
        crops = m.crop_regions(image, boxes)
        expected = image[1:2, 1:3]
        np.testing.assert_array_equal(crops[0], expected)

    def test_clipping_out_of_bounds_crops(self):
        image = np.arange(25).reshape(5, 5)
        # Box extends past image boundaries (width=5, height=5)
        boxes = np.array([[3.0, 3.0, 10.0, 10.0]])
        crops = m.crop_regions(image, boxes)
        # Should clip to image boundaries: y in [3, 5], x in [3, 5]
        expected = image[3:5, 3:5]
        np.testing.assert_array_equal(crops[0], expected)

    def test_empty_boxes(self):
        image = np.arange(25).reshape(5, 5)
        boxes = np.empty((0, 4))
        crops = m.crop_regions(image, boxes)
        self.assertEqual(len(crops), 0)

    def test_completely_out_of_bounds_box(self):
        image = np.arange(25).reshape(5, 5)
        boxes = np.array([[10.0, 10.0, 5.0, 5.0]])
        crops = m.crop_regions(image, boxes)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
