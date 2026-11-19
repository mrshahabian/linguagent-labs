"""Tests for 36 - grid and anchor-based object detection (YOLO)."""

import importlib
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestEncodeGridTargets(unittest.TestCase):
    def test_marks_the_cell_containing_the_center(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0]])
        class_ids = torch.tensor([1])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        # center is (30, 50); cell size is 40 -> col 0, row 1
        self.assertEqual(target[1, 0, 0].item(), 1.0)

    def test_empty_cells_stay_zero(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0]])
        class_ids = torch.tensor([0])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        self.assertEqual(target[6, 6, 0].item(), 0.0)

    def test_class_is_one_hot_encoded(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0]])
        class_ids = torch.tensor([3])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        class_scores = target[1, 0, 5:]
        self.assertEqual(class_scores.tolist(), [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_multiple_boxes_in_different_cells(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0], [200.0, 150.0, 260.0, 200.0]])
        class_ids = torch.tensor([1, 3])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        self.assertEqual(int(target[..., 0].sum().item()), 2)


class TestDecodeGridPredictions(unittest.TestCase):
    def test_round_trip_recovers_the_original_box(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0]])
        class_ids = torch.tensor([1])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        detections = m.decode_grid_predictions(target, grid_size=7, img_size=280, conf_threshold=0.5)
        self.assertEqual(len(detections), 1)
        box, confidence, class_id = detections[0]
        self.assertTrue(torch.allclose(box, boxes[0], atol=1e-3))
        self.assertEqual(confidence, 1.0)
        self.assertEqual(class_id, 1)

    def test_below_threshold_cells_are_skipped(self):
        grid = torch.zeros(7, 7, 10)
        grid[3, 3, 0] = 0.2
        detections = m.decode_grid_predictions(grid, grid_size=7, img_size=280, conf_threshold=0.5)
        self.assertEqual(detections, [])

    def test_reports_the_highest_scoring_class(self):
        grid = torch.zeros(4, 4, 9)
        grid[2, 2, 0] = 1.0
        grid[2, 2, 1:5] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        grid[2, 2, 5:] = torch.tensor([0.1, 0.9, 0.3, 0.0])
        detections = m.decode_grid_predictions(grid, grid_size=4, img_size=100, conf_threshold=0.5)
        self.assertEqual(detections[0][2], 1)


class TestBestAnchorForBox(unittest.TestCase):
    ANCHORS = [(10, 10), (50, 20), (20, 60)]

    def test_square_box_matches_square_anchor(self):
        self.assertEqual(m.best_anchor_for_box((12, 11), self.ANCHORS), 0)

    def test_wide_box_matches_wide_anchor(self):
        self.assertEqual(m.best_anchor_for_box((45, 18), self.ANCHORS), 1)

    def test_tall_box_matches_tall_anchor(self):
        self.assertEqual(m.best_anchor_for_box((18, 55), self.ANCHORS), 2)

    def test_matches_by_scale_and_aspect_ratio_together(self):
        # comparable scale and aspect ratio (2.5:1) to anchor 1 (50x20),
        # without being identical to it
        self.assertEqual(m.best_anchor_for_box((40, 16), self.ANCHORS), 1)


class TestYoloLossComponents(unittest.TestCase):
    def test_perfect_prediction_is_zero_loss(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 90.0]])
        class_ids = torch.tensor([1])
        target = m.encode_grid_targets(boxes, class_ids, grid_size=7, img_size=280, num_classes=5)
        losses = m.yolo_loss_components(target.clone(), target)
        self.assertAlmostEqual(losses["total"].item(), 0.0, places=5)

    def test_false_objectness_in_empty_cell_only_hits_noobj(self):
        target = torch.zeros(4, 4, 8)
        pred = target.clone()
        pred[0, 0, 0] = 0.9
        losses = m.yolo_loss_components(pred, target, lambda_noobj=0.5)
        self.assertAlmostEqual(losses["noobj"].item(), 0.5 * 0.9 ** 2, places=5)
        self.assertAlmostEqual(losses["coord"].item(), 0.0)
        self.assertAlmostEqual(losses["obj"].item(), 0.0)

    def test_lambda_coord_scales_the_coordinate_term(self):
        target = torch.zeros(2, 2, 7)
        target[0, 0, 0] = 1.0
        target[0, 0, 1:5] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        pred = target.clone()
        pred[0, 0, 1] = 0.6  # off by 0.1 in x

        low = m.yolo_loss_components(pred, target, lambda_coord=1.0)
        high = m.yolo_loss_components(pred, target, lambda_coord=5.0)
        self.assertAlmostEqual(high["coord"].item(), 5 * low["coord"].item(), places=5)

    def test_total_is_the_sum_of_components(self):
        target = torch.zeros(3, 3, 7)
        target[1, 1, 0] = 1.0
        target[1, 1, 5] = 1.0
        pred = torch.rand(3, 3, 7)
        losses = m.yolo_loss_components(pred, target)
        expected_total = losses["coord"] + losses["obj"] + losses["noobj"] + losses["class"]
        self.assertAlmostEqual(losses["total"].item(), expected_total.item(), places=5)


class TestMatchDetectionsToGroundTruth(unittest.TestCase):
    def setUp(self):
        self.gt_boxes = [
            torch.tensor([10.0, 10.0, 50.0, 90.0]),
            torch.tensor([200.0, 150.0, 260.0, 200.0]),
        ]

    def test_good_matches_are_labelled_correct(self):
        detections = [
            (torch.tensor([11.0, 11.0, 49.0, 89.0]), 0.9, 1),
            (torch.tensor([201.0, 151.0, 259.0, 199.0]), 0.95, 3),
        ]
        confidences, labels = m.match_detections_to_ground_truth(detections, self.gt_boxes)
        self.assertEqual(labels, [1, 1])

    def test_duplicate_detection_of_the_same_box_is_a_false_positive(self):
        detections = [
            (torch.tensor([11.0, 11.0, 49.0, 89.0]), 0.9, 1),
            (torch.tensor([12.0, 12.0, 48.0, 88.0]), 0.8, 1),
        ]
        confidences, labels = m.match_detections_to_ground_truth(detections, self.gt_boxes)
        self.assertEqual(labels, [1, 0])

    def test_no_overlap_is_a_false_positive(self):
        detections = [(torch.tensor([0.0, 0.0, 5.0, 5.0]), 0.7, 2)]
        confidences, labels = m.match_detections_to_ground_truth(detections, self.gt_boxes)
        self.assertEqual(labels, [0])

    def test_results_sorted_by_confidence_descending(self):
        detections = [
            (torch.tensor([0.0, 0.0, 5.0, 5.0]), 0.3, 2),
            (torch.tensor([11.0, 11.0, 49.0, 89.0]), 0.9, 1),
        ]
        confidences, labels = m.match_detections_to_ground_truth(detections, self.gt_boxes)
        self.assertEqual(confidences, [0.9, 0.3])

    def test_below_iou_threshold_does_not_count_as_a_match(self):
        # heavily shifted box - low IoU with gt_boxes[0]
        detections = [(torch.tensor([30.0, 30.0, 70.0, 110.0]), 0.9, 1)]
        confidences, labels = m.match_detections_to_ground_truth(
            detections, self.gt_boxes, iou_threshold=0.9)
        self.assertEqual(labels, [0])


if __name__ == "__main__":
    unittest.main()
