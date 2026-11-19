"""Tests for 27 - convolution, detection and video.

Convolution is checked against F.conv2d rather than against hand-computed
numbers, so the exercise is pinned to real PyTorch semantics (including
padding and stride conventions) instead of my arithmetic.
"""

import importlib
import os
import unittest

import torch
import torch.nn.functional as F

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestConvOutputSize(unittest.TestCase):
    def test_no_padding_no_stride(self):
        self.assertEqual(m.conv_output_size(7, 3), 5)

    def test_same_padding_keeps_the_size(self):
        self.assertEqual(m.conv_output_size(32, 3, stride=1, padding=1), 32)

    def test_stride_halves_it(self):
        self.assertEqual(m.conv_output_size(224, 3, stride=2, padding=1), 112)

    def test_dilation_enlarges_the_kernel(self):
        # effective kernel = 2*(3-1)+1 = 5
        self.assertEqual(m.conv_output_size(9, 3, dilation=2), 5)

    def test_matches_pytorch_across_a_grid(self):
        for size in (8, 9, 16):
            for kernel in (1, 3, 5):
                for stride in (1, 2, 3):
                    for padding in (0, 1, 2):
                        x = torch.zeros(1, 1, size, size)
                        weight = torch.zeros(1, 1, kernel, kernel)
                        try:
                            expected = F.conv2d(x, weight, stride=stride,
                                                padding=padding).shape[-1]
                        except RuntimeError:
                            continue
                        self.assertEqual(
                            m.conv_output_size(size, kernel, stride, padding),
                            expected,
                            f"size={size} k={kernel} s={stride} p={padding}")

    def test_a_kernel_that_does_not_fit_raises(self):
        with self.assertRaises(ValueError):
            m.conv_output_size(3, 5)

    def test_kernel_equal_to_input_gives_one(self):
        self.assertEqual(m.conv_output_size(5, 5), 1)


class TestReceptiveField(unittest.TestCase):
    def test_a_single_layer(self):
        self.assertEqual(m.receptive_field([(3, 1)]), 3)

    def test_two_stacked_3x3_equal_one_5x5(self):
        """The VGG argument, in one assertion."""
        self.assertEqual(m.receptive_field([(3, 1), (3, 1)]), 5)

    def test_three_stacked_3x3_equal_one_7x7(self):
        self.assertEqual(m.receptive_field([(3, 1)] * 3), 7)

    def test_stride_multiplies_later_layers(self):
        self.assertEqual(m.receptive_field([(3, 2), (3, 1)]), 7)

    def test_no_layers_is_one_pixel(self):
        self.assertEqual(m.receptive_field([]), 1)

    def test_1x1_convolutions_do_not_grow_it(self):
        self.assertEqual(m.receptive_field([(3, 1), (1, 1), (1, 1)]), 3)

    def test_a_deep_strided_stack(self):
        # r=1; (3,2): r=3 j=2; (3,2): r=7 j=4; (3,1): r=15
        self.assertEqual(m.receptive_field([(3, 2), (3, 2), (3, 1)]), 15)


class TestConv2dViaUnfold(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 3, 8, 9)
        self.weight = torch.randn(4, 3, 3, 3)
        self.bias = torch.randn(4)

    def assert_matches(self, **kwargs):
        torch.testing.assert_close(
            m.conv2d_via_unfold(self.x, self.weight, self.bias, **kwargs),
            F.conv2d(self.x, self.weight, self.bias, **kwargs),
            atol=1e-4, rtol=1e-4)

    def test_plain(self):
        self.assert_matches()

    def test_with_padding(self):
        self.assert_matches(padding=1)

    def test_with_stride(self):
        self.assert_matches(stride=2)

    def test_with_stride_and_padding(self):
        self.assert_matches(stride=2, padding=1)

    def test_without_bias(self):
        torch.testing.assert_close(
            m.conv2d_via_unfold(self.x, self.weight),
            F.conv2d(self.x, self.weight), atol=1e-4, rtol=1e-4)

    def test_a_1x1_convolution_is_a_channel_mix(self):
        weight = torch.randn(5, 3, 1, 1)
        torch.testing.assert_close(
            m.conv2d_via_unfold(self.x, weight),
            F.conv2d(self.x, weight), atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        out = m.conv2d_via_unfold(self.x, self.weight, self.bias, padding=1)
        self.assertEqual(out.shape, (2, 4, 8, 9))

    def test_it_is_differentiable(self):
        weight = self.weight.clone().requires_grad_(True)
        m.conv2d_via_unfold(self.x, weight).sum().backward()
        self.assertIsNotNone(weight.grad)


class TestBoxIou(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                   [5.0, 5.0, 15.0, 15.0],
                                   [20.0, 20.0, 30.0, 30.0]])

    def test_shape(self):
        self.assertEqual(m.box_iou(self.boxes, self.boxes[:2]).shape, (3, 2))

    def test_identical_boxes_are_one(self):
        torch.testing.assert_close(m.box_iou(self.boxes, self.boxes).diagonal(),
                                   torch.ones(3))

    def test_disjoint_boxes_are_zero(self):
        self.assertEqual(m.box_iou(self.boxes[:1], self.boxes[2:]).item(), 0.0)

    def test_known_partial_overlap(self):
        """25 intersection over 175 union."""
        value = m.box_iou(self.boxes[:1], self.boxes[1:2]).item()
        self.assertAlmostEqual(value, 25.0 / 175.0, places=5)

    def test_a_contained_box(self):
        outer = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        inner = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        self.assertAlmostEqual(m.box_iou(outer, inner).item(), 0.25, places=5)

    def test_touching_boxes_do_not_count_as_overlapping(self):
        """Negative width times negative height would give a false positive."""
        left = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        right = torch.tensor([[6.0, 6.0, 10.0, 10.0]])
        self.assertEqual(m.box_iou(left, right).item(), 0.0)

    def test_it_is_symmetric(self):
        torch.testing.assert_close(m.box_iou(self.boxes, self.boxes),
                                   m.box_iou(self.boxes, self.boxes).T)

    def test_degenerate_boxes_do_not_produce_nan(self):
        point = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.assertFalse(torch.isnan(m.box_iou(point, point)).any())


class TestNms(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                   [1.0, 1.0, 11.0, 11.0],
                                   [50.0, 50.0, 60.0, 60.0]])
        self.scores = torch.tensor([0.9, 0.8, 0.7])

    def test_suppresses_the_overlapping_duplicate(self):
        kept = m.nms(self.boxes, self.scores, iou_threshold=0.5)
        self.assertEqual(kept.tolist(), [0, 2])

    def test_a_high_threshold_keeps_everything(self):
        kept = m.nms(self.boxes, self.scores, iou_threshold=0.99)
        self.assertEqual(sorted(kept.tolist()), [0, 1, 2])

    def test_a_zero_threshold_keeps_only_disjoint_boxes(self):
        kept = m.nms(self.boxes, self.scores, iou_threshold=0.0)
        self.assertEqual(kept.tolist(), [0, 2])

    def test_results_are_in_descending_score_order(self):
        scores = torch.tensor([0.1, 0.9, 0.5])
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                              [10.0, 10.0, 11.0, 11.0],
                              [20.0, 20.0, 21.0, 21.0]])
        kept = m.nms(boxes, scores, 0.5)
        self.assertEqual(kept.tolist(), [1, 2, 0])

    def test_the_highest_scoring_box_always_survives(self):
        kept = m.nms(self.boxes, self.scores, 0.5)
        self.assertIn(int(self.scores.argmax()), kept.tolist())

    def test_it_keeps_the_best_not_the_first(self):
        """Order of input must not matter - score does."""
        scores = torch.tensor([0.4, 0.95, 0.7])
        kept = m.nms(self.boxes, scores, 0.5)
        self.assertEqual(kept[0].item(), 1)

    def test_empty_input(self):
        kept = m.nms(torch.zeros(0, 4), torch.zeros(0), 0.5)
        self.assertEqual(kept.numel(), 0)

    def test_a_single_box(self):
        kept = m.nms(self.boxes[:1], self.scores[:1], 0.5)
        self.assertEqual(kept.tolist(), [0])

    def test_returns_long_indices(self):
        self.assertEqual(m.nms(self.boxes, self.scores, 0.5).dtype,
                         torch.long)


class TestPatchify(unittest.TestCase):
    def test_shape(self):
        out = m.patchify(torch.randn(2, 3, 8, 8), 4)
        self.assertEqual(out.shape, (2, 4, 48))

    def test_the_vit_base_configuration(self):
        """224/16 = 14, so 196 tokens of 768 dims."""
        out = m.patchify(torch.randn(1, 3, 224, 224), 16)
        self.assertEqual(out.shape, (1, 196, 768))

    def test_indivisible_sizes_raise(self):
        with self.assertRaises(ValueError):
            m.patchify(torch.randn(1, 3, 10, 10), 4)

    def test_patches_are_non_overlapping_and_row_major(self):
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        out = m.patchify(image, 2)
        torch.testing.assert_close(out[0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
        torch.testing.assert_close(out[0, 1], torch.tensor([2.0, 3.0, 6.0, 7.0]))

    def test_every_pixel_appears_exactly_once(self):
        image = torch.randn(1, 3, 8, 8)
        out = m.patchify(image, 4)
        self.assertEqual(out.numel(), image.numel())
        torch.testing.assert_close(out.sum(), image.sum(), atol=1e-4, rtol=1e-4)

    def test_a_patch_covering_the_whole_image(self):
        out = m.patchify(torch.randn(1, 3, 4, 4), 4)
        self.assertEqual(out.shape, (1, 1, 48))


class TestDepthwiseSeparable(unittest.TestCase):
    def test_standard_count(self):
        standard, _ = m.depthwise_separable_params(64, 128, 3)
        self.assertEqual(standard, 128 * 64 * 9)

    def test_separable_count(self):
        _, separable = m.depthwise_separable_params(64, 128, 3)
        self.assertEqual(separable, 64 * 9 + 64 * 128)

    def test_separable_is_much_smaller(self):
        standard, separable = m.depthwise_separable_params(256, 256, 3)
        self.assertGreater(standard / separable, 8.0)

    def test_a_1x1_kernel_saves_nothing(self):
        standard, separable = m.depthwise_separable_params(32, 64, 1)
        self.assertGreater(separable, standard * 0.9)

    def test_it_matches_the_real_layers(self):
        standard = torch.nn.Conv2d(64, 128, 3, bias=False)
        depthwise = torch.nn.Conv2d(64, 64, 3, groups=64, bias=False)
        pointwise = torch.nn.Conv2d(64, 128, 1, bias=False)
        counts = m.depthwise_separable_params(64, 128, 3)
        self.assertEqual(counts[0], standard.weight.numel())
        self.assertEqual(counts[1], depthwise.weight.numel()
                         + pointwise.weight.numel())


class TestNormaliseBatch(unittest.TestCase):
    def test_shape_is_unchanged(self):
        images = torch.randn(2, 3, 4, 6)
        self.assertEqual(m.normalise_batch(images, [0.5] * 3, [0.5] * 3).shape,
                         images.shape)

    def test_it_normalises_per_channel_not_per_column(self):
        """The bug: a (C,) tensor broadcasts against WIDTH, not channels."""
        images = torch.zeros(1, 3, 2, 5)
        images[0, 0] = 1.0
        images[0, 1] = 2.0
        images[0, 2] = 3.0
        out = m.normalise_batch(images, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        torch.testing.assert_close(out, torch.zeros(1, 3, 2, 5))

    def test_non_square_images_work(self):
        """A square image hides the bug; this one does not."""
        images = torch.randn(1, 3, 4, 7)
        m.normalise_batch(images, [0.0] * 3, [1.0] * 3)

    def test_known_values(self):
        images = torch.full((1, 1, 2, 2), 3.0)
        out = m.normalise_batch(images, [1.0], [2.0])
        torch.testing.assert_close(out, torch.full((1, 1, 2, 2), 1.0))

    def test_imagenet_statistics(self):
        images = torch.rand(2, 3, 8, 12)
        out = m.normalise_batch(images, [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        self.assertTrue(torch.isfinite(out).all())

    def test_it_accepts_tensors_as_well_as_lists(self):
        images = torch.randn(1, 3, 4, 5)
        torch.testing.assert_close(
            m.normalise_batch(images, torch.zeros(3), torch.ones(3)), images)

    def test_it_does_not_mutate_the_input(self):
        images = torch.randn(1, 3, 4, 5)
        before = images.clone()
        m.normalise_batch(images, [0.5] * 3, [0.5] * 3)
        torch.testing.assert_close(images, before)


class TestSampleFrames(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(m.sample_frames(100, 8)), 8)

    def test_indices_are_in_range(self):
        indices = m.sample_frames(10, 4)
        self.assertGreaterEqual(indices.min().item(), 0)
        self.assertLessEqual(indices.max().item(), 9)

    def test_indices_are_increasing(self):
        indices = m.sample_frames(100, 8).tolist()
        self.assertEqual(indices, sorted(indices))

    def test_it_spans_the_whole_clip(self):
        """Taking the first N frames would fail this."""
        indices = m.sample_frames(100, 4)
        self.assertGreater(indices[-1].item(), 50)

    def test_it_does_not_start_at_zero_bias(self):
        """The midpoint offset centres each sample in its segment."""
        self.assertGreater(m.sample_frames(100, 4)[0].item(), 0)

    def test_a_clip_shorter_than_the_request_repeats_frames(self):
        indices = m.sample_frames(3, 8)
        self.assertEqual(len(indices), 8)
        self.assertLessEqual(indices.max().item(), 2)

    def test_one_sample(self):
        self.assertEqual(len(m.sample_frames(10, 1)), 1)

    def test_sampling_every_frame(self):
        self.assertEqual(m.sample_frames(5, 5).tolist(), [0, 1, 2, 3, 4])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            m.sample_frames(10, 0)
        with self.assertRaises(ValueError):
            m.sample_frames(0, 4)


class TestFrameDifference(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(m.frame_difference(torch.randn(5, 3, 4, 4)).shape,
                         (4, 3, 4, 4))

    def test_a_static_clip_differences_to_zero(self):
        clip = torch.ones(4, 3, 2, 2)
        torch.testing.assert_close(m.frame_difference(clip),
                                   torch.zeros(3, 3, 2, 2))

    def test_values(self):
        clip = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
        torch.testing.assert_close(m.frame_difference(clip),
                                   torch.ones(3, 1, 1, 1))

    def test_stride(self):
        clip = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)
        out = m.frame_difference(clip, stride=2)
        self.assertEqual(out.shape, (3, 1, 1, 1))
        torch.testing.assert_close(out, torch.full((3, 1, 1, 1), 2.0))

    def test_a_clip_too_short_for_the_stride_raises(self):
        with self.assertRaises(ValueError):
            m.frame_difference(torch.randn(2, 1, 1, 1), stride=3)

    def test_stride_zero_raises(self):
        with self.assertRaises(ValueError):
            m.frame_difference(torch.randn(5, 1, 1, 1), stride=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
