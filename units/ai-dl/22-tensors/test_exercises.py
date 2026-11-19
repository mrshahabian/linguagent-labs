"""Tests for 22 - tensors.

Everything here runs on CPU in well under a second. Where a test is about
memory rather than values it inspects strides and storage pointers, which
are exact, rather than timing anything.
"""

import importlib
import os
import unittest

import torch

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestAsTensorLike(unittest.TestCase):
    def test_matches_dtype(self):
        reference = torch.zeros(3, dtype=torch.float64)
        self.assertEqual(m.as_tensor_like([1, 2, 3], reference).dtype,
                         torch.float64)

    def test_matches_device(self):
        reference = torch.zeros(3)
        self.assertEqual(m.as_tensor_like([1, 2], reference).device,
                         reference.device)

    def test_keeps_values(self):
        out = m.as_tensor_like([1, 2, 3], torch.zeros(3))
        torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 3.0]))

    def test_accepts_a_tensor_too(self):
        out = m.as_tensor_like(torch.tensor([1, 2]), torch.zeros(2))
        self.assertEqual(out.dtype, torch.float32)

    def test_half_precision_reference(self):
        out = m.as_tensor_like([1.0], torch.zeros(1, dtype=torch.float16))
        self.assertEqual(out.dtype, torch.float16)


class TestSharesStorage(unittest.TestCase):
    def test_basic_slicing_is_a_view(self):
        x = torch.arange(10)
        self.assertTrue(m.shares_storage(x[2:5], x))

    def test_view_and_permute_are_views(self):
        x = torch.arange(12).reshape(3, 4)
        self.assertTrue(m.shares_storage(x.view(4, 3), x))
        self.assertTrue(m.shares_storage(x.permute(1, 0), x))

    def test_expand_is_a_view(self):
        x = torch.arange(3).reshape(3, 1)
        self.assertTrue(m.shares_storage(x.expand(3, 5), x))

    def test_fancy_indexing_copies(self):
        x = torch.arange(10)
        self.assertFalse(m.shares_storage(x[[0, 2, 4]], x))

    def test_mask_indexing_copies(self):
        x = torch.arange(10)
        self.assertFalse(m.shares_storage(x[x > 5], x))

    def test_clone_copies(self):
        x = torch.arange(10)
        self.assertFalse(m.shares_storage(x.clone(), x))


class TestSafeNormalise(unittest.TestCase):
    def test_zero_mean_unit_variance(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = m.safe_normalise(x)
        self.assertAlmostEqual(out.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(out.std().item(), 1.0, places=3)

    def test_does_not_mutate_the_input(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        before = x.clone()
        m.safe_normalise(x)
        torch.testing.assert_close(x, before)

    def test_constant_input_does_not_divide_by_zero(self):
        out = m.safe_normalise(torch.full((4,), 7.0))
        self.assertTrue(torch.isfinite(out).all())

    def test_works_on_a_leaf_that_requires_grad(self):
        """In-place ops on a leaf requiring grad raise; this must not."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = m.safe_normalise(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestFlattenBatch(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(m.flatten_batch(torch.zeros(2, 3, 4, 5)).shape,
                         (2, 60))

    def test_already_flat_is_unchanged(self):
        self.assertEqual(m.flatten_batch(torch.zeros(6, 7)).shape, (6, 7))

    def test_works_on_a_non_contiguous_tensor(self):
        """x.view(B, -1) raises here; reshape/flatten do not."""
        x = torch.zeros(2, 3, 4).permute(0, 2, 1)
        self.assertFalse(x.is_contiguous())
        self.assertEqual(m.flatten_batch(x).shape, (2, 12))

    def test_values_are_preserved_in_order(self):
        x = torch.arange(8).reshape(2, 2, 2)
        torch.testing.assert_close(m.flatten_batch(x),
                                   torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))


class TestChannelsLast(unittest.TestCase):
    def test_shape_is_unchanged(self):
        x = torch.zeros(2, 3, 4, 5)
        self.assertEqual(m.to_channels_last(x).shape, (2, 3, 4, 5))

    def test_strides_move(self):
        """Channels-last means the channel stride is 1."""
        out = m.to_channels_last(torch.zeros(2, 3, 4, 5))
        self.assertEqual(out.stride(), (60, 1, 15, 3))

    def test_flag_is_set(self):
        out = m.to_channels_last(torch.zeros(2, 3, 4, 5))
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def test_values_survive(self):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        torch.testing.assert_close(m.to_channels_last(x), x)


class TestBatchedPairwiseDistance(unittest.TestCase):
    def test_shape(self):
        out = m.batched_pairwise_distance(torch.randn(2, 3, 5),
                                          torch.randn(2, 4, 5))
        self.assertEqual(out.shape, (2, 3, 4))

    def test_matches_cdist(self):
        a, b = torch.randn(2, 6, 4), torch.randn(2, 5, 4)
        torch.testing.assert_close(m.batched_pairwise_distance(a, b),
                                   torch.cdist(a, b), atol=1e-4, rtol=1e-4)

    def test_distance_to_self_is_zero(self):
        a = torch.randn(1, 4, 3)
        diagonal = m.batched_pairwise_distance(a, a).diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(diagonal, torch.zeros(1, 4), atol=1e-5,
                                   rtol=0)

    def test_known_values(self):
        a = torch.tensor([[[0.0, 0.0]]])
        b = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        torch.testing.assert_close(m.batched_pairwise_distance(a, b),
                                   torch.tensor([[[5.0, 0.0]]]))

    def test_no_nan_gradients_at_zero_distance(self):
        """sqrt(0) has an infinite derivative - a classic NaN-loss source."""
        a = torch.zeros(1, 2, 3, requires_grad=True)
        m.batched_pairwise_distance(a, a.detach()).sum().backward()
        self.assertFalse(torch.isnan(a.grad).any(),
                         "gradient is NaN where the distance is 0")


class TestMaskedSoftmax(unittest.TestCase):
    def test_masked_positions_are_zero(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, True, False]])
        out = m.masked_softmax(scores, mask)
        self.assertEqual(out[0, 2].item(), 0.0)

    def test_unmasked_positions_sum_to_one(self):
        scores = torch.randn(3, 5)
        mask = torch.tensor([[True, True, False, False, False]] * 3)
        out = m.masked_softmax(scores, mask)
        torch.testing.assert_close(out.sum(-1), torch.ones(3))

    def test_masking_happens_before_the_exponential(self):
        """Softmax-then-zero leaves the row summing to less than 1."""
        scores = torch.tensor([[1.0, 1.0, 100.0]])
        mask = torch.tensor([[True, True, False]])
        out = m.masked_softmax(scores, mask)
        torch.testing.assert_close(out, torch.tensor([[0.5, 0.5, 0.0]]))

    def test_a_fully_masked_row_is_zero_not_nan(self):
        out = m.masked_softmax(torch.randn(1, 4),
                               torch.zeros(1, 4, dtype=torch.bool))
        self.assertFalse(torch.isnan(out).any())
        torch.testing.assert_close(out, torch.zeros(1, 4))

    def test_all_unmasked_matches_plain_softmax(self):
        scores = torch.randn(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        torch.testing.assert_close(m.masked_softmax(scores, mask),
                                   torch.softmax(scores, -1))

    def test_works_on_a_non_final_dim(self):
        scores = torch.randn(4, 3)
        mask = torch.ones(4, 3, dtype=torch.bool)
        torch.testing.assert_close(m.masked_softmax(scores, mask, dim=0),
                                   torch.softmax(scores, 0))


class TestCountParameters(unittest.TestCase):
    def test_linear_layer(self):
        # 3*4 weights + 4 biases
        self.assertEqual(m.count_parameters(torch.nn.Linear(3, 4)), 16)

    def test_counts_elements_not_tensors(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 20),
                                    torch.nn.Linear(20, 1))
        self.assertEqual(m.count_parameters(model), 10 * 20 + 20 + 20 + 1)

    def test_frozen_parameters_are_excluded(self):
        model = torch.nn.Linear(3, 4)
        model.weight.requires_grad_(False)
        self.assertEqual(m.count_parameters(model), 4)

    def test_trainable_only_false_counts_everything(self):
        model = torch.nn.Linear(3, 4)
        model.weight.requires_grad_(False)
        self.assertEqual(m.count_parameters(model, trainable_only=False), 16)

    def test_buffers_are_not_parameters(self):
        model = torch.nn.BatchNorm1d(4)   # 4 weight + 4 bias, plus buffers
        self.assertEqual(m.count_parameters(model), 8)


class TestMemoryBytes(unittest.TestCase):
    def test_matches_torch(self):
        for shape, dtype in [((3, 4), torch.float32), ((10,), torch.uint8),
                             ((2, 3, 4), torch.float64)]:
            expected = torch.zeros(shape, dtype=dtype)
            self.assertEqual(m.memory_bytes(shape, dtype),
                             expected.numel() * expected.element_size())

    def test_float16_is_half_of_float32(self):
        self.assertEqual(m.memory_bytes((1000,), torch.float16) * 2,
                         m.memory_bytes((1000,), torch.float32))

    def test_an_image_batch(self):
        self.assertEqual(m.memory_bytes((32, 3, 224, 224)), 19_267_584)

    def test_scalar(self):
        self.assertEqual(m.memory_bytes(()), 4)


class TestStableLogSumExp(unittest.TestCase):
    def test_matches_torch(self):
        x = torch.randn(4, 7)
        torch.testing.assert_close(m.stable_logsumexp(x),
                                   torch.logsumexp(x, -1))

    def test_does_not_overflow(self):
        """The naive exp().sum().log() is inf here."""
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        out = m.stable_logsumexp(x)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.logsumexp(x, -1))

    def test_shape_drops_the_reduced_dim(self):
        self.assertEqual(m.stable_logsumexp(torch.randn(3, 5)).shape, (3,))

    def test_other_dims(self):
        x = torch.randn(3, 5)
        torch.testing.assert_close(m.stable_logsumexp(x, dim=0),
                                   torch.logsumexp(x, 0))

    def test_large_negative_values(self):
        x = torch.full((1, 3), -1e4)
        torch.testing.assert_close(m.stable_logsumexp(x),
                                   torch.logsumexp(x, -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
