"""Tests for 12 - broadcasting and vectorisation."""

import importlib
import os
import unittest

import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestBroadcastShape(unittest.TestCase):
    def test_pads_from_the_left(self):
        self.assertEqual(m.broadcast_shape((3, 1), (1, 4)), (3, 4))
        self.assertEqual(m.broadcast_shape((5,), (3, 5)), (3, 5))

    def test_scalar_broadcasts_with_anything(self):
        self.assertEqual(m.broadcast_shape((), (2, 3)), (2, 3))

    def test_incompatible_returns_none(self):
        self.assertIsNone(m.broadcast_shape((3,), (4,)))
        self.assertIsNone(m.broadcast_shape((2, 3), (2, 4)))

    def test_agrees_with_numpy(self):
        for a, b in [((3, 1), (1, 4)), ((5,), (3, 5)), ((), (2, 3)),
                     ((2, 1, 4), (3, 1))]:
            self.assertEqual(m.broadcast_shape(a, b),
                             np.broadcast_shapes(a, b))


class TestPairwiseDistances(unittest.TestCase):
    def test_shape(self):
        a, b = np.zeros((4, 3)), np.zeros((5, 3))
        self.assertEqual(m.pairwise_distances(a, b).shape, (4, 5))

    def test_values(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        b = np.array([[0.0, 0.0]])
        np.testing.assert_allclose(m.pairwise_distances(a, b),
                                   [[0.0], [5.0]])

    def test_matches_a_loop(self):
        rng = np.random.RandomState(0)
        a, b = rng.randn(6, 3), rng.randn(4, 3)
        expected = np.array([[np.linalg.norm(x - y) for y in b] for x in a])
        np.testing.assert_allclose(m.pairwise_distances(a, b), expected)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        rng = np.random.RandomState(1)
        out = m.softmax(rng.randn(5, 7), axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(5))

    def test_is_numerically_stable(self):
        """Without subtracting the max, exp(1000) overflows to inf."""
        out = m.softmax(np.array([1000.0, 1000.0]))
        np.testing.assert_allclose(out, [0.5, 0.5])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_respects_the_axis(self):
        x = np.arange(6.0).reshape(2, 3)
        np.testing.assert_allclose(m.softmax(x, axis=0).sum(axis=0),
                                   np.ones(3))

    def test_shift_invariant(self):
        x = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(m.softmax(x), m.softmax(x + 100))


class TestOneHot(unittest.TestCase):
    def test_shape_and_values(self):
        out = m.one_hot(np.array([0, 2, 1]), 3)
        np.testing.assert_array_equal(
            out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    def test_dtype_is_float32(self):
        self.assertEqual(m.one_hot(np.array([0]), 2).dtype, np.float32)

    def test_rows_sum_to_one(self):
        out = m.one_hot(np.array([0, 1, 1, 2]), 4)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4))


class TestMaskedMean(unittest.TestCase):
    def test_ignores_masked_positions(self):
        x = np.array([1.0, 100.0, 3.0])
        mask = np.array([True, False, True])
        self.assertAlmostEqual(float(m.masked_mean(x, mask)), 2.0)

    def test_along_an_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [True, True]])
        np.testing.assert_allclose(m.masked_mean(x, mask, axis=1), [1.0, 3.5])

    def test_all_masked_gives_nan_not_an_error(self):
        x = np.array([1.0, 2.0])
        mask = np.array([False, False])
        self.assertTrue(np.isnan(m.masked_mean(x, mask)))


class TestIoU(unittest.TestCase):
    def test_identical_boxes_score_one(self):
        box = np.array([[0.0, 0.0, 2.0, 2.0]])
        np.testing.assert_allclose(m.iou(box, box), [[1.0]])

    def test_disjoint_boxes_score_zero(self):
        a = np.array([[0.0, 0.0, 1.0, 1.0]])
        b = np.array([[5.0, 5.0, 6.0, 6.0]])
        np.testing.assert_allclose(m.iou(a, b), [[0.0]])

    def test_half_overlap(self):
        a = np.array([[0.0, 0.0, 2.0, 2.0]])     # area 4
        b = np.array([[1.0, 0.0, 3.0, 2.0]])     # area 4, overlap 2
        np.testing.assert_allclose(m.iou(a, b), [[2 / 6]])

    def test_shape_is_the_full_cross_product(self):
        a = np.zeros((3, 4)) + np.array([0.0, 0.0, 1.0, 1.0])
        b = np.zeros((5, 4)) + np.array([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(m.iou(a, b).shape, (3, 5))

    def test_touching_edges_do_not_overlap(self):
        a = np.array([[0.0, 0.0, 1.0, 1.0]])
        b = np.array([[1.0, 0.0, 2.0, 1.0]])
        np.testing.assert_allclose(m.iou(a, b), [[0.0]])


class TestBatchedDot(unittest.TestCase):
    def test_values(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(m.batched_dot(a, b), [17.0, 53.0])

    def test_shape_is_one_dimensional(self):
        rng = np.random.RandomState(2)
        self.assertEqual(m.batched_dot(rng.randn(9, 3), rng.randn(9, 3)).shape,
                         (9,))

    def test_matches_einsum(self):
        rng = np.random.RandomState(3)
        a, b = rng.randn(7, 5), rng.randn(7, 5)
        np.testing.assert_allclose(m.batched_dot(a, b),
                                   np.einsum("nd,nd->n", a, b))


class TestTopK(unittest.TestCase):
    def test_returns_indices_highest_first(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        np.testing.assert_array_equal(m.top_k(scores, 2), [1, 3])

    def test_ties_break_towards_the_lower_index(self):
        scores = np.array([0.5, 0.5, 0.1])
        np.testing.assert_array_equal(m.top_k(scores, 2), [0, 1])

    def test_k_larger_than_the_input(self):
        scores = np.array([0.2, 0.4])
        np.testing.assert_array_equal(m.top_k(scores, 10), [1, 0])

    def test_k_zero(self):
        self.assertEqual(len(m.top_k(np.array([1.0, 2.0]), 0)), 0)


class TestNormaliseRows(unittest.TestCase):
    def test_rows_become_unit_length(self):
        x = np.array([[3.0, 4.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            np.linalg.norm(m.normalise_rows(x), axis=1), [1.0, 1.0])

    def test_zero_rows_stay_zero_and_do_not_produce_nan(self):
        x = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = m.normalise_rows(x)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[0], [0.0, 0.0])

    def test_direction_is_preserved(self):
        x = np.array([[2.0, 0.0]])
        np.testing.assert_allclose(m.normalise_rows(x), [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
