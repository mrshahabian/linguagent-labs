"""Tests for 13 - views, copies and dtypes."""

import importlib
import os
import unittest

import numpy as np

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestViews(unittest.TestCase):
    def test_basic_slicing_is_a_view(self):
        x = np.arange(10)
        self.assertTrue(m.is_view_of(x[2:5], x))

    def test_transpose_and_reshape_are_views(self):
        x = np.arange(12).reshape(3, 4)
        self.assertTrue(m.is_view_of(x.T, x))
        self.assertTrue(m.is_view_of(x.reshape(4, 3), x))

    def test_fancy_indexing_is_a_copy(self):
        x = np.arange(10)
        self.assertFalse(m.is_view_of(x[[0, 2, 4]], x))

    def test_boolean_masking_is_a_copy(self):
        x = np.arange(10)
        self.assertFalse(m.is_view_of(x[x > 5], x))

    def test_unrelated_arrays(self):
        self.assertFalse(m.is_view_of(np.arange(4), np.arange(4)))


class TestMutation(unittest.TestCase):
    def test_safe_scale_leaves_the_input_alone(self):
        x = np.array([1.0, 2.0])
        out = m.safe_scale(x, 3.0)
        np.testing.assert_allclose(out, [3.0, 6.0])
        np.testing.assert_allclose(x, [1.0, 2.0])

    def test_safe_scale_does_not_touch_the_parent_of_a_view(self):
        """The nastier version: mutating a view corrupts its base."""
        base = np.ones((3, 3))
        row = base[1]
        m.safe_scale(row, 5.0)
        np.testing.assert_allclose(base, np.ones((3, 3)))

    def test_scale_in_place_mutates_and_returns_none(self):
        x = np.array([1.0, 2.0])
        self.assertIsNone(m.scale_in_place(x, 3.0))
        np.testing.assert_allclose(x, [3.0, 6.0])

    def test_scale_in_place_writes_through_a_view(self):
        base = np.ones((2, 2))
        m.scale_in_place(base[0], 7.0)
        np.testing.assert_allclose(base, [[7.0, 7.0], [1.0, 1.0]])

    def test_zero_negatives_copies(self):
        x = np.array([-1.0, 2.0, -3.0])
        out = m.zero_negatives(x)
        np.testing.assert_allclose(out, [0.0, 2.0, 0.0])
        np.testing.assert_allclose(x, [-1.0, 2.0, -3.0])


class TestContiguity(unittest.TestCase):
    def test_a_transpose_is_not_c_contiguous(self):
        x = np.arange(12).reshape(3, 4)
        self.assertFalse(x.T.flags["C_CONTIGUOUS"])

    def test_makes_it_contiguous(self):
        x = np.arange(12).reshape(3, 4)
        out = m.to_contiguous(x.T)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(out, x.T)

    def test_already_contiguous_is_not_copied(self):
        x = np.arange(12).reshape(3, 4)
        self.assertTrue(m.is_view_of(m.to_contiguous(x), x))


class TestSlidingWindowView(unittest.TestCase):
    def test_shape_and_contents(self):
        out = m.sliding_window_view(np.arange(5), 3)
        np.testing.assert_array_equal(
            out, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_is_a_view_not_a_copy(self):
        x = np.arange(5)
        self.assertTrue(m.is_view_of(m.sliding_window_view(x, 2), x))

    def test_window_equal_to_length(self):
        out = m.sliding_window_view(np.arange(3), 3)
        np.testing.assert_array_equal(out, [[0, 1, 2]])

    def test_rejects_impossible_sizes(self):
        with self.assertRaises(ValueError):
            m.sliding_window_view(np.arange(3), 0)
        with self.assertRaises(ValueError):
            m.sliding_window_view(np.arange(3), 9)


class TestDtypes(unittest.TestCase):
    def test_naive_sum_of_int8_overflows(self):
        """Establishes the problem the next test solves."""
        x = np.full(10, 100, dtype=np.int8)
        self.assertNotEqual(int(np.sum(x, dtype=np.int8)), 1000)

    def test_safe_sum_does_not_overflow(self):
        x = np.full(10, 100, dtype=np.int8)
        self.assertEqual(int(m.safe_sum(x)), 1000)

    def test_safe_sum_of_floats(self):
        x = np.full(1000, 0.1, dtype=np.float32)
        self.assertAlmostEqual(float(m.safe_sum(x)), 100.0, places=3)

    def test_downcast_that_fits(self):
        out = m.downcast_safely(np.array([0, 100, 255]), np.uint8)
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out, [0, 100, 255])

    def test_downcast_that_would_wrap_raises(self):
        with self.assertRaises(ValueError):
            m.downcast_safely(np.array([300]), np.uint8)

    def test_downcast_of_a_negative_into_unsigned_raises(self):
        with self.assertRaises(ValueError):
            m.downcast_safely(np.array([-1]), np.uint8)


class TestMemoryBytes(unittest.TestCase):
    def test_matches_numpy(self):
        for shape, dtype in [((3, 4), np.float32), ((10,), np.uint8),
                             ((2, 3, 4), np.float64)]:
            self.assertEqual(m.memory_bytes(shape, dtype),
                             np.zeros(shape, dtype=dtype).nbytes)

    def test_float32_is_half_of_float64(self):
        self.assertEqual(m.memory_bytes((1000,), np.float32) * 2,
                         m.memory_bytes((1000,), np.float64))

    def test_scalar_shape(self):
        self.assertEqual(m.memory_bytes((), np.float32), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
