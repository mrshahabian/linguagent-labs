"""Tests for 59 — Two-sum to four-sum
"""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestTwoSum(unittest.TestCase):
    def test_basic_two_sum(self):
        self.assertEqual(m.two_sum([2, 7, 11, 15], 9), (0, 1))

    def test_two_sum_unsorted(self):
        self.assertEqual(m.two_sum([3, 2, 4], 6), (1, 2))

    def test_two_sum_no_solution(self):
        self.assertIsNone(m.two_sum([1, 2, 3], 7))

    def test_two_sum_same_element_not_reused(self):
        # Target is 6, but we only have one 3. Should return None.
        self.assertIsNone(m.two_sum([3, 1, 4], 6))


class TestTwoSumSorted(unittest.TestCase):
    def test_basic_sorted_pairs(self):
        self.assertEqual(
            m.two_sum_sorted([1, 2, 3, 4, 5, 6], 7),
            [(1, 6), (2, 5), (3, 4)]
        )

    def test_sorted_pairs_with_duplicates(self):
        # Duplicates should be skipped to return only unique pairs
        self.assertEqual(
            m.two_sum_sorted([1, 1, 2, 2, 3, 3, 4, 4], 5),
            [(1, 4), (2, 3)]
        )

    def test_sorted_no_solution(self):
        self.assertEqual(m.two_sum_sorted([1, 2, 3], 10), [])

    def test_sorted_negative_numbers(self):
        self.assertEqual(
            m.two_sum_sorted([-4, -3, -2, -1, 0, 1, 2], -2),
            [(-4, 2), (-3, 1), (-2, 0)]
        )


class TestThreeSum(unittest.TestCase):
    def test_basic_three_sum(self):
        self.assertEqual(
            m.three_sum([-1, 0, 1, 2, -1, -4], 0),
            [(-1, -1, 2), (-1, 0, 1)]
        )

    def test_three_sum_no_triplets(self):
        self.assertEqual(m.three_sum([1, 2, 3], 100), [])

    def test_three_sum_all_zeros(self):
        self.assertEqual(m.three_sum([0, 0, 0, 0], 0), [(0, 0, 0)])

    def test_three_sum_with_target(self):
        self.assertEqual(
            m.three_sum([1, 2, 3, 4, 5], 9),
            [(1, 3, 5), (2, 3, 4)]
        )


class TestFourSum(unittest.TestCase):
    def test_basic_four_sum(self):
        self.assertEqual(
            m.four_sum([1, 0, -1, 0, -2, 2], 0),
            [(-2, -1, 1, 2), (-2, 0, 0, 2), (-1, 0, 0, 1)]
        )

    def test_four_sum_no_solution(self):
        self.assertEqual(m.four_sum([1, 2, 3, 4], 100), [])

    def test_four_sum_all_same(self):
        self.assertEqual(m.four_sum([2, 2, 2, 2, 2], 8), [(2, 2, 2, 2)])

    def test_four_sum_large_target(self):
        self.assertEqual(
            m.four_sum([1, 2, 3, 4, 5, 6], 14),
            [(1, 2, 5, 6), (1, 3, 4, 6), (2, 3, 4, 5)]
        )


class TestCountPairsWithDifference(unittest.TestCase):
    def test_positive_difference(self):
        # Pairs with diff 2: (3, 1), (4, 2), (5, 3)
        self.assertEqual(m.count_pairs_with_difference([1, 2, 3, 4, 5], 2), 3)

    def test_positive_difference_with_duplicates(self):
        # Unique pairs with diff 1: (2, 1), (3, 2)
        self.assertEqual(m.count_pairs_with_difference([1, 1, 2, 2, 3, 3], 1), 2)

    def test_zero_difference(self):
        # Elements appearing at least twice: 1 and 3
        self.assertEqual(m.count_pairs_with_difference([1, 1, 2, 3, 3, 3, 4], 0), 2)

    def test_zero_difference_no_duplicates(self):
        self.assertEqual(m.count_pairs_with_difference([1, 2, 3, 4], 0), 0)

    def test_negative_difference_raises_value_error(self):
        with self.assertRaises(ValueError):
            m.count_pairs_with_difference([1, 2, 3], -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
