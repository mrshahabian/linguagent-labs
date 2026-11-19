"""Tests for 58 - Move zeros, in place. Read them; they are the
specification.
"""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestMoveZerosToEnd(unittest.TestCase):
    def test_typical_case(self):
        nums = [0, 1, 0, 3, 12]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [1, 3, 12, 0, 0])
        self.assertIs(result, nums)  # Must be in-place

    def test_all_zeros(self):
        nums = [0, 0, 0]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [0, 0, 0])

    def test_no_zeros(self):
        nums = [1, 2, 3]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [1, 2, 3])

    def test_empty(self):
        nums = []
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [])

    def test_single_zero(self):
        nums = [0]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [0])

    def test_single_nonzero(self):
        nums = [5]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [5])

    def test_zeros_at_front_and_back(self):
        nums = [0, 9, 0, 8, 0]
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [9, 8, 0, 0, 0])

    def test_large_array(self):
        nums = [0] * 50 + [1] * 50
        result = m.move_zeros_to_end(nums)
        self.assertEqual(result, [1] * 50 + [0] * 50)


class TestMoveValueToEnd(unittest.TestCase):
    def test_move_integer_value(self):
        nums = [2, 1, 2, 3, 2]
        result = m.move_value_to_end(nums, 2)
        self.assertEqual(result, [1, 3, 2, 2, 2])

    def test_move_string_value(self):
        items = ["apple", "banana", "apple", "cherry"]
        result = m.move_value_to_end(items, "apple")
        self.assertEqual(result, ["banana", "cherry", "apple", "apple"])

    def test_equality_vs_identity(self):
        # 0.0 == 0 is True, but 0.0 is 0 is False.
        # The implementation must use == to match target.
        nums = [0.0, 1, 0, 2]
        result = m.move_value_to_end(nums, 0)
        # Both 0.0 and 0 should be treated as target and moved to the end
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 2)
        self.assertEqual(set(result[2:]), {0.0, 0})

    def test_no_matches(self):
        nums = [1, 2, 3]
        self.assertEqual(m.move_value_to_end(nums, 99), [1, 2, 3])

    def test_all_matches(self):
        nums = [7, 7, 7]
        self.assertEqual(m.move_value_to_end(nums, 7), [7, 7, 7])

    def test_none_matches_empty(self):
        self.assertEqual(m.move_value_to_end([], 42), [])

    def test_custom_objects(self):
        class Dummy:
            def __init__(self, val):
                self.val = val
            def __eq__(self, other):
                if isinstance(other, Dummy):
                    return self.val == other.val
                return False
            def __hash__(self):
                return hash(self.val)

        items = [Dummy(1), Dummy(2), Dummy(1)]
        result = m.move_value_to_end(items, Dummy(1))
        self.assertEqual(result[0].val, 2)
        self.assertEqual(result[1].val, 1)
        self.assertEqual(result[2].val, 1)


class TestStablePartition(unittest.TestCase):
    def test_partition_even_odd(self):
        # Evens go to the end (predicate returns True for evens)
        nums = [1, 2, 3, 4, 5, 6]
        is_even = lambda x: x % 2 == 0
        result = m.stable_partition(nums, is_even)
        # Odds (False) must be at the front, preserving relative order: [1, 3, 5]
        self.assertEqual(result[:3], [1, 3, 5])
        # Evens (True) must be at the back
        self.assertEqual(set(result[3:]), {2, 4, 6})

    def test_partition_strings_by_length(self):
        # Strings with length > 3 go to the end
        words = ["cat", "elephant", "dog", "tiger", "ox"]
        is_long = lambda s: len(s) > 3
        result = m.stable_partition(words, is_long)
        # Short words (False) at front: ["cat", "dog", "ox"]
        self.assertEqual(result[:3], ["cat", "dog", "ox"])
        # Long words (True) at back
        self.assertEqual(set(result[3:]), {"elephant", "tiger"})

    def test_all_satisfy(self):
        nums = [2, 4, 6]
        is_even = lambda x: x % 2 == 0
        self.assertEqual(set(m.stable_partition(nums, is_even)), {2, 4, 6})

    def test_none_satisfy(self):
        nums = [1, 3, 5]
        is_even = lambda x: x % 2 == 0
        self.assertEqual(m.stable_partition(nums, is_even), [1, 3, 5])

    def test_empty_list(self):
        self.assertEqual(m.stable_partition([], lambda x: True), [])

    def test_single_element_satisfies(self):
        self.assertEqual(m.stable_partition([2], lambda x: x % 2 == 0), [2])

    def test_single_element_does_not_satisfy(self):
        self.assertEqual(m.stable_partition([1], lambda x: x % 2 == 0), [1])


class TestMinSwapsToGroup(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(m.min_swaps_to_group([]), 0)

    def test_no_zeros(self):
        self.assertEqual(m.min_swaps_to_group([1, 2, 3]), 0)

    def test_all_zeros(self):
        self.assertEqual(m.min_swaps_to_group([0, 0, 0]), 0)

    def test_already_grouped_left(self):
        self.assertEqual(m.min_swaps_to_group([0, 0, 1, 1, 1]), 0)

    def test_already_grouped_right(self):
        self.assertEqual(m.min_swaps_to_group([1, 1, 0, 0, 0]), 0)

    def test_typical_case_1(self):
        # Total zeros = 2. Windows of size 2:
        # [1, 0], [0, 1], [1, 0], [0, 1] -> max zeros in window is 1. Swaps = 2 - 1 = 1.
        self.assertEqual(m.min_swaps_to_group([1, 0, 1, 0, 1]), 1)

    def test_typical_case_2(self):
        # Total zeros = 4. Windows of size 4:
        # [0, 1, 0, 0] has 3 zeros. Swaps = 4 - 3 = 1.
        self.assertEqual(m.min_swaps_to_group([0, 1, 0, 0, 1, 1, 0]), 1)

    def test_does_not_modify_input(self):
        nums = [1, 0, 1, 0, 1]
        m.min_swaps_to_group(nums)
        self.assertEqual(nums, [1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
