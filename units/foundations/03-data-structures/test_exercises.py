"""Tests for 03 - core data structures. Read them; they are the
specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestFirstAndLast(unittest.TestCase):
    def test_several_items(self):
        self.assertEqual(m.first_and_last([1, 2, 3, 4]), (1, 4))

    def test_single_item_is_both_first_and_last(self):
        self.assertEqual(m.first_and_last([9]), (9, 9))

    def test_works_on_strings_too(self):
        self.assertEqual(m.first_and_last(["a", "b", "c"]), ("a", "c"))


class TestUniqueSorted(unittest.TestCase):
    def test_removes_duplicates(self):
        self.assertEqual(m.unique_sorted([3, 1, 2, 1, 3]), [1, 2, 3])

    def test_already_sorted_and_unique(self):
        self.assertEqual(m.unique_sorted([1, 2, 3]), [1, 2, 3])

    def test_empty_list(self):
        self.assertEqual(m.unique_sorted([]), [])


class TestEvensOnly(unittest.TestCase):
    def test_mixed_list(self):
        self.assertEqual(m.evens_only([1, 2, 3, 4, 5, 6]), [2, 4, 6])

    def test_preserves_original_order(self):
        self.assertEqual(m.evens_only([6, 3, 4, 1, 2]), [6, 4, 2])

    def test_no_evens(self):
        self.assertEqual(m.evens_only([1, 3, 5]), [])


class TestWordLengths(unittest.TestCase):
    def test_several_words(self):
        self.assertEqual(m.word_lengths(["a", "bb", "ccc"]), {"a": 1, "bb": 2, "ccc": 3})

    def test_empty_list(self):
        self.assertEqual(m.word_lengths([]), {})

    def test_duplicate_words_collapse_to_one_key(self):
        """A dict can't have two entries with the same key - the second
        occurrence just overwrites the first (with the same value here)."""
        self.assertEqual(m.word_lengths(["hi", "hi"]), {"hi": 2})


class TestMergeDicts(unittest.TestCase):
    def test_disjoint_keys_combine(self):
        self.assertEqual(m.merge_dicts({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

    def test_conflicting_key_b_wins(self):
        self.assertEqual(m.merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4}),
                          {"a": 1, "b": 3, "c": 4})

    def test_originals_are_not_mutated(self):
        a = {"x": 1}
        b = {"y": 2}
        m.merge_dicts(a, b)
        self.assertEqual(a, {"x": 1})
        self.assertEqual(b, {"y": 2})


class TestGroupByFirstLetter(unittest.TestCase):
    def test_groups_correctly(self):
        self.assertEqual(
            m.group_by_first_letter(["cat", "car", "dog", "deer"]),
            {"c": ["cat", "car"], "d": ["dog", "deer"]},
        )

    def test_preserves_relative_order_within_a_group(self):
        result = m.group_by_first_letter(["bob", "amy", "bea", "abe"])
        self.assertEqual(result["b"], ["bob", "bea"])
        self.assertEqual(result["a"], ["amy", "abe"])

    def test_empty_list(self):
        self.assertEqual(m.group_by_first_letter([]), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
