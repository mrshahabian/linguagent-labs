"""Tests for 07 - tooling and complexity basics. Read them; they are the
specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestLinearSearch(unittest.TestCase):
    def test_found_in_the_middle(self):
        self.assertEqual(m.linear_search([5, 3, 8, 1], 8), 2)

    def test_found_at_index_zero(self):
        self.assertEqual(m.linear_search([5, 3, 8, 1], 5), 0)

    def test_not_found_returns_negative_one(self):
        self.assertEqual(m.linear_search([5, 3, 8, 1], 99), -1)

    def test_empty_list(self):
        self.assertEqual(m.linear_search([], 1), -1)

    def test_finds_the_first_occurrence(self):
        self.assertEqual(m.linear_search([1, 2, 2, 3], 2), 1)


class TestBinarySearch(unittest.TestCase):
    def test_found_in_the_middle(self):
        self.assertEqual(m.binary_search([1, 3, 5, 7, 9, 11], 7), 3)

    def test_found_at_the_start(self):
        self.assertEqual(m.binary_search([1, 3, 5, 7, 9, 11], 1), 0)

    def test_found_at_the_end(self):
        self.assertEqual(m.binary_search([1, 3, 5, 7, 9, 11], 11), 5)

    def test_not_present_between_elements(self):
        self.assertEqual(m.binary_search([1, 3, 5, 7, 9, 11], 4), -1)

    def test_empty_list(self):
        self.assertEqual(m.binary_search([], 1), -1)

    def test_single_element_found(self):
        self.assertEqual(m.binary_search([42], 42), 0)

    def test_single_element_not_found(self):
        self.assertEqual(m.binary_search([42], 1), -1)


class TestHasDuplicatePair(unittest.TestCase):
    def test_has_a_duplicate(self):
        self.assertTrue(m.has_duplicate_pair([1, 2, 3, 2]))

    def test_no_duplicates(self):
        self.assertFalse(m.has_duplicate_pair([1, 2, 3, 4]))

    def test_empty_list_has_no_duplicates(self):
        self.assertFalse(m.has_duplicate_pair([]))

    def test_duplicate_at_the_ends(self):
        self.assertTrue(m.has_duplicate_pair([1, 2, 3, 1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
