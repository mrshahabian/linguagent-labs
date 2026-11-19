"""Tests for 01 - syntax, values and control flow. Read them; they are the
specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestClassifyNumber(unittest.TestCase):
    def test_negative(self):
        self.assertEqual(m.classify_number(-5), "negative")

    def test_zero(self):
        self.assertEqual(m.classify_number(0), "zero")

    def test_positive(self):
        self.assertEqual(m.classify_number(5), "positive")

    def test_negative_float(self):
        self.assertEqual(m.classify_number(-0.5), "negative")


class TestFizzbuzz(unittest.TestCase):
    def test_small_range(self):
        self.assertEqual(m.fizzbuzz(5), ["1", "2", "Fizz", "4", "Buzz"])

    def test_multiple_of_fifteen_is_fizzbuzz_not_fizz_then_buzz(self):
        self.assertEqual(m.fizzbuzz(15)[-1], "FizzBuzz")

    def test_n_is_included(self):
        """fizzbuzz(1) must include 1, not stop before it."""
        self.assertEqual(m.fizzbuzz(1), ["1"])

    def test_length_matches_n(self):
        self.assertEqual(len(m.fizzbuzz(30)), 30)


class TestCountVowels(unittest.TestCase):
    def test_counts_lowercase(self):
        self.assertEqual(m.count_vowels("banana"), 3)

    def test_counts_uppercase_too(self):
        self.assertEqual(m.count_vowels("HELLO"), 2)

    def test_no_vowels(self):
        self.assertEqual(m.count_vowels("rhythm"), 0)

    def test_empty_string(self):
        self.assertEqual(m.count_vowels(""), 0)


class TestGradeForScore(unittest.TestCase):
    def test_a_grade(self):
        self.assertEqual(m.grade_for_score(95), "A")

    def test_boundary_is_inclusive(self):
        """90 is an A, not a B - the boundary belongs to the grade above it."""
        self.assertEqual(m.grade_for_score(90), "A")
        self.assertEqual(m.grade_for_score(89), "B")

    def test_f_grade(self):
        self.assertEqual(m.grade_for_score(40), "F")

    def test_every_boundary(self):
        self.assertEqual(m.grade_for_score(80), "B")
        self.assertEqual(m.grade_for_score(70), "C")
        self.assertEqual(m.grade_for_score(60), "D")
        self.assertEqual(m.grade_for_score(59), "F")


class TestSumUpTo(unittest.TestCase):
    def test_typical_case(self):
        self.assertEqual(m.sum_up_to(10), 55)

    def test_n_equals_one(self):
        self.assertEqual(m.sum_up_to(1), 1)

    def test_n_equals_zero_sums_nothing(self):
        """The while loop should never execute - the sum of nothing is 0."""
        self.assertEqual(m.sum_up_to(0), 0)


class TestIsLeapYear(unittest.TestCase):
    def test_ordinary_leap_year(self):
        self.assertTrue(m.is_leap_year(2024))

    def test_ordinary_non_leap_year(self):
        self.assertFalse(m.is_leap_year(2023))

    def test_century_not_divisible_by_400_is_not_a_leap_year(self):
        """1900 is divisible by 4 and by 100, but not by 400."""
        self.assertFalse(m.is_leap_year(1900))

    def test_century_divisible_by_400_is_a_leap_year(self):
        self.assertTrue(m.is_leap_year(2000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
