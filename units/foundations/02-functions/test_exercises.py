"""Tests for 02 - functions and scope. Read them; they are the
specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestGreet(unittest.TestCase):
    def test_default_greeting(self):
        self.assertEqual(m.greet("Ana"), "Hello, Ana!")

    def test_custom_greeting(self):
        self.assertEqual(m.greet("Ana", "Hi"), "Hi, Ana!")

    def test_custom_greeting_by_keyword(self):
        self.assertEqual(m.greet("Ana", greeting="Yo"), "Yo, Ana!")


class TestAddAll(unittest.TestCase):
    def test_several_numbers(self):
        self.assertEqual(m.add_all(1, 2, 3), 6)

    def test_one_number(self):
        self.assertEqual(m.add_all(5), 5)

    def test_no_arguments_sums_to_zero(self):
        self.assertEqual(m.add_all(), 0)


class TestBuildProfile(unittest.TestCase):
    def test_name_only(self):
        self.assertEqual(m.build_profile("Al"), {"name": "Al"})

    def test_name_plus_details(self):
        self.assertEqual(
            m.build_profile("Al", age=30, city="NY"),
            {"name": "Al", "age": 30, "city": "NY"},
        )

    def test_details_do_not_overwrite_name_key_accidentally(self):
        """name is a positional argument, not a keyword - it stays "Al"
        even though details is a completely separate dict."""
        profile = m.build_profile("Al", age=30)
        self.assertEqual(profile["name"], "Al")


class TestTotalPrice(unittest.TestCase):
    def test_no_tax_by_default(self):
        self.assertEqual(m.total_price([10, 20]), 30)

    def test_tax_applied_to_the_sum(self):
        self.assertAlmostEqual(m.total_price([10, 20], tax_rate=0.1), 33.0)

    def test_empty_list_is_zero(self):
        self.assertEqual(m.total_price([]), 0)


class TestFactorial(unittest.TestCase):
    def test_base_case(self):
        self.assertEqual(m.factorial(0), 1)

    def test_typical_case(self):
        self.assertEqual(m.factorial(5), 120)

    def test_one(self):
        self.assertEqual(m.factorial(1), 1)


class TestSumDigits(unittest.TestCase):
    def test_multi_digit_number(self):
        self.assertEqual(m.sum_digits(123), 6)

    def test_single_digit_is_the_base_case(self):
        self.assertEqual(m.sum_digits(7), 7)

    def test_zero(self):
        self.assertEqual(m.sum_digits(0), 0)

    def test_number_with_repeated_digits(self):
        self.assertEqual(m.sum_digits(999), 27)


if __name__ == "__main__":
    unittest.main(verbosity=2)
