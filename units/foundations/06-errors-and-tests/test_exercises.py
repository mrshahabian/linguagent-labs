"""Tests for 06 - errors, exceptions and your first tests. Read them; they
are the specification, and also the running example this module's README
talks you through."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSafeDivide(unittest.TestCase):
    def test_ordinary_division(self):
        self.assertEqual(m.safe_divide(10, 2), 5.0)

    def test_division_by_zero_returns_none(self):
        self.assertIsNone(m.safe_divide(10, 0))

    def test_does_not_raise(self):
        """The whole point: calling this must never raise, even on the
        bad input that would normally crash a plain a / b."""
        try:
            m.safe_divide(1, 0)
        except ZeroDivisionError:
            self.fail("safe_divide raised ZeroDivisionError instead of catching it")


class TestParseInt(unittest.TestCase):
    def test_valid_integer_string(self):
        self.assertEqual(m.parse_int("42"), 42)

    def test_invalid_string_returns_none(self):
        self.assertIsNone(m.parse_int("abc"))

    def test_negative_number_string(self):
        self.assertEqual(m.parse_int("-7"), -7)


class TestValidateAge(unittest.TestCase):
    def test_valid_age_is_returned(self):
        self.assertEqual(m.validate_age(25), 25)

    def test_zero_is_valid(self):
        self.assertEqual(m.validate_age(0), 0)

    def test_negative_age_raises_the_specific_exception(self):
        with self.assertRaises(m.NegativeAgeError):
            m.validate_age(-1)

    def test_error_message_names_the_bad_value(self):
        with self.assertRaises(m.NegativeAgeError) as caught:
            m.validate_age(-5)
        self.assertIn("-5", str(caught.exception))


class TestWithdrawOrRaise(unittest.TestCase):
    def test_ordinary_withdrawal(self):
        self.assertEqual(m.withdraw_or_raise(100, 30), 70)

    def test_withdrawing_everything_is_allowed(self):
        self.assertEqual(m.withdraw_or_raise(100, 100), 0)

    def test_overdrawing_raises_the_specific_exception(self):
        with self.assertRaises(m.InsufficientFundsError):
            m.withdraw_or_raise(100, 1000)


class TestAverage(unittest.TestCase):
    def test_ordinary_list(self):
        self.assertEqual(m.average([1, 2, 3, 4]), 2.5)

    def test_single_number(self):
        self.assertEqual(m.average([7]), 7)

    def test_empty_list_raises_value_error_not_zero_division_error(self):
        """The bug this function guards against: sum([]) / len([]) would
        raise a confusing ZeroDivisionError with no context. This should
        raise a clear ValueError instead."""
        with self.assertRaises(ValueError):
            m.average([])

    def test_empty_list_error_message_is_informative(self):
        with self.assertRaises(ValueError) as caught:
            m.average([])
        self.assertIn("empty", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
