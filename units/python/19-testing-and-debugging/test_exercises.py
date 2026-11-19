"""Tests for 19 - testing and debugging patterns."""

import importlib
import logging
import os
import unittest
from unittest.mock import Mock

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestIsPrime(unittest.TestCase):
    """One parametrized test via subTest, instead of a method per case.

    Each case runs as its own labelled sub-test - one failure is reported
    with which (n, expected) pair failed, and the rest of the table still
    runs instead of the whole test method stopping at the first failure.
    """

    CASES = [
        (0, False), (1, False), (2, True), (3, True), (4, False),
        (17, True), (18, False), (19, True), (25, False), (97, True),
    ]

    def test_table(self):
        for n, expected in self.CASES:
            with self.subTest(n=n):
                self.assertEqual(m.is_prime(n), expected)


class TestFetchFirstSuccess(unittest.TestCase):
    def test_returns_first_non_none_result(self):
        client = Mock()
        client.get.side_effect = [None, "found", "unreachable"]
        self.assertEqual(m.fetch_first_success(client, [1, 2, 3]), "found")

    def test_skips_ids_whose_call_raises(self):
        client = Mock()
        client.get.side_effect = [Exception("boom"), None, "found"]
        self.assertEqual(m.fetch_first_success(client, [1, 2, 3]), "found")

    def test_all_fail_returns_none(self):
        client = Mock()
        client.get.side_effect = [None, None, None]
        self.assertIsNone(m.fetch_first_success(client, [1, 2, 3]))

    def test_calls_client_with_each_id_in_order(self):
        client = Mock()
        client.get.side_effect = [None, None, "found"]
        m.fetch_first_success(client, [10, 20, 30])
        client.get.assert_any_call(10)
        client.get.assert_any_call(20)
        client.get.assert_any_call(30)

    def test_stops_at_first_success(self):
        client = Mock()
        client.get.side_effect = ["found", "should not be reached"]
        m.fetch_first_success(client, [1, 2])
        self.assertEqual(client.get.call_count, 1)


class TestTTLCache(unittest.TestCase):
    def test_returns_value_before_expiry(self):
        now = {"t": 0}
        cache = m.TTLCache(now_fn=lambda: now["t"])
        cache.set("a", "value", ttl=10)
        now["t"] = 5
        self.assertEqual(cache.get("a"), "value")

    def test_returns_none_after_expiry(self):
        now = {"t": 0}
        cache = m.TTLCache(now_fn=lambda: now["t"])
        cache.set("a", "value", ttl=10)
        now["t"] = 11
        self.assertIsNone(cache.get("a"))

    def test_missing_key_returns_none(self):
        cache = m.TTLCache(now_fn=lambda: 0)
        self.assertIsNone(cache.get("missing"))

    def test_expired_entry_is_removed_not_just_hidden(self):
        now = {"t": 0}
        cache = m.TTLCache(now_fn=lambda: now["t"])
        cache.set("a", "value", ttl=10)
        now["t"] = 11
        cache.get("a")
        self.assertNotIn("a", cache._store)


class TestDivideSafely(unittest.TestCase):
    def test_normal_division(self):
        self.assertEqual(m.divide_safely(6, 3), 2.0)

    def test_zero_denominator_returns_none(self):
        self.assertIsNone(m.divide_safely(1, 0))

    def test_zero_denominator_logs_a_warning(self):
        with self.assertLogs(level="WARNING") as captured:
            m.divide_safely(1, 0)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].levelno, logging.WARNING)

    def test_normal_division_does_not_log(self):
        with self.assertRaises(AssertionError):
            with self.assertLogs(level="WARNING"):
                m.divide_safely(6, 3)


class TestFindFirstFailure(unittest.TestCase):
    def test_finds_the_boundary(self):
        self.assertEqual(m.find_first_failure(20, lambda i: i >= 7), 7)

    def test_boundary_at_start(self):
        self.assertEqual(m.find_first_failure(10, lambda i: True), 0)

    def test_no_failure_returns_n(self):
        self.assertEqual(m.find_first_failure(10, lambda i: False), 10)

    def test_boundary_at_last_index(self):
        self.assertEqual(m.find_first_failure(10, lambda i: i >= 9), 9)

    def test_uses_few_predicate_calls(self):
        calls = []

        def predicate(i):
            calls.append(i)
            return i >= 500

        m.find_first_failure(1000, predicate)
        self.assertLess(len(calls), 15)  # log2(1000) =~ 10; a linear scan would be 500+


class TestMatmulChecked(unittest.TestCase):
    def test_multiplies_compatible_matrices(self):
        result = m.matmul_checked([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(result, [[19, 22], [43, 50]])

    def test_raises_on_shape_mismatch(self):
        with self.assertRaises(m.MatrixShapeError):
            m.matmul_checked([[1, 2]], [[1, 2]])

    def test_error_message_names_both_shapes(self):
        with self.assertRaises(m.MatrixShapeError) as ctx:
            m.matmul_checked([[1, 2, 3]], [[1, 2], [3, 4]])
        message = str(ctx.exception)
        self.assertIn("1x3", message)
        self.assertIn("2x2", message)

    def test_matrix_shape_error_is_a_value_error(self):
        self.assertTrue(issubclass(m.MatrixShapeError, ValueError))


if __name__ == "__main__":
    unittest.main()
