"""Tests for 14 - performance and memory.

Where a test measures behaviour rather than output it counts *operations*,
not seconds - wall-clock assertions are flaky on shared machines.
"""

import importlib
import os
import sys
import time
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class Probe:
    """A value that counts how many times it was compared for equality."""

    def __init__(self, value):
        self.value = value
        self.comparisons = 0

    def __eq__(self, other):
        self.comparisons += 1
        return isinstance(other, Probe) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class TestCommonElements(unittest.TestCase):
    def test_preserves_order_of_the_first_argument(self):
        self.assertEqual(m.common_elements([3, 1, 2], [2, 3]), [3, 2])

    def test_deduplicates(self):
        self.assertEqual(m.common_elements([1, 1, 2], [1, 2]), [1, 2])

    def test_no_overlap(self):
        self.assertEqual(m.common_elements([1, 2], [3, 4]), [])

    def test_uses_a_set_not_a_linear_scan(self):
        """With a list membership test this is O(n*m) equality calls."""
        needles = [Probe(i) for i in range(50)]
        haystack = [Probe(i) for i in range(50, 100)]
        m.common_elements(needles, haystack)
        total = sum(p.comparisons for p in haystack)
        self.assertLess(total, 200,
                        "membership looks linear - convert b to a set")


class TestJoinWords(unittest.TestCase):
    def test_joins(self):
        self.assertEqual(m.join_words(["a", "b", "c"]), "a b c")

    def test_custom_separator(self):
        self.assertEqual(m.join_words(["a", "b"], "-"), "a-b")

    def test_empty(self):
        self.assertEqual(m.join_words([]), "")

    def test_handles_a_large_input_quickly(self):
        """Quadratic concatenation on 200k words takes visible seconds."""
        words = ["word"] * 200_000
        started = time.perf_counter()
        out = m.join_words(words)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(len(out), 200_000 * 5 - 1)


class TestCollatz(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(m.collatz_length(1), 1)
        self.assertEqual(m.collatz_length(2), 2)
        self.assertEqual(m.collatz_length(6), 9)

    def test_is_memoised(self):
        """Without caching this recursion repeats enormous amounts of work."""
        started = time.perf_counter()
        for n in range(1, 30_000):
            m.collatz_length(n)
        self.assertLess(time.perf_counter() - started, 5.0)


class TestSlots(unittest.TestCase):
    def test_stores_and_reads(self):
        point = m.Point(1, 2)
        self.assertEqual((point.x, point.y), (1, 2))

    def test_has_no_instance_dict(self):
        self.assertFalse(hasattr(m.Point(1, 2), "__dict__"))

    def test_rejects_unexpected_attributes(self):
        with self.assertRaises(AttributeError):
            m.Point(1, 2).z = 3

    def test_is_smaller_than_a_plain_object(self):
        class Plain:
            def __init__(self, x, y):
                self.x, self.y = x, y

        slotted = sys.getsizeof(m.Point(1, 2))
        plain = sys.getsizeof(Plain(1, 2)) + sys.getsizeof(Plain(1, 2).__dict__)
        self.assertLess(slotted, plain)


class TestRunningMax(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(m.running_max([1, 3, 2, 5, 4], 3), [3, 5, 5])

    def test_window_of_one_is_the_input(self):
        self.assertEqual(m.running_max([4, 2, 7], 1), [4, 2, 7])

    def test_window_equal_to_length(self):
        self.assertEqual(m.running_max([4, 2, 7], 3), [7])

    def test_descending_input(self):
        self.assertEqual(m.running_max([5, 4, 3, 2], 2), [5, 4, 3])

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ValueError):
            m.running_max([1, 2], 0)

    def test_is_linear_not_quadratic(self):
        values = list(range(200_000, 0, -1))
        started = time.perf_counter()
        m.running_max(values, 1000)
        self.assertLess(time.perf_counter() - started, 3.0,
                        "looks like max() per window - use a deque")


class TestInsertSorted(unittest.TestCase):
    def test_keeps_order(self):
        data = [1, 3, 5]
        m.insert_sorted(data, 4)
        self.assertEqual(data, [1, 3, 4, 5])

    def test_insert_at_the_ends(self):
        data = [2, 3]
        m.insert_sorted(data, 1)
        m.insert_sorted(data, 9)
        self.assertEqual(data, [1, 2, 3, 9])

    def test_duplicates_are_allowed(self):
        data = [1, 2]
        m.insert_sorted(data, 2)
        self.assertEqual(data, [1, 2, 2])


class TestRankByFrequency(unittest.TestCase):
    def test_most_frequent_first(self):
        self.assertEqual(m.rank_by_frequency("aaabbc"),
                         [("a", 3), ("b", 2), ("c", 1)])

    def test_ties_break_alphabetically(self):
        self.assertEqual(m.rank_by_frequency("ba"), [("a", 1), ("b", 1)])

    def test_limit(self):
        self.assertEqual(m.rank_by_frequency("aaabbc", limit=2),
                         [("a", 3), ("b", 2)])

    def test_empty(self):
        self.assertEqual(m.rank_by_frequency(""), [])


class TestStreaming(unittest.TestCase):
    def test_total_size(self):
        rows = [{"size": 1}, {"size": 2}]
        self.assertEqual(m.total_size(rows), 3)

    def test_total_size_streams(self):
        """A generator must not pull the whole source into memory first."""
        pulled = []

        def source():
            for i in range(5):
                pulled.append(i)
                yield {"size": i}

        self.assertEqual(m.total_size(source()), 10)
        self.assertEqual(pulled, [0, 1, 2, 3, 4])

    def test_first_match(self):
        rows = [{"n": 1}, {"n": 2}, {"n": 3}]
        self.assertEqual(m.first_match(rows, lambda r: r["n"] > 1), {"n": 2})

    def test_first_match_returns_none_when_absent(self):
        self.assertIsNone(m.first_match([{"n": 1}], lambda r: r["n"] > 9))

    def test_first_match_stops_early(self):
        """Must not evaluate the predicate for every row."""
        calls = []

        def predicate(row):
            calls.append(row["n"])
            return row["n"] == 2

        m.first_match([{"n": i} for i in range(100)], predicate)
        self.assertEqual(calls, [0, 1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
