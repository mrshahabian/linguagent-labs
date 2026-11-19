"""Tests for 11 - iteration, closures and decorators."""

import importlib
import itertools
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestCountdown(unittest.TestCase):
    def test_counts_down(self):
        self.assertEqual(list(m.Countdown(3)), [3, 2, 1])

    def test_zero_is_empty(self):
        self.assertEqual(list(m.Countdown(0)), [])

    def test_is_restartable(self):
        """__iter__ must hand out a fresh iterator each time."""
        countdown = m.Countdown(3)
        self.assertEqual(list(countdown), [3, 2, 1])
        self.assertEqual(list(countdown), [3, 2, 1])


class TestSliding(unittest.TestCase):
    def test_windows_overlap(self):
        self.assertEqual(list(m.sliding([1, 2, 3, 4], 2)),
                         [(1, 2), (2, 3), (3, 4)])

    def test_window_larger_than_input_yields_nothing(self):
        self.assertEqual(list(m.sliding([1, 2], 5)), [])

    def test_window_equal_to_input(self):
        self.assertEqual(list(m.sliding([1, 2], 2)), [(1, 2)])

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            list(m.sliding([1, 2, 3], 0))


class TestChunked(unittest.TestCase):
    def test_batches(self):
        self.assertEqual(list(m.chunked(range(7), 3)),
                         [[0, 1, 2], [3, 4, 5], [6]])

    def test_empty_input(self):
        self.assertEqual(list(m.chunked([], 3)), [])

    def test_works_on_an_infinite_iterable(self):
        """Must not call len() or materialise the input."""
        batches = m.chunked(itertools.count(), 2)
        self.assertEqual(next(batches), [0, 1])
        self.assertEqual(next(batches), [2, 3])

    def test_consumes_lazily(self):
        source = iter(range(100))
        batches = m.chunked(source, 2)
        next(batches)
        # Only the first batch should have been pulled.
        self.assertEqual(next(source), 2)


class TestFlatten(unittest.TestCase):
    def test_nested_lists(self):
        self.assertEqual(list(m.flatten([1, [2, [3, [4]]], 5])), [1, 2, 3, 4, 5])

    def test_tuples_count_as_containers(self):
        self.assertEqual(list(m.flatten([1, (2, 3)])), [1, 2, 3])

    def test_strings_are_leaves(self):
        """Recursing into a string never terminates: 'a'[0] is 'a'."""
        self.assertEqual(list(m.flatten(["ab", ["cd"]])), ["ab", "cd"])

    def test_empty(self):
        self.assertEqual(list(m.flatten([])), [])


class TestUnique(unittest.TestCase):
    def test_preserves_first_occurrence_order(self):
        self.assertEqual(list(m.unique([3, 1, 3, 2, 1])), [3, 1, 2])

    def test_is_lazy(self):
        source = iter([1, 1, 2, 3, 4, 5])
        result = m.unique(source)
        self.assertEqual(next(result), 1)
        self.assertEqual(next(result), 2)
        # 3 onwards must still be unconsumed.
        self.assertEqual(next(source), 3)


class TestClosures(unittest.TestCase):
    def test_each_counter_returns_its_own_index(self):
        """The late-binding trap: a closure captures the variable, not the
        value it had when the closure was made."""
        counters = m.make_counters(3)
        self.assertEqual([fn() for fn in counters], [0, 1, 2])

    def test_none(self):
        self.assertEqual(m.make_counters(0), [])


class TestRetry(unittest.TestCase):
    def test_returns_on_first_success(self):
        calls = []

        @m.retry(3)
        def always_fine():
            calls.append(1)
            return "ok"

        self.assertEqual(always_fine(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_until_it_succeeds(self):
        calls = []

        @m.retry(3)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 3)

    def test_reraises_after_the_last_attempt(self):
        @m.retry(2)
        def always_fails():
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            always_fails()

    def test_only_catches_the_named_exceptions(self):
        @m.retry(3, exceptions=(ValueError,))
        def wrong_error():
            raise KeyError("not retried")

        with self.assertRaises(KeyError):
            wrong_error()

    def test_preserves_metadata(self):
        """functools.wraps, or the decorated function loses its identity."""
        @m.retry(2)
        def documented():
            """A docstring worth keeping."""
            return 1

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "A docstring worth keeping.")


class TestCounted(unittest.TestCase):
    def test_counts_calls(self):
        @m.counted
        def work(x):
            return x * 2

        self.assertEqual(work.calls, 0)
        work(1)
        work(2)
        self.assertEqual(work.calls, 2)

    def test_still_returns_the_value(self):
        @m.counted
        def work(x):
            return x * 2

        self.assertEqual(work(21), 42)

    def test_preserves_metadata(self):
        @m.counted
        def named():
            pass

        self.assertEqual(named.__name__, "named")


class TestAtomic(unittest.TestCase):
    def test_keeps_changes_when_the_block_succeeds(self):
        data = {"a": 1}
        with m.atomic(data) as scratch:
            scratch["b"] = 2
        self.assertEqual(data, {"a": 1, "b": 2})

    def test_rolls_back_when_the_block_raises(self):
        data = {"a": 1}
        with self.assertRaises(ValueError):
            with m.atomic(data) as scratch:
                scratch["b"] = 2
                del scratch["a"]
                raise ValueError("abort")
        self.assertEqual(data, {"a": 1})

    def test_rolls_back_in_place(self):
        """The caller's own dict must be restored, not a replacement."""
        data = {"a": 1}
        with self.assertRaises(ValueError):
            with m.atomic(data):
                data["b"] = 2
                raise ValueError("abort")
        self.assertEqual(data, {"a": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
