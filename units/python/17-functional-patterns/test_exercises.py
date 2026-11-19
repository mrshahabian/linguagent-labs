"""Tests for 17 - functional patterns: itertools, functools, decorators."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestFlatten(unittest.TestCase):
    def test_flattens_one_level(self):
        self.assertEqual(m.flatten([[1, 2], [3], [4, 5]]), [1, 2, 3, 4, 5])

    def test_empty_sublists(self):
        self.assertEqual(m.flatten([[], [1], []]), [1])

    def test_empty_input(self):
        self.assertEqual(m.flatten([]), [])


class TestGroupAdjacent(unittest.TestCase):
    def test_merges_only_consecutive_runs(self):
        result = m.group_adjacent(["a", "a", "b", "a"])
        self.assertEqual(result, [("a", ["a", "a"]), ("b", ["b"]), ("a", ["a"])])

    def test_all_same_is_one_group(self):
        result = m.group_adjacent([1, 1, 1])
        self.assertEqual(result, [(1, [1, 1, 1])])

    def test_no_repeats_is_one_group_each(self):
        result = m.group_adjacent([1, 2, 3])
        self.assertEqual(result, [(1, [1]), (2, [2]), (3, [3])])


class TestGroupAll(unittest.TestCase):
    def test_merges_non_adjacent_matches(self):
        result = m.group_all(["a", "a", "b", "a"], key=lambda x: x)
        self.assertEqual(result, {"a": ["a", "a", "a"], "b": ["b"]})

    def test_groups_by_key_function(self):
        result = m.group_all([1, 2, 3, 4, 5, 6], key=lambda n: n % 2)
        self.assertEqual(result, {0: [2, 4, 6], 1: [1, 3, 5]})

    def test_empty_input(self):
        self.assertEqual(m.group_all([], key=lambda x: x), {})


class TestAllPairs(unittest.TestCase):
    def test_cartesian_product(self):
        result = m.all_pairs([1, 2], ["x", "y"])
        self.assertEqual(result, [(1, "x"), (1, "y"), (2, "x"), (2, "y")])

    def test_empty_a_gives_empty_result(self):
        self.assertEqual(m.all_pairs([], ["x"]), [])


class TestRunningTotal(unittest.TestCase):
    def test_cumulative_sum(self):
        self.assertEqual(m.running_total([1, 2, 3, 4]), [1, 3, 6, 10])

    def test_empty_input(self):
        self.assertEqual(m.running_total([]), [])

    def test_single_element(self):
        self.assertEqual(m.running_total([5]), [5])


class TestProductOf(unittest.TestCase):
    def test_multiplies_all(self):
        self.assertEqual(m.product_of([1, 2, 3, 4]), 24)

    def test_empty_is_one(self):
        self.assertEqual(m.product_of([]), 1)

    def test_single_element(self):
        self.assertEqual(m.product_of([7]), 7)


class TestMakeMultiplier(unittest.TestCase):
    def test_multiplies_by_n(self):
        times_three = m.make_multiplier(3)
        self.assertEqual(times_three(4), 12)

    def test_returns_a_reusable_callable(self):
        times_two = m.make_multiplier(2)
        self.assertEqual([times_two(n) for n in [1, 2, 3]], [2, 4, 6])

    def test_independent_instances(self):
        times_two = m.make_multiplier(2)
        times_five = m.make_multiplier(5)
        self.assertEqual(times_two(10), 20)
        self.assertEqual(times_five(10), 50)


class TestDescribe(unittest.TestCase):
    def test_int(self):
        self.assertEqual(m.describe(42), "the integer 42")

    def test_str(self):
        self.assertEqual(m.describe("hi"), "a string of length 2")

    def test_list(self):
        self.assertEqual(m.describe([1, 2, 3]), "a list of 3 items")

    def test_unregistered_type_uses_default(self):
        self.assertEqual(m.describe(3.14), "a float")


class TestRetry(unittest.TestCase):
    def test_succeeds_after_transient_failures(self):
        state = {"calls": 0}

        @m.retry(3)
        def flaky():
            state["calls"] += 1
            if state["calls"] < 3:
                raise ValueError("not yet")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(state["calls"], 3)

    def test_gives_up_after_exhausting_attempts(self):
        state = {"calls": 0}

        @m.retry(2)
        def always_fails():
            state["calls"] += 1
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            always_fails()
        self.assertEqual(state["calls"], 2)

    def test_preserves_function_name(self):
        @m.retry(1)
        def named_function():
            return 1

        self.assertEqual(named_function.__name__, "named_function")

    def test_succeeds_first_try_calls_once(self):
        state = {"calls": 0}

        @m.retry(5)
        def works():
            state["calls"] += 1
            return "ok"

        self.assertEqual(works(), "ok")
        self.assertEqual(state["calls"], 1)


class TestCallCounter(unittest.TestCase):
    def test_counts_calls(self):
        @m.call_counter
        def double(x):
            return x * 2

        self.assertEqual(double(1), 2)
        self.assertEqual(double(2), 4)
        self.assertEqual(double.calls, 2)

    def test_starts_at_zero(self):
        @m.call_counter
        def noop():
            return None

        self.assertEqual(noop.calls, 0)

    def test_independent_across_functions(self):
        @m.call_counter
        def f():
            return None

        @m.call_counter
        def g():
            return None

        f()
        f()
        g()
        self.assertEqual(f.calls, 2)
        self.assertEqual(g.calls, 1)


if __name__ == "__main__":
    unittest.main()
