"""Tests for 10 - the data model. Read them; they are the specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestVec(unittest.TestCase):
    def test_repr_is_unambiguous(self):
        self.assertEqual(repr(m.Vec(1, 2)), "Vec(1, 2)")

    def test_repr_is_used_inside_containers(self):
        self.assertEqual(repr([m.Vec(0, 0)]), "[Vec(0, 0)]")

    def test_arithmetic(self):
        self.assertEqual(m.Vec(1, 2) + m.Vec(3, 4), m.Vec(4, 6))
        self.assertEqual(m.Vec(5, 5) - m.Vec(1, 2), m.Vec(4, 3))
        self.assertEqual(m.Vec(1, 2) * 3, m.Vec(3, 6))

    def test_reflected_multiplication(self):
        """3 * v must work, not just v * 3."""
        self.assertEqual(3 * m.Vec(1, 2), m.Vec(3, 6))

    def test_abs_is_the_magnitude(self):
        self.assertAlmostEqual(abs(m.Vec(3, 4)), 5.0)

    def test_hashable_and_usable_in_a_set(self):
        self.assertEqual(len({m.Vec(1, 2), m.Vec(1, 2), m.Vec(3, 4)}), 2)

    def test_unpacks(self):
        x, y = m.Vec(7, 8)
        self.assertEqual((x, y), (7, 8))

    def test_comparison_with_a_foreign_type_raises_cleanly(self):
        """__eq__ returning NotImplemented for a type it doesn't recognise
        falls back to "not equal" - no exception."""
        self.assertNotEqual(m.Vec(1, 2), "not a vector")

    def test_arithmetic_with_a_foreign_type_raises_cleanly(self):
        """__add__ returning NotImplemented lets Python raise TypeError
        itself, once neither side knows how to add."""
        with self.assertRaises(TypeError):
            m.Vec(1, 2) + 5


class TestBag(unittest.TestCase):
    def test_len_counts_every_item(self):
        self.assertEqual(len(m.Bag("aab")), 3)

    def test_empty_bag_is_falsy(self):
        """Truthiness falls back to __len__ when __bool__ is absent."""
        self.assertFalse(m.Bag())
        self.assertTrue(m.Bag("a"))

    def test_contains(self):
        bag = m.Bag("aab")
        self.assertIn("a", bag)
        self.assertNotIn("z", bag)

    def test_contains_does_not_iterate(self):
        """__contains__ must answer directly, not scan every item."""
        bag = m.Bag("aab")
        original = type(bag).__iter__
        try:
            type(bag).__iter__ = lambda self: (_ for _ in ()).throw(
                AssertionError("`in` fell back to iteration"))
            self.assertIn("a", bag)
        finally:
            type(bag).__iter__ = original

    def test_iterates_with_multiplicity(self):
        self.assertEqual(sorted(m.Bag("aab")), ["a", "a", "b"])

    def test_add(self):
        bag = m.Bag()
        bag.add("x")
        bag.add("x")
        self.assertEqual(len(bag), 2)


class TestVersion(unittest.TestCase):
    def test_sorts_numerically_not_lexicographically(self):
        got = sorted(m.Version(v) for v in ["1.10.0", "1.2.0", "1.9.9"])
        self.assertEqual([v.text for v in got], ["1.2.0", "1.9.9", "1.10.0"])

    def test_total_ordering_fills_in_the_rest(self):
        a, b = m.Version("1.0"), m.Version("2.0")
        self.assertTrue(a < b)
        self.assertTrue(b > a)
        self.assertTrue(a <= b)
        self.assertTrue(b >= a)

    def test_missing_components_are_zero(self):
        self.assertEqual(m.Version("1.0"), m.Version("1.0.0"))
        self.assertTrue(m.Version("1.0") < m.Version("1.0.1"))

    def test_equal_versions_hash_equal(self):
        """The contract: a == b implies hash(a) == hash(b). Exercises
        __hash__ alone - __eq__ can still be a stub when this passes."""
        self.assertEqual(hash(m.Version("1.0")), hash(m.Version("1.0.0")))

    def test_equal_versions_dedupe_in_a_set(self):
        """A set only collapses two entries if they are both equal and hash
        equal, so this needs __eq__ working too, not just __hash__."""
        self.assertEqual(len({m.Version("1.0"), m.Version("1.0.0")}), 1)


class TestRateLimiter(unittest.TestCase):
    def test_instances_are_callable(self):
        limiter = m.RateLimiter(limit=2)
        self.assertTrue(limiter("a"))
        self.assertTrue(limiter("a"))
        self.assertFalse(limiter("a"))

    def test_keys_are_counted_separately(self):
        limiter = m.RateLimiter(limit=1)
        self.assertTrue(limiter("a"))
        self.assertTrue(limiter("b"))
        self.assertFalse(limiter("a"))


class TestCollecting(unittest.TestCase):
    def test_binds_the_list_and_keeps_it(self):
        with m.collecting() as items:
            items.append(1)
            items.append(2)
        self.assertEqual(items, [1, 2])

    def test_exit_runs_even_when_the_body_raises(self):
        manager = m.collecting(suppress=ValueError)
        with manager as items:
            items.append(1)
            raise ValueError("handled")
        self.assertEqual(manager.items, [1])

    def test_only_the_named_exception_is_suppressed(self):
        with self.assertRaises(KeyError):
            with m.collecting(suppress=ValueError):
                raise KeyError("not suppressed")

    def test_nothing_is_suppressed_by_default(self):
        with self.assertRaises(ValueError):
            with m.collecting():
                raise ValueError("propagates")


class TestDescriptor(unittest.TestCase):
    def test_valid_values_round_trip(self):
        config = m.Config(width=10, height=20)
        self.assertEqual((config.width, config.height), (10, 20))

    def test_rejects_non_positive_on_construction(self):
        with self.assertRaises(ValueError):
            m.Config(width=0, height=5)

    def test_rejects_non_positive_on_assignment(self):
        config = m.Config(width=1, height=1)
        with self.assertRaises(ValueError):
            config.height = -3
        self.assertEqual(config.height, 1)

    def test_error_names_the_attribute(self):
        """__set_name__ is how the descriptor learns what it was assigned to."""
        with self.assertRaises(ValueError) as caught:
            m.Config(width=5, height=-1)
        self.assertIn("height", str(caught.exception))

    def test_instances_do_not_share_state(self):
        a, b = m.Config(1, 1), m.Config(2, 2)
        a.width = 99
        self.assertEqual(b.width, 2)


class TestLazy(unittest.TestCase):
    def test_builds_on_first_access(self):
        lazy = m.Lazy(value=lambda: 42)
        self.assertEqual(lazy.build_count, 0)
        self.assertEqual(lazy.value, 42)
        self.assertEqual(lazy.build_count, 1)

    def test_caches_after_the_first_access(self):
        """__getattr__ is only called when normal lookup fails."""
        lazy = m.Lazy(value=lambda: 42)
        lazy.value, lazy.value, lazy.value
        self.assertEqual(lazy.build_count, 1)

    def test_unknown_attribute_raises_attribute_error(self):
        lazy = m.Lazy(value=lambda: 1)
        with self.assertRaises(AttributeError):
            lazy.missing


if __name__ == "__main__":
    unittest.main(verbosity=2)
