"""Tests for 16 - typing, dataclasses and protocols."""

import dataclasses
import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestSafeDivide(unittest.TestCase):
    def test_normal_division(self):
        self.assertEqual(m.safe_divide(6, 3), 2.0)

    def test_zero_denominator_returns_none(self):
        self.assertIsNone(m.safe_divide(1, 0))

    def test_does_not_raise(self):
        try:
            m.safe_divide(5, 0)
        except ZeroDivisionError:
            self.fail("safe_divide must catch ZeroDivisionError, not raise it")


class TestStack(unittest.TestCase):
    def test_push_pop_is_lifo(self):
        stack = m.Stack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 2)

    def test_peek_does_not_remove(self):
        stack = m.Stack()
        stack.push("x")
        self.assertEqual(stack.peek(), "x")
        self.assertEqual(len(stack), 1)

    def test_len_tracks_size(self):
        stack = m.Stack()
        self.assertEqual(len(stack), 0)
        stack.push(1)
        stack.push(2)
        self.assertEqual(len(stack), 2)

    def test_is_empty(self):
        stack = m.Stack()
        self.assertTrue(stack.is_empty())
        stack.push(1)
        self.assertFalse(stack.is_empty())

    def test_pop_empty_raises(self):
        stack = m.Stack()
        with self.assertRaises(IndexError):
            stack.pop()

    def test_peek_empty_raises(self):
        stack = m.Stack()
        with self.assertRaises(IndexError):
            stack.peek()

    def test_two_instances_do_not_share_state(self):
        a, b = m.Stack(), m.Stack()
        a.push(1)
        self.assertTrue(b.is_empty())


class TestFirstMatch(unittest.TestCase):
    def test_finds_first_even(self):
        self.assertEqual(m.first_match([1, 3, 4, 6], lambda x: x % 2 == 0), 4)

    def test_no_match_returns_none(self):
        self.assertIsNone(m.first_match([1, 3, 5], lambda x: x % 2 == 0))

    def test_empty_iterable_returns_none(self):
        self.assertIsNone(m.first_match([], lambda x: True))


class TestPoint(unittest.TestCase):
    def test_distance_to(self):
        p1, p2 = m.Point(0, 0), m.Point(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_frozen_prevents_mutation(self):
        p = m.Point(1, 2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.x = 99

    def test_equality_is_by_value(self):
        self.assertEqual(m.Point(1, 2), m.Point(1, 2))
        self.assertNotEqual(m.Point(1, 2), m.Point(2, 1))

    def test_is_hashable(self):
        seen = {m.Point(0, 0), m.Point(0, 0), m.Point(1, 1)}
        self.assertEqual(len(seen), 2)


class TestSemVer(unittest.TestCase):
    def test_prerelease_excluded_from_equality(self):
        self.assertEqual(m.SemVer(1, 2, 3, "rc1"), m.SemVer(1, 2, 3, "rc2"))

    def test_ordering_by_release_core(self):
        self.assertTrue(m.SemVer(1, 0, 0) < m.SemVer(1, 2, 0))
        self.assertTrue(m.SemVer(1, 2, 3) < m.SemVer(2, 0, 0))

    def test_sort_versions(self):
        versions = [m.SemVer(2, 0, 0), m.SemVer(1, 5, 0), m.SemVer(1, 0, 0)]
        result = m.sort_versions(versions)
        self.assertEqual([(v.major, v.minor, v.patch) for v in result],
                          [(1, 0, 0), (1, 5, 0), (2, 0, 0)])


class TestShoppingCart(unittest.TestCase):
    def test_default_starts_empty(self):
        cart = m.ShoppingCart()
        self.assertEqual(cart.total_items(), 0)

    def test_add_item(self):
        cart = m.ShoppingCart()
        cart.add_item("apple")
        cart.add_item("bread")
        self.assertEqual(cart.total_items(), 2)

    def test_instances_do_not_share_the_default_list(self):
        a, b = m.ShoppingCart(), m.ShoppingCart()
        a.add_item("apple")
        self.assertEqual(b.total_items(), 0)
        self.assertIsNot(a.items, b.items)


class TestPositiveInt(unittest.TestCase):
    def test_accepts_positive(self):
        self.assertEqual(m.PositiveInt(5).value, 5)

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            m.PositiveInt(0)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            m.PositiveInt(-3)


class TestSupportsAreaProtocol(unittest.TestCase):
    def test_circle_area(self):
        self.assertAlmostEqual(m.Circle(2).area(), 12.566370614359172)

    def test_rectangle_area(self):
        self.assertEqual(m.Rectangle(3, 4).area(), 12)

    def test_total_area_mixes_unrelated_classes(self):
        shapes = [m.Circle(1), m.Rectangle(2, 3)]
        expected = m.Circle(1).area() + m.Rectangle(2, 3).area()
        self.assertAlmostEqual(m.total_area(shapes), expected)

    def test_satisfies_protocol_structurally(self):
        self.assertTrue(m.satisfies_supports_area(m.Circle(1)))
        self.assertTrue(m.satisfies_supports_area(m.Rectangle(1, 1)))

    def test_does_not_satisfy_protocol(self):
        self.assertFalse(m.satisfies_supports_area(object()))
        self.assertFalse(m.satisfies_supports_area(42))


if __name__ == "__main__":
    unittest.main()
