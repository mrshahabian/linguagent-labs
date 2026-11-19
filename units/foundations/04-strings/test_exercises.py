"""Tests for 04 - strings and basic text processing. Read them; they are
the specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestShout(unittest.TestCase):
    def test_uppercases_and_appends(self):
        self.assertEqual(m.shout("hello"), "HELLO!!!")

    def test_already_uppercase(self):
        self.assertEqual(m.shout("HI"), "HI!!!")


class TestCountWords(unittest.TestCase):
    def test_several_words(self):
        self.assertEqual(m.count_words("the cat sat on the mat"), 6)

    def test_single_word(self):
        self.assertEqual(m.count_words("hello"), 1)

    def test_extra_whitespace_does_not_inflate_the_count(self):
        """split() with no arguments collapses runs of whitespace."""
        self.assertEqual(m.count_words("  the   cat  "), 2)


class TestReverseWords(unittest.TestCase):
    def test_three_words(self):
        self.assertEqual(m.reverse_words("the cat sat"), "sat cat the")

    def test_single_word_is_unchanged(self):
        self.assertEqual(m.reverse_words("hello"), "hello")

    def test_reverses_words_not_letters(self):
        """A common mistake: reversing the whole string reverses the
        letters too. "ab cd" reversed by word is "cd ab", not "dc ba"."""
        self.assertEqual(m.reverse_words("ab cd"), "cd ab")


class TestIsPalindrome(unittest.TestCase):
    def test_classic_palindrome_with_spaces_and_mixed_case(self):
        self.assertTrue(m.is_palindrome("Never Odd or Even"))

    def test_not_a_palindrome(self):
        self.assertFalse(m.is_palindrome("hello"))

    def test_single_character(self):
        self.assertTrue(m.is_palindrome("a"))

    def test_empty_string(self):
        self.assertTrue(m.is_palindrome(""))


class TestInitials(unittest.TestCase):
    def test_two_names(self):
        self.assertEqual(m.initials("Ada Lovelace"), "AL")

    def test_already_uppercase_input(self):
        self.assertEqual(m.initials("GRACE HOPPER"), "GH")

    def test_three_names(self):
        self.assertEqual(m.initials("Grace Brewster Hopper"), "GBH")


class TestSlugify(unittest.TestCase):
    def test_two_words(self):
        self.assertEqual(m.slugify("Hello World"), "hello-world")

    def test_already_lowercase(self):
        self.assertEqual(m.slugify("already lower"), "already-lower")

    def test_single_word(self):
        self.assertEqual(m.slugify("Title"), "title")


if __name__ == "__main__":
    unittest.main(verbosity=2)
