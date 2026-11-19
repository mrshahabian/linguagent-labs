"""07 - tooling and complexity basics

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 07            test your work
    ../run 07 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""


def linear_search(items, target):
    """Return the index of the first occurrence of target in items, or -1
    if it isn't present. Checks every item, in order - O(n)."""
    raise NotImplementedError("your code here")


def binary_search(sorted_items, target):
    """Return the index of target in sorted_items (which must already be
    sorted ascending), or -1 if it isn't present. Halves the search space
    each step - O(log n)."""
    raise NotImplementedError("your code here")


def has_duplicate_pair(numbers):
    """True if any value appears more than once, checking every pair -
    deliberately the O(n^2) version. Compare this to what you already
    know from Module 03: len(numbers) != len(set(numbers)) answers the
    same question in O(n)."""
    raise NotImplementedError("your code here")
