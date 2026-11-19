"""Reference solution for the ICA pre-certification Container task.

Copy the class body into container.py once you have made your own attempt.

The design in one line: **keep the list sorted at all times**, so the median is
an index lookup instead of a search.

    add         bisect.insort   O(log n) search + O(n) memmove
    delete      bisect_left     O(log n) search + O(n) memmove
    get_median  index lookup    O(1)

See docs/02-median-structures.md for why this beats both the naive
sort-on-every-query approach and the two-heap approach *for this task*.
"""

import bisect


class Container:
    """A container of integers supporting add, delete and median."""

    def __init__(self):
        # Invariant: always sorted ascending. Every method below either
        # preserves it or relies on it.
        self._values = []

    def add(self, value: int) -> None:
        """Insert `value`, keeping the list sorted."""
        bisect.insort(self._values, value)

    def delete(self, value: int) -> bool:
        """Remove one occurrence of `value`. True if it was there."""
        index = bisect.bisect_left(self._values, value)
        if index < len(self._values) and self._values[index] == value:
            self._values.pop(index)
            return True
        return False

    def get_median(self) -> int:
        """The middle value; for an even count, the leftmost of the two.

        Raises on an empty container, as the spec requires.
        """
        if not self._values:
            raise IndexError("get_median() on an empty container")
        return self._values[(len(self._values) - 1) // 2]
