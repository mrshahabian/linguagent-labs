"""10 - the data model

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 10            test your work
    ../run 10 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import functools


class Vec:
    """A 2D vector supporting +, -, scalar *, ==, abs() and unpacking."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        raise NotImplementedError("your code here")

    def __eq__(self, other):
        raise NotImplementedError("your code here")

    def __hash__(self):
        raise NotImplementedError("your code here")

    def __add__(self, other):
        raise NotImplementedError("your code here")

    def __sub__(self, other):
        raise NotImplementedError("your code here")

    def __mul__(self, scalar):
        raise NotImplementedError("your code here")

    __rmul__ = __mul__

    def __abs__(self):
        raise NotImplementedError("your code here")

    def __iter__(self):
        raise NotImplementedError("your code here")


class Bag:
    """A multiset. Empty bags are falsy; `in` is O(1)."""

    def __init__(self, items=()):
        self._counts = {}
        for item in items:
            self._counts[item] = self._counts.get(item, 0) + 1

    def add(self, item):
        raise NotImplementedError("your code here")

    def __len__(self):
        # Total items, not distinct ones - so truthiness follows from it.
        raise NotImplementedError("your code here")

    def __contains__(self, item):
        # Without this, `in` would fall back to iterating: O(n) instead of O(1).
        raise NotImplementedError("your code here")

    def __iter__(self):
        raise NotImplementedError("your code here")

    def __repr__(self):
        raise NotImplementedError("your code here")


@functools.total_ordering
class Version:
    """A dotted version that sorts numerically, not lexicographically."""

    def __init__(self, text):
        self.text = text
        self.parts = tuple(int(part) for part in text.split("."))

    def __repr__(self):
        raise NotImplementedError("your code here")

    def _padded(self, other):
        raise NotImplementedError("your code here")

    def __eq__(self, other):
        raise NotImplementedError("your code here")

    def __lt__(self, other):
        raise NotImplementedError("your code here")

    def __hash__(self):
        # 1.0 and 1.0.0 compare equal, so they must hash equal: strip
        # trailing zeros before hashing.
        raise NotImplementedError("your code here")


class RateLimiter:
    """A callable object: `limiter(name)` is True while under the limit."""

    def __init__(self, limit):
        self.limit = limit
        self.seen = {}

    def __call__(self, key):
        raise NotImplementedError("your code here")


class collecting:
    """Context manager that gathers appended values and optionally swallows
    one exception type."""

    def __init__(self, suppress=()):
        self.suppress = suppress
        self.items = []

    def __enter__(self):
        # Whatever this returns is what `as` binds.
        raise NotImplementedError("your code here")

    def __exit__(self, exc_type, exc, tb):
        # A truthy return value suppresses the exception. Returning None or
        # False lets it propagate.
        raise NotImplementedError("your code here")


class Positive:
    """A data descriptor validating that the value stays positive."""

    def __set_name__(self, owner, name):
        # Called at class creation with the attribute name it was bound to.
        # your code here. This one runs at class-creation
        # time, so it must not raise.
        pass

    def __get__(self, obj, objtype=None):
        raise NotImplementedError("your code here")

    def __set__(self, obj, value):
        raise NotImplementedError("your code here")


class Config:

    width = Positive()
    height = Positive()

    def __init__(self, width, height):
        self.width = width
        self.height = height


class Lazy:
    """Builds each attribute on first access, then caches it.

    __getattr__ runs only when normal lookup fails, so once the value is in
    __dict__ it is never called again.
    """

    def __init__(self, **builders):
        # Bypass __setattr__ concerns by writing straight to __dict__.
        self.__dict__["_builders"] = builders
        self.__dict__["build_count"] = 0

    def __getattr__(self, name):
        raise NotImplementedError("your code here")
