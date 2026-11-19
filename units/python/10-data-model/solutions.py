"""Reference answers for 10 - the data model."""

import functools


# ----------------------------------------------------------------- 1. Vec

class Vec:
    """A 2D vector supporting +, -, scalar *, ==, abs() and unpacking."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vec({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vec(self.x * scalar, self.y * scalar)

    # 3 * v as well as v * 3
    __rmul__ = __mul__

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __iter__(self):
        yield self.x
        yield self.y


# --------------------------------------------------------------- 2. Bag

class Bag:
    """A multiset. Empty bags are falsy; `in` is O(1)."""

    def __init__(self, items=()):
        self._counts = {}
        for item in items:
            self._counts[item] = self._counts.get(item, 0) + 1

    def add(self, item):
        self._counts[item] = self._counts.get(item, 0) + 1

    def __len__(self):
        # Total items, not distinct ones - so truthiness follows from it.
        return sum(self._counts.values())

    def __contains__(self, item):
        # Without this, `in` would fall back to iterating: O(n) instead of O(1).
        return item in self._counts

    def __iter__(self):
        for item, count in self._counts.items():
            for _ in range(count):
                yield item

    def __repr__(self):
        return f"Bag({sorted(self)!r})"


# ----------------------------------------------------------- 3. Version

@functools.total_ordering
class Version:
    """A dotted version that sorts numerically, not lexicographically."""

    def __init__(self, text):
        self.text = text
        self.parts = tuple(int(part) for part in text.split("."))

    def __repr__(self):
        return f"Version({self.text!r})"

    def _padded(self, other):
        width = max(len(self.parts), len(other.parts))
        pad = lambda parts: parts + (0,) * (width - len(parts))
        return pad(self.parts), pad(other.parts)

    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        mine, theirs = self._padded(other)
        return mine == theirs

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        mine, theirs = self._padded(other)
        return mine < theirs

    def __hash__(self):
        # 1.0 and 1.0.0 compare equal, so they must hash equal: strip
        # trailing zeros before hashing.
        parts = list(self.parts)
        while parts and parts[-1] == 0:
            parts.pop()
        return hash(tuple(parts))


# --------------------------------------------------------- 4. RateLimiter

class RateLimiter:
    """A callable object: `limiter(name)` is True while under the limit."""

    def __init__(self, limit):
        self.limit = limit
        self.seen = {}

    def __call__(self, key):
        count = self.seen.get(key, 0)
        if count >= self.limit:
            return False
        self.seen[key] = count + 1
        return True


# ------------------------------------------------------------ 5. collecting

class collecting:
    """Context manager that gathers appended values and optionally swallows
    one exception type."""

    def __init__(self, suppress=()):
        self.suppress = suppress
        self.items = []

    def __enter__(self):
        # Whatever this returns is what `as` binds.
        return self.items

    def __exit__(self, exc_type, exc, tb):
        # A truthy return value suppresses the exception. Returning None or
        # False lets it propagate.
        return exc_type is not None and issubclass(exc_type, self.suppress)


# ------------------------------------------------------------ 6. Positive

class Positive:
    """A data descriptor validating that the value stays positive."""

    def __set_name__(self, owner, name):
        # Called at class creation with the attribute name it was bound to.
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive, got {value}")
        obj.__dict__[self.name] = value


class Config:
    width = Positive()
    height = Positive()

    def __init__(self, width, height):
        self.width = width
        self.height = height


# --------------------------------------------------------------- 7. Lazy

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
        builders = self.__dict__["_builders"]
        if name not in builders:
            raise AttributeError(name)
        value = builders[name]()
        self.__dict__["build_count"] += 1
        self.__dict__[name] = value
        return value
