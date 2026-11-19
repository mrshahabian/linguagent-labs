"""Reference answers for 14 - performance and memory."""

import bisect
import functools
from collections import Counter, deque


def common_elements(a, b):
    """Elements of `a` that also appear in `b`, in a's order, deduplicated.

    The point: converting b to a set turns an O(n*m) scan into O(n+m).
    `x in list` is a linear scan; `x in set` is a hash lookup.
    """
    lookup = set(b)
    seen = set()
    out = []
    for item in a:
        if item in lookup and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def join_words(words, separator=" "):
    """Join words without quadratic string concatenation.

    `result += word` in a loop rebuilds the whole string each time, because
    str is immutable - O(n^2) total. str.join sizes the buffer once.
    """
    return separator.join(words)


@functools.lru_cache(maxsize=None)
def collatz_length(n):
    """Number of steps for n to reach 1 under the Collatz map.

    Memoised: the recursion revisits the same values constantly, so caching
    turns an exponential-looking workload into a linear one.
    """
    if n == 1:
        return 1
    if n % 2 == 0:
        return 1 + collatz_length(n // 2)
    return 1 + collatz_length(3 * n + 1)


class Point:
    """A class with __slots__: no per-instance __dict__.

    Saves a few hundred bytes per instance and speeds attribute access.
    The trade-off is that you cannot add new attributes at runtime, and
    multiple inheritance gets fiddly.
    """

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def running_max(values, window):
    """Maximum of each sliding window of `window` values.

    A deque of *indices* kept in decreasing value order gives O(n) total:
    each index is pushed and popped at most once. The naive max() per window
    is O(n * window).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    candidates = deque()
    for index, value in enumerate(values):
        # Drop indices that have fallen out of the window on the left.
        while candidates and candidates[0] <= index - window:
            candidates.popleft()
        # Anything smaller than the incoming value can never be a max again.
        while candidates and values[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)
        if index >= window - 1:
            out.append(values[candidates[0]])
    return out


def insert_sorted(sorted_list, value):
    """Insert into an already-sorted list, keeping it sorted.

    bisect.insort finds the position in O(log n) instead of re-sorting the
    whole list in O(n log n) after every append.
    """
    bisect.insort(sorted_list, value)
    return sorted_list


def rank_by_frequency(items, limit=None):
    """(item, count) pairs, most frequent first, ties alphabetically.

    Counter.most_common does not define tie order, so sort explicitly when
    the tie-break matters.
    """
    counts = Counter(items)
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:limit] if limit is not None else ranked


def total_size(rows):
    """Sum the "size" field of every row, without building an intermediate list.

    A generator expression inside sum() streams; a list comprehension
    materialises every value first. Same result, constant memory instead of
    O(n).
    """
    return sum(row["size"] for row in rows)


def first_match(rows, predicate):
    """The first row satisfying predicate, or None.

    next() over a generator stops at the first hit. Filtering into a list
    and taking [0] evaluates the predicate for every row.
    """
    return next((row for row in rows if predicate(row)), None)
