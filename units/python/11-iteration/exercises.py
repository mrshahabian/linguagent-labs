"""11 - iteration, closures and decorators

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 11            test your work
    ../run 11 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""

import functools
import itertools
from contextlib import contextmanager


class Countdown:
    """An iterable that counts down from n to 1, restartable.

    __iter__ returns a *fresh* iterator each time, which is why the object
    can be looped over twice. A generator object cannot.
    """

    def __init__(self, start):
        self.start = start

    def __iter__(self):
        raise NotImplementedError("your code here")


def sliding(sequence, size):
    """Yield overlapping windows of `size` as tuples.

    sliding([1,2,3,4], 2) -> (1,2), (2,3), (3,4)
    Nothing is yielded when the sequence is shorter than the window.
    """
    raise NotImplementedError("your code here")


def chunked(iterable, size):
    """Yield consecutive lists of at most `size` items.

    Works on any iterable, including infinite ones - so it must not call
    len() or materialise the input.
    """
    raise NotImplementedError("your code here")


def flatten(nested):
    """Yield every non-iterable leaf of arbitrarily nested lists/tuples.

    Strings are leaves, not containers - otherwise "ab" recurses forever
    because "a"[0] is "a".
    """
    raise NotImplementedError("your code here")


def unique(iterable):
    """Yield items the first time they are seen, preserving order.

    Lazy: consuming two items must not consume the whole input.
    """
    raise NotImplementedError("your code here")


def make_counters(count):
    """Return `count` functions; the i-th returns i when called.

    The classic trap:

        [lambda: i for i in range(3)]   # every one returns 2

    A closure captures the *variable*, not its value at creation time. Bind
    the value with a default argument (or a factory function).
    """
    raise NotImplementedError("your code here")


def retry(times, exceptions=(Exception,)):
    """Decorator factory: re-call the function until it stops raising.

    Re-raises the last exception once the attempts are used up. Keeps the
    wrapped function's name and docstring via functools.wraps.
    """
    raise NotImplementedError("your code here")


def counted(func):
    """Decorator recording how many times the function was called.

    The count lives on the wrapper as `.calls`.
    """
    raise NotImplementedError("your code here")


@contextmanager
def atomic(mapping):
    """Apply changes to `mapping` only if the block completes.

    On the way in, take a shallow copy. On the way out, either keep the
    mutations or restore the snapshot. The exception still propagates.
    """
    raise NotImplementedError("your code here")
