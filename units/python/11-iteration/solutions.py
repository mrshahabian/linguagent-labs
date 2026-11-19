"""Reference answers for 11 - iteration, closures and decorators."""

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
        current = self.start
        while current > 0:
            yield current
            current -= 1


def sliding(sequence, size):
    """Yield overlapping windows of `size` as tuples.

    sliding([1,2,3,4], 2) -> (1,2), (2,3), (3,4)
    Nothing is yielded when the sequence is shorter than the window.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(len(sequence) - size + 1):
        yield tuple(sequence[start:start + size])


def chunked(iterable, size):
    """Yield consecutive lists of at most `size` items.

    Works on any iterable, including infinite ones - so it must not call
    len() or materialise the input.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def flatten(nested):
    """Yield every non-iterable leaf of arbitrarily nested lists/tuples.

    Strings are leaves, not containers - otherwise "ab" recurses forever
    because "a"[0] is "a".
    """
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def unique(iterable):
    """Yield items the first time they are seen, preserving order.

    Lazy: consuming two items must not consume the whole input.
    """
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item


def make_counters(count):
    """Return `count` functions; the i-th returns i when called.

    The classic trap:

        [lambda: i for i in range(3)]   # every one returns 2

    A closure captures the *variable*, not its value at creation time. Bind
    the value with a default argument (or a factory function).
    """
    return [lambda i=index: i for index in range(count)]


def retry(times, exceptions=(Exception,)):
    """Decorator factory: re-call the function until it stops raising.

    Re-raises the last exception once the attempts are used up. Keeps the
    wrapped function's name and docstring via functools.wraps.
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last = exc
            raise last
        wrapper.attempts = times
        return wrapper
    return decorate


def counted(func):
    """Decorator recording how many times the function was called.

    The count lives on the wrapper as `.calls`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


@contextmanager
def atomic(mapping):
    """Apply changes to `mapping` only if the block completes.

    On the way in, take a shallow copy. On the way out, either keep the
    mutations or restore the snapshot. The exception still propagates.
    """
    snapshot = dict(mapping)
    try:
        yield mapping
    except Exception:
        mapping.clear()
        mapping.update(snapshot)
        raise
