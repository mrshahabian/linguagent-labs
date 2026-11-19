"""Reference answers for 19 - testing and debugging patterns."""

import logging
import time

logger = logging.getLogger(__name__)


def is_prime(n):
    """True if n is prime.

    The function itself is deliberately simple - the point of this
    exercise is test_exercises.py's use of subTest to run it against a
    whole table of (input, expected) cases as one parametrized test,
    rather than one test method per case.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def fetch_first_success(client, ids):
    """Try client.get(id) for each id in order; return the first success.

    Ids whose call raises are skipped, not fatal - one bad id should not
    stop the search. Taking `client` as a parameter (dependency
    injection) rather than importing a concrete HTTP client at module
    level is what makes this testable at all: a test passes a
    unittest.mock.Mock instead of hitting a real network.
    """
    for item_id in ids:
        try:
            result = client.get(item_id)
        except Exception:
            continue
        if result is not None:
            return result
    return None


class TTLCache:
    """A cache where entries expire after a time-to-live, in seconds.

    now_fn defaults to time.time but is injectable - exactly like
    `client` above. Time-based logic is notoriously hard to test
    honestly (a test that calls time.sleep(61) to prove an entry expired
    after 60s is slow and still flaky under load); injecting the clock
    lets a test substitute a fake one it controls exactly.
    """

    def __init__(self, now_fn=time.time):
        self._now = now_fn
        self._store = {}

    def set(self, key, value, ttl):
        self._store[key] = (value, self._now() + ttl)

    def get(self, key):
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if self._now() > expires_at:
            del self._store[key]
            return None
        return value


def divide_safely(a, b):
    """Divide a by b; log a warning and return None instead of raising on b == 0.

    logger.warning, not print - a caller (or test) can control or
    capture logging output without the function needing to know how its
    caller wants messages handled. unittest's assertLogs is the
    standard way to assert a specific message was logged at a specific
    level, without capturing stdout.
    """
    if b == 0:
        logger.warning("division by zero: %r / %r", a, b)
        return None
    return a / b


def find_first_failure(n, predicate):
    """Binary-search range(n) for the first index where predicate(i) is True.

    predicate is assumed MONOTONIC: False for a while, then True from
    some point on (never back to False) - exactly the assumption behind
    `git bisect`. Instead of checking every index in order (O(n), and
    each check might be an expensive test run or a slow query),
    bisection finds the boundary in O(log n) checks. Returns n if
    predicate is False everywhere.
    """
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


class MatrixShapeError(ValueError):
    """Raised when two matrices' shapes are incompatible for multiplication."""


def matmul_checked(a, b):
    """Multiply two matrices (lists of lists); raise MatrixShapeError on mismatch.

    A generic "shapes don't match" or an IndexError from the multiply
    loop itself both fail without saying *what* was wrong. Checking the
    shapes up front and naming both of them in the message is the
    difference between a five-second fix and a debugging session - the
    same principle as this repo's own test failure messages.
    """
    rows_a, cols_a = len(a), len(a[0]) if a else 0
    rows_b, cols_b = len(b), len(b[0]) if b else 0
    if cols_a != rows_b:
        raise MatrixShapeError(
            f"cannot multiply a {rows_a}x{cols_a} matrix by a {rows_b}x{cols_b} "
            f"matrix: {cols_a} (columns of a) != {rows_b} (rows of b)"
        )
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]
