"""06 - errors, exceptions and your first tests

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 06            test your work
    ../run 06 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""


def safe_divide(a, b):
    """a / b, or None if b is 0 - catch the error instead of crashing."""
    raise NotImplementedError("your code here")


def parse_int(text):
    """Parse text as an int, or None if it isn't a valid integer."""
    raise NotImplementedError("your code here")


class NegativeAgeError(Exception):
    """Raised when an age is negative - a custom, specific exception type
    instead of a bare ValueError, so callers can catch exactly this."""

    pass


def validate_age(age):
    """Return age if it's >= 0, otherwise raise NegativeAgeError with a
    message that names the bad value."""
    raise NotImplementedError("your code here")


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the available balance."""

    pass


def withdraw_or_raise(balance, amount):
    """Return balance - amount, or raise InsufficientFundsError if amount
    is more than balance."""
    raise NotImplementedError("your code here")


def average(numbers):
    """The mean of numbers. Raise ValueError with a clear message for an
    empty list, instead of letting a confusing ZeroDivisionError escape."""
    raise NotImplementedError("your code here")
