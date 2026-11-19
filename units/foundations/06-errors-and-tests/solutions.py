"""Reference answers for 06 - errors, exceptions and your first tests."""


def safe_divide(a, b):
    """a / b, or None if b is 0 - catch the error instead of crashing."""
    try:
        return a / b
    except ZeroDivisionError:
        return None


def parse_int(text):
    """Parse text as an int, or None if it isn't a valid integer."""
    try:
        return int(text)
    except ValueError:
        return None


class NegativeAgeError(Exception):
    """Raised when an age is negative - a custom, specific exception type
    instead of a bare ValueError, so callers can catch exactly this."""


def validate_age(age):
    """Return age if it's >= 0, otherwise raise NegativeAgeError with a
    message that names the bad value."""
    if age < 0:
        raise NegativeAgeError(f"age cannot be negative, got {age}")
    return age


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the available balance."""


def withdraw_or_raise(balance, amount):
    """Return balance - amount, or raise InsufficientFundsError if amount
    is more than balance."""
    if amount > balance:
        raise InsufficientFundsError(
            f"cannot withdraw {amount}, balance is only {balance}"
        )
    return balance - amount


def average(numbers):
    """The mean of numbers. Raise ValueError with a clear message for an
    empty list, instead of letting a confusing ZeroDivisionError escape."""
    if not numbers:
        raise ValueError("average() of an empty list is undefined")
    return sum(numbers) / len(numbers)
