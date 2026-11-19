"""Reference answers for 02 - functions and scope."""


def greet(name, greeting="Hello"):
    """Return "{greeting}, {name}!" - greeting defaults to "Hello"."""
    return f"{greeting}, {name}!"


def add_all(*numbers):
    """Sum any number of positional arguments, e.g. add_all(1, 2, 3) == 6.
    add_all() with no arguments returns 0."""
    total = 0
    for n in numbers:
        total += n
    return total


def build_profile(name, **details):
    """Return a dict with "name" plus every keyword argument given,
    e.g. build_profile("Al", age=30) == {"name": "Al", "age": 30}."""
    profile = {"name": name}
    profile.update(details)
    return profile


def total_price(prices, tax_rate=0.0):
    """Sum prices, then apply tax: total * (1 + tax_rate)."""
    total = 0
    for price in prices:
        total += price
    return total * (1 + tax_rate)


def factorial(n):
    """n! = n * (n-1) * ... * 1, and 0! = 1. Implemented recursively."""
    if n == 0:
        return 1
    return n * factorial(n - 1)


def sum_digits(n):
    """Sum the digits of a non-negative integer, recursively:
    sum_digits(123) == 1 + 2 + 3 == 6."""
    if n < 10:
        return n
    return n % 10 + sum_digits(n // 10)
