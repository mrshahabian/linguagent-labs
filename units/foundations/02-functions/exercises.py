"""02 - functions and scope

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 02            test your work
    ../run 02 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""


def greet(name, greeting="Hello"):
    """Return "{greeting}, {name}!" - greeting defaults to "Hello"."""
    raise NotImplementedError("your code here")


def add_all(*numbers):
    """Sum any number of positional arguments, e.g. add_all(1, 2, 3) == 6.
    add_all() with no arguments returns 0."""
    raise NotImplementedError("your code here")


def build_profile(name, **details):
    """Return a dict with "name" plus every keyword argument given,
    e.g. build_profile("Al", age=30) == {"name": "Al", "age": 30}."""
    raise NotImplementedError("your code here")


def total_price(prices, tax_rate=0.0):
    """Sum prices, then apply tax: total * (1 + tax_rate)."""
    raise NotImplementedError("your code here")


def factorial(n):
    """n! = n * (n-1) * ... * 1, and 0! = 1. Implemented recursively."""
    raise NotImplementedError("your code here")


def sum_digits(n):
    """Sum the digits of a non-negative integer, recursively:
    sum_digits(123) == 1 + 2 + 3 == 6."""
    raise NotImplementedError("your code here")
