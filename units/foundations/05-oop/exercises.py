"""05 - basic OOP

Implement each function so the tests in test_exercises.py pass. Read
README.md first; every exercise below is covered there.

    ../run 05            test your work
    ../run 05 --check    watch the reference solutions pass

The docstrings are the specification. Standard library only.
"""


class Car:
    """A car with a make and model."""

    def __init__(self, make, model):
        self.make = make
        self.model = model

    def describe(self):
        """Return "{make} {model}"."""
        raise NotImplementedError("your code here")


class BankAccount:
    """A bank account that tracks a balance."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        """Add amount to the balance."""
        raise NotImplementedError("your code here")

    def withdraw(self, amount):
        """Subtract amount from the balance. Raise ValueError if amount
        is more than the current balance - never go negative."""
        raise NotImplementedError("your code here")


class Employee:
    """Every Employee works at the same company - that's a class
    attribute, shared by every instance unless one overrides it."""

    company = "Acme"

    def __init__(self, name):
        self.name = name

    def intro(self):
        """Return "{name} works at {company}"."""
        raise NotImplementedError("your code here")


class Animal:
    """The base class every animal shares an __init__ with."""

    def __init__(self, name):
        self.name = name

    def speak(self):
        """The generic animal sound - subclasses override this."""
        raise NotImplementedError("your code here")


class Dog(Animal):
    """Inherits __init__ from Animal - only speak() is different."""

    def speak(self):
        raise NotImplementedError("your code here")


class Cat(Animal):
    """Inherits __init__ from Animal - only speak() is different."""

    def speak(self):
        raise NotImplementedError("your code here")
