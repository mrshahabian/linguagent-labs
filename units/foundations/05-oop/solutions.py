"""Reference answers for 05 - basic OOP."""


class Car:
    """A car with a make and model."""

    def __init__(self, make, model):
        self.make = make
        self.model = model

    def describe(self):
        """Return "{make} {model}"."""
        return f"{self.make} {self.model}"


class BankAccount:
    """A bank account that tracks a balance."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        """Add amount to the balance."""
        self.balance += amount

    def withdraw(self, amount):
        """Subtract amount from the balance. Raise ValueError if amount
        is more than the current balance - never go negative."""
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount


class Employee:
    """Every Employee works at the same company - that's a class
    attribute, shared by every instance unless one overrides it."""

    company = "Acme"

    def __init__(self, name):
        self.name = name

    def intro(self):
        """Return "{name} works at {company}"."""
        return f"{self.name} works at {self.company}"


class Animal:
    """The base class every animal shares an __init__ with."""

    def __init__(self, name):
        self.name = name

    def speak(self):
        """The generic animal sound - subclasses override this."""
        return "..."


class Dog(Animal):
    """Inherits __init__ from Animal - only speak() is different."""

    def speak(self):
        return "Woof!"


class Cat(Animal):
    """Inherits __init__ from Animal - only speak() is different."""

    def speak(self):
        return "Meow!"
