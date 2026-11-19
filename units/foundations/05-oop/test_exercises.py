"""Tests for 05 - basic OOP. Read them; they are the specification."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class TestCar(unittest.TestCase):
    def test_describe(self):
        car = m.Car("Toyota", "Corolla")
        self.assertEqual(car.describe(), "Toyota Corolla")

    def test_instances_are_independent(self):
        a = m.Car("Toyota", "Corolla")
        b = m.Car("Honda", "Civic")
        self.assertEqual(a.describe(), "Toyota Corolla")
        self.assertEqual(b.describe(), "Honda Civic")


class TestBankAccount(unittest.TestCase):
    def test_starts_at_the_given_balance(self):
        acc = m.BankAccount("Ana", 100)
        self.assertEqual(acc.balance, 100)

    def test_default_balance_is_zero(self):
        acc = m.BankAccount("Ana")
        self.assertEqual(acc.balance, 0)

    def test_deposit_increases_balance(self):
        acc = m.BankAccount("Ana", 100)
        acc.deposit(50)
        self.assertEqual(acc.balance, 150)

    def test_withdraw_decreases_balance(self):
        acc = m.BankAccount("Ana", 100)
        acc.withdraw(30)
        self.assertEqual(acc.balance, 70)

    def test_withdraw_more_than_balance_raises_and_does_not_change_it(self):
        acc = m.BankAccount("Ana", 100)
        with self.assertRaises(ValueError):
            acc.withdraw(1000)
        self.assertEqual(acc.balance, 100)

    def test_withdrawing_exactly_the_balance_is_allowed(self):
        acc = m.BankAccount("Ana", 100)
        acc.withdraw(100)
        self.assertEqual(acc.balance, 0)


class TestEmployee(unittest.TestCase):
    def test_intro_uses_the_shared_company(self):
        emp = m.Employee("Al")
        self.assertEqual(emp.intro(), "Al works at Acme")

    def test_company_is_shared_across_instances(self):
        """company is a class attribute - every Employee sees the same
        value unless something overrides it on the instance."""
        a = m.Employee("Al")
        b = m.Employee("Bo")
        self.assertEqual(a.company, b.company)

    def test_name_is_not_shared(self):
        a = m.Employee("Al")
        b = m.Employee("Bo")
        self.assertNotEqual(a.name, b.name)


class TestAnimalDogCat(unittest.TestCase):
    def test_animal_speaks_generically(self):
        self.assertEqual(m.Animal("Generic").speak(), "...")

    def test_dog_overrides_speak(self):
        self.assertEqual(m.Dog("Rex").speak(), "Woof!")

    def test_cat_overrides_speak(self):
        self.assertEqual(m.Cat("Whiskers").speak(), "Meow!")

    def test_dog_inherits_init_from_animal(self):
        """Dog does not redefine __init__ - it uses Animal's, unchanged."""
        dog = m.Dog("Rex")
        self.assertEqual(dog.name, "Rex")

    def test_cat_inherits_init_from_animal(self):
        cat = m.Cat("Whiskers")
        self.assertEqual(cat.name, "Whiskers")

    def test_dog_is_an_animal(self):
        self.assertIsInstance(m.Dog("Rex"), m.Animal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
