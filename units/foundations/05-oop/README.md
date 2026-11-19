# 05 — Basic OOP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/05-oop/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/oop?utm_source=github&utm_medium=labs)

*~40 minutes · 17 tests · Junior*

A class is a blueprint for building objects that bundle data (attributes)
with the behaviour that acts on it (methods). Every function you've written
so far has taken its data as an argument and handed back a result with no
memory of it; a class lets an object **carry its own state** between
method calls. This is deliberately the gentle version of that idea — the
senior-tier Python 1 module builds on this to cover the dunder protocols
(`__eq__`, `__hash__`, and friends) that assume you're already comfortable
with what's here.

Every class below already has a working `__init__` — that's not an
oversight. Reading five real constructors, back to back, is how the
pattern actually sinks in; the exercises are the *methods*, which is where
a class's actual behaviour lives.

Work in `exercises.py`. Check your work with `python -m labkit check 05`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/05-oop/`. No need to finish the whole
file before checking your work.

## What you'll build

Five classes in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Class | You'll implement | What it teaches |
|---|---|---|
| `Car` | `describe` | The basic shape: `__init__`, instance attributes, a method |
| `BankAccount` | `deposit`, `withdraw` | Methods that change an object's own state, and guarding against a bad one |
| `Employee` | `intro` | Class attributes — shared by every instance, unlike instance attributes |
| `Animal` | `speak` | The base of an inheritance chain |
| `Dog`, `Cat` | `speak` (each) | Inheriting `__init__` for free; overriding just the one method that differs |

---

## 1. `__init__`, instance attributes, and a method

```python
>>> class Car:
...     def __init__(self, make, model):
...         self.make = make
...         self.model = model
...     def describe(self):
...         return f"{self.make} {self.model}"
...
>>> car = Car("Toyota", "Corolla")
>>> car.make
'Toyota'
>>> car.describe()
'Toyota Corolla'
```

`__init__` runs automatically when you write `Car("Toyota", "Corolla")` —
`self` is the new object being built, and every `self.x = ...` inside
`__init__` creates an **instance attribute**: data that belongs to this
one object. A method (like `describe`) is just a function defined inside
the class that always takes `self` as its first parameter, so it can read
and use that object's own attributes.

**In `exercises.py`:** `Car.describe` is this section's target — `__init__`
is already written for you.

**Test it:** `python3 -m unittest test_exercises.TestCar -v`

## 2. Methods that change an object's own state

```python
>>> class BankAccount:
...     def __init__(self, owner, balance=0):
...         self.owner = owner
...         self.balance = balance
...     def deposit(self, amount):
...         self.balance += amount
...
>>> acc = BankAccount("Ana", 100)
>>> acc.deposit(50)
>>> acc.balance
150
```

`deposit` doesn't `return` anything useful — its job is to **mutate**
`self.balance`, the object's own state, so the change is visible the next
time anyone reads `acc.balance`. This is the other half of what objects
are for, beyond bundling data: methods that change that data over the
object's lifetime, the way a real bank balance changes over time rather
than being recomputed from scratch on every access.

**In `exercises.py`:** `BankAccount.deposit` and `BankAccount.withdraw` are
this section's targets — `withdraw` should raise `ValueError` rather than
letting the balance go negative.

**Test it:** `python3 -m unittest test_exercises.TestBankAccount -v`

## 3. Class attributes

```python
>>> class Employee:
...     company = "Acme"
...     def __init__(self, name):
...         self.name = name
...
>>> a = Employee("Al")
>>> b = Employee("Bo")
>>> a.company, b.company
('Acme', 'Acme')
>>> Employee.company = "Beta"
>>> a.company
'Beta'
```

`company = "Acme"`, written directly inside the class body (not inside
`__init__`, and with no `self.`), is a **class attribute** — it belongs to
the class itself, not to any one instance, so every `Employee` shares the
exact same value. Change it on the class (`Employee.company = "Beta"`) and
every instance sees the new value immediately, since they were never
storing their own separate copy. `self.name`, by contrast, is an instance
attribute — each `Employee` genuinely has its own.

**In `exercises.py`:** `Employee.intro` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestEmployee -v`

## 4–5. Inheritance: sharing `__init__`, overriding one method

```python
>>> class Animal:
...     def __init__(self, name):
...         self.name = name
...     def speak(self):
...         return "..."
...
>>> class Dog(Animal):
...     def speak(self):
...         return "Woof!"
...
>>> rex = Dog("Rex")
>>> rex.name
'Rex'
>>> rex.speak()
'Woof!'
```

`class Dog(Animal):` makes `Dog` a **subclass** of `Animal` — it inherits
everything `Animal` has, including `__init__`, without writing it again.
`Dog("Rex")` calls `Animal.__init__` (since `Dog` didn't define its own),
which is why `rex.name` works with zero code in `Dog` for it. Defining
`speak` inside `Dog` **overrides** `Animal`'s version specifically for
`Dog` instances — this is the actual point of inheritance: share what's
identical, override only what's different.

**In `exercises.py`:** `Animal.speak`, `Dog.speak` and `Cat.speak` are this
section's targets — three small overrides, not three classes' worth of
`__init__`.

**Test it:** `python3 -m unittest test_exercises.TestAnimalDogCat -v`

---

## The exercises

| Class | What it teaches |
|---|---|
| `Car` | `__init__`; instance attributes; a method |
| `BankAccount` | Mutating state through a method; guarding a bad call |
| `Employee` | Class attributes vs. instance attributes |
| `Animal` / `Dog` / `Cat` | Inheritance; overriding one method, sharing the rest |

## Check your understanding

1. **What does `self` actually refer to inside a method?**
   <details><summary>Answer</summary>The specific instance the method was
   called on — <code>car.describe()</code> calls
   <code>describe(car)</code> under the hood, so inside the method,
   <code>self</code> is <code>car</code>.</details>

2. **Why does `deposit` change `self.balance` directly instead of
   returning a new balance for the caller to store?**
   <details><summary>Answer</summary>The whole point of an object is
   that it carries its own state between calls — mutating
   <code>self.balance</code> means every later read of
   <code>acc.balance</code> reflects the deposit, without the caller
   having to remember to reassign anything.</details>

3. **If `Employee.company = "Beta"` changes what every existing `Employee`
   instance sees, why doesn't changing `some_employee.name` do the same
   for other instances?**
   <details><summary>Answer</summary><code>company</code> is a class
   attribute, stored once on the class and shared by every instance that
   hasn't overridden it. <code>name</code> is an instance attribute,
   assigned separately inside <code>__init__</code> for each
   object.</details>

4. **`Dog` never defines `__init__`. Where does `Dog("Rex").name` come
   from?**
   <details><summary>Answer</summary>From <code>Animal.__init__</code>,
   which <code>Dog</code> inherits automatically since it doesn't define
   its own. Inheritance means "everything not overridden comes from the
   parent."</details>

5. **What's the actual difference between `Dog` and `Cat` in this
   module?**
   <details><summary>Answer</summary>Only their <code>speak</code> method
   — both inherit the exact same <code>__init__</code> from
   <code>Animal</code>, and both are instances of <code>Animal</code>
   too.</details>

## Interview questions this module answers

- *What's the difference between a class attribute and an instance
  attribute?*
- *What does `self` mean, mechanically?*
- *What does a subclass inherit, and what does overriding a method
  actually do?*
- *Why would a method mutate `self` instead of returning a new value?*
