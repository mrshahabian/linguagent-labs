# 06 — Errors, exceptions and your first tests

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/06-errors-and-tests/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/errors-and-tests?utm_source=github&utm_medium=labs)

*~40 minutes · 17 tests · Junior*

Two sides of the same idea: things go wrong, and you need a disciplined
way both to signal that ("raise the right exception") and to check that
your code handles it correctly ("write a test for it"). You've been
*reading* real test code since Module 01 — every `test_exercises.py` in
this repo is a `unittest.TestCase`. This module is where that stops being
background scenery and becomes something you can read fluently yourself.

Work in `exercises.py`. Check your work with `python -m labkit check 06`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/06-errors-and-tests/`. No need to
finish the whole file before checking your work.

## What you'll build

Five functions (plus two tiny custom exception classes) in `exercises.py`,
in this order. Keep this table open as you go — each section below tells
you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `safe_divide` | `try`/`except`, catching one specific error | Any operation that can fail in a predictable, named way |
| `parse_int` | `try`/`except` around a conversion | Validating untrusted input (form fields, CLI args, API bodies) |
| `NegativeAgeError` + `validate_age` | Custom exception classes | Letting callers catch *exactly* your error, not every `ValueError` in the codebase |
| `InsufficientFundsError` + `withdraw_or_raise` | `raise` with a message | Failing loudly and specifically instead of returning a sentinel like `None` |
| `average` | Turning a confusing error into a clear one | Guarding an edge case before it produces a bad stack trace three calls away |

---

## 1. `try` / `except`: catching one specific error

```python
>>> 10 / 0
Traceback (most recent call last):
ZeroDivisionError: division by zero
>>> def safe_divide(a, b):
...     try:
...         return a / b
...     except ZeroDivisionError:
...         return None
...
>>> safe_divide(10, 2)
5.0
>>> safe_divide(10, 0)
```

`try:` wraps code that might raise; `except ZeroDivisionError:` runs only
if *that specific* exception type is raised inside the `try` block —
anything else propagates normally, uncaught. Naming the exact exception
type (not a bare `except:`) is deliberate: catching everything, including
bugs you didn't anticipate, hides real problems instead of handling the
one you actually planned for.

**In `exercises.py`:** `safe_divide` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestSafeDivide -v`

## 2. The same pattern, guarding a conversion

```python
>>> int("42")
42
>>> int("abc")
Traceback (most recent call last):
ValueError: invalid literal for int() with base 10: 'abc'
```

`int(text)` raises `ValueError` on anything that isn't a valid integer
string — the same `try`/`except` shape from Section 1 applies here, just
catching `ValueError` instead of `ZeroDivisionError`. This exact pattern —
attempt a conversion, catch the specific failure, return `None` (or some
other clear "it didn't work" signal) — is how you validate input that
might not be in the shape you expect.

**In `exercises.py`:** `parse_int` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestParseInt -v`

## 3. Defining your own exception type

```python
>>> class NegativeAgeError(Exception):
...     pass
...
>>> def validate_age(age):
...     if age < 0:
...         raise NegativeAgeError(f"age cannot be negative, got {age}")
...     return age
...
>>> validate_age(-5)
Traceback (most recent call last):
NegativeAgeError: age cannot be negative, got -5
```

`class NegativeAgeError(Exception):` defines a brand new exception type by
inheriting from the built-in `Exception` — `pass` (or, better, a
docstring explaining when it's raised) is all the body needs, since it
inherits everything else. `raise NegativeAgeError("...")` stops execution
immediately and hands that exception up to whoever calls this function. The
payoff: a caller can `except NegativeAgeError:` and catch *only* this
specific problem, without accidentally also swallowing an unrelated
`ValueError` from somewhere else in the same `try` block.

**In `exercises.py`:** `NegativeAgeError` (the class body — just a
docstring, nothing to implement) and `validate_age` are this section's
targets.

**Test it:** `python3 -m unittest test_exercises.TestValidateAge -v`

## 4. Raising instead of returning a sentinel

```python
>>> class InsufficientFundsError(Exception):
...     pass
...
>>> def withdraw_or_raise(balance, amount):
...     if amount > balance:
...         raise InsufficientFundsError(
...             f"cannot withdraw {amount}, balance is only {balance}")
...     return balance - amount
...
>>> withdraw_or_raise(100, 30)
70
>>> withdraw_or_raise(100, 1000)
Traceback (most recent call last):
InsufficientFundsError: cannot withdraw 1000, balance is only 100
```

An alternative design here would be returning `None` (or `-1`, or some
other sentinel value) to mean "this failed" — but a sentinel is easy to
forget to check, and easy to confuse with a genuinely valid result.
Raising a specific exception makes the failure impossible to silently
ignore: the caller either handles `InsufficientFundsError` explicitly, or
the program stops with a traceback that names exactly what went wrong.

**In `exercises.py`:** `InsufficientFundsError` and `withdraw_or_raise` are
this section's targets.

**Test it:** `python3 -m unittest test_exercises.TestWithdrawOrRaise -v`

## 5. Turning a confusing error into a clear one

```python
>>> numbers = []
>>> sum(numbers) / len(numbers)
Traceback (most recent call last):
ZeroDivisionError: division by zero
```

That traceback is technically correct and practically useless — nothing
in it says "you passed an empty list." A better version checks for the
problem case explicitly and raises something that actually explains it:

```python
>>> def average(numbers):
...     if not numbers:
...         raise ValueError("average() of an empty list is undefined")
...     return sum(numbers) / len(numbers)
...
>>> average([])
Traceback (most recent call last):
ValueError: average() of an empty list is undefined
```

This is a habit worth having generally: when you can see a bad input
coming, check for it and raise something informative *before* it reaches
whatever line would have failed confusingly. A future reader (including
you, in three weeks) will thank you for the message.

**In `exercises.py`:** `average` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestAverage -v`

## Reading your first `unittest.TestCase`

Open `test_exercises.py` in this folder — every test file in this repo has
the same shape, and this is the first time it's worth reading one closely
rather than skimming past it:

```python
class TestSafeDivide(unittest.TestCase):
    def test_ordinary_division(self):
        self.assertEqual(m.safe_divide(10, 2), 5.0)

    def test_division_by_zero_returns_none(self):
        self.assertIsNone(m.safe_divide(10, 0))
```

A class inheriting from `unittest.TestCase` groups related tests. Each
method whose name starts with `test_` is one independent test — `unittest`
finds and runs every one of them automatically, in no particular order.
`self.assertEqual(a, b)` fails the test (with a clear diff) if `a != b`;
`self.assertIsNone(x)`, `self.assertRaises(SomeError)` and friends are the
same idea for other common checks. A test method's name is documentation —
`test_division_by_zero_returns_none` says exactly what's being verified,
readable without opening the function it's testing.

## The exercises

| Function | What it teaches |
|---|---|
| `safe_divide` | `try`/`except`, one specific exception type |
| `parse_int` | The same pattern, guarding a conversion |
| `NegativeAgeError` / `validate_age` | Defining a custom exception |
| `InsufficientFundsError` / `withdraw_or_raise` | Raising instead of returning a sentinel |
| `average` | Replacing a confusing error with an informative one |

## Check your understanding

1. **Why catch `ZeroDivisionError` specifically in `safe_divide`, instead
   of a bare `except:`?**
   <details><summary>Answer</summary>A bare <code>except:</code> catches
   *everything*, including bugs you never anticipated — it would silently
   swallow a real problem (like a typo raising <code>NameError</code>)
   the same way it swallows the division by zero you actually meant to
   handle.</details>

2. **What does a custom exception class like `NegativeAgeError` actually
   need in its body, at minimum?**
   <details><summary>Answer</summary>Nothing beyond inheriting from
   <code>Exception</code> — <code>class NegativeAgeError(Exception):
   pass</code> is a complete, working exception type. A docstring
   explaining when it's raised is good practice, not a requirement.</details>

3. **Why raise `InsufficientFundsError` instead of just returning `None`
   when a withdrawal is too large?**
   <details><summary>Answer</summary>A sentinel like <code>None</code> is
   easy to forget to check for, and easy to confuse with a genuinely
   valid result. Raising makes the failure impossible to silently
   ignore.</details>

4. **What's wrong with the plain `sum(numbers) / len(numbers)` version of
   `average`, even though it's technically correct?**
   <details><summary>Answer</summary>On an empty list it raises
   <code>ZeroDivisionError</code>, whose message says nothing about what
   the caller actually did wrong — passed an empty list — leaving them to
   figure that out themselves.</details>

5. **In `test_exercises.py`, what does `unittest` use to find which
   methods are tests?**
   <details><summary>Answer</summary>The method name — anything starting
   with <code>test_</code> inside a class that inherits from
   <code>unittest.TestCase</code> is discovered and run
   automatically.</details>

## Interview questions this module answers

- *Why catch a specific exception instead of using a bare `except:`?*
- *When would you define a custom exception class instead of reusing
  `ValueError`?*
- *Why raise an exception instead of returning `None` on failure?*
- *What makes a function `unittest` discovers as a test?*
