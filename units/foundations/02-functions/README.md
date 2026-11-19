# 02 — Functions and scope

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/02-functions/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/functions?utm_source=github&utm_medium=labs)

*~35 minutes · 19 tests · Junior*

A function is a named, reusable chunk of code that takes some inputs and
gives back a result. Module 01 already used them without naming the idea;
this module is about the machinery around that — how arguments actually
get passed in, where a variable does and doesn't exist, and how a function
can call itself.

Work in `exercises.py`. Check your work with `python -m labkit check 02`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/02-functions/`. No need to finish the
whole file before checking your work.

## What you'll build

Six functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `greet` | Default parameter values | Optional configuration — most function calls don't need to specify everything |
| `add_all` | `*args` — variable positional arguments | Functions that accept "any number of" something, like `print` itself |
| `build_profile` | `**kwargs` — variable keyword arguments | Passing a flexible bag of named options through, like a request's query params |
| `total_price` | Local scope | Why a variable inside a function doesn't leak out, and why that's a feature |
| `factorial` | Recursion — a function calling itself | The natural shape for anything defined in terms of a smaller version of itself |
| `sum_digits` | Recursion on a number, not a count | Breaking a value down (not just counting down) recursively |

---

## 1. Default parameter values

```python
>>> def greet(name, greeting="Hello"):
...     return f"{greeting}, {name}!"
...
>>> greet("Ana")
'Hello, Ana!'
>>> greet("Ana", "Hi")
'Hi, Ana!'
>>> greet("Ana", greeting="Yo")
'Yo, Ana!'
```

`greeting="Hello"` gives that parameter a value to fall back on if the
caller doesn't supply one — the caller can still override it, either
positionally (`greet("Ana", "Hi")`) or by name (`greeting="Yo"`). Naming it
makes the call self-documenting, which matters once a function has more
than one or two parameters.

**In `exercises.py`:** `greet` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestGreet -v`

## 2. `*args` — any number of positional arguments

```python
>>> def add_all(*numbers):
...     total = 0
...     for n in numbers:
...         total += n
...     return total
...
>>> add_all(1, 2, 3)
6
>>> add_all()
0
```

`*numbers` collects every positional argument the caller passes — zero, one,
or a hundred — into a single tuple named `numbers` inside the function. This
is exactly how a built-in like `print(a, b, c)` accepts any number of
things to print; `*args` is the mechanism that makes that possible in your
own functions too. The name `numbers` is a choice here, not special syntax
— `*args` is only a convention, not a keyword.

**In `exercises.py`:** `add_all` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestAddAll -v`

## 3. `**kwargs` — any number of keyword arguments

```python
>>> def build_profile(name, **details):
...     profile = {"name": name}
...     profile.update(details)
...     return profile
...
>>> build_profile("Al", age=30, city="NY")
{'name': 'Al', 'age': 30, 'city': 'NY'}
```

`**details` is the keyword equivalent of `*args`: every `key=value` the
caller passes beyond the named parameters gets collected into a `dict`
named `details`. `name` stays separate because it's declared explicitly as
its own parameter — only the *extra*, unanticipated keyword arguments end
up in `**details`. This is the pattern behind any function that takes a
flexible set of optional settings without you having to name every single
one in the signature up front.

**In `exercises.py`:** `build_profile` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestBuildProfile -v`

## 4. Local scope

```python
>>> def total_price(prices, tax_rate=0.0):
...     total = 0
...     for price in prices:
...         total += price
...     return total * (1 + tax_rate)
...
>>> total_price([10, 20])
30.0
>>> total
Traceback (most recent call last):
NameError: name 'total' is not defined
```

`total` only exists **while `total_price` is running**. The moment the
function returns, that variable is gone — trying to read `total` from
outside raises `NameError`, even though the function itself used it freely.
This is deliberate, not a limitation: every call to `total_price` gets its
own fresh `total`, starting at `0`, with zero risk of one call's leftover
value leaking into the next. A variable assigned inside a function is
**local** to that function unless you go out of your way (the `global`
keyword, which you should reach for rarely) to say otherwise.

**In `exercises.py`:** `total_price` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestTotalPrice -v`

## 5. Recursion: a function calling itself

```python
>>> def factorial(n):
...     if n == 0:
...         return 1
...     return n * factorial(n - 1)
...
>>> factorial(5)
120
```

`factorial(5)` calls `factorial(4)`, which calls `factorial(3)`, and so on
down to `factorial(0)` — the **base case**, which returns directly instead
of calling itself again. Without a base case (or if it's never reached),
this recurses forever until Python gives up with a `RecursionError`. Every
recursive function needs exactly this shape: one case that stops, and one
case that makes the problem smaller and calls itself again.

**In `exercises.py`:** `factorial` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestFactorial -v`

## 6. Recursion on a value, not a count

```python
>>> def sum_digits(n):
...     if n < 10:
...         return n
...     return n % 10 + sum_digits(n // 10)
...
>>> sum_digits(123)
6
```

`factorial` shrinks by subtracting 1 each time; `sum_digits` shrinks a
different way — `n // 10` drops the last digit (integer division discards
the remainder), so `123 → 12 → 1`, and the base case is "down to a single
digit." `n % 10` peels off exactly the digit that `// 10` just dropped, and
each level adds its digit to the sum coming back up from the smaller call.
The shape is the same as `factorial` (base case, recursive case that makes
the problem smaller) even though what "smaller" means is different.

**In `exercises.py`:** `sum_digits` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestSumDigits -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `greet` | Default parameter values |
| `add_all` | `*args` |
| `build_profile` | `**kwargs` |
| `total_price` | Local scope |
| `factorial` | Recursion — base case + recursive case |
| `sum_digits` | Recursion that shrinks a value, not a count |

## Check your understanding

1. **What's the difference between `greet("Ana", "Hi")` and
   `greet("Ana", greeting="Hi")`?**
   <details><summary>Answer</summary>Nothing, for this function —
   <code>"Hi"</code> reaches the same parameter either positionally or by
   keyword. Naming it is more explicit, which matters more as a function
   gains more parameters.</details>

2. **Why does `add_all()` (no arguments) return `0` instead of raising an
   error?**
   <details><summary>Answer</summary><code>*numbers</code> collects zero
   arguments into an empty tuple when none are passed. The loop over an
   empty tuple simply never runs, so <code>total</code> stays at its
   starting value, <code>0</code>.</details>

3. **After calling `total_price([10, 20])`, why does `total` not exist in
   the calling code?**
   <details><summary>Answer</summary><code>total</code> is a local
   variable — it's created fresh each call and destroyed when the function
   returns. It was never visible outside <code>total_price</code> in the
   first place.</details>

4. **What are the two things every recursive function needs, and what
   happens if one is missing?**
   <details><summary>Answer</summary>A base case that stops without
   recursing, and a recursive case that makes the problem smaller. Miss
   the base case (or never actually reach it) and the function calls
   itself forever, until Python raises <code>RecursionError</code>.</details>

5. **In `sum_digits`, what does `n // 10` do, and why is it different from
   `n / 10`?**
   <details><summary>Answer</summary><code>//</code> is integer division —
   it divides and drops any remainder/fraction, so <code>123 // 10 ==
   12</code>. <code>/</code> is regular division and would give
   <code>12.3</code>, a float, which never shrinks down to a single digit
   the way the recursion needs.</details>

## Interview questions this module answers

- *What's the difference between `*args` and `**kwargs`?*
- *Why doesn't this variable exist outside the function I defined it in?*
- *What are the two parts every recursive function needs?*
- *When would you use recursion instead of a loop?*
