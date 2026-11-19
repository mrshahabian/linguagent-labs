# 01 — Syntax, values and control flow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/01-syntax/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/syntax?utm_source=github&utm_medium=labs)

*~30 minutes · 23 tests · Junior*

Every program is built from the same handful of moves: store a value, ask a
question about it, and decide what to do based on the answer. This module is
those moves — `if`, `for`, `while`, and the operators that go inside them —
with nothing else in the way yet. Every later module in this repo, junior
through senior, assumes you can read and write these without thinking about
the syntax.

Work in `exercises.py`. Check your work with `python -m labkit check 01`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/01-syntax/`. No need to finish the whole
file before checking your work.

## What you'll build

Six functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `classify_number` | `if`/`elif`/`else` | Any branch on a value — routing a request, validating input |
| `fizzbuzz` | `for` loops and `%` (modulo) | Iterating a fixed number of times and reacting to each value |
| `count_vowels` | Iterating over a string, `in` | Scanning text character by character |
| `grade_for_score` | Chained conditions and boundaries | Thresholds — rate limits, pass/fail cutoffs, pricing tiers |
| `sum_up_to` | `while` loops and the accumulator pattern | Looping until a condition changes, not a fixed count |
| `is_leap_year` | `and`/`or` and operator precedence | Combining several conditions into one correct rule |

---

## 1. `if` / `elif` / `else`

```python
>>> def classify_number(n):
...     if n < 0:
...         return "negative"
...     elif n == 0:
...         return "zero"
...     else:
...         return "positive"
...
>>> classify_number(-5)
'negative'
>>> classify_number(0)
'zero'
```

Python checks each condition top to bottom and runs the **first** branch
that's `True` — the rest are skipped entirely, even if they'd also be
`True`. `elif` is short for "else if"; you can chain as many as you need,
and the final `else` (optional) catches everything nothing else matched.
Comparison operators — `<`, `>`, `<=`, `>=`, `==`, `!=` — each return a
`bool` (`True`/`False`), which is exactly what `if` is asking a question
about.

**In `exercises.py`:** `classify_number` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestClassifyNumber -v`

## 2. `for` loops and the modulo operator

```python
>>> for i in range(1, 6):
...     print(i, i % 3)
...
1 1
2 2
3 0
4 1
5 2
```

`for x in range(1, n + 1)` runs the loop body once for each integer from
`1` up to and **including** `n` — `range`'s upper bound is exclusive, which
is why the `+ 1` is there; forgetting it is the single most common
off-by-one mistake in this module. `%` (modulo) gives the remainder after
division — `i % 3 == 0` means "`i` is a multiple of 3." That's the whole
trick behind FizzBuzz: check the more specific rule (`% 15`, both 3 and 5)
before the more general ones (`% 3`, `% 5`), or the specific case never
gets a chance to run.

**In `exercises.py`:** `fizzbuzz` is this section's target — build a list,
one entry per number from `1` to `n`, and `return` it at the end.

**Test it:** `python3 -m unittest test_exercises.TestFizzbuzz -v`

## 3. Looping over a string, and `in`

```python
>>> for ch in "cat":
...     print(ch)
...
c
a
t
>>> "a" in "cat"
True
>>> "z" in "cat"
False
```

A string is iterable — `for ch in word` visits one character at a time, in
order, same shape as looping over a list. `in` asks membership: is this
character present anywhere in that string (or that collection)? Combining
the two — loop over every character, ask `in` about each one, keep a
running count — is a pattern you'll use constantly, well beyond counting
vowels.

**In `exercises.py`:** `count_vowels` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestCountVowels -v`

## 4. Chained conditions and where the boundary belongs

```python
>>> def grade_for_score(score):
...     if score >= 90:
...         return "A"
...     elif score >= 80:
...         return "B"
...     else:
...         return "F"
...
>>> grade_for_score(90)
'A'
>>> grade_for_score(89)
'B'
```

`90` is an `A`, not a `B` — `>=` includes the boundary itself. The order of
the checks matters here too, same as FizzBuzz: because Python stops at the
first `True` branch, checking `score >= 90` first is what makes `95` land
in `"A"` instead of also matching `>= 80` and stopping there. Write the
checks in the wrong order (loosest first) and everything above the lowest
threshold silently gets the wrong grade — worth tracing through by hand
once so the failure mode is obvious before it's just a red test.

**In `exercises.py`:** `grade_for_score` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestGradeForScore -v`

## 5. `while` loops and the accumulator pattern

```python
>>> total = 0
>>> i = 1
>>> while i <= 3:
...     total += i
...     i += 1
...
>>> total
6
```

`while condition:` keeps running the body for as long as `condition` is
`True` — unlike `for`, there's no built-in notion of "how many times"; you
manage that yourself, which is why `i += 1` is inside the loop. Forget it,
and `i` never changes, the condition never becomes `False`, and the loop
never ends. `total = 0` before the loop, then `total += i` inside it, is
the **accumulator pattern**: a variable that starts at an identity value
(`0` for a sum) and builds up one step at a time. You'll meet this same
shape again, doing far more interesting things than adding numbers, in
Python 5 — Performance.

**In `exercises.py`:** `sum_up_to` is this section's target — use a `while`
loop even though `for i in range(1, n + 1)` would also work; the point of
this exercise is the `while` mechanics, not the shortest solution.

**Test it:** `python3 -m unittest test_exercises.TestSumUpTo -v`

## 6. `and` / `or`, and getting the rule right

```python
>>> year = 1900
>>> year % 4 == 0
True
>>> year % 4 == 0 and year % 100 != 0
False
>>> (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
False
```

`and` is `True` only if both sides are; `or` is `True` if either side is.
`not` flips a `bool`. Python evaluates left to right and, crucially, stops
early — `False and anything` skips evaluating "anything" entirely, since
the answer's already decided. The leap year rule is a good test of reading
comprehension as much as syntax: divisible by 4, **except** century years
(divisible by 100), **except** those are leap years anyway if divisible by
400. Get the parentheses wrong and `1900` (not a leap year) or `2000` (a
leap year) will silently come out wrong while every ordinary year still
passes — exactly the kind of bug that hides until someone hits the edge
case.

**In `exercises.py`:** `is_leap_year` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestIsLeapYear -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `classify_number` | `if`/`elif`/`else`; comparison operators |
| `fizzbuzz` | `for` and `range`; `%`; checking the specific case first |
| `count_vowels` | Iterating a string; `in` |
| `grade_for_score` | Chained conditions; where a boundary belongs |
| `sum_up_to` | `while`; the accumulator pattern |
| `is_leap_year` | `and`/`or`; getting a multi-part rule right |

## Check your understanding

Answer these out loud or in writing before moving on — if you can't, revisit
the section above it before starting Module 2.

1. **Why does `elif` matter here, instead of writing three separate `if`
   statements?**
   <details><summary>Answer</summary>Separate <code>if</code>s each get
   checked independently, so more than one could run. <code>elif</code>
   guarantees only the first matching branch runs, because Python stops
   checking once one is <code>True</code>.</details>

2. **What does `range(1, 5)` actually produce, and why does `fizzbuzz`
   use `range(1, n + 1)` instead of `range(1, n)`?**
   <details><summary>Answer</summary><code>range(1, 5)</code> produces
   1, 2, 3, 4 — the stop value is exclusive. To include <code>n</code>
   itself, the stop value has to be <code>n + 1</code>.</details>

3. **If you swap the order of the checks in `grade_for_score` so `score
   >= 60` comes first, what breaks?**
   <details><summary>Answer</summary>Every score of 60 or above would
   return <code>"D"</code>, since that's now the first matching branch —
   the <code>A</code>/<code>B</code>/<code>C</code> checks would never be
   reached.</details>

4. **What's the difference between `for` and `while`, and how do you know
   which one a problem calls for?**
   <details><summary>Answer</summary><code>for</code> is for iterating a
   known sequence or count; <code>while</code> is for repeating until a
   condition changes, when you don't know the number of iterations up
   front.</details>

5. **In `is_leap_year`, why is `1900 % 100 != 0` False, and why does that
   make the overall rule return `False` for 1900 with the given
   parentheses?**
   <details><summary>Answer</summary>1900 divided by 100 has no remainder,
   so <code>1900 % 100 != 0</code> is <code>False</code>. That makes the
   first <code>and</code> clause <code>False</code>, and since 1900 isn't
   divisible by 400 either, the final <code>or</code> has nothing to make
   it <code>True</code>.</details>

## Interview questions this module answers

- *Why did my `if`/`elif` chain return the wrong branch?*
- *What's the difference between `for` and `while`, and when do you reach
  for each?*
- *What does `range(1, n)` actually iterate over?*
- *Why does short-circuit evaluation (`and`/`or` stopping early) matter?*
