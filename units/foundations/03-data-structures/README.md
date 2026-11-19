# 03 — Core data structures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/03-data-structures/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/data-structures?utm_source=github&utm_medium=labs)

*~40 minutes · 18 tests · Junior*

Four containers cover most of what you'll ever reach for: a `list` when
order and duplicates matter, a `set` when only membership does, a `dict`
when you need to look things up by name, and a `tuple` for a small,
fixed-shape bundle. This is the single most load-bearing module in the
foundations track — nearly everything later in this repo, at every tier,
is built on comfort with these four.

Work in `exercises.py`. Check your work with `python -m labkit check 03`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/foundations/03-data-structures/`. No need to
finish the whole file before checking your work.

## What you'll build

Six functions in `exercises.py`, in this order. Keep this table open as you
go — each section below tells you which row it's building:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `first_and_last` | Indexing, incl. negative indices | Grabbing the ends of a sequence without knowing its length |
| `unique_sorted` | `set` for dedup, `sorted` | Removing duplicates from a list of IDs, tags, or results |
| `evens_only` | List comprehensions | Filtering a collection down to what you actually need |
| `word_lengths` | Dict comprehensions | Building a lookup table from a collection in one line |
| `merge_dicts` | Combining dicts; `dict.update` | Layering config: defaults, then overrides |
| `group_by_first_letter` | Building a dict of lists incrementally | Bucketing records by a key — the shape behind almost any "group by" |

---

## 1. Indexing, including negative indices

```python
>>> items = [10, 20, 30, 40]
>>> items[0]
10
>>> items[-1]
40
>>> items[-2]
30
```

`items[0]` is the first element — Python counts from `0`, not `1`.
`items[-1]` is the **last** element, `items[-2]` the second-to-last, and so
on — negative indices count from the end, which is why `items[-1]` is
almost always better than `items[len(items) - 1]`: it's shorter, and it
can't be thrown off by an off-by-one in the length calculation.

**In `exercises.py`:** `first_and_last` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestFirstAndLast -v`

## 2. `set` for "just the distinct values", then `sorted`

```python
>>> set([3, 1, 2, 1, 3])
{1, 2, 3}
>>> sorted({3, 1, 2})
[1, 2, 3]
```

A `set` holds only distinct values — building one from a list with
duplicates is the standard way to deduplicate. Sets are also unordered, so
if the order of the result matters (it usually does, for a human reading
it or a test comparing it), pass the set to `sorted()` afterward, which
always returns a list. `set(items)` then `sorted(...)` is a two-step
pattern worth having reflexively.

**In `exercises.py`:** `unique_sorted` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestUniqueSorted -v`

## 3. List comprehensions

```python
>>> numbers = [1, 2, 3, 4, 5, 6]
>>> [n for n in numbers if n % 2 == 0]
[2, 4, 6]
```

`[expression for item in iterable if condition]` is a `for` loop and an
`if` check compressed into one expression that builds a new list — read it
as "for each item in the iterable, if the condition holds, keep the
expression's value." It's not a different tool from the `for` loop in
Module 01; it's the same idea, written as an expression instead of a
statement, for the common case of "build a new list by filtering and/or
transforming an existing one."

**In `exercises.py`:** `evens_only` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestEvensOnly -v`

## 4. Dict comprehensions

```python
>>> words = ["a", "bb", "ccc"]
>>> {word: len(word) for word in words}
{'a': 1, 'bb': 2, 'ccc': 3}
```

Same idea as a list comprehension, but building a `dict`: `{key_expr:
value_expr for item in iterable}`. This is the fast way to turn a
collection into a lookup table — instead of scanning a list every time you
need to find something by name, build a dict once and look it up in
constant time from then on.

**In `exercises.py`:** `word_lengths` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestWordLengths -v`

## 5. Combining dicts

```python
>>> defaults = {"timeout": 30, "retries": 3}
>>> overrides = {"retries": 5}
>>> merged = dict(defaults)
>>> merged.update(overrides)
>>> merged
{'timeout': 30, 'retries': 5}
```

`dict(defaults)` makes a **copy** — updating `merged` doesn't touch
`defaults`. `.update(overrides)` then writes every key from `overrides`
into `merged`, overwriting any key that already existed. This is exactly
the shape of layering configuration: start from defaults, apply
overrides on top, and whichever source is applied *last* wins on any
conflict.

**In `exercises.py`:** `merge_dicts` is this section's target — copy `a`,
then update with `b`, so `b`'s values win.

**Test it:** `python3 -m unittest test_exercises.TestMergeDicts -v`

## 6. Building a dict of lists

```python
>>> groups = {}
>>> for word in ["cat", "car", "dog"]:
...     letter = word[0]
...     if letter not in groups:
...         groups[letter] = []
...     groups[letter].append(word)
...
>>> groups
{'c': ['cat', 'car'], 'd': ['dog']}
```

This is the general shape behind "group these things by some key": for
each item, figure out its key, make sure that key has an empty list
waiting in the dict (`if letter not in groups: groups[letter] = []`), then
append the item to it. The `if ... not in` check matters — without it,
`groups[letter].append(...)` on a key that doesn't exist yet raises
`KeyError`, since you can't append to a list that was never created.

**In `exercises.py`:** `group_by_first_letter` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestGroupByFirstLetter -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `first_and_last` | Indexing, incl. negative indices |
| `unique_sorted` | `set` for dedup; `sorted` |
| `evens_only` | List comprehensions |
| `word_lengths` | Dict comprehensions |
| `merge_dicts` | Combining dicts; who wins on conflict |
| `group_by_first_letter` | Building a dict of lists incrementally |

## Check your understanding

1. **Why is `items[-1]` usually better than `items[len(items) - 1]`?**
   <details><summary>Answer</summary>It's shorter and can't be thrown off
   by an off-by-one error in computing the length — Python computes the
   "from the end" position for you.</details>

2. **`unique_sorted([3, 1, 2, 1])` returns `[1, 2, 3]`, not `[3, 1, 2]` or
   `{1, 2, 3}`. Why both details?**
   <details><summary>Answer</summary><code>set(...)</code> removes
   duplicates but has no defined order, so the result is passed through
   <code>sorted()</code> to get a predictable, ordered <code>list</code>
   back, not a <code>set</code>.</details>

3. **What's the relationship between a list comprehension and a `for`
   loop with an `if` inside it?**
   <details><summary>Answer</summary>They're the same idea — a list
   comprehension is that loop written as a single expression that builds
   a new list, for the common case of filtering and/or transforming an
   existing collection.</details>

4. **In `merge_dicts(a, b)`, why does the function start with
   `dict(a)` instead of just using `a` directly?**
   <details><summary>Answer</summary><code>dict(a)</code> makes a copy.
   Using <code>a</code> directly and then calling <code>.update(b)</code>
   on it would mutate the caller's original dictionary — a surprising
   side effect for a function that looks like it just returns a new
   value.</details>

5. **In `group_by_first_letter`, what would happen if the
   `if letter not in groups` check were removed?**
   <details><summary>Answer</summary>The first time a new letter is seen,
   <code>groups[letter].append(word)</code> would raise
   <code>KeyError</code>, because there's no list at that key yet to
   append to.</details>

## Interview questions this module answers

- *When would you use a `set` instead of a `list`?*
- *What's the difference between a list comprehension and a `for` loop?*
- *How do you merge two dictionaries, and which one "wins" on a
  conflicting key?*
- *How would you group a list of items by some property of each one?*
