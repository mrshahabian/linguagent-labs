# 07 — Tooling and complexity basics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/07-tooling-and-complexity/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/foundations/tooling-and-complexity?utm_source=github&utm_medium=labs)

*~40 minutes · 16 tests · Junior*

Two things every working programmer needs that aren't really "syntax":
enough of the surrounding tooling (a virtual environment, `pip`, `git`) to
not be lost the moment you leave a single script, and enough intuition for
*how fast* code runs to know when a loop inside a loop is a problem before
it becomes one in production. This module closes out the foundations track
— from here, Python 5 — Performance in the mid tier takes the complexity
half of this much further.

Work in `exercises.py`. Check your work with `python -m labkit check 07`, or open the notebook (`lab.ipynb`) in Colab.

The **algorithm** sections below (1–3) end with **Test it**, same as every
other module. The **tooling** section (4) has no exercise — there's
nothing to unit-test about running a terminal command — read it, then try
the commands yourself in a real terminal.

## What you'll build

Three functions in `exercises.py`, in this order:

| Function | What it teaches | Shows up in real code as |
|---|---|---|
| `linear_search` | Checking every item — O(n) | The only option on an unsorted collection |
| `binary_search` | Halving the search space — O(log n) | Why sorted data is worth keeping sorted |
| `has_duplicate_pair` | A nested loop — O(n²) | The shape to notice and usually avoid |

---

## 1. Linear search — checking every item

```python
>>> def linear_search(items, target):
...     for i, item in enumerate(items):
...         if item == target:
...             return i
...     return -1
...
>>> linear_search([5, 3, 8, 1], 8)
2
>>> linear_search([5, 3, 8, 1], 99)
-1
```

`enumerate(items)` gives you `(index, item)` pairs as you loop, so you can
return *where* the match was found, not just whether one exists. In the
worst case — the target is last, or missing entirely — this checks every
single element once: the number of steps grows in direct proportion to the
size of `items`. That's what **O(n)** means: roughly, "the work scales
linearly with the input size." It's not a precise measurement of speed, it's
a shape — twice the input, roughly twice the work.

**In `exercises.py`:** `linear_search` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestLinearSearch -v`

## 2. Binary search — halving the search space

```python
>>> def binary_search(sorted_items, target):
...     low, high = 0, len(sorted_items) - 1
...     while low <= high:
...         mid = (low + high) // 2
...         if sorted_items[mid] == target:
...             return mid
...         elif sorted_items[mid] < target:
...             low = mid + 1
...         else:
...             high = mid - 1
...     return -1
...
>>> binary_search([1, 3, 5, 7, 9, 11], 7)
3
```

This only works because the input is **sorted** — that's what lets it
throw away half the remaining possibilities on every step, instead of
checking one at a time. Look at the middle element: if it's the target,
done; if the target is bigger, everything to the left can be discarded (it's
all smaller); if smaller, discard the right half. Each step halves the
search space, which is what **O(log n)** means: to search a list a
thousand times bigger, you need only about ten more steps, not a thousand
more. A sorted list of a million items takes at most ~20 comparisons to
search this way — a linear search could take a million.

**In `exercises.py`:** `binary_search` is this section's target.

**Test it:** `python3 -m unittest test_exercises.TestBinarySearch -v`

## 3. A nested loop — the shape to notice

```python
>>> def has_duplicate_pair(numbers):
...     for i in range(len(numbers)):
...         for j in range(i + 1, len(numbers)):
...             if numbers[i] == numbers[j]:
...                 return True
...     return False
...
>>> has_duplicate_pair([1, 2, 3, 2])
True
```

For every element, this checks it against every element *after* it — a
loop inside a loop, over roughly the same data twice. That's **O(n²)**:
double the input, and the work roughly *quadruples*, not doubles, because
both loops grow. This isn't automatically wrong — for a handful of items
it's completely fine — but it's the shape worth noticing before it's
running over a collection ten times bigger than you tested it on. You
already know a faster way to ask the same question: Module 03's
`unique_sorted` used `set(items)` to get the distinct values in one pass —
`len(numbers) != len(set(numbers))` answers "is there a duplicate?" in
O(n), no nested loop required.

**In `exercises.py`:** `has_duplicate_pair` is this section's target —
write the nested-loop version deliberately, even knowing the faster one,
so the shape is one you'll recognize on sight afterward.

**Test it:** `python3 -m unittest test_exercises.TestHasDuplicatePair -v`

## 4. Tooling: venv, pip, and git — enough to not be lost

No exercise here — this is background you'll use constantly rather than
test, so it's presented as reading, not a stub. Try each command for real
in a terminal as you read it.

**A virtual environment** keeps one project's installed packages separate
from every other project's, so upgrading a package for one thing doesn't
silently break another:

```bash
python3 -m venv .venv          # create one, once, per project
source .venv/bin/activate      # turn it on for this terminal session
```

Once activated, `python3` and `pip` inside that terminal both point at the
virtual environment, not your system-wide Python. This whole repo, at
every tier, assumes you've done this — that's what `.venv/` in the repo
root is.

**`pip`** installs packages into whichever environment is currently
active:

```bash
pip install numpy               # install one package
pip install -r requirements.txt # install everything a project lists
```

**Running a script** is `python3 path/to/file.py` from a terminal — for
this repo specifically, every module also has its own `run` script (or a
shared one, like `units/foundations/run` for this whole track) that runs
the tests for you, which is what you've been using all along.

**`git`**, at the level you need to not be lost in *this* repo:

```bash
git status              # what's changed since the last commit
git add <file>           # stage a file's changes to be committed
git commit -m "message"  # save the staged changes, with a message
git log --oneline        # see recent commit history
```

You don't need much more than this to follow along with a project day to
day — branches, merges, and rebasing are worth learning, but they're not a
prerequisite for anything else in this repo.

---

## The exercises

| Function | What it teaches |
|---|---|
| `linear_search` | Checking every item — O(n) |
| `binary_search` | Halving the search space on sorted data — O(log n) |
| `has_duplicate_pair` | A nested loop — O(n²), and the O(n) alternative |

## Check your understanding

1. **Why does `binary_search` require its input to already be sorted, when
   `linear_search` doesn't care?**
   <details><summary>Answer</summary>Binary search decides which half to
   discard by comparing to the middle element — that logic only works if
   everything to one side is guaranteed smaller and everything to the
   other side is guaranteed bigger, which is exactly what "sorted"
   means.</details>

2. **Roughly how many more steps does binary search need to search a list
   1,000 times bigger?**
   <details><summary>Answer</summary>About 10 more — <code>log₂(1000)</code>
   is roughly 10, since each step halves the remaining space. Compare
   that to linear search, which would need up to 1,000 times more
   steps.</details>

3. **Why is `has_duplicate_pair`'s inner loop written as
   `range(i + 1, len(numbers))` instead of `range(len(numbers))`?**
   <details><summary>Answer</summary>Starting from <code>i + 1</code>
   avoids comparing an element to itself and avoids checking the same pair
   twice in both orders — <code>(numbers[i], numbers[j])</code> and
   <code>(numbers[j], numbers[i])</code> would otherwise both get
   checked.</details>

4. **What's the O(n) alternative to `has_duplicate_pair`, and why is it
   faster?**
   <details><summary>Answer</summary><code>len(numbers) !=
   len(set(numbers))</code> — building a <code>set</code> is one pass
   over the data, versus the nested loop's roughly n² comparisons.</details>

5. **Why does this repo need a virtual environment per project instead of
   just installing packages globally?**
   <details><summary>Answer</summary>So that one project's package
   versions can't silently conflict with or upgrade-break another
   project's — each virtual environment has its own independent set of
   installed packages.</details>

## Interview questions this module answers

- *What's the difference between O(n) and O(log n), concretely?*
- *Why does binary search need sorted input?*
- *What's the giveaway that code is O(n²), and how would you avoid it
  here?*
- *What does a virtual environment actually isolate?*

---

**You've finished the foundations track.** From here, continue to
[Python 1 — The data model](../../python/10-data-model/), the first module
of the mid tier — it assumes everything covered across these seven
modules and moves on to how Python's own dunder protocols work.
