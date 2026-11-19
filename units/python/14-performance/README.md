# 14 — Performance and memory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/14-performance/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/performance?utm_source=github&utm_medium=labs)

*~50 minutes · 32 tests*

Interview performance questions are almost never about micro-optimisation.
They are about picking a data structure whose complexity matches the access
pattern, and about not accidentally writing quadratic code that looks linear.

The four moves in this module cover most of it: **use the right container**,
**do not rebuild immutable objects in a loop**, **cache repeated work**, and
**stream instead of materialising**.

Work in `exercises.py`. Check your work with `python -m labkit check 14`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/14-performance/`. No need to finish the whole
file before checking your work.

## What you'll build

Nine functions (and one class) in `exercises.py`, in this order. Keep this
table open as you go — each section below tells you which row it's building:

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `common_elements` | the function body | Filtering a batch against an allow-list or a fixed vocabulary — keeping only the tokens that exist in an embedding table without scanning the table per token |
| `join_words` | the function body | Assembling a prompt string or a log line from many pieces — naive `+=` concatenation is a classic way to make a data-loading or prompt-building step accidentally quadratic |
| `collatz_length` | the recursion body (`@functools.lru_cache` is already applied above it) | Memoising a pure, repeatedly-called function — the same decorator is how you'd cache a tokenizer's `encode()` on a small fixed vocabulary, or a hash-consed feature lookup |
| `Point` | nothing — it's given complete, as a worked example | Memory-efficient value objects at scale: millions of small structs (coordinates, graph nodes, bounding boxes) where per-instance `__dict__` overhead dominates |
| `running_max` | the function body (monotonic deque) | Streaming max/min over a sliding window — a running max of GPU memory or per-batch loss over the last *N* steps, updated once per step instead of re-scanning the window |
| `insert_sorted` | the function body (`bisect.insort`) | Maintaining a sorted structure incrementally — a top-k leaderboard or a sorted buffer of scored candidates, inserted one at a time |
| `rank_by_frequency` | the function body (`Counter` + explicit sort key) | Building a vocabulary or label distribution from a corpus, with a tie-break that has to be deterministic rather than whatever order a hash table produces |
| `total_size` | the function body (generator inside `sum`) | Aggregating over a dataset too large to hold in memory — summing example counts or byte sizes across sharded dataset files as they're read |
| `first_match` | the function body (`next()` with a default) | Short-circuiting a search over a large or unbounded stream — pulling the first example matching a filter out of a data loader without exhausting it |

---

## 1. The complexity table worth memorising

| Operation | `list` | `set` / `dict` | `deque` |
|---|---|---|---|
| `x in c` | **O(n)** | O(1) | O(n) |
| append / add | O(1)* | O(1) | O(1) |
| pop from end | O(1) | — | O(1) |
| pop from front | **O(n)** | — | **O(1)** |
| insert in middle | O(n) | — | O(n) |
| indexing | O(1) | — | O(n) |

\* amortised — lists over-allocate, so most appends are free and occasional
ones copy.

`common_elements` is the whole first row in one function: `x in b` where `b`
is a list makes the function O(n·m). One `set(b)` makes it O(n+m). The test
counts `__eq__` calls rather than seconds, so it fails on the algorithm, not
on a slow machine.

`deque` matters specifically for FIFO. `list.pop(0)` shifts every remaining
element; that is the difference between a linear and a quadratic queue drain.

```
>>> common_elements([3, 1, 2], [2, 3])
[3, 2]
>>> from collections import deque
>>> d = deque([1, 2, 3])
>>> d.popleft()          # O(1) - list.pop(0) here would be O(n): every
1                        # remaining element shifts down one slot
>>> d
deque([2, 3])
```

**In `exercises.py`:** `common_elements` is this section's target — convert
`b` to a `set` before the membership test.

**Test it:** `python3 -m unittest test_exercises.TestCommonElements -v`

## 2. Strings are immutable, so `+=` in a loop is quadratic

```python
out = ""
for word in words:
    out += word + " "        # allocates and copies the whole string, n times
```

Each `+=` builds a *new* string containing everything so far — total work is
1 + 2 + … + n. `str.join` walks the sequence once to size the buffer, then
fills it:

```python
" ".join(words)
```

For 200k words that is milliseconds instead of seconds. (CPython has an
optimisation that sometimes rescues `+=` when the string has a refcount of 1,
but it is an implementation detail and it does not survive being passed
around. Do not rely on it.)

The same reasoning applies to any immutable in a loop: build a list and
convert once, rather than concatenating tuples or re-creating frozensets.

```
>>> join_words(["a", "b", "c"])
'a b c'
>>> join_words(["a", "b"], "-")
'a-b'
```

**In `exercises.py`:** `join_words` — one call to `separator.join(words)`.

**Test it:** `python3 -m unittest test_exercises.TestJoinWords -v`

## 3. Caching: `functools.lru_cache`

```python
@functools.lru_cache(maxsize=None)
def collatz_length(n):
    ...
```

The Collatz recursion revisits the same values relentlessly — every
trajectory funnels into the same tail. Memoising turns an
exponential-looking workload into a linear one for one line of code.

Things to know before you reach for it:

- **Arguments must be hashable.** No lists, no dicts, no NumPy arrays.
- **`maxsize=None` never evicts.** That is unbounded memory keyed by
  arguments — fine for a bounded integer domain, a leak for user input.
- **The cache is per-function, not per-instance.** Decorating a method holds
  a reference to `self` in the cache, so the instances never get collected.
  Use `functools.cached_property` for per-instance caching.
- **`cache_info()` / `cache_clear()`** exist and are the first things to
  check when a cached function looks like it is not helping.
- **Only cache pure functions.** Caching something that reads a file or a
  clock produces stale answers that are very hard to trace.

```
>>> collatz_length(6)
9
>>> collatz_length.cache_info()
CacheInfo(hits=0, misses=9, maxsize=None, currsize=9)
>>> collatz_length(6)             # every value on the 6 -> 1 trajectory was
9                                 # cached individually, so this is a hit
>>> collatz_length.cache_info()
CacheInfo(hits=1, misses=9, maxsize=None, currsize=9)
```

**In `exercises.py`:** `collatz_length`'s recursion body — the
`@functools.lru_cache` decorator is already applied above the stub, so you
only need to write the base case and the two recursive branches.

**Test it:** `python3 -m unittest test_exercises.TestCollatz -v`

## 4. Amortising with the right algorithm

### The monotonic deque: `running_max`

`running_max` is the classic. The obvious version:

```python
[max(values[i:i + w]) for i in range(len(values) - w + 1)]     # O(n·w)
```

The monotonic-deque version keeps *indices* in decreasing value order:

```python
while candidates and candidates[0] <= index - window:   # left of the window
    candidates.popleft()
while candidates and values[candidates[-1]] <= value:   # can never win again
    candidates.pop()
candidates.append(index)
```

Each index is pushed once and popped at most once, so the total is **O(n)**
regardless of window size. The invariant — *the deque holds exactly those
indices that could still be the maximum of some future window, in decreasing
value order* — is the thing to say out loud; the code follows from it.

Store indices, not values, so you can tell when an entry falls out on the
left.

```
>>> running_max([1, 3, 2, 5, 4], 3)
[3, 5, 5]
```

**In `exercises.py`:** `running_max`.

**Test it:** `python3 -m unittest test_exercises.TestRunningMax -v`

### `bisect.insort`: `insert_sorted`

`insert_sorted` is the small sibling: `bisect.insort` finds the position in
O(log n) and inserts, versus appending and re-sorting the whole list every
time — the difference between maintaining a sorted structure and rebuilding
it from scratch on every write. It is the same move behind a top-k
leaderboard or a sorted buffer of scored candidates that grows one item at a
time.

```
>>> data = [2, 3]
>>> insert_sorted(data, 1)
[1, 2, 3]
>>> insert_sorted(data, 9)
[1, 2, 3, 9]
```

**In `exercises.py`:** `insert_sorted` — one call to `bisect.insort`.

**Test it:** `python3 -m unittest test_exercises.TestInsertSorted -v`

## 5. Sorting with an explicit key

```python
sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
```

A tuple key gives you multi-level sorting; negating a numeric field flips
just that level. Python's sort is **stable**, which is the other way to do
this: sort by the secondary key first, then by the primary.

`Counter.most_common()` does not define its tie order, so when the tie-break
is part of the specification — as it usually is in a coding assessment —
sort explicitly.

```
>>> rank_by_frequency("aaabbc")
[('a', 3), ('b', 2), ('c', 1)]
>>> rank_by_frequency("ba")        # tied at count 1 -> break alphabetically
[('a', 1), ('b', 1)]
>>> rank_by_frequency("aaabbc", limit=2)
[('a', 3), ('b', 2)]
```

**In `exercises.py`:** `rank_by_frequency` — `Counter(items)` then
`sorted(..., key=lambda pair: (-pair[1], pair[0]))`, sliced by `limit`.

**Test it:** `python3 -m unittest test_exercises.TestRankByFrequency -v`

## 6. Streaming instead of materialising

```python
sum(row["size"] for row in rows)      # generator expression — constant memory
sum([row["size"] for row in rows])    # list comprehension — O(n) memory
```

Same answer, and for a source that is itself a generator the first version
never holds more than one row in memory at a time. The tests check this by
recording which rows were pulled — laziness is observable behaviour, not a
style preference.

### Generator expressions: `total_size`

A generator argument is single-pass: once `total_size` has walked it, it is
exhausted, which is itself proof nothing built a hidden list along the way.

```
>>> gen = (row for row in [{"size": 1}, {"size": 2}])
>>> total_size(gen)
3
>>> total_size(gen)      # gen is already exhausted - not silently re-read
0
```

**In `exercises.py`:** `total_size` — `sum(row["size"] for row in rows)`.

**Test it:** `python3 -m unittest test_exercises -k total_size -v`

This is narrower than the whole `TestStreaming` class, which also holds the
tests for `first_match` below — the `-k total_size` filter matches only
`test_total_size` and `test_total_size_streams`.

### Stopping early with `next()`: `first_match`

`first_match` makes the other half of the point: `next(x for x in rows if
pred(x))` stops at the first hit, while `[x for x in rows if pred(x)][0]`
evaluates the predicate for every row — including ones after the match, and
including all of them if there is no match. That difference is the reason
this works at all on an unbounded source:

```
>>> import itertools
>>> rows = ({"n": n} for n in itertools.count())   # infinite generator
>>> first_match(rows, lambda r: r["n"] == 3)
{'n': 3}
```

A version that filtered into a list first would never return here — it
would spin forever trying to exhaust `itertools.count()`.

**In `exercises.py`:** `first_match` — `next((row for row in rows if
predicate(row)), None)`.

**Test it:** `python3 -m unittest test_exercises -k first_match -v`

Again narrower than the full `TestStreaming` class, for the same reason as
above — this matches only the three `test_first_match*` methods.

## 7. `__slots__`

```python
class Point:
    __slots__ = ("x", "y")
```

By default every instance carries a `__dict__`, which costs a few hundred
bytes and one extra indirection per attribute access. `__slots__` replaces it
with fixed descriptor-backed storage: smaller, faster, and it turns typos
into `AttributeError` instead of silently creating a new attribute.

The trade-offs: no new attributes at runtime, no `__dict__`, no default
`__weakref__` (add `"__weakref__"` to the slots if you need it), and
inheritance needs care — a subclass without `__slots__` gets a `__dict__`
back, undoing the saving. Worth it when you have millions of instances;
irrelevant when you have ten.

```
>>> p = Point(1, 2)
>>> p.x, p.y
(1, 2)
>>> hasattr(p, "__dict__")
False
>>> p.z = 3
Traceback (most recent call last):
    ...
AttributeError: 'Point' object has no attribute 'z'
```

**In `exercises.py`:** nothing to fill in — `Point` is given complete, as a
worked example rather than a stub. `TestSlots` characterises the trade-off
in code (no `__dict__`, rejecting a new attribute, smaller than an
equivalent plain class) instead of testing something you wrote.

**Test it:** `python3 -m unittest test_exercises.TestSlots -v`

## 8. Measure before you optimise

Guessing wrong about where time goes is the norm. The tools, in order of
reach:

```python
timeit.timeit("...", number=1000)     # micro-benchmarks, handles warm-up
python -m cProfile -s cumtime app.py  # where the time actually went
```

plus `line_profiler` for line-level detail and `tracemalloc` for allocation.

```
>>> import timeit
>>> timeit.timeit(lambda: join_words(["word"] * 1000), number=1000)
0.0069...
```

The interview-safe statement is: *profile first, fix the top entry, measure
again* — and be able to name the tool you would use.

This section has no matching exercise or test — profiling is a skill you
demonstrate by narrating your process out loud, not something a unit test
can check. None of the nine rows in the table above are exercised here;
they're each covered by their own section's **Test it** above.

---

## The exercises

| Function | What it teaches |
|---|---|
| `common_elements` | Set membership vs linear scan |
| `join_words` | Quadratic string concatenation |
| `collatz_length` | `lru_cache` and its caveats |
| `Point` | `__slots__`, and what it costs |
| `running_max` | Monotonic deque; O(n) sliding window |
| `insert_sorted` | `bisect.insort` vs re-sorting |
| `rank_by_frequency` | `Counter`; explicit tuple sort keys |
| `total_size` | Generator expressions inside `sum` |
| `first_match` | `next()` with a default; stopping early |

## Check your understanding

1. **Why is checking membership in a `set` O(1) but O(n) in a `list`?**
   <details><summary>Answer</summary>A <code>set</code> is hash-based —
   it computes where an item *should* be and checks directly. A
   <code>list</code> has no such structure, so membership means scanning
   every element until a match or the end.</details>

2. **Why is `result += word` in a loop quadratic, when a single `+=` is
   cheap?**
   <details><summary>Answer</summary>Strings are immutable — each
   <code>+=</code> allocates a brand new string and copies everything so
   far into it. Over n iterations, that's 1 + 2 + ... + n characters
   copied in total, which is O(n²).</details>

3. **`lru_cache` speeds up `collatz_length`. What's a situation where
   reaching for it would be the wrong call?**
   <details><summary>Answer</summary>When arguments are rarely repeated
   (the cache just wastes memory holding entries that are never reused),
   when arguments are unhashable, or when the function has side effects
   or depends on mutable state the cache can't see changing.</details>

4. **What invariant does the monotonic deque in `running_max` maintain,
   and why does that make it O(n) overall instead of O(n·k)?**
   <details><summary>Answer</summary>It keeps the deque's values in
   decreasing order, popping smaller elements from the back before
   pushing a new one — since each element is pushed and popped at most
   once across the whole run, the total work is O(n), rather than
   rescanning the last k elements at every step.</details>

5. **`Point` uses `__slots__`. What's the actual trade-off — what do you
   gain, and what do you give up?**
   <details><summary>Answer</summary>You gain lower memory per instance
   and slightly faster attribute access, since there's no per-instance
   <code>__dict__</code>. You give up the ability to add new attributes
   at runtime, and lose the default <code>__weakref__</code> unless you
   add it explicitly.</details>

6. **Why does `total_size` use a generator expression instead of a list
   comprehension inside `sum(...)`?**
   <details><summary>Answer</summary>A generator expression produces
   values one at a time as <code>sum</code> consumes them, never holding
   the whole sequence in memory. A list comprehension would build the
   entire intermediate list first, for no benefit when it's only ever
   summed once.</details>

## Interview questions this module answers

- *This function is slow on large inputs. What would you change?*
- *When would you use a `deque` over a `list`?*
- *What is the complexity of `x in my_list`?*
- *How does `lru_cache` work, and when is it a bad idea?*
- *Maximum of every sliding window, better than O(n·k).*
- *What does `__slots__` do?*
- *How would you profile this?*
