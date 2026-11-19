# The Container task, and choosing a data structure

## The spec

The program starts with an empty container of integers.

- `ADD <value>` — add the integer to the container.
- `DELETE <value>` — remove one occurrence if present, return `True`;
  otherwise return `False`.
- `GET_MEDIAN` — the value in the middle after sorting. **If the length is
  even, return the leftmost of the two middle values.** If the container is
  empty, raise a runtime exception.

Three things the prose implies and the tests confirm:

1. **Duplicates are kept.** `add(5); add(5)` gives two fives, and the first
   `delete(5)` returns `True`, the second `True`, the third `False`.
2. **`delete` reports, it does not raise.** Missing value → `False`.
3. **`get_median` raises, it does not report.** Empty → exception, not `None`.
   (The docstring comments in the tests say "Median of [] is None", but the
   assertion is `assertRaises`. The assertion wins.)

## The index formula — get this right first

For a sorted list of length `n`, the "leftmost of the two middle" median is at

```python
index = (n - 1) // 2
```

Check it at both parities:

| list | n | `(n-1)//2` | value | correct? |
|---|---|---|---|---|
| `[1, 2, 4, 5]` | 4 | 1 | 2 | ✓ leftmost of 2 and 4 |
| `[2, 4, 5]` | 3 | 1 | 4 | ✓ true middle |
| `[10, 30]` | 2 | 0 | 10 | ✓ leftmost |
| `[1]` | 1 | 0 | 1 | ✓ |

Common wrong versions:

| Wrong | Gives | Fails on |
|---|---|---|
| `n // 2` | rightmost of the two middle | any even-length case: `[10,30]` → 30 |
| `len(v) // 2 - 1` | too far left on odd lengths | `[2,4,5]` → 2 |
| `(n + 1) // 2` | off by one for odd n | `[2,4,5]` → 5 |

Write `(n - 1) // 2` and verify it on `[10, 30]` before anything else. One
minute here saves debugging six failing tests that all share one root cause.

## Three implementations

### 1. Unsorted list, sort on every query

```python
def add(self, value):    self._values.append(value)
def delete(self, value):
    if value in self._values:
        self._values.remove(value)
        return True
    return False
def get_median(self):
    if not self._values: raise IndexError
    ordered = sorted(self._values)
    return ordered[(len(ordered) - 1) // 2]
```

| op | cost |
|---|---|
| add | O(1) |
| delete | O(n) |
| get_median | **O(n log n)** |

**This passes this task.** The inputs are small (`test_10` is ~400 operations
on ≤200 elements) and `sorted()` is very fast C code. Do not be too proud to
write it if you are short on time — a passing naive solution beats an elegant
unfinished one.

Its weakness is a read-heavy workload: a query loop that calls `get_median()`
`n` times costs O(n² log n). That is the shape that times out.

### 2. Always-sorted list with `bisect` — the recommended answer

Keep the invariant "`self._values` is always sorted", and the median becomes a
single index lookup.

```python
import bisect

class Container:
    def __init__(self):
        self._values = []

    def add(self, value):
        bisect.insort(self._values, value)

    def delete(self, value):
        i = bisect.bisect_left(self._values, value)
        if i < len(self._values) and self._values[i] == value:
            self._values.pop(i)
            return True
        return False

    def get_median(self):
        if not self._values:
            raise IndexError("get_median() on an empty container")
        return self._values[(len(self._values) - 1) // 2]
```

| op | cost |
|---|---|
| add | O(log n) search + O(n) memmove |
| delete | O(log n) search + O(n) memmove |
| get_median | **O(1)** |

The O(n) memmove is a `memmove` in C, not a Python loop — in practice it is
fast for anything up to tens of thousands of elements. This is the right answer
for this task: short, obviously correct, and O(1) on the operation the task is
named after.

**The `bisect_left` idiom is worth memorising:**

```python
i = bisect.bisect_left(xs, value)
if i < len(xs) and xs[i] == value:      # both halves are needed
    ...
```

`bisect_left` returns the *insertion point*, which is `len(xs)` when the value
is larger than everything. Testing `i < len(xs)` first avoids an `IndexError`;
testing `xs[i] == value` distinguishes "found it" from "here is where it
would go". Getting this wrong is the single most common bug in this task —
usually as `if xs[i] == value` without the bounds check, which crashes on
`delete()` of a value larger than any present.

### 3. Two heaps — the classic interview answer, and wrong here

The textbook streaming-median structure: a max-heap of the lower half and a
min-heap of the upper half, rebalanced so the median is always a heap top.
O(log n) insert, O(1) median.

**It is a poor fit for this task, because of `delete`.** Heaps support removing
the *top*, not an arbitrary value. Supporting `delete(value)` requires lazy
deletion — a counter of pending removals, plus cleanup passes at both tops,
plus size bookkeeping that no longer matches the heap lengths. It is roughly
60 lines instead of 15, and every one of them is a chance to introduce a bug
under time pressure.

Know it exists, be able to say *why you did not use it*, and use `bisect`.
"I chose the sorted list because the operation set includes arbitrary deletion,
which heaps don't support without lazy-deletion bookkeeping" is a strong answer
in a follow-up conversation.

### What about `SortedList`?

`sortedcontainers.SortedList` gives O(log n) for all three. It is not in the
standard library, so assume it is unavailable. Mentioning it is a good
follow-up answer; relying on it is a failed build.

## Choosing, in general

The reasoning that transfers to any ICA task:

1. **Which operation is called most, and which is in the tightest loop?** Here
   `get_median` is in the name and in every test — so make *it* the O(1) one.
2. **What does the operation set forbid?** Arbitrary `delete` rules out plain
   heaps. That single constraint picks the structure.
3. **What is the input size?** ≤200 elements means "correct and simple" beats
   "asymptotically optimal". Check the largest test before optimising.

## Pitfalls checklist

- [ ] `(n - 1) // 2`, not `n // 2` — verify on `[10, 30]` → `10`.
- [ ] Empty container **raises**; missing delete **returns `False`**.
- [ ] Duplicates are independent items; deleting one leaves the rest.
- [ ] `bisect_left` bounds check: `i < len(xs) and xs[i] == value`.
- [ ] Don't `sorted()` inside `add` — that is O(n log n) per insert. Use
      `insort`.
- [ ] Don't keep a separate "is sorted" flag you forget to invalidate.

## Drills

1. Write the median index formula and verify it against all four rows of the
   table above without running code.
2. Implement version 1 (naive) from scratch in under five minutes. Run
   `./run`. It should pass all 13.
3. Now implement version 2 with `bisect`. Run again.
4. Write the `bisect_left` delete idiom from memory. What happens if you drop
   the `i < len(xs)` check, and which test catches it?
5. In `test_10`, 200 items are added and the median queried 51 times. Estimate
   the operation count for versions 1 and 2. At what container size would
   version 1 start to matter?
6. Sketch the two-heap version's `delete`. How do you keep the two heap sizes
   balanced when the deleted value is in the middle of one of them?

<details>
<summary>Answers</summary>

1. See the table — the key case is even length, where `(n-1)//2` picks the
   left of the pair.
4. Without the bounds check, `delete(value)` for a value greater than every
   element gives `i == len(xs)` and `xs[i]` raises `IndexError`.
   `test_07_simpleDeletes2` catches it (`delete(5)` on `[3, 4]`), as does
   `test_10`.
5. Version 1: 200 appends + 51 sorts of ~150 elements ≈ 51 × 150 × log₂150 ≈
   58k comparisons — trivial. Version 2: 200 insorts (~8 comparisons + a
   memmove each) + 51 O(1) lookups. Version 1 starts to hurt once queries are
   frequent *and* n is in the tens of thousands — i.e. when `n_queries × n log n`
   stops fitting in the time limit.
6. You need a count of pending deletions per heap (a `dict`), a logical size
   counter separate from `len(heap)`, a cleanup loop that pops tops that are
   already logically deleted, and a rebalance that uses the logical sizes.
   Roughly four times the code of the `bisect` version, which is exactly why
   it is the wrong choice here.

</details>
