# 19 — Testing and debugging patterns

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/19-testing-and-debugging/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/python/testing-and-debugging?utm_source=github&utm_medium=labs)

*~45 minutes · 23 tests*

Every other module in this repo asks you to make tests pass. This one is
about **writing** them — the techniques that make a test suite actually
useful: mocking out the parts you don't control, injecting a fake clock
instead of sleeping, capturing logs instead of scraping stdout, and turning
"debug this by staring at it" into a search you can automate.

`test_exercises.py` in this module is itself the primary teaching material
for two of the seven sections — read it, not just the README.

Work in `exercises.py`. Check your work with `python -m labkit check 19`, or open the notebook (`lab.ipynb`) in Colab.

Each section below ends with **Test it** — the exact command for just that
piece, run from `units/python/19-testing-and-debugging/`.

## What you'll build

| Name | You'll implement | Shows up in real code as |
|---|---|---|
| `is_prime` | `is_prime` | The subject of a table-driven test using `subTest` |
| `fetch_first_success` | `fetch_first_success` | A function shaped for mocking — the client is a parameter, not an import |
| `TTLCache` | `set`, `get` | A time-based cache tested with a fake clock instead of `time.sleep` |
| `divide_safely` | `divide_safely` | Logging a warning instead of printing, so tests can assert on it |
| `find_first_failure` | `find_first_failure` | `git bisect`'s algorithm, generalised to any monotonic predicate |
| `MatrixShapeError`, `matmul_checked` | `matmul_checked` | An error message that names the actual mismatch |

---

## 1. Parametrized tests: `subTest`

```python
CASES = [(0, False), (1, False), (2, True), (3, True), ...]

def test_table(self):
    for n, expected in self.CASES:
        with self.subTest(n=n):
            self.assertEqual(m.is_prime(n), expected)
```

Look at `TestIsPrime` in `test_exercises.py`. Without `subTest`, the first
failing case in the loop would raise and **stop the whole test method** —
you would fix it, rerun, and find out about the *next* failing case one at
a time. `subTest` catches each iteration's assertion separately: a failing
`n=8` is reported as its own labelled failure, and the loop keeps going
through the rest of the table regardless. One test method, one table, full
visibility into every case at once — table-driven testing, in the
vocabulary this repo uses elsewhere for "the spec as executable tests."

**In `exercises.py`:** `is_prime`.

**Test it:** `python3 -m unittest test_exercises.TestIsPrime -v`

## 2. Mocking: functions shaped to be testable

```python
def fetch_first_success(client, ids):
    for item_id in ids:
        try:
            result = client.get(item_id)
        except Exception:
            continue
        if result is not None:
            return result
    return None
```

`client` is a parameter, not a module-level `import requests` used
directly inside the function. That one design choice — **dependency
injection** — is what makes this testable without a network:

```python
from unittest.mock import Mock

client = Mock()
client.get.side_effect = [Exception("boom"), None, "found"]
fetch_first_success(client, [1, 2, 3])   # "found"
```

`Mock().get` is a stand-in callable; `side_effect` set to a list makes
each successive call return the next item, or raise it if the item is an
exception instance. `client.get.assert_any_call(10)` afterwards checks
*how* the mock was used, not just what came back — read
`TestFetchFirstSuccess` in `test_exercises.py` for the full pattern,
including asserting the search stops at the first success rather than
continuing.

A function that reaches into a global `requests` module directly has no
such seam — mocking it means monkeypatching the import itself, which is
exactly the awkward, brittle pattern dependency injection avoids.

**In `exercises.py`:** `fetch_first_success`.

**Test it:** `python3 -m unittest test_exercises.TestFetchFirstSuccess -v`

## 3. Injectable clocks: testing time without waiting

```python
class TTLCache:
    def __init__(self, now_fn=time.time):
        self._now = now_fn
```

Same idea as `client` above, applied to `time.time`. A test that wants to
prove an entry expires after 60 seconds has two options: call
`time.sleep(61)` (slow, and still flaky if the process is scheduled out
for a moment near the boundary), or inject a fake clock it controls
exactly:

```python
now = {"t": 0}
cache = TTLCache(now_fn=lambda: now["t"])
cache.set("a", "value", ttl=10)
now["t"] = 11
cache.get("a")   # None - "11 seconds later" took zero wall-clock time
```

This is the general answer to "how do I test something that depends on
the current time" — inject the clock, don't call the real one.

**In `exercises.py`:** `TTLCache.set`, `TTLCache.get`.

**Test it:** `python3 -m unittest test_exercises.TestTTLCache -v`

## 4. Logging instead of printing, and `assertLogs`

```python
logger = logging.getLogger(__name__)

def divide_safely(a, b):
    if b == 0:
        logger.warning("division by zero: %r / %r", a, b)
        return None
    return a / b
```

`logger.warning(...)` instead of `print(...)` because logging is
structured and controllable: a caller can set the level, redirect it, or
turn it off, none of which is true of stdout. `unittest.TestCase.assertLogs`
is the matching testing tool — it captures records emitted inside the
`with` block at or above a given level, and **raises `AssertionError` if
nothing was logged**, which is what makes it useful both for "this should
log a warning" and, as its own negative-space test, "this should log
nothing":

```python
with self.assertLogs(level="WARNING") as captured:
    divide_safely(1, 0)
self.assertEqual(captured.records[0].levelno, logging.WARNING)
```

**In `exercises.py`:** `divide_safely`.

**Test it:** `python3 -m unittest test_exercises.TestDivideSafely -v`

## 5. Bisection as a debugging technique

```python
def find_first_failure(n, predicate):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

This is `git bisect`'s algorithm, stripped down to the loop underneath
it. Given a **monotonic** predicate — false for a while, then true from
some point on, never flipping back — bisection finds the exact boundary
in `O(log n)` calls to `predicate` instead of checking every index in
order. `git bisect` applies this to "which commit introduced the bug"
with the predicate being "does this commit's build pass the failing
test"; the same shape applies any time you're narrowing down *where*
something started going wrong, over a range of indices, dates, or
config versions, not just commits.

```python
>>> find_first_failure(1000, lambda i: i >= 743)
743                      # found in ~10 calls, not 743
```

**In `exercises.py`:** `find_first_failure`.

**Test it:** `python3 -m unittest test_exercises.TestFindFirstFailure -v`

## 6. Error messages that name the mistake

```python
class MatrixShapeError(ValueError):
    """Raised when two matrices' shapes are incompatible for multiplication."""

def matmul_checked(a, b):
    ...
    if cols_a != rows_b:
        raise MatrixShapeError(
            f"cannot multiply a {rows_a}x{cols_a} matrix by a {rows_b}x{cols_b} "
            f"matrix: {cols_a} (columns of a) != {rows_b} (rows of b)"
        )
```

A bare `IndexError` from the multiply loop itself, or a generic
`"shapes don't match"`, both fail without saying *what* was wrong. This
checks the shapes up front and names **both** of them, plus which
dimensions were compared — the same principle behind every assertion
message in this repo's own test suites (`"membership looks linear -
convert b to a set"` from module 14, for instance). A subclass of
`ValueError` rather than a bare `raise ValueError(...)` also lets a
caller catch specifically `MatrixShapeError` without also swallowing
unrelated value errors.

```python
>>> matmul_checked([[1, 2, 3]], [[1, 2], [3, 4]])
Traceback (most recent call last):
    ...
MatrixShapeError: cannot multiply a 1x3 matrix by a 2x2 matrix: 3 (columns of a) != 2 (rows of b)
```

**In `exercises.py`:** `matmul_checked`.

**Test it:** `python3 -m unittest test_exercises.TestMatmulChecked -v`

---

## The exercises

| Function / class | What it teaches |
|---|---|
| `is_prime` | `subTest` — table-driven tests with per-case visibility |
| `fetch_first_success` | Dependency injection as the thing that makes mocking possible |
| `TTLCache` | Injecting a clock instead of sleeping in tests |
| `divide_safely` | `logging` over `print`; `assertLogs` |
| `find_first_failure` | Bisection as a general debugging technique |
| `matmul_checked` | Error messages that name the actual mismatch |

Every function in this table has a stub in `exercises.py` and test
coverage in `test_exercises.py` — a clean 1:1 mapping.

## Check your understanding

1. **What does `self.subTest(n=n)` buy you over a plain `for` loop with
   no `subTest`?**
   <details><summary>Answer</summary>Without <code>subTest</code>, the
   first failing iteration raises and stops the whole test method —
   later cases in the loop never run that pass. <code>subTest</code>
   catches each iteration's assertion failure separately and reports it
   labelled with the loop variables, so a run shows every failing case
   in the table at once instead of one at a time across repeated
   test-fix-rerun cycles.</details>

2. **Why is `fetch_first_success(client, ids)` easier to test than a
   version that imports a concrete HTTP client at module level and uses
   it directly?**
   <details><summary>Answer</summary>Taking <code>client</code> as a
   parameter means a test can pass a
   <code>unittest.mock.Mock</code> instead — no network call happens,
   and the test controls exactly what each call returns via
   <code>side_effect</code>. A function that reaches for a
   module-level import directly has no such seam; testing it means
   monkeypatching the import itself, which is more brittle.</details>

3. **Why does `TTLCache` take `now_fn` as a parameter instead of calling
   `time.time()` directly inside `set`/`get`?**
   <details><summary>Answer</summary>So a test can substitute a fake
   clock it fully controls, advancing "time" instantly by changing what
   the fake returns — instead of calling <code>time.sleep()</code> for
   real, which is slow and can be flaky under load near a boundary.</details>

4. **What does `unittest.assertLogs` actually assert if nothing inside
   the `with` block calls a logger?**
   <details><summary>Answer</summary>It raises <code>AssertionError</code>
   — <code>assertLogs</code> requires at least one matching log record
   to have been emitted inside the block, so it can also be used
   negatively (wrapped in <code>assertRaises(AssertionError)</code>) to
   prove that a code path does <em>not</em> log anything.</details>

5. **Why does bisection need the predicate to be monotonic, and what
   goes wrong if it isn't?**
   <details><summary>Answer</summary>Bisection decides which half of the
   remaining range to discard based on a single sample at the midpoint,
   trusting that "true" stays true for everything past that point (and
   "false" stays false before it). If the predicate can flip back and
   forth, the half it discards might still contain the boundary it's
   looking for, and the search can return a wrong or arbitrary
   result.</details>

## Interview questions this module answers

- *How would you test a function that calls an external API?*
- *How do you test time-based logic without making the test slow or
  flaky?*
- *What's the advantage of `subTest` over one test method per case, or
  over a loop with no `subTest` at all?*
- *A build has been failing since one of the last 500 commits — how do
  you find which one, fast?*
- *What makes an error message actually useful for debugging, versus
  technically correct?*
