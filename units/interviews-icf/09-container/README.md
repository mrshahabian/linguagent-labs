# ICA Pre-certification — Container

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/09-container/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/container?utm_source=github&utm_medium=labs)

CodeSignal's pre-certification practice task, reproduced so you can run it
locally. Two things to get out of it:

1. **The environment.** RUN modes, `main.sh`, partial credit, what
   `basicTests.py` is for. This is the actual point of the pre-cert task.
2. **The task.** A container of integers with `add`, `delete` and
   `get_median` — small, but with three real decisions in it.

## Running it

```
./run              all 13 tests against container.py
./run basic        the 3 playground tests only
./run graded       the 10 scored tests only
./run --check      run everything against solution.py, to see them pass
./main.sh          the same shape as the platform's "Run Custom Script"
```

One single test, the fast loop worth memorising:

```
cd tests && python3 -m unittest basicTests.DemoICATest.test_basic1
```

## The task

The program starts with an empty container.

- `ADD <value>` — add the integer.
- `DELETE <value>` — remove one occurrence if present and return `True`,
  otherwise `False`.
- `GET_MEDIAN` — the middle value once sorted; **for an even count, the
  leftmost of the two middle values**. Raise a runtime exception if empty.

Limits: 3 seconds, 4 GB, and `@timeout(0.1)` per test.

Write your answer in `container.py`. `solution.py` has the reference — read it
after you have attempted it.

## Layout

```
container.py          YOU WRITE HERE
solution.py           reference answer
tests/basicTests.py   3 tests, unscored, yours to modify
tests/containerTests.py  10 scored tests, read-only on the platform
timeout_decorator.py  local shim for the package the platform provides
docs/                 the teaching material
```

## Where to start

- `docs/01-platform.md` — the environment: RUN modes, `main.sh`, partial
  credit, and why you submit early and often. **Read this first**; it is the
  part that transfers to every future assessment.
- `docs/02-median-structures.md` — the task itself: the median index formula,
  three implementations with their trade-offs, and why the textbook two-heap
  answer is the wrong choice here.

## The three things this task actually teaches

**1. The index formula is the whole problem.**

```python
index = (len(values) - 1) // 2
```

`n // 2` gives the *rightmost* of the two middle values and fails every
even-length test. Verify on `[10, 30] → 10` before writing anything else.

**2. Read the tests; they are the specification.** The prose says
"raise a runtime exception"; the docstring comments in the tests say
"Median of [] is None". They disagree — and the assertion,
`assertRaises(Exception, ...)`, is what is checked. Prose is ambiguous, tests
are not.

**3. Pick the structure from the operation set, not from the name.**
"Streaming median" pattern-matches to two heaps. But `delete(value)` removes an
*arbitrary* value, which heaps don't support without lazy-deletion bookkeeping.
That one requirement makes a `bisect`-maintained sorted list both simpler and
better here:

| | add | delete | get_median | lines |
|---|---|---|---|---|
| unsorted + sort per query | O(1) | O(n) | O(n log n) | ~12 |
| **sorted list + `bisect`** | O(n) | O(n) | **O(1)** | ~15 |
| two heaps + lazy deletion | O(log n) | O(log n)* | O(1) | ~60 |

The middle row is the right answer for this task. The top row also passes —
if you are short on time, write it, submit it, then improve it.

## Check your understanding

1. **Why does `n // 2` fail the even-length median test, and what's the
   correct formula?**
   <details><summary>Answer</summary><code>n // 2</code> gives the
   *rightmost* of the two middle values for an even count.
   <code>(len(values) - 1) // 2</code> gives the leftmost, which is what
   the spec asks for — verify on <code>[10, 30]</code>, which should
   give <code>10</code>.</details>

2. **The prose says "raise a runtime exception" for an empty container;
   a test comment says "Median of [] is None." Which one is actually
   checked?**
   <details><summary>Answer</summary>The test's assertion is what's
   graded — read what `assertRaises` or the equivalent actually checks,
   not the prose summary. Where they disagree, the test wins.</details>

3. **"Streaming median" sounds like a two-heaps problem. Why is a
   `bisect`-maintained sorted list the better choice here specifically?**
   <details><summary>Answer</summary><code>delete(value)</code> removes
   an *arbitrary* value, which two heaps don't support without extra
   lazy-deletion bookkeeping. That one requirement in the operation set
   — not the task's name — is what should drive the structure
   choice.</details>

4. **Why is the "unsorted list + sort per query" approach still worth
   writing if you're short on time, given it's not the best answer?**
   <details><summary>Answer</summary>It passes every test — partial
   credit is scored per test, not per elegance. Submitting a working
   O(n log n) answer early, then improving it if time remains, beats
   running out of time on a better approach that never gets
   submitted.</details>

## Related

- `../08-drills/` — typing-speed drills for the primitives.
- `../20-registry/` — a five-level package-registry mock, the standard
  build-from-a-spec shape.
- `../21-scheduler/` — a full five-level ICF assessment with a
  runnable simulator, for when you want the long-form version.
- `../33-trees/` — a NumPy debugging mock, for assessments
  built on `numpy` rather than plain standard library.
- `../34-stations/` — a shorter NumPy build-from-spec mock.
- `../34-stations/` — a NumPy build-from-spec mock, shorter than
  `21-scheduler`.
