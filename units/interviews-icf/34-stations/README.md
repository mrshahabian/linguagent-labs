# Mock Assessment 3 — Weather Station Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/34-stations/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/icf/stations?utm_source=github&utm_medium=labs)

*45–60 minutes · 26 tests*

A three-level CodeSignal ICF-style exercise built around NumPy instead of
pure standard-library Python: `NamedTuple`, recursion and list
comprehensions on the Python side; `np.array`, broadcasting, `NaN`
semantics, `np.bincount`, `np.nonzero`/`np.where` and
`np.random.RandomState` on the NumPy side.

You maintain a small weather-station network. Stations report a Celsius
reading most days but not every day. Across three levels you turn that
sparse data into a dense matrix and summarise it, bucket it into a
histogram and find the outlier station, and aggregate it up a tree of
regional groups.

| Level | Adds | The one new idea |
|---|---|---|
| 1 | Build the station×day matrix; per-station stats | `NaN` propagates through `np.min`/`np.max`/`np.sum` unless you mask it out first |
| 2 | Histogram, outlier station, reproducible audit sample | broadcasting replaces a pairwise loop; `RandomState(seed)`, not global `np.random`, for determinism |
| 3 | Recursive station-group aggregation | recursion carries the same NaN-masking rule down through a tree |

## This module needs NumPy

Every other mock in this repo is standard-library only; this one is the
exception, because the real assessment it practices is. Install it if
`python3 -c "import numpy"` fails:

```
pip install numpy
```

## Before you start

Same room setup as the other mocks:

- Phone in another room. Notifications off. One monitor.
- **Disable Copilot / any AI completion in your editor.** Check it, don't
  assume it.
- No browser tabs except `https://numpy.org/doc/stable/` and
  `https://docs.python.org/3/`.
- Do not open `solutions/` until you're done, or stuck for fifteen minutes
  on one level.

## Running it

```
./run              every level test, against your code in levels/
./run 1            just level 1 (also 2, 3)
./run --check      the same tests against solutions/, to see them pass
python3 -m stations.demo                          # explore your own code
STATIONS_TARGET=solutions python3 -m stations.demo  # explore the reference
```

`stations.demo` is not graded — it prints the worked-example matrix, stats,
histogram and tree aggregation so you can see intermediate values while
you're debugging, the same way `print(render(...))` works in
[`../21-scheduler/`](../21-scheduler/).

## Two ways to use this

**As a course.** Start at `docs/00-orientation.md`. `01-prereq-python.md`
and `02-prereq-numpy.md` are short and every example in them is runnable —
paste them into a REPL before you touch `levels/`. Then `10-level1.md`
through `12-level3.md`, one at a time. Under an hour end to end.

**As a mock.** 45–60 minutes, timer running. Read only the spec at the top
of each `docs/1*.md`, write in `levels/`, stay out of `solutions/`.

## Layout

```
stations/      api.py (the shared NamedTuples and two helpers — read it),
               demo.py (interactive, ungraded — see intermediate values)
levels/        YOU WRITE HERE. Three stubs.
solutions/     reference implementations, commented.
tests/         the worked example (tests/fixtures.py) as executable tests.
docs/          the course.
```

## Suggested checkpoints

| Elapsed | You should be |
|---|---|
| 0:15 | Level 1 green |
| 0:30 | Level 2 green |
| 0:55 | Level 3 green |

## The one idea this mock is really testing

**`NaN` doesn't announce itself.** It doesn't raise, it doesn't warn, it
just poisons whatever reduction touches it and comes out the other end
looking like a normal (wrong) float. Every level's correctness rests on
masking it out — `row[~np.isnan(row)]` — before calling `np.min`, `np.max`
or `np.sum`, and level 3 is just level 1's masking rule applied recursively
down a tree instead of once per row.

## Afterwards

Write down, before you look at anything: which level cost the most time,
what kind of bug it was (a spec-reading bug vs. a NumPy-semantics bug), and
whether `docs/20-common-bugs.md` would have caught it faster than your own
debugging did.

## Check your understanding

1. **Why does `station_stats` mask a row with `row[~np.isnan(row)]`
   before calling `np.min`/`np.max`/`np.sum`, instead of calling them on
   the raw row directly?**
   <details><summary>Answer</summary>`np.min`, `np.max` and `np.sum` all
   propagate NaN: if even one cell in the row is NaN, the result is NaN,
   silently, with no warning or exception. Masking first drops the
   missing days entirely, so the reduction only ever sees the days that
   actually have a reading — and a station with *no* valid days is
   handled as its own explicit case (reporting zeros) rather than
   letting `np.min` raise on an empty array.</details>

2. **Why does the outlier-detection level use `RandomState(seed)`
   instead of just calling `np.random.choice(...)` directly?**
   <details><summary>Answer</summary><code>np.random.choice</code> draws
   from NumPy's global random state, which is shared and mutated by
   anything else that happens to call a NumPy random function first —
   including a previous test. A <code>RandomState(seed)</code> instance
   is a private generator: the same seed reproduces the same audit
   sample regardless of what else ran before it.</details>

3. **Why does broadcasting replace the pairwise loop in the
   outlier-detection level, rather than just being a faster way to write
   the same loop?**
   <details><summary>Answer</summary>The pairwise version computes one
   station's distance to "the rest" via an explicit loop over station
   pairs; broadcasting expresses the same comparison as one vectorised
   operation across the whole matrix at once, without ever materialising
   an explicit pairwise loop in Python. It's not just a syntax
   difference — it's replacing an O(n) Python loop with a single NumPy
   call operating on the whole array.</details>

4. **Level 3 aggregates readings recursively up a tree. What rule from
   level 1 has to be reapplied at every node, and why would skipping it
   at just one node corrupt the whole result?**
   <details><summary>Answer</summary>The same NaN-masking rule from
   level 1 — a node's own readings and its children's aggregated
   readings both have to be masked before being combined. Skipping the
   mask at even one node lets a NaN slip into that node's result, which
   then poisons every ancestor's aggregate above it, since NaN
   propagates through further reductions rather than being
   contained.</details>

## Related

- [`../08-drills/`](../08-drills/) and [`../21-scheduler/drills/`](../21-scheduler/drills/)
  — typing-speed drills for the plain-Python primitives
  (`02-prereq-sorting.md`-style material); this module doesn't have its
  own drill set.
- [`../09-container/`](../09-container/) — a shorter, standard-library
  warm-up on choosing a data structure.
- [`../20-registry/`](../20-registry/), [`../21-scheduler/`](../21-scheduler/) —
  the standard-library mocks this one is styled after.
- [`../33-trees/`](../33-trees/) — the other NumPy-based unit, debugging
  instead of building.
