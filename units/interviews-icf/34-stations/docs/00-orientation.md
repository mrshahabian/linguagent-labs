# Orientation

## What this exercise is

You maintain a small weather-station network. Stations report a Celsius
reading most days but not every day — a dropped packet just means no
`Reading` for that station on that day. Across three levels you:

1. turn that sparse list of readings into a dense NumPy matrix and summarise
   it per station,
2. bucket every valid reading into a histogram, find the outlier station, and
   pull a reproducible audit sample,
3. aggregate readings up a tree of regional groups, recursively.

| Level | Adds | The one new idea |
|---|---|---|
| 1 | Build the station×day matrix; per-station stats | `NaN` doesn't compare equal to anything — mask it out before `np.min`/`np.max`/`np.sum`, don't hand them a row that still has `NaN` in it |
| 2 | Histogram, outlier station, reproducible audit sample | broadcasting replaces a pairwise loop; `np.random.RandomState(seed)`, never the global `np.random`, for determinism |
| 3 | Recursive station-group aggregation | recursion carries the same NaN-masking rule down through a tree — a leaf and a branch use the same formula |

Each level is self-contained in `levels/levelN.py`; later levels import the
*types* from earlier levels but not each other's functions.

## Why this shape

This mirrors the other mocks in this repo: the interesting part is reading a
precise specification, not algorithmic cleverness. The twist here is that the
spec is written in terms of NumPy semantics you have to know cold —
**`NaN` propagates through nearly every reduction unless you mask it out
first.** That single fact is behind most of the bugs this mock is designed to
surface.

## How to use this repo

```
stations/     api.py — the shared types (Reading, StationStats, StationGroup,
              GroupStats) and two small helpers. Given code; read it.
levels/       YOU WRITE HERE. Three stubs.
solutions/    reference implementations, commented.
tests/        the worked example from fixtures.py, as executable tests.
docs/         this course.
```

Commands:

```
./run             every level test, against your code in levels/
./run 1           just level 1 (also 2, 3)
./run --check     the same tests against solutions/, to see them pass
```

## The recommended path

1. **`01-prereq-python.md`** — `NamedTuple`, recursion, list comprehensions:
   the three plain-Python tools every level leans on.
2. **`02-prereq-numpy.md`** — arrays, `NaN`, broadcasting, `bincount`,
   `nonzero`/`where`, `RandomState`. Skim what you already know, read closely
   what you don't.
3. `10-level1.md`, `11-level2.md`, `12-level3.md` — one at a time. Read the
   doc, write the code, `./run <level>`, then check `solutions/` only once
   green or stuck for fifteen minutes.
4. **`20-common-bugs.md`** — read after level 1, and again at the end.

## The one thing to internalise first

```python
>>> import numpy as np
>>> row = np.array([1.0, 2.0, float("nan")])
>>> np.min(row)
np.float64(nan)
>>> np.max(row)
np.float64(nan)
>>> np.sum(row)
np.float64(nan)
```

One `NaN` anywhere in the array silently poisons the whole reduction — no
exception, no warning, just a wrong answer that looks like a valid float
until you print it. Every level's stats have to be computed on
`row[~np.isnan(row)]`, never on `row` itself.
