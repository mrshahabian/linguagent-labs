# Common bugs, symptom-first

## `StationStats` field is `nan` instead of a number

You called `np.min`/`np.max`/`np.sum` on a row that still contains `NaN`.
Mask first: `valid = row[~np.isnan(row)]`, then reduce `valid`, not `row`.
See `10-level1.md`.

## `ValueError: zero-size array to reduction operation minimum which has no identity`

`np.min`/`np.max` on an empty array. This is `east` in the worked example —
a station with zero valid readings. Check `valid.size == 0` and return the
zeroed `StationStats` before calling any reduction.

## Histogram is shorter than `high - low`

Missing `minlength=high - low` in the `np.bincount` call. `bincount` only
returns as many bins as the largest index needs; it does not know about
`high` unless you tell it.

## `empty_bins` / `most_anomalous_station` comparison fails against a plain list, values look right when printed

You returned the tuple `np.where(...)` gives you instead of unwrapping it
with `[0]`. `assertEqual(np.where(hist == 0), [0, 4])` fails even though the
numbers are correct, because the left side is `(array([0, 4]),)`.

## `audit_sample` gives a different answer on a second run with the same seed

You used the module-level `np.random.seed(seed)` / `np.random.choice(...)`
instead of a fresh `np.random.RandomState(seed)`. The global generator's
state depends on everything that has called into `np.random` before it —
including other tests in the same process.

## `find_regional_outliers` finds different outliers depending on how deep the tree is

You're comparing each node to its immediate parent's mean instead of the
one fixed `group` (root) mean, computed once before recursion starts. On a
two-level tree parent and root are the same node, so the bug is invisible
until you test a deeper tree — which is why the worked example uses a
three-level one.

## Recursion returns `None` somewhere in the tree

A common cause: the recursive case doesn't `return` the result of folding in
each child, it just calls `aggregate(child, ...)` and discards it. Every
recursive call's return value has to be combined into the running total
before the function returns.

## All rows of `build_matrix`'s output are identical no matter what's in `readings`

`[[float("nan")] * num_days] * len(station_ids)` instead of
`[[float("nan")] * num_days for _ in station_ids]` — the first repeats
*references* to one list; the second builds `len(station_ids)` independent
ones. See `01-prereq-python.md`.
