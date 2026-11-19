# Level 1 — build the matrix, then per-station stats

Write your answer in `levels/level1.py`. Check with `./run 1`.

## The spec

You're given `station_ids` (a list, defines row order), `num_days`, and
`readings` (a list of `Reading(station_id, day, celsius)` — at most one per
station per day, not necessarily one per station per day).

**`build_matrix(station_ids, num_days, readings) -> np.ndarray`**

Return a `(len(station_ids), num_days)` array. Row `i` is `station_ids[i]`.
A cell with no matching `Reading` is `NaN`. A cell with a matching `Reading`
is that reading's `celsius`.

**`station_stats(matrix, station_ids) -> dict[str, StationStats]`**

One `StationStats(minimum, maximum, total, valid_count)` per station,
computed over that station's non-`NaN` days only. A station with zero valid
days reports `StationStats(0.0, 0.0, 0.0, 0)`.

## Worked example

Four stations, five days (`.` = missing):

```
north  1.0  3.0  5.0    .  7.0
south  2.0    .  2.0  2.0    .
east      .    .    .    .    .
west  10.0 12.0 11.0  9.0 13.0
```

```
station_stats(matrix, station_ids) == {
    "north": StationStats(1.0, 7.0, 16.0, 4),
    "south": StationStats(2.0, 2.0, 6.0, 3),
    "east":  StationStats(0.0, 0.0, 0.0, 0),
    "west":  StationStats(9.0, 13.0, 55.0, 5),
}
```

## Deriving the code

**`build_matrix`.** Start every cell at `NaN`, then overwrite the ones you
have data for. A dict from station id to row index turns "find station X's
row" from a linear scan into a lookup — build it once, not inside the loop
over readings.

```python
index = {station_id: row for row, station_id in enumerate(station_ids)}
grid = [[float("nan")] * num_days for _ in station_ids]
for reading in readings:
    grid[index[reading.station_id]][reading.day] = reading.celsius
return np.array(grid)
```

(See `01-prereq-python.md` for why the grid comprehension has to be
`[[float("nan")] * num_days for _ in station_ids]` and not
`[[float("nan")] * num_days] * len(station_ids)`.)

**`station_stats`.** For each row, mask out `NaN` *before* reducing:

```python
valid = row[~np.isnan(row)]
if valid.size == 0:
    stats[station_id] = StationStats(0.0, 0.0, 0.0, 0)
else:
    stats[station_id] = StationStats(
        minimum=float(np.min(valid)),
        maximum=float(np.max(valid)),
        total=float(np.sum(valid)),
        valid_count=int(valid.size),
    )
```

The `valid.size == 0` branch is not defensive padding — it's load-bearing.
`np.min([])` raises `ValueError: zero-size array to reduction operation
minimum which has no identity`. `east` never reports, so the tests hit this
branch directly; skip it and level 1 crashes on its own worked example.

## Pitfalls

- **Calling `np.min`/`np.max`/`np.sum` on the raw row.** Passes for `west`
  (no `NaN`s), silently returns `nan` for `north` and `south`, and crashes
  outright for `east`. If your `StationStats` for `north` comes back with
  `minimum=nan`, this is why.
- **Filtering with `row != row`.** Reads like "not equal to itself, so it's
  NaN, so drop it" and even sort of works for *finding* NaN, but
  `row[row == row]` (its natural complement) is easy to get backwards under
  time pressure. Use `np.isnan` — it says what it means.
- **The shared-row list bug.** If `build_matrix` returns all-identical rows
  no matter what's in `readings`, you built the grid with
  `[[float("nan")] * num_days] * len(station_ids)`.
- **Returning `np.int` types instead of the spec's Python `float`/`int`.**
  `StationStats` fields are compared with `==` against plain Python numbers
  in the tests; `float(...)` and `int(...)` around the NumPy scalars keep
  that comparison simple. (In practice `np.float64(1.0) == 1.0` is `True`
  too, but don't rely on the exception, get in the habit of unwrapping.)
