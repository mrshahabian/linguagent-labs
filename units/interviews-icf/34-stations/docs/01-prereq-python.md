# Prereq: NamedTuple, recursion, list comprehensions

Three plain-Python tools, no NumPy yet. Every level uses all three.

## `typing.NamedTuple`

A `NamedTuple` is an immutable, positional-or-keyword tuple with named
fields. `stations/api.py` defines four of them (`Reading`, `StationStats`,
`StationGroup`, `GroupStats`). Two things matter for this mock:

**Equality is structural.** Two `StationStats` with the same field values are
`==`, which is why the tests can write
`self.assertEqual(stats["north"], StationStats(1.0, 7.0, 16.0, 4))` instead of
checking each field.

```python
>>> from stations.api import StationStats
>>> StationStats(1.0, 7.0, 16.0, 4) == StationStats(minimum=1.0, maximum=7.0, total=16.0, valid_count=4)
True
```

**A field can default to an empty collection, safely.** `StationGroup.children`
defaults to `()`, not `[]` — a mutable default on a plain function or
dataclass is a classic trap, but `NamedTuple` fields are always immutable, so
`StationGroup("leaf", ("a", "b"))` is a legal leaf node without having to pass
`children` every time.

## Recursion, on a tree

`StationGroup` is recursive in its own type: `children` is a tuple of more
`StationGroup`. Level 3 asks you to write a function over that shape. The
pattern is always:

```python
def visit(node):
    result = handle(node)               # base case: this node's own work
    for child in node.children:
        result = combine(result, visit(child))   # recursive case
    return result
```

A leaf (`children == ()`) makes the `for` loop run zero times, so it falls
out of the recursive case for free — you don't need a separate `if not
node.children: ...` branch.

```python
>>> from stations.api import StationGroup
>>> def count_stations(node):
...     return len(node.stations) + sum(count_stations(c) for c in node.children)
...
>>> leaf = StationGroup("leaf", ("a", "b"))
>>> root = StationGroup("root", (), (leaf,))
>>> count_stations(root)
2
```

## List comprehensions, as the "loop that builds a collection"

Three shapes you'll want without thinking about them:

```python
# transform
celsius = [reading.celsius for reading in readings]

# transform + filter
station_ids_reporting = [r.station_id for r in readings if r.day == 0]

# build a fixed-size grid, one row per station
grid = [[float("nan")] * num_days for _ in station_ids]
```

That last one is the one to get right in level 1: `[[float("nan")] * num_days
for _ in station_ids]` makes a *new* list for every station. Writing
`[[float("nan")] * num_days] * len(station_ids)` instead gives you
`len(station_ids)` references to the *same* inner list — mutate row 0 and
every row changes with it. This is the classic "2D list of lists" trap and it
will cost you a debugging session if you haven't seen it before.

```python
>>> bad = [[0] * 3] * 2
>>> bad[0][0] = 9
>>> bad
[[9, 0, 0], [9, 0, 0]]
>>> good = [[0] * 3 for _ in range(2)]
>>> good[0][0] = 9
>>> good
[[9, 0, 0], [0, 0, 0]]
```
