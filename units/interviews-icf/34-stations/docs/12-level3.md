# Level 3 — recursive station-group aggregation

Write your answer in `levels/level3.py`. Check with `./run 3`.

## The spec

A `StationGroup(name, stations, children)` is one node of a regional
hierarchy — `stations` are ids it owns directly, `children` are more
`StationGroup`s, and a leaf has `children == ()`. A node can have both its
own stations *and* children.

**`aggregate(group, matrix, index) -> GroupStats`**

Combine every reading in `group.stations`, plus every reading in every
descendant, into one `GroupStats(total, count, mean)`. `index` is
`{station_id: row}` — use it to find a station's row in `matrix` (see
`stations.api.station_index`, which builds exactly this dict). A group with
zero valid readings anywhere below it has `mean = 0.0`.

**`find_regional_outliers(group, matrix, index, threshold) -> list[str]`**

The *names* of every subgroup at any depth below `group` (not `group`
itself) whose own aggregate mean differs from `group`'s aggregate mean by
more than `threshold`. Depth-first, in child order. `group`'s mean is
computed once, at the top, and used as the fixed baseline for every
descendant — not each node compared to its immediate parent.

## Worked example

```
ROOT
  MID
    NE = {north, east}
    SW = {south, west}
```

Using level 1's matrix: `aggregate(NE) = GroupStats(16.0, 4, 4.0)`,
`aggregate(SW) = GroupStats(61.0, 8, 7.625)`, and since `MID`'s only child is
the whole tree, `aggregate(ROOT).mean == aggregate(MID).mean == 77/12 ≈
6.417`.

`find_regional_outliers(ROOT, matrix, index, threshold=2.0)`: `MID` differs
from the root mean by `0` (it *is* the root's only child, so it can't
diverge) — not flagged. Recurse into `MID`'s children: `NE` differs by
`|4.0 - 6.417| ≈ 2.417 > 2.0` — flagged. `SW` differs by `|7.625 - 6.417| ≈
1.208`, under threshold — not flagged. Result: `["NE"]`.

Tighten the threshold to `1.0` and `SW` crosses it too: `["NE", "SW"]`.

## Deriving the code

**`aggregate`.** This is the recursion pattern from `01-prereq-python.md`:
handle this node's own stations (the base-case work), then fold in every
child's aggregate (the recursive case). A leaf's `for child in
group.children` loop simply runs zero times — no special leaf branch needed.

```python
def aggregate(group, matrix, index):
    total = 0.0
    count = 0
    for station_id in group.stations:
        row = matrix[index[station_id]]
        valid = row[~np.isnan(row)]
        total += float(np.sum(valid))
        count += int(valid.size)
    for child in group.children:
        child_stats = aggregate(child, matrix, index)
        total += child_stats.total
        count += child_stats.count
    mean = total / count if count else 0.0
    return GroupStats(total=total, count=count, mean=mean)
```

Notice the per-station block is level 1's `station_stats` masking logic,
unchanged — recursion doesn't invent a new rule for NaN, it just applies the
old one at every node.

**`find_regional_outliers`.** Compute the baseline once, outside the
recursion, then walk the tree comparing every descendant against that fixed
number:

```python
def find_regional_outliers(group, matrix, index, threshold):
    root_mean = aggregate(group, matrix, index).mean

    def walk(node):
        outliers = []
        for child in node.children:
            if abs(aggregate(child, matrix, index).mean - root_mean) > threshold:
                outliers.append(child.name)
            outliers.extend(walk(child))
        return outliers

    return walk(group)
```

The inner `walk` closes over `root_mean` and `matrix`/`index` rather than
threading them through every recursive call as extra parameters — either is
correct, but the closure keeps the recursive signature to just `node`.

## Pitfalls

- **Comparing each node to its parent instead of the fixed root.** Passes
  a two-level tree (parent and root coincide) and fails a three-level one.
  `root_mean` is computed exactly once, before recursion starts.
- **Re-deriving stats by hand in `find_regional_outliers` instead of calling
  `aggregate`.** Works until the two implementations drift — call
  `aggregate` again for each child rather than duplicating its logic.
- **A leaf-only base case that skips a node's own `stations` when it also has
  children.** The spec allows both on one node (see `mixed` in the tests) —
  don't structure the recursion as "if leaf: sum stations, else: sum
  children", do both unconditionally.
- **Including `group` itself in the result.** The spec says descendants,
  not `group` — `walk` starts from `group.children`, never yielding
  `group.name`.
